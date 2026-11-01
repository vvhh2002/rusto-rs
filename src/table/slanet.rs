//! SLANet Plus preprocessing and the fixed 50-token Paddle dictionary.
//! Reference: PaddleX table_recognition processors (Apache-2.0).
use super::{BBox, TableCell, TableRow, TableStructure};
use crate::{engine::EngineError, image_impl::Mat};
use ndarray::{Array4, ArrayD};
#[cfg(feature = "use-opencv")]
use opencv::prelude::*;
use std::collections::HashMap;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TablePrediction {
    pub structure: TableStructure,
    /// Mean probability of decoded structure tokens, excluding SOS/EOS.
    pub confidence: f32,
}

fn error(message: &str) -> EngineError {
    EngineError::OutputError(message.into())
}

pub(super) fn preprocess(img: &Mat) -> Result<ArrayD<f32>, EngineError> {
    let (w, h) = (img.cols(), img.rows());
    if w <= 0 || h <= 0 {
        return Err(EngineError::InvalidInputShape);
    }
    const SIDE: usize = 488;
    let ratio = SIDE as f64 / w.max(h) as f64;
    let nw = ((w as f64 * ratio).round_ties_even() as usize).clamp(1, SIDE);
    let nh = ((h as f64 * ratio).round_ties_even() as usize).clamp(1, SIDE);
    let mean = [0.485, 0.456, 0.406];
    let std = [0.229, 0.224, 0.225];
    let mut tensor = Array4::<f32>::zeros((1, 3, SIDE, SIDE));
    // Half-pixel bilinear interpolation, without downsampling antialiasing.
    // Paddle normalizes BGR, then pads the normalized tensor with zero.
    for y in 0..nh {
        let sy = ((y as f64 + 0.5) * h as f64 / nh as f64 - 0.5).max(0.0);
        let y0 = (sy.floor() as i32).min(h - 1);
        let y1 = (y0 + 1).min(h - 1);
        let fy = sy - sy.floor();
        for x in 0..nw {
            let sx = ((x as f64 + 0.5) * w as f64 / nw as f64 - 0.5).max(0.0);
            let x0 = (sx.floor() as i32).min(w - 1);
            let x1 = (x0 + 1).min(w - 1);
            let fx = sx - sx.floor();
            #[cfg(not(feature = "use-opencv"))]
            let pixels = [
                img.get_pixel(x0 as u32, y0 as u32),
                img.get_pixel(x1 as u32, y0 as u32),
                img.get_pixel(x0 as u32, y1 as u32),
                img.get_pixel(x1 as u32, y1 as u32),
            ];
            #[cfg(feature = "use-opencv")]
            let pixels = [
                *img.at_2d::<opencv::core::Vec3b>(y0, x0)?,
                *img.at_2d::<opencv::core::Vec3b>(y0, x1)?,
                *img.at_2d::<opencv::core::Vec3b>(y1, x0)?,
                *img.at_2d::<opencv::core::Vec3b>(y1, x1)?,
            ];
            for c in 0..3 {
                #[cfg(not(feature = "use-opencv"))]
                let k = 2 - c;
                #[cfg(feature = "use-opencv")]
                let k = c;
                let top = pixels[0][k] as f64 * (1.0 - fx) + pixels[1][k] as f64 * fx;
                let bottom = pixels[2][k] as f64 * (1.0 - fx) + pixels[3][k] as f64 * fx;
                let value = (top * (1.0 - fy) + bottom * fy).round().clamp(0., 255.) as f32;
                tensor[[0, c, y, x]] = (value / 255.0 - mean[c]) / std[c];
            }
        }
    }
    Ok(tensor.into_dyn())
}

pub(super) fn decode(
    outputs: &HashMap<String, ArrayD<f32>>,
    w: i32,
    h: i32,
) -> Result<TablePrediction, EngineError> {
    if w <= 0 || h <= 0 {
        return Err(EngineError::InvalidInputShape);
    }
    let find = |last| {
        let tensors: Vec<_> = outputs
            .values()
            .filter(|t| t.ndim() == 3 && t.shape()[0] == 1 && t.shape()[2] == last)
            .collect();
        if tensors.len() == 1 {
            Ok(tensors[0])
        } else {
            Err(error(
                "Expected unique SLANet Plus probability [1,T,50] and box [1,T,8] tensors",
            ))
        }
    };
    let probs = find(50)?;
    let boxes = find(8)?;
    let n = probs.shape()[1];
    if n == 0
        || n > 1024
        || boxes.shape()[1] != n
        || probs.iter().chain(boxes.iter()).any(|v| !v.is_finite())
    {
        return Err(error(
            "Invalid SLANet Plus tensor values or sequence length",
        ));
    }
    let mut rows: Vec<TableRow> = Vec::new();
    let mut row: Option<Vec<TableCell>> = None;
    let mut pending: Option<TableCell> = None;
    let mut open_attributes = false;
    let mut section: Option<usize> = None;
    let mut ended = false;
    let mut score = 0.0;
    let mut count = 0;
    let long = w.max(h) as f32;
    for i in 0..n {
        let mut token = 0;
        for j in 1..50 {
            if probs[[0, i, j]] > probs[[0, i, token]] {
                token = j;
            }
        }
        if token == 49 {
            ended = true;
            break;
        }
        if token == 0 {
            if i == 0 {
                continue;
            }
            return Err(error("Unexpected SOS in SLANet sequence"));
        }
        let probability = probs[[0, i, token]];
        if !(0.0..=1.0).contains(&probability) {
            return Err(error("Invalid token probability"));
        }
        score += probability;
        count += 1;
        match token {
            1 | 3 if section.is_none() && row.is_none() && pending.is_none() => {
                section = Some(token);
            }
            2 | 4 if section == Some(token - 1) && row.is_none() && pending.is_none() => {
                section = None;
            }
            5 if row.is_none() && pending.is_none() => {
                row = Some(Vec::new());
            }
            6 if row.is_some() && pending.is_none() => {
                if rows.len() >= 512 {
                    return Err(error("Too many table rows"));
                }
                rows.push(TableRow {
                    cells: row.take().unwrap(),
                });
            }
            7 | 48 if row.is_some() && pending.is_none() => {
                let x: Vec<_> = (0..4).map(|k| boxes[[0, i, 2 * k]] * long).collect();
                let y: Vec<_> = (0..4).map(|k| boxes[[0, i, 2 * k + 1]] * long).collect();
                let bbox = BBox {
                    x_min: x
                        .iter()
                        .copied()
                        .fold(f32::INFINITY, f32::min)
                        .clamp(0., w as f32) as i32,
                    y_min: y
                        .iter()
                        .copied()
                        .fold(f32::INFINITY, f32::min)
                        .clamp(0., h as f32) as i32,
                    x_max: x
                        .iter()
                        .copied()
                        .fold(f32::NEG_INFINITY, f32::max)
                        .clamp(0., w as f32) as i32,
                    y_max: y
                        .iter()
                        .copied()
                        .fold(f32::NEG_INFINITY, f32::max)
                        .clamp(0., h as f32) as i32,
                };
                if bbox.x_max <= bbox.x_min || bbox.y_max <= bbox.y_min {
                    return Err(error("Degenerate table cell box"));
                }
                let cell = TableCell {
                    bbox,
                    row: rows.len(),
                    col: 0,
                    row_span: 1,
                    col_span: 1,
                    text: String::new(),
                };
                if token == 48 {
                    row.as_mut().unwrap().push(cell);
                } else {
                    pending = Some(cell);
                    open_attributes = true;
                }
            }
            8 if pending.is_some() && open_attributes => {
                open_attributes = false;
            }
            9 if pending.is_some() && !open_attributes => {
                row.as_mut().unwrap().push(pending.take().unwrap());
            }
            10..=28
                if pending.is_some()
                    && open_attributes
                    && pending.as_ref().unwrap().col_span == 1 =>
            {
                pending.as_mut().unwrap().col_span = token - 8;
            }
            29..=47
                if pending.is_some()
                    && open_attributes
                    && pending.as_ref().unwrap().row_span == 1 =>
            {
                pending.as_mut().unwrap().row_span = token - 27;
            }
            _ => return Err(error("Malformed or unsupported SLANet structure tokens")),
        }
    }
    if !ended
        || count == 0
        || rows.is_empty()
        || row.is_some()
        || pending.is_some()
        || section.is_some()
    {
        return Err(error("Incomplete SLANet structure sequence"));
    }
    let num_rows = rows.len();
    let mut occupied = vec![vec![false; 256]; num_rows];
    let mut num_cols = 0;
    let mut cells = 0;
    for (r, row) in rows.iter_mut().enumerate() {
        let mut col = 0;
        for cell in &mut row.cells {
            cells += 1;
            while col < 256 && occupied[r][col] {
                col += 1;
            }
            let end = col + cell.col_span;
            if cells > 4096 || end > 256 || r + cell.row_span > num_rows {
                return Err(error("Table span exceeds grid bounds"));
            }
            cell.col = col;
            for slots in occupied.iter_mut().skip(r).take(cell.row_span) {
                for slot in slots.iter_mut().take(end).skip(col) {
                    if *slot {
                        return Err(error("Overlapping table spans"));
                    }
                    *slot = true;
                }
            }
            num_cols = num_cols.max(end);
            col = end;
        }
    }
    let structure = TableStructure {
        rows,
        num_rows,
        num_cols,
    };
    structure.validate(w, h)?;
    Ok(TablePrediction {
        structure,
        confidence: score / count as f32,
    })
}

impl TableStructure {
    /// Reject incomplete/overlapping grids and boxes outside the crop. A valid
    /// topology is necessary, but does not prove that a model understood a table.
    pub fn validate(&self, width: i32, height: i32) -> Result<(), EngineError> {
        if width <= 0
            || height <= 0
            || self.num_rows == 0
            || self.num_rows > 512
            || self.num_cols == 0
            || self.num_cols > 256
            || self.rows.len() != self.num_rows
        {
            return Err(error("Invalid table dimensions"));
        }
        let mut grid = vec![vec![false; self.num_cols]; self.num_rows];
        let mut count = 0;
        for (r, row) in self.rows.iter().enumerate() {
            let mut previous_end = 0;
            for cell in &row.cells {
                count += 1;
                let end_r = cell
                    .row
                    .checked_add(cell.row_span)
                    .ok_or_else(|| error("Invalid row span"))?;
                let end_c = cell
                    .col
                    .checked_add(cell.col_span)
                    .ok_or_else(|| error("Invalid column span"))?;
                let b = &cell.bbox;
                if count > 4096
                    || cell.row != r
                    || cell.col < previous_end
                    || cell.row_span == 0
                    || cell.col_span == 0
                    || end_r > self.num_rows
                    || end_c > self.num_cols
                    || b.x_min < 0
                    || b.y_min < 0
                    || b.x_max > width
                    || b.y_max > height
                    || b.x_min >= b.x_max
                    || b.y_min >= b.y_max
                {
                    return Err(error("Invalid table cell or span"));
                }
                for slots in grid.iter_mut().take(end_r).skip(r) {
                    for slot in slots.iter_mut().take(end_c).skip(cell.col) {
                        if *slot {
                            return Err(error("Overlapping table grid cells"));
                        }
                        *slot = true;
                    }
                }
                previous_end = end_c;
            }
        }
        if grid.iter().flatten().any(|v| !v) {
            return Err(error("Incomplete table grid"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn tensors(tokens: &[usize]) -> HashMap<String, ArrayD<f32>> {
        let mut p = ndarray::Array3::zeros((1, tokens.len(), 50));
        let mut b = ndarray::Array3::zeros((1, tokens.len(), 8));
        for (i, &token) in tokens.iter().enumerate() {
            p[[0, i, token]] = 0.99;
            for (j, v) in [0.1, 0.1, 0.4, 0.1, 0.4, 0.3, 0.1, 0.3]
                .into_iter()
                .enumerate()
            {
                b[[0, i, j]] = v;
            }
        }
        HashMap::from([
            ("any_probability_name".into(), p.into_dyn()),
            ("any_box_name".into(), b.into_dyn()),
        ])
    }
    #[test]
    fn decodes_cells_and_spans_in_crop_pixels() {
        let result = decode(
            &tensors(&[3, 5, 7, 10, 8, 9, 6, 5, 48, 48, 6, 4, 49]),
            200,
            100,
        )
        .unwrap();
        assert_eq!(
            (result.structure.num_rows, result.structure.num_cols),
            (2, 2)
        );
        assert_eq!(result.structure.rows[0].cells[0].col_span, 2);
        assert_eq!(result.structure.rows[0].cells[0].bbox.x_max, 80);
        assert_eq!(result.structure.rows[0].cells[0].bbox.y_max, 60);
        assert!((result.confidence - 0.99).abs() < 1e-6);
        let result = decode(
            &tensors(&[3, 5, 7, 29, 8, 9, 48, 6, 5, 48, 6, 4, 49]),
            200,
            100,
        )
        .unwrap();
        assert_eq!(result.structure.rows[1].cells[0].col, 1);
    }
    #[test]
    fn rejects_truncation_bad_topology_and_nonfinite_outputs() {
        for tokens in [
            &[3, 5, 48, 6, 4][..],
            &[3, 5, 7, 29, 8, 9, 6, 4, 49],
            &[3, 5, 48, 48, 6, 5, 48, 6, 4, 49],
            &[3, 5, 8, 9, 6, 4, 49],
        ] {
            assert!(decode(&tensors(tokens), 200, 100).is_err());
        }
        let mut outputs = tensors(&[3, 5, 48, 6, 4, 49]);
        outputs.get_mut("any_box_name").unwrap()[[0, 2, 0]] = f32::NAN;
        assert!(decode(&outputs, 200, 100).is_err());
        let outputs = HashMap::from([(
            "unsupported_slanext".into(),
            ndarray::Array3::zeros((1, 10, 10)).into_dyn(),
        )]);
        assert!(decode(&outputs, 200, 100).is_err());
    }
    #[test]
    #[cfg(not(feature = "use-opencv"))]
    fn preprocessing_preserves_aspect_ratio_bgr_normalization_and_zero_padding() {
        let img = Mat::from_rgb8(2, 1, vec![255, 0, 0, 255, 0, 0]).unwrap();
        let t = preprocess(&img).unwrap();
        assert_eq!(t.shape(), [1, 3, 488, 488]);
        assert!((t[[0, 0, 0, 0]] - (0.0 - 0.485) / 0.229).abs() < 1e-5);
        assert!((t[[0, 2, 0, 0]] - (1.0 - 0.406) / 0.225).abs() < 1e-5);
        assert_eq!(t[[0, 0, 244, 0]], 0.0);
        assert_eq!(t[[0, 2, 487, 487]], 0.0);
    }
    #[test]
    fn real_paddle_captures_reject_incomplete_grids_and_decode_complete_grid() {
        // Only model geometry/tokens are stored, never the source document text.
        let cases: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/fixtures/slanet-plus-real.json"))
                .unwrap();
        for case in cases.as_array().unwrap() {
            let sequence = case["sequence"].as_array().unwrap();
            let mut probabilities = ndarray::Array3::zeros((1, sequence.len(), 50));
            let mut boxes = ndarray::Array3::zeros((1, sequence.len(), 8));
            for (i, item) in sequence.iter().enumerate() {
                probabilities[[0, i, item["token"].as_u64().unwrap() as usize]] =
                    item["probability"].as_f64().unwrap() as f32;
                if let Some(values) = item["box"].as_array() {
                    for (j, value) in values.iter().enumerate() {
                        boxes[[0, i, j]] = value.as_f64().unwrap() as f32;
                    }
                }
            }
            let outputs = HashMap::from([
                ("structure".into(), probabilities.into_dyn()),
                ("coordinates".into(), boxes.into_dyn()),
            ]);
            let result = decode(
                &outputs,
                case["size"][0].as_i64().unwrap() as i32,
                case["size"][1].as_i64().unwrap() as i32,
            );
            assert_eq!(
                result.is_ok(),
                case["expected_valid"].as_bool().unwrap(),
                "{}: {:?}",
                case["id"],
                result
            );
            if let Ok(prediction) = result {
                assert_eq!(
                    (prediction.structure.num_rows, prediction.structure.num_cols),
                    (11, 5)
                );
                assert_eq!(
                    prediction
                        .structure
                        .rows
                        .iter()
                        .map(|r| r.cells.len())
                        .sum::<usize>(),
                    51
                );
            }
        }
    }
}
