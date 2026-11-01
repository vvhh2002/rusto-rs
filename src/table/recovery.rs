//! Optional structure enrichment. OCR text and source geometry are always kept.
use super::{TablePrediction, TableStructure, TableStructureRecognizer};
use crate::{engine::EngineError, image_impl::Mat, TextResult};
#[cfg(feature = "use-opencv")]
use opencv::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy)]
pub struct TableRecoveryPolicy {
    pub min_structure_confidence: f32,
    /// Fraction of each OCR box that must lie in its assigned cell.
    pub min_text_coverage: f32,
    /// Required separation between the best and second-best cell coverage.
    pub ambiguity_margin: f32,
}
impl Default for TableRecoveryPolicy {
    fn default() -> Self {
        Self {
            min_structure_confidence: 0.85,
            min_text_coverage: 0.8,
            ambiguity_margin: 0.1,
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TableRecoveryStatus {
    Restored,
    Disabled,
    InvalidPolicy,
    InvalidRegion,
    ModelError,
    LowConfidence,
    InvalidStructure,
    NoText,
    UnmatchedText,
    AmbiguousText,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellAssignment {
    pub row: usize,
    pub col: usize,
    pub source_indices: Vec<usize>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableRecovery {
    pub status: TableRecoveryStatus,
    pub reason: Option<String>,
    /// Unmodified input, in crop-pixel coordinates, including on success.
    pub original_ocr: Vec<TextResult>,
    pub plain_text: String,
    pub structure: Option<TableStructure>,
    /// Each nonempty source box appears exactly once on success.
    pub assignments: Vec<CellAssignment>,
}
impl TableRecovery {
    pub fn fallback(
        ocr: &[TextResult],
        status: TableRecoveryStatus,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            status,
            reason: Some(reason.into()),
            original_ocr: ocr.to_vec(),
            plain_text: plain_text(ocr),
            structure: None,
            assignments: vec![],
        }
    }
    pub fn to_html(&self) -> Option<String> {
        self.structure.as_ref().map(TableStructure::to_html)
    }
}
pub(crate) fn plain_text(ocr: &[TextResult]) -> String {
    crate::text_layout::merge_line_fragments(ocr, 0.5, 0.4)
        .iter()
        .map(|t| t.text.as_str())
        .collect::<Vec<_>>()
        .join("\n")
}

impl TableStructureRecognizer {
    /// `ocr` must describe this same crop. No second OCR pass is performed.
    pub fn recover(
        &mut self,
        image: &Mat,
        ocr: &[TextResult],
        policy: &TableRecoveryPolicy,
    ) -> TableRecovery {
        if image.cols() <= 0 || image.rows() <= 0 {
            return TableRecovery::fallback(
                ocr,
                TableRecoveryStatus::InvalidRegion,
                "Empty table crop",
            );
        }
        let prediction = self.recognize_with_score(image);
        recover_prediction(ocr, prediction, image.cols(), image.rows(), policy)
    }
}

/// Recover a grid from a model prediction, including custom inference backends.
/// Any uncertain assignment rejects the entire structure rather than moving text
/// into the nearest unrelated cell. This does not certify semantic correctness.
pub fn recover_prediction(
    ocr: &[TextResult],
    prediction: Result<TablePrediction, EngineError>,
    width: i32,
    height: i32,
    policy: &TableRecoveryPolicy,
) -> TableRecovery {
    use TableRecoveryStatus::*;
    let fallback = |status, reason: String| TableRecovery::fallback(ocr, status, reason);
    if [
        policy.min_structure_confidence,
        policy.min_text_coverage,
        policy.ambiguity_margin,
    ]
    .iter()
    .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
        || policy.min_text_coverage == 0.0
    {
        return fallback(InvalidPolicy, "Invalid recovery thresholds".into());
    }
    let prediction = match prediction {
        Ok(p) => p,
        Err(e @ EngineError::OutputError(_)) => return fallback(InvalidStructure, e.to_string()),
        Err(e) => return fallback(ModelError, e.to_string()),
    };
    if !prediction.confidence.is_finite()
        || prediction.confidence > 1.0
        || prediction.confidence < policy.min_structure_confidence
    {
        return fallback(LowConfidence, "Structure confidence below threshold".into());
    }
    let mut structure = prediction.structure;
    if let Err(e) = structure.validate(width, height) {
        return fallback(InvalidStructure, e.to_string());
    }
    let cells: Vec<_> = structure.rows.iter().flat_map(|r| r.cells.iter()).collect();
    let mut members = vec![Vec::new(); cells.len()];
    let mut text_count = 0;
    for (source_index, text) in ocr.iter().enumerate() {
        if text.text.trim().is_empty() {
            continue;
        }
        text_count += 1;
        let f = &text.frame;
        let right = f.left + f.width;
        let bottom = f.top + f.height;
        if [f.left, f.top, f.width, f.height, right, bottom]
            .iter()
            .any(|v| !v.is_finite())
            || f.width <= 0.0
            || f.height <= 0.0
        {
            return fallback(UnmatchedText, "Invalid source OCR geometry".into());
        }
        let cx = f.left + f.width / 2.0;
        let cy = f.top + f.height / 2.0;
        let mut scores: Vec<_> = cells
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let b = &c.bbox;
                let overlap_w = (right.min(b.x_max as f32) - f.left.max(b.x_min as f32)).max(0.0);
                let overlap_h = (bottom.min(b.y_max as f32) - f.top.max(b.y_min as f32)).max(0.0);
                let coverage = overlap_w * overlap_h / (f.width * f.height);
                let centered = cx >= b.x_min as f32
                    && cx <= b.x_max as f32
                    && cy >= b.y_min as f32
                    && cy <= b.y_max as f32;
                (i, coverage, centered)
            })
            .collect();
        scores.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        let best = scores[0];
        if !best.2 || best.1 < policy.min_text_coverage {
            return fallback(
                UnmatchedText,
                format!("Source box {source_index} does not fit a cell"),
            );
        }
        if scores
            .get(1)
            .is_some_and(|s| best.1 - s.1 <= policy.ambiguity_margin)
        {
            return fallback(
                AmbiguousText,
                format!("Source box {source_index} fits multiple cells"),
            );
        }
        members[best.0].push(source_index);
    }
    if text_count == 0 {
        return fallback(
            NoText,
            "No OCR text available to verify cell assignments".into(),
        );
    }
    let mut assignments = Vec::new();
    for (cell, indices) in structure
        .rows
        .iter_mut()
        .flat_map(|r| r.cells.iter_mut())
        .zip(members)
    {
        let source: Vec<_> = indices.iter().map(|&i| ocr[i].clone()).collect();
        let order = crate::text_layout::line_groups(&source, 0.5)
            .into_iter()
            .flatten()
            .map(|i| indices[i])
            .collect();
        cell.text = plain_text(&source);
        assignments.push(CellAssignment {
            row: cell.row,
            col: cell.col,
            source_indices: order,
        });
    }
    TableRecovery {
        status: Restored,
        reason: None,
        original_ocr: ocr.to_vec(),
        plain_text: plain_text(ocr),
        structure: Some(structure),
        assignments,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        table::{BBox, TableCell, TableRow},
        Frame,
    };
    fn source(s: &str, x: f32) -> TextResult {
        let points = [(x, 5.), (x + 30., 5.), (x + 30., 20.), (x, 20.)];
        TextResult {
            text: s.into(),
            score: 0.9,
            frame: Frame::from_points(&points),
            box_points: points,
        }
    }
    fn prediction() -> TablePrediction {
        TablePrediction {
            confidence: 0.95,
            structure: TableStructure {
                num_rows: 1,
                num_cols: 2,
                rows: vec![TableRow {
                    cells: (0..2)
                        .map(|c| TableCell {
                            bbox: BBox {
                                x_min: c * 50,
                                y_min: 0,
                                x_max: (c + 1) * 50,
                                y_max: 30,
                            },
                            row: 0,
                            col: c as usize,
                            row_span: 1,
                            col_span: 1,
                            text: String::new(),
                        })
                        .collect(),
                }],
            },
        }
    }
    #[test]
    fn restores_multilingual_text_and_source_mapping_without_re_ocr() {
        for (a, b) in [
            ("文档", "归档"),
            ("文件", "歸檔"),
            ("Document", "Archive"),
            ("Été", "Français"),
            ("Größe", "Straße"),
            ("문서", "저장"),
            ("書類", "保存"),
        ] {
            let original = vec![source(b, 60.), source(a, 5.)];
            let result = recover_prediction(
                &original,
                Ok(prediction()),
                100,
                30,
                &TableRecoveryPolicy::default(),
            );
            assert_eq!(result.status, TableRecoveryStatus::Restored);
            assert_eq!(result.original_ocr, original);
            assert_eq!(result.assignments[0].source_indices, [1]);
            assert_eq!(result.assignments[1].source_indices, [0]);
            assert!(result.to_html().unwrap().contains(a));
        }
    }
    #[test]
    fn errors_and_low_confidence_preserve_every_source() {
        let original = vec![source("keep me", 5.)];
        let mut low = prediction();
        low.confidence = 0.1;
        for (p, status) in [
            (
                Err(EngineError::Engine("offline".into())),
                TableRecoveryStatus::ModelError,
            ),
            (Ok(low), TableRecoveryStatus::LowConfidence),
        ] {
            let result = recover_prediction(&original, p, 100, 30, &TableRecoveryPolicy::default());
            assert_eq!(result.status, status);
            assert_eq!(result.original_ocr, original);
            assert_eq!(result.plain_text, "keep me");
            assert!(result.structure.is_none());
        }
    }
    #[test]
    fn rejects_unmatched_ambiguous_and_invalid_grids() {
        let original = vec![source("boundary", 40.)];
        let result = recover_prediction(
            &original,
            Ok(prediction()),
            100,
            30,
            &TableRecoveryPolicy::default(),
        );
        assert_eq!(result.status, TableRecoveryStatus::UnmatchedText);
        let mut p = prediction();
        p.structure.rows[0].cells[0].bbox.x_max = 100;
        let result = recover_prediction(
            &[source("ambiguous", 60.)],
            Ok(p),
            100,
            30,
            &TableRecoveryPolicy::default(),
        );
        assert_eq!(result.status, TableRecoveryStatus::AmbiguousText);
        let mut p = prediction();
        p.structure.rows[0].cells[0].row_span = 2;
        let result = recover_prediction(&original, Ok(p), 100, 30, &TableRecoveryPolicy::default());
        assert_eq!(result.status, TableRecoveryStatus::InvalidStructure);
    }
    #[test]
    fn html_escapes_text_and_policy_cannot_disable_finite_checks() {
        let original = vec![source("<&\"", 5.)];
        let result = recover_prediction(
            &original,
            Ok(prediction()),
            100,
            30,
            &TableRecoveryPolicy::default(),
        );
        assert!(result.to_html().unwrap().contains("&lt;&amp;&quot;"));
        let p = TableRecoveryPolicy {
            min_text_coverage: f32::NAN,
            ..Default::default()
        };
        assert_eq!(
            recover_prediction(&original, Ok(prediction()), 100, 30, &p).status,
            TableRecoveryStatus::InvalidPolicy
        );
    }
}
