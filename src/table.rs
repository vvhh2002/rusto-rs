use crate::engine::{EngineError, EngineSession, InferenceSession};
use crate::types::EngineConfig;
use ndarray::{Array3, ArrayD};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub mod recovery;
mod slanet;
pub use slanet::TablePrediction;

#[cfg(feature = "use-opencv")]
use opencv::prelude::*;

#[cfg(not(feature = "use-opencv"))]
use crate::image_impl::Mat;

/// Bounding box for table cells
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BBox {
    pub x_min: i32,
    pub y_min: i32,
    pub x_max: i32,
    pub y_max: i32,
}

/// Table cell information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableCell {
    pub bbox: BBox,
    pub row: usize,
    pub col: usize,
    pub row_span: usize,
    pub col_span: usize,
    pub text: String,
}

/// Table row
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableRow {
    pub cells: Vec<TableCell>,
}

/// Complete table structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableStructure {
    pub rows: Vec<TableRow>,
    pub num_rows: usize,
    pub num_cols: usize,
}

impl TableStructure {
    /// Generate HTML table from structure
    pub fn to_html(&self) -> String {
        let mut html = String::from("<table>\n");

        for row in &self.rows {
            html.push_str("  <tr>\n");
            for cell in &row.cells {
                let colspan = if cell.col_span > 1 {
                    format!(" colspan=\"{}\"", cell.col_span)
                } else {
                    String::new()
                };
                let rowspan = if cell.row_span > 1 {
                    format!(" rowspan=\"{}\"", cell.row_span)
                } else {
                    String::new()
                };

                // Escape HTML special characters
                let text = cell
                    .text
                    .replace('&', "&amp;")
                    .replace('<', "&lt;")
                    .replace('>', "&gt;")
                    .replace('"', "&quot;");

                html.push_str(&format!("    <td{}{}>{}</td>\n", colspan, rowspan, text));
            }
            html.push_str("  </tr>\n");
        }

        html.push_str("</table>");
        html
    }

    /// Generate markdown table from structure (simpler format, no spans)
    pub fn to_markdown(&self) -> String {
        if self.rows.is_empty() {
            return String::new();
        }

        let mut md = String::new();

        // Header row
        if let Some(first_row) = self.rows.first() {
            md.push_str("| ");
            for cell in &first_row.cells {
                md.push_str(&format!("{} | ", cell.text.trim()));
            }
            md.push('\n');

            // Separator
            md.push('|');
            for _ in &first_row.cells {
                md.push_str("---|");
            }
            md.push('\n');
        }

        // Data rows
        for row in self.rows.iter().skip(1) {
            md.push_str("| ");
            for cell in &row.cells {
                md.push_str(&format!("{} | ", cell.text.trim()));
            }
            md.push('\n');
        }

        md
    }
}

/// Configuration for table detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableDetectorConfig {
    pub model_path: std::path::PathBuf,
    pub conf_threshold: f32,
    pub iou_threshold: f32,
    pub model_type: TableModelType,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TableModelType {
    Wireless, // Borderless tables
    Wired,    // Bordered tables
}

impl Default for TableDetectorConfig {
    fn default() -> Self {
        Self {
            model_path: std::path::PathBuf::from("models/DocOCR/rtdetr-wireless.rten"),
            conf_threshold: 0.5,
            iou_threshold: 0.5,
            model_type: TableModelType::Wireless,
        }
    }
}

/// RT-DETR table detector
pub struct TableDetector {
    _session: EngineSession,
    _config: TableDetectorConfig,
}

impl TableDetector {
    pub fn new(config: TableDetectorConfig) -> Result<Self, EngineError> {
        let engine_config = EngineConfig::default();
        let session = EngineSession::from_path(&config.model_path, &engine_config)?;

        Ok(Self {
            _session: session,
            _config: config,
        })
    }

    pub fn with_custom_session(
        config: TableDetectorConfig,
        session: impl InferenceSession + 'static,
    ) -> Self {
        Self {
            _session: EngineSession::from_custom(session),
            _config: config,
        }
    }

    /// Detect table cells in an image
    pub fn detect(&mut self, img: &Mat) -> Result<Vec<BBox>, EngineError> {
        // Preprocess image for RT-DETR (640x640)
        let (input_tensor, scale_x, scale_y, orig_w, orig_h) = self.preprocess(img)?;

        // Run inference
        let mut inputs = HashMap::new();
        inputs.insert("image".to_string(), input_tensor);

        let outputs = self._session.run_with_inputs(inputs)?;

        // Postprocess: decode bounding boxes
        self.postprocess(outputs, scale_x, scale_y, orig_w, orig_h)
    }

    fn preprocess(&self, image: &Mat) -> Result<(ArrayD<f32>, f32, f32, i32, i32), EngineError> {
        #[cfg(feature = "use-opencv")]
        let (orig_h, orig_w) = (image.rows(), image.cols());

        #[cfg(not(feature = "use-opencv"))]
        let size = image.size()?;
        #[cfg(not(feature = "use-opencv"))]
        let (orig_h, orig_w) = (size.height, size.width);

        // RT-DETR expects 640x640 input
        let target_size = 640;
        let scale = (target_size as f32) / (orig_h.max(orig_w) as f32);
        let new_h = (orig_h as f32 * scale) as i32;
        let new_w = (orig_w as f32 * scale) as i32;

        #[cfg(not(feature = "use-opencv"))]
        {
            use crate::image_impl::{resize, Size, INTER_LINEAR};

            let mut resized = Mat::default();
            resize(image, &mut resized, Size::new(new_w, new_h), INTER_LINEAR)?;

            // ImageNet normalization
            let mean = [0.485f32, 0.456, 0.406];
            let std = [0.229f32, 0.224, 0.225];

            // Pad to 640x640
            let mut data = vec![0.0f32; 3 * target_size * target_size];

            for y in 0..new_h as usize {
                for x in 0..new_w as usize {
                    let pixel = resized.get_pixel(x as u32, y as u32);
                    for c in 0..3 {
                        let val = pixel[c] as f32 / 255.0;
                        let normalized = (val - mean[c]) / std[c];
                        data[c * target_size * target_size + y * target_size + x] = normalized;
                    }
                }
            }

            let arr = Array3::from_shape_vec((3, target_size, target_size), data)
                .map_err(|e| EngineError::Preprocess(e.to_string()))?;

            Ok((
                arr.insert_axis(ndarray::Axis(0)).into_dyn(),
                scale,
                scale,
                orig_w,
                orig_h,
            ))
        }

        #[cfg(feature = "use-opencv")]
        {
            use opencv::{core, imgproc};

            let mut resized = Mat::default();
            imgproc::resize(
                image,
                &mut resized,
                core::Size::new(new_w, new_h),
                0.0,
                0.0,
                imgproc::INTER_LINEAR,
            )?;

            // ImageNet normalization
            let mean = [0.485f32, 0.456, 0.406];
            let std = [0.229f32, 0.224, 0.225];

            let hw = target_size * target_size;
            let mut data = vec![0.0f32; 3 * hw];

            for y in 0..new_h as usize {
                for x in 0..new_w as usize {
                    let pix = resized.at_2d::<core::Vec3b>(y as i32, x as i32)?;
                    let idx = y * target_size + x;
                    for c in 0..3 {
                        let val = pix[c] as f32 / 255.0;
                        let normalized = (val - mean[c]) / std[c];
                        data[c * hw + idx] = normalized;
                    }
                }
            }

            let arr = Array3::from_shape_vec((3, target_size, target_size), data)
                .map_err(|e| EngineError::Preprocess(e.to_string()))?;

            Ok((
                arr.insert_axis(ndarray::Axis(0)).into_dyn(),
                scale,
                scale,
                orig_w,
                orig_h,
            ))
        }
    }

    fn postprocess(
        &self,
        outputs: HashMap<String, ArrayD<f32>>,
        scale_x: f32,
        scale_y: f32,
        orig_w: i32,
        orig_h: i32,
    ) -> Result<Vec<BBox>, EngineError> {
        // RT-DETR typically outputs: boxes [N, 4] and scores [N, num_classes]
        // Find the output tensors
        let boxes = outputs
            .values()
            .next()
            .ok_or_else(|| EngineError::OutputError("No output tensor found".to_string()))?;

        // A cell detector without a verified output contract must not masquerade
        // as a successful empty detection. SLANet Plus does not need this model.
        let _ = (scale_x, scale_y, orig_w, orig_h, boxes);
        Err(EngineError::OutputError(
            "RT-DETR cell decoding is not implemented; use SLANet Plus joint structure and boxes"
                .into(),
        ))
    }
}

/// Configuration for table structure recognition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableStructureConfig {
    pub model_path: std::path::PathBuf,
    pub model_type: TableModelType,
}

impl Default for TableStructureConfig {
    fn default() -> Self {
        Self {
            model_path: std::path::PathBuf::from("models/DocOCR/slanet-plus.onnx"),
            model_type: TableModelType::Wireless,
        }
    }
}

/// SLANet table structure recognizer
pub struct TableStructureRecognizer {
    _session: EngineSession,
    _config: TableStructureConfig,
}

impl TableStructureRecognizer {
    pub fn new(config: TableStructureConfig) -> Result<Self, EngineError> {
        let engine_config = EngineConfig::default();
        let session = EngineSession::from_path(&config.model_path, &engine_config)?;

        Ok(Self {
            _session: session,
            _config: config,
        })
    }

    pub fn with_custom_session(
        config: TableStructureConfig,
        session: impl InferenceSession + 'static,
    ) -> Self {
        Self {
            _session: EngineSession::from_custom(session),
            _config: config,
        }
    }

    /// Decode SLANet Plus output in crop-pixel coordinates. Unsupported models,
    /// malformed grids and truncated sequences return an error (never a fake cell).
    pub fn recognize(&mut self, img: &Mat) -> Result<TableStructure, EngineError> {
        Ok(self.recognize_with_score(img)?.structure)
    }

    pub fn recognize_with_score(&mut self, img: &Mat) -> Result<TablePrediction, EngineError> {
        let input = slanet::preprocess(img)?;
        // Use the actual single input name (Paddle exports it as `x`).
        let outputs = self._session.run_all(input)?;
        slanet::decode(&outputs, img.cols(), img.rows())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_html_generation() {
        let structure = TableStructure {
            rows: vec![
                TableRow {
                    cells: vec![
                        TableCell {
                            bbox: BBox {
                                x_min: 0,
                                y_min: 0,
                                x_max: 100,
                                y_max: 30,
                            },
                            row: 0,
                            col: 0,
                            row_span: 1,
                            col_span: 1,
                            text: "Header 1".to_string(),
                        },
                        TableCell {
                            bbox: BBox {
                                x_min: 100,
                                y_min: 0,
                                x_max: 200,
                                y_max: 30,
                            },
                            row: 0,
                            col: 1,
                            row_span: 1,
                            col_span: 1,
                            text: "Header 2".to_string(),
                        },
                    ],
                },
                TableRow {
                    cells: vec![
                        TableCell {
                            bbox: BBox {
                                x_min: 0,
                                y_min: 30,
                                x_max: 100,
                                y_max: 60,
                            },
                            row: 1,
                            col: 0,
                            row_span: 1,
                            col_span: 1,
                            text: "Data 1".to_string(),
                        },
                        TableCell {
                            bbox: BBox {
                                x_min: 100,
                                y_min: 30,
                                x_max: 200,
                                y_max: 60,
                            },
                            row: 1,
                            col: 1,
                            row_span: 1,
                            col_span: 1,
                            text: "Data 2".to_string(),
                        },
                    ],
                },
            ],
            num_rows: 2,
            num_cols: 2,
        };

        let html = structure.to_html();
        assert!(html.contains("<table>"));
        assert!(html.contains("<td>Header 1</td>"));
        assert!(html.contains("<td>Data 2</td>"));
        assert!(html.contains("</table>"));
    }
}
