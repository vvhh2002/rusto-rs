use ndarray::{Array2, Array3, Array4};
use std::time::Instant;

#[cfg(feature = "use-opencv")]
use opencv::prelude::*;

#[cfg(not(feature = "use-opencv"))]
use crate::image_impl::Mat;

use crate::engine::{EngineError, EngineSession, InferenceSession};
use crate::types::LayoutConfig;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BBox {
    pub x_min: i32,
    pub y_min: i32,
    pub x_max: i32,
    pub y_max: i32,
}

/// Layout element types
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum LayoutType {
    Text = 0,
    Title = 1,
    Figure = 2,
    FigureCaption = 3,
    Table = 4,
    TableCaption = 5,
    Header = 6,
    Footer = 7,
    Reference = 8,
    Equation = 9,
}

impl LayoutType {
    pub fn from_class(class_id: usize) -> Option<Self> {
        match class_id {
            0 => Some(LayoutType::Text),
            1 => Some(LayoutType::Title),
            2 => Some(LayoutType::Figure),
            3 => Some(LayoutType::FigureCaption),
            4 => Some(LayoutType::Table),
            5 => Some(LayoutType::TableCaption),
            6 => Some(LayoutType::Header),
            7 => Some(LayoutType::Footer),
            8 => Some(LayoutType::Reference),
            9 => Some(LayoutType::Equation),
            _ => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            LayoutType::Text => "text",
            LayoutType::Title => "title",
            LayoutType::Figure => "figure",
            LayoutType::FigureCaption => "figure_caption",
            LayoutType::Table => "table",
            LayoutType::TableCaption => "table_caption",
            LayoutType::Header => "header",
            LayoutType::Footer => "footer",
            LayoutType::Reference => "reference",
            LayoutType::Equation => "equation",
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LayoutRegion {
    pub bbox: BBox,
    pub layout_type: LayoutType,
    pub confidence: f32,
}

pub struct LayoutOutput {
    pub regions: Vec<LayoutRegion>,
    pub elapse: f64,
}

/// Export contract, including labels, inputs and output coordinate space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutModelProfile {
    Legacy,
    PPDocLayoutS,
}

pub struct LayoutDetector {
    session: EngineSession,
    config: LayoutConfig,
    profile: LayoutModelProfile,
}

impl LayoutDetector {
    pub fn new(config: LayoutConfig) -> Result<Self, EngineError> {
        let session = EngineSession::from_path(&config.model_path, &config.engine_cfg)?;
        Ok(Self {
            session,
            config,
            profile: LayoutModelProfile::Legacy,
        })
    }

    /// Construct layout detector with a custom inference session
    pub fn with_custom_session(
        config: LayoutConfig,
        session: impl InferenceSession + 'static,
    ) -> Self {
        Self {
            session: EngineSession::from_custom(session),
            config,
            profile: LayoutModelProfile::Legacy,
        }
    }

    /// PP-DocLayout-S: 23 labels, RGB 480x480, image + scale_factor inputs,
    /// post-NMS boxes already expressed in original-image pixels.
    pub fn pp_doclayout_s(model_path: impl Into<std::path::PathBuf>) -> Result<Self, EngineError> {
        let mut config = LayoutConfig::default(model_path.into());
        config.conf_thresh = 0.5;
        Ok(Self::new(config)?.with_profile(LayoutModelProfile::PPDocLayoutS))
    }

    pub fn with_profile(mut self, profile: LayoutModelProfile) -> Self {
        self.profile = profile;
        if profile == LayoutModelProfile::PPDocLayoutS {
            self.config.target_size = 480;
        }
        self
    }

    pub fn detect(&mut self, image: &Mat) -> Result<LayoutOutput, EngineError> {
        let start = Instant::now();
        if !(1..=4096).contains(&self.config.target_size)
            || self.config.mean.iter().any(|v| !v.is_finite())
            || self.config.std.iter().any(|v| !v.is_finite() || *v <= 0.0)
            || [self.config.conf_thresh, self.config.iou_thresh]
                .iter()
                .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
        {
            return Err(EngineError::Preprocess(
                "Invalid layout configuration".into(),
            ));
        }

        // Store original size
        #[cfg(feature = "use-opencv")]
        let (orig_h, orig_w) = (image.rows(), image.cols());

        #[cfg(not(feature = "use-opencv"))]
        let size = image.size()?;
        #[cfg(not(feature = "use-opencv"))]
        let (orig_h, orig_w) = (size.height, size.width);

        if orig_w <= 0 || orig_h <= 0 {
            return Err(EngineError::InvalidInputShape);
        }
        // Preprocess
        let (input_tensor, scale_w, scale_h) = self.preprocess(image)?;

        // Prepare inputs
        let mut inputs = std::collections::HashMap::new();
        inputs.insert("image".to_string(), input_tensor.into_dyn());

        // im_shape should be [H, W] in the resized space
        let target_size = self.config.target_size as f32;
        let im_shape = Array2::from_shape_vec((1, 2), vec![target_size, target_size])
            .map_err(EngineError::ShapeError)?;
        if self.profile == LayoutModelProfile::Legacy {
            inputs.insert("im_shape".to_string(), im_shape.into_dyn());
        }

        let scale_factor = Array2::from_shape_vec((1, 2), vec![scale_h, scale_w])
            .map_err(EngineError::ShapeError)?;
        inputs.insert("scale_factor".to_string(), scale_factor.into_dyn());

        // Run inference
        let outputs = self.session.run_with_inputs(inputs)?;

        // Postprocess
        let regions = self.postprocess(outputs, orig_w, orig_h, scale_w, scale_h)?;

        let elapse = start.elapsed().as_secs_f64();

        Ok(LayoutOutput { regions, elapse })
    }

    fn preprocess(&self, image: &Mat) -> Result<(Array4<f32>, f32, f32), EngineError> {
        #[cfg(feature = "use-opencv")]
        let (h, w) = (image.rows(), image.cols());

        #[cfg(not(feature = "use-opencv"))]
        let size = image.size()?;
        #[cfg(not(feature = "use-opencv"))]
        let (h, w) = (size.height, size.width);

        // Layout model expects fixed size (e.g. 640x640 or 800x800)
        let target_size = self.config.target_size;
        let new_h = target_size;
        let new_w = target_size;
        let scale_w = target_size as f32 / w as f32;
        let scale_h = target_size as f32 / h as f32;

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
                if self.profile == LayoutModelProfile::PPDocLayoutS {
                    imgproc::INTER_CUBIC
                } else {
                    imgproc::INTER_LINEAR
                },
            )?;

            // Normalize with ImageNet mean/std
            let mean = self.config.mean;
            let std = self.config.std;

            let rows = resized.rows() as usize;
            let cols = resized.cols() as usize;
            let hw = rows * cols;

            let mut data = vec![0.0f32; 3 * hw];

            for i in 0..rows {
                for j in 0..cols {
                    let pix = resized.at_2d::<core::Vec3b>(i as i32, j as i32)?;
                    let idx = i * cols + j;
                    for c in 0..3 {
                        let val = pix[2 - c] as f32 / 255.0;
                        data[c * hw + idx] = (val - mean[c]) / std[c];
                    }
                }
            }

            let arr = Array3::from_shape_vec((3, new_h as usize, new_w as usize), data)
                .map_err(|e| EngineError::Preprocess(e.to_string()))?;

            Ok((arr.insert_axis(ndarray::Axis(0)), scale_w, scale_h))
        }

        #[cfg(not(feature = "use-opencv"))]
        {
            use crate::image_impl::{resize, Size, INTER_LINEAR};

            let mut resized = Mat::default();
            if self.profile == LayoutModelProfile::PPDocLayoutS {
                resized = resize_paddle_cubic(image, new_w, new_h)?;
            } else {
                resize(image, &mut resized, Size::new(new_w, new_h), INTER_LINEAR)?;
            }

            let mean = self.config.mean;
            let std = self.config.std;

            let mut data = vec![0.0f32; 3 * new_h as usize * new_w as usize];

            for y in 0..new_h as usize {
                for x in 0..new_w as usize {
                    let pixel = resized.get_pixel(x as u32, y as u32);
                    for c in 0..3 {
                        let val = pixel[c] as f32 / 255.0;
                        data[c * (new_h as usize * new_w as usize) + y * (new_w as usize) + x] =
                            (val - mean[c]) / std[c];
                    }
                }
            }

            let arr = Array3::from_shape_vec((3, new_h as usize, new_w as usize), data)
                .map_err(|e| EngineError::Preprocess(e.to_string()))?;

            Ok((arr.insert_axis(ndarray::Axis(0)), scale_w, scale_h))
        }
    }

    fn postprocess(
        &self,
        outputs: std::collections::HashMap<String, ndarray::ArrayD<f32>>,
        _orig_w: i32,
        _orig_h: i32,
        scale_w: f32,
        scale_h: f32,
    ) -> Result<Vec<LayoutRegion>, EngineError> {
        // Find the output tensor containing boxes
        // Usually it's the one with shape [N, 6]
        let mut boxes_tensor = None;
        for tensor in outputs.values() {
            let shape = tensor.shape();
            if shape.len() == 2 && shape[1] == 6 {
                boxes_tensor = Some(tensor);
                break;
            }
        }

        let boxes_tensor = match boxes_tensor {
            Some(t) => t,
            None => {
                return Err(EngineError::OutputError(
                    "No suitable output tensor found".to_string(),
                ))
            }
        };

        let mut raw_boxes = Vec::new();
        for row in boxes_tensor.outer_iter() {
            if row.iter().any(|v| !v.is_finite())
                || row[0] < 0.0
                || row[0].fract() != 0.0
                || !(0.0..=1.0).contains(&row[1])
                || row[4] <= row[2]
                || row[5] <= row[3]
            {
                continue;
            }
            let class_id = row[0] as usize;
            let score = row[1];
            if score < self.config.conf_thresh {
                continue;
            }
            let x1 = row[2];
            let y1 = row[3];
            let x2 = row[4];
            let y2 = row[5];

            raw_boxes.push([class_id as f32, score, x1, y1, x2, y2]);
        }

        // Filter by confidence (handled by NMS or manual filter)
        // Assuming model output is already filtered or we filter here
        // NMS
        let indices = crate::geometry::nms(&raw_boxes, self.config.iou_thresh, 0.95);

        let mut regions = Vec::new();
        for idx in indices {
            let row = &raw_boxes[idx];
            let class_id = row[0] as usize;
            let score = row[1];
            let x1 = row[2];
            let y1 = row[3];
            let x2 = row[4];
            let y2 = row[5];

            // Map class ID to LayoutType
            // Mapping based on PP-DocLayout_plus-L
            // 0: Title, 1: Text, 2: Abandon, 3: Figure, 4: FigureCaption, 5: Table, 6: TableCaption,
            // 7: TableFootnote, 8: IsolateFormula, 9: FormulaCaption, 13: InlineFormula, 14: IsolatedFormula, 15: OcrText
            let layout_type = if self.profile == LayoutModelProfile::PPDocLayoutS {
                match class_id {
                    0 | 11 => LayoutType::Title,
                    1 | 16 | 18 | 20 | 21 => LayoutType::Figure,
                    6 | 17 => LayoutType::FigureCaption,
                    7 | 19 => LayoutType::Equation,
                    8 => LayoutType::Table,
                    9 => LayoutType::TableCaption,
                    10 | 12 => LayoutType::Reference,
                    13 => LayoutType::Header,
                    15 => LayoutType::Footer,
                    2..=5 | 14 | 22 => LayoutType::Text,
                    _ => continue,
                }
            } else {
                match class_id {
                    0 => LayoutType::Title,
                    1 | 15 => LayoutType::Text,
                    2 => continue, // Abandon
                    3 => LayoutType::Figure,
                    4 => LayoutType::FigureCaption,
                    5 => LayoutType::Table,
                    6 => LayoutType::TableCaption,
                    7 => LayoutType::Reference, // TableFootnote -> Reference?
                    8 | 13 | 14 => LayoutType::Equation,
                    9 => LayoutType::Text, // FormulaCaption -> Text
                    _ => LayoutType::Text, // Default to Text
                }
            };

            // PP-DocLayout-S performs inverse scaling inside its exported graph.
            let (scale_w, scale_h) = if self.profile == LayoutModelProfile::PPDocLayoutS {
                (1.0, 1.0)
            } else {
                (scale_w, scale_h)
            };
            // Scale back to original image and clip to bounds
            let x_min = ((x1 / scale_w) as i32).max(0).min(_orig_w);
            let y_min = ((y1 / scale_h) as i32).max(0).min(_orig_h);
            let x_max = ((x2 / scale_w) as i32).max(0).min(_orig_w);
            let y_max = ((y2 / scale_h) as i32).max(0).min(_orig_h);

            // Skip invalid boxes
            if x_max <= x_min || y_max <= y_min {
                continue;
            }

            regions.push(LayoutRegion {
                bbox: BBox {
                    x_min,
                    y_min,
                    x_max,
                    y_max,
                },
                layout_type,
                confidence: score,
            });
        }

        Ok(regions)
    }
}

/// OpenCV-style half-pixel cubic sampling (A=-0.75), with fixed 11-bit
/// coefficients and no antialiasing. `image::resize` widens its filter while
/// downsampling, which materially changes the input to PP-DocLayout-S.
#[cfg(not(feature = "use-opencv"))]
fn resize_paddle_cubic(image: &Mat, width: i32, height: i32) -> Result<Mat, EngineError> {
    fn axis(source: i32, target: i32) -> Vec<([i32; 4], [i32; 4])> {
        (0..target)
            .map(|i| {
                let coordinate = (i as f64 + 0.5) * source as f64 / target as f64 - 0.5;
                let base = coordinate.floor() as i32;
                let x = (coordinate - coordinate.floor()) as f32;
                let a = -0.75_f32;
                let c0 = ((a * (x + 1.0) - 5.0 * a) * (x + 1.0) + 8.0 * a) * (x + 1.0) - 4.0 * a;
                let c1 = ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
                let z = 1.0 - x;
                let c2 = ((a + 2.0) * z - (a + 3.0)) * z * z + 1.0;
                let c3 = 1.0 - c0 - c1 - c2;
                (
                    [base - 1, base, base + 1, base + 2].map(|j| j.clamp(0, source - 1)),
                    [c0, c1, c2, c3].map(|c| (c * 2048.0).round_ties_even() as i32),
                )
            })
            .collect()
    }
    let horizontal = axis(image.cols(), width);
    let vertical = axis(image.rows(), height);
    let mut pixels = vec![0u8; width as usize * height as usize * 3];
    for (y, (ys, yc)) in vertical.iter().enumerate() {
        for (x, (xs, xc)) in horizontal.iter().enumerate() {
            let mut sums = [0i64; 3];
            for j in 0..4 {
                for i in 0..4 {
                    let pixel = image.get_pixel(xs[i] as u32, ys[j] as u32);
                    let weight = xc[i] as i64 * yc[j] as i64;
                    for c in 0..3 {
                        sums[c] += pixel[c] as i64 * weight;
                    }
                }
            }
            for c in 0..3 {
                pixels[(y * width as usize + x) * 3 + c] =
                    ((sums[c] + (1 << 21)) >> 22).clamp(0, 255) as u8;
            }
        }
    }
    Mat::from_rgb8(width as u32, height as u32, pixels)
        .map_err(|e| EngineError::ImageError(e.to_string()))
}

#[cfg(all(test, not(feature = "use-opencv")))]
mod tests {
    use super::*;
    use ndarray::ArrayD;
    use std::collections::HashMap;
    struct Session;
    impl InferenceSession for Session {
        fn run(&mut self, _: ArrayD<f32>) -> Result<ArrayD<f32>, EngineError> {
            unreachable!()
        }
        fn run_all(&mut self, _: ArrayD<f32>) -> Result<HashMap<String, ArrayD<f32>>, EngineError> {
            unreachable!()
        }
        fn run_with_inputs(
            &mut self,
            inputs: HashMap<String, ArrayD<f32>>,
        ) -> Result<HashMap<String, ArrayD<f32>>, EngineError> {
            assert_eq!(inputs.len(), 2);
            assert!(!inputs.contains_key("im_shape"));
            assert_eq!(inputs["image"].shape(), [1, 3, 480, 480]);
            assert_eq!(inputs["scale_factor"][[0, 0]], 4.8);
            assert_eq!(inputs["scale_factor"][[0, 1]], 2.4);
            assert!((inputs["image"][[0, 0, 0, 0]] - (1.0 - 0.485) / 0.229).abs() < 1e-4);
            // Already in original-image pixels. The low score and invalid box must disappear.
            let boxes = ndarray::arr2(&[
                [8., 0.95, 20., 10., 180., 90.],
                [8., 0.1, 0., 0., 100., 50.],
                [8., 0.99, 20., 20., 10., 10.],
                [8., f32::NAN, 0., 0., 1., 1.],
            ])
            .into_dyn();
            Ok(HashMap::from([("boxes".into(), boxes)]))
        }
    }
    #[test]
    fn small_layout_profile_uses_correct_inputs_labels_and_original_pixel_boxes() {
        let image = Mat::from_rgb8(200, 100, [255, 0, 0].repeat(200 * 100)).unwrap();
        let mut detector =
            LayoutDetector::with_custom_session(LayoutConfig::default("unused".into()), Session)
                .with_profile(LayoutModelProfile::PPDocLayoutS);
        let out = detector.detect(&image).unwrap();
        assert_eq!(out.regions.len(), 1);
        assert_eq!(out.regions[0].layout_type, LayoutType::Table);
        assert_eq!(out.regions[0].bbox.x_min, 20);
        assert_eq!(out.regions[0].bbox.y_max, 90);
    }
    #[test]
    fn cubic_resize_matches_opencv_reference_without_downsample_antialiasing() {
        let cases: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/cubic-resize.json")).unwrap();
        for case in cases.as_array().unwrap() {
            let w = case["size"][0].as_u64().unwrap() as u32;
            let h = case["size"][1].as_u64().unwrap() as u32;
            let pixels = (0..w * h * 3)
                .map(|i| ((i * 37 + 13) % 256) as u8)
                .collect();
            let image = Mat::from_rgb8(w, h, pixels).unwrap();
            let resized = resize_paddle_cubic(
                &image,
                case["target"][0].as_i64().unwrap() as i32,
                case["target"][1].as_i64().unwrap() as i32,
            )
            .unwrap();
            let reference = case["pixels"].as_array().unwrap();
            for (i, &pixel) in resized.to_rgb8().as_raw().iter().enumerate() {
                assert!(
                    (pixel as i32 - reference[i].as_i64().unwrap() as i32).abs() <= 1,
                    "pixel {i}"
                );
            }
        }
    }
}
