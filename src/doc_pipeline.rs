use crate::{
    config::InitializeConfig,
    engine::EngineError,
    image_impl::Mat,
    layout::{BBox, LayoutDetector, LayoutOutput, LayoutRegion, LayoutType},
    rusto_ocr::{OcrRunOptions, RustO},
    table::recovery::{plain_text, TableRecovery, TableRecoveryPolicy, TableRecoveryStatus},
    table::{TableDetectorConfig, TableStructureConfig, TableStructureRecognizer},
    types::LayoutConfig,
    TextResult,
};
#[cfg(feature = "use-opencv")]
use opencv::prelude::*;
use std::path::Path;

pub struct DocPipelineConfig {
    pub layout: LayoutConfig,
    pub ocr: InitializeConfig,
    /// Legacy cell-detector setting. SLANet Plus predicts its own cell boxes.
    pub table_detector: Option<TableDetectorConfig>,
    pub table_recognizer: Option<TableStructureConfig>,
}

pub struct DocPipeline {
    layout: Option<LayoutDetector>,
    ocr: RustO,
    table_recognizer: Option<TableStructureRecognizer>,
    recovery_enabled: bool,
    recovery_policy: TableRecoveryPolicy,
    initialization_warnings: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct DocBlock {
    pub region: LayoutRegion,
    /// Plain text is populated for tables too, including on recovery failure.
    pub text: Option<String>,
    pub html: Option<String>,
    /// Indices into `DocResult::original_ocr` (each source belongs to one block).
    pub source_indices: Vec<usize>,
}
#[derive(Debug, Clone)]
pub struct DocTableResult {
    pub region: LayoutRegion,
    /// Translate recovery-local source indices back to the full page.
    pub source_indices: Vec<usize>,
    pub recovery: TableRecovery,
}
pub struct DocResult {
    pub blocks: Vec<DocBlock>,
    /// Unmerged, unmodified full-page OCR boxes. Never replaced by table OCR.
    pub original_ocr: Vec<TextResult>,
    pub plain_text: String,
    pub tables: Vec<DocTableResult>,
    pub warnings: Vec<String>,
}

impl DocPipeline {
    pub fn new(config: DocPipelineConfig) -> Result<Self, EngineError> {
        // OCR is required; layout and table models are optional enrichments.
        let ocr = RustO::initialize(config.ocr)?;
        let mut warnings = Vec::new();
        let layout = match LayoutDetector::new(config.layout) {
            Ok(model) => Some(model),
            Err(e) => {
                warnings.push(format!("Layout disabled: {e}"));
                None
            }
        };
        let recognizer =
            config
                .table_recognizer
                .and_then(|c| match TableStructureRecognizer::new(c) {
                    Ok(model) => Some(model),
                    Err(e) => {
                        warnings.push(format!("Table recovery disabled: {e}"));
                        None
                    }
                });
        if config.table_detector.is_some() {
            warnings
                .push("SLANet Plus uses its own boxes; legacy cell detector is not loaded".into());
        }
        let mut pipeline = Self::with_components(ocr, layout, recognizer);
        pipeline.initialization_warnings = warnings;
        Ok(pipeline)
    }

    /// Supply an explicit layout preset or custom inference sessions.
    pub fn with_components(
        ocr: RustO,
        layout: Option<LayoutDetector>,
        recognizer: Option<TableStructureRecognizer>,
    ) -> Self {
        Self {
            ocr,
            layout,
            table_recognizer: recognizer,
            recovery_enabled: true,
            recovery_policy: TableRecoveryPolicy::default(),
            initialization_warnings: vec![],
        }
    }
    pub fn set_table_recovery_enabled(&mut self, enabled: bool) {
        self.recovery_enabled = enabled;
    }
    pub fn set_table_recovery_policy(&mut self, policy: TableRecoveryPolicy) {
        self.recovery_policy = policy;
    }
    pub fn run<P: AsRef<Path>>(&mut self, path: P) -> Result<DocResult, EngineError> {
        self.run_with_options(path, &OcrRunOptions::default())
    }
    pub fn run_with_options<P: AsRef<Path>>(
        &mut self,
        path: P,
        options: &OcrRunOptions,
    ) -> Result<DocResult, EngineError> {
        options.validate()?;
        let image = crate::image_impl::imread(path)?;
        let ocr = self
            .ocr
            .run_on_mat_with_options(&image, options)?
            .to_text_results();
        let layout = match self.layout.as_mut() {
            Some(model) => model.detect(&image),
            None => Ok(LayoutOutput {
                regions: vec![],
                elapse: 0.0,
            }),
        };
        let mut result = enrich_document(
            &image,
            ocr,
            layout,
            self.table_recognizer.as_mut(),
            self.recovery_enabled,
            &self.recovery_policy,
        );
        result
            .warnings
            .splice(0..0, self.initialization_warnings.clone());
        Ok(result)
    }
}

fn clip_region(region: &mut LayoutRegion, w: i32, h: i32) -> bool {
    let b = &mut region.bbox;
    b.x_min = b.x_min.clamp(0, w);
    b.x_max = b.x_max.clamp(0, w);
    b.y_min = b.y_min.clamp(0, h);
    b.y_max = b.y_max.clamp(0, h);
    b.x_min < b.x_max && b.y_min < b.y_max && region.confidence.is_finite()
}
fn crop(image: &Mat, b: &BBox) -> Result<Mat, EngineError> {
    #[cfg(not(feature = "use-opencv"))]
    {
        crate::image_impl::crop(
            image,
            crate::image_impl::Rect {
                x: b.x_min,
                y: b.y_min,
                width: (b.x_max - b.x_min) as u32,
                height: (b.y_max - b.y_min) as u32,
            },
        )
        .map_err(|e| EngineError::ImageError(e.to_string()))
    }
    #[cfg(feature = "use-opencv")]
    {
        let region =
            opencv::core::Rect::new(b.x_min, b.y_min, b.x_max - b.x_min, b.y_max - b.y_min);
        Ok(Mat::roi(image, region)?.try_clone()?)
    }
}
fn coverage(text: &TextResult, b: &BBox) -> f32 {
    let f = &text.frame;
    if f.width <= 0. || f.height <= 0. {
        return 0.0;
    }
    let x = ((f.left + f.width).min(b.x_max as f32) - f.left.max(b.x_min as f32)).max(0.0);
    let y = ((f.top + f.height).min(b.y_max as f32) - f.top.max(b.y_min as f32)).max(0.0);
    let result = x * y / (f.width * f.height);
    if result.is_finite() {
        result
    } else {
        0.0
    }
}
fn enrich_document(
    image: &Mat,
    ocr: Vec<TextResult>,
    layout: Result<LayoutOutput, EngineError>,
    mut recognizer: Option<&mut TableStructureRecognizer>,
    enabled: bool,
    policy: &TableRecoveryPolicy,
) -> DocResult {
    let mut warnings = vec![];
    let mut regions = match layout {
        Ok(result) => result.regions,
        Err(e) => {
            warnings.push(format!("Layout failed; preserving OCR: {e}"));
            vec![]
        }
    };
    let (w, h) = (image.cols().max(0), image.rows().max(0));
    regions.retain_mut(|r| clip_region(r, w, h));
    regions.sort_by_key(|r| (r.bbox.y_min, r.bbox.x_min));
    let mut members = vec![Vec::new(); regions.len()];
    let mut unassigned = Vec::new();
    for (i, text) in ocr.iter().enumerate() {
        let target = regions
            .iter()
            .enumerate()
            .filter_map(|(j, r)| {
                let score = coverage(text, &r.bbox);
                (score >= 0.5).then_some((j, score, r.layout_type == LayoutType::Table))
            })
            .max_by(|a, b| a.2.cmp(&b.2).then(a.1.total_cmp(&b.1)).then(b.0.cmp(&a.0)));
        if let Some((j, _, _)) = target {
            members[j].push(i);
        } else {
            unassigned.push(i);
        }
    }
    let mut blocks = vec![];
    let mut tables = vec![];
    for (region, indices) in regions.into_iter().zip(members) {
        let source: Vec<_> = indices.iter().map(|&i| ocr[i].clone()).collect();
        let text = plain_text(&source);
        let mut html = None;
        if region.layout_type == LayoutType::Table {
            let local: Vec<_> = source
                .iter()
                .cloned()
                .map(|mut t| {
                    t.frame.left -= region.bbox.x_min as f32;
                    t.frame.top -= region.bbox.y_min as f32;
                    for point in &mut t.box_points {
                        point.0 -= region.bbox.x_min as f32;
                        point.1 -= region.bbox.y_min as f32;
                    }
                    t
                })
                .collect();
            let recovery = if !enabled || recognizer.is_none() {
                TableRecovery::fallback(
                    &local,
                    TableRecoveryStatus::Disabled,
                    "Table model unavailable or recovery disabled",
                )
            } else {
                match crop(image, &region.bbox) {
                    Ok(cropped) => recognizer
                        .as_deref_mut()
                        .unwrap()
                        .recover(&cropped, &local, policy),
                    Err(e) => TableRecovery::fallback(
                        &local,
                        TableRecoveryStatus::InvalidRegion,
                        e.to_string(),
                    ),
                }
            };
            html = recovery.to_html();
            tables.push(DocTableResult {
                region: region.clone(),
                source_indices: indices.clone(),
                recovery,
            });
        }
        let block_text = if source.is_empty() && region.layout_type == LayoutType::Figure {
            None
        } else {
            Some(text)
        };
        blocks.push(DocBlock {
            region,
            text: block_text,
            html,
            source_indices: indices,
        });
    }
    // Layout can miss text or return no regions. Preserve every uncovered box.
    for i in unassigned {
        let f = &ocr[i].frame;
        blocks.push(DocBlock {
            region: LayoutRegion {
                bbox: BBox {
                    x_min: f.left as i32,
                    y_min: f.top as i32,
                    x_max: (f.left + f.width) as i32,
                    y_max: (f.top + f.height) as i32,
                },
                layout_type: LayoutType::Text,
                confidence: ocr[i].score,
            },
            text: Some(ocr[i].text.clone()),
            html: None,
            source_indices: vec![i],
        });
    }
    blocks.sort_by_key(|b| (b.region.bbox.y_min, b.region.bbox.x_min));
    DocResult {
        plain_text: plain_text(&ocr),
        original_ocr: ocr,
        blocks,
        tables,
        warnings,
    }
}

impl DocResult {
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();
        let mut last_was_caption = false;

        for (idx, block) in self.blocks.iter().enumerate() {
            if let Some(html) = &block.html {
                md.push_str(&format!("\n{}\n\n", html));
                last_was_caption = false;
                continue;
            }
            if let Some(text) = &block.text {
                if text.trim().is_empty() {
                    continue;
                }

                match block.region.layout_type {
                    LayoutType::Title => {
                        // Detect title level based on font size heuristics
                        // For now, use bbox height as proxy for font size
                        let height = block.region.bbox.y_max - block.region.bbox.y_min;
                        let level = if height > 100 {
                            1 // # Main title
                        } else if height > 60 {
                            2 // ## Section
                        } else {
                            3 // ### Subsection
                        };
                        let prefix = "#".repeat(level);
                        md.push_str(&format!("{} {}\n\n", prefix, text.trim()));
                    }
                    LayoutType::Header => {
                        // Headers are typically page headers - can be omitted or included
                        // For RapidDoc compatibility, include them but don't make them prominent
                        md.push_str(&format!("{}\n\n", text.trim()));
                    }
                    LayoutType::Footer => {
                        // Footers (page numbers, etc.) - typically skipped in content
                        // but included for completeness
                        md.push_str(&format!("---\n{}\n\n", text.trim()));
                    }
                    LayoutType::FigureCaption => {
                        // Figure captions should be in italics and on separate line
                        md.push_str(&format!("*{}*\n\n", text.trim()));
                        last_was_caption = true;
                    }
                    LayoutType::TableCaption => {
                        // Table captions should be in italics
                        md.push_str(&format!("*{}*\n\n", text.trim()));
                        last_was_caption = true;
                    }
                    LayoutType::Equation => {
                        // Equations - treat as regular text for now (LaTeX formatting ignored)
                        md.push_str(&format!("{}\n\n", text.trim()));
                    }
                    LayoutType::Reference => {
                        // References - keep as plain text but maybe add indicator
                        md.push_str(&format!("{}\n\n", text.trim()));
                    }
                    LayoutType::Text => {
                        // Check if this looks like a list
                        if Self::is_list_line(text) {
                            // Format as markdown list
                            md.push_str(&format!("{}\n", text.trim()));
                        } else {
                            // Regular paragraph
                            md.push_str(&format!("{}\n\n", text.trim()));
                            last_was_caption = false;
                        }
                    }
                    _ => {
                        md.push_str(&format!("{}\n\n", text.trim()));
                        last_was_caption = false;
                    }
                }
            } else {
                // Blocks without text (figures, tables)
                match block.region.layout_type {
                    LayoutType::Figure => {
                        // Check if next block is a figure caption
                        let has_caption = idx + 1 < self.blocks.len()
                            && matches!(
                                self.blocks[idx + 1].region.layout_type,
                                LayoutType::FigureCaption
                            );

                        if !has_caption && !last_was_caption {
                            // No caption, show placeholder
                            md.push_str("![Figure]\n\n");
                        } else if !last_was_caption {
                            // Caption comes after, just show image
                            md.push_str("![Figure]\n");
                        }
                        // If caption was before, it's already added
                        last_was_caption = false;
                    }
                    LayoutType::Table => {
                        // Check if we have HTML table markup (from table recognition)
                        if let Some(html) = &block.html {
                            // Insert HTML table directly (RapidDoc style)
                            md.push_str(&format!("\n{}\n\n", html));
                        } else {
                            // No table recognition - show placeholder
                            md.push_str(
                                "| Table | (Structure recognition not yet implemented) |\n",
                            );
                            md.push_str(
                                "|-------|---------------------------------------------|\n\n",
                            );
                        }
                        last_was_caption = false;
                    }
                    _ => {}
                }
            }
        }

        md
    }

    /// Detect if a line looks like a list item
    fn is_list_line(text: &str) -> bool {
        let trimmed = text.trim();
        // Check for common list patterns
        trimmed.starts_with("- ")
            || trimmed.starts_with("* ")
            || trimmed.starts_with("• ")
            || (trimmed.len() > 2
                && trimmed.chars().next().unwrap().is_ascii_digit()
                && (trimmed.chars().nth(1) == Some('.') || trimmed.chars().nth(1) == Some(')')))
    }
}

#[cfg(all(test, not(feature = "use-opencv")))]
mod recovery_tests {
    use super::*;
    use crate::Frame;
    fn source(text: &str, x: f32, y: f32) -> TextResult {
        let points = [(x, y), (x + 15., y), (x + 15., y + 10.), (x, y + 10.)];
        TextResult {
            text: text.into(),
            score: 0.95,
            frame: Frame::from_points(&points),
            box_points: points,
        }
    }
    fn image() -> Mat {
        Mat::from_rgb8(100, 100, vec![255; 100 * 100 * 3]).unwrap()
    }
    fn region(kind: LayoutType, x0: i32, y0: i32, x1: i32, y1: i32) -> LayoutRegion {
        LayoutRegion {
            bbox: BBox {
                x_min: x0,
                y_min: y0,
                x_max: x1,
                y_max: y1,
            },
            layout_type: kind,
            confidence: 0.9,
        }
    }
    #[test]
    fn missing_or_failed_layout_preserves_all_ocr() {
        let ocr = vec![source("first", 5., 5.), source("last", 5., 80.)];
        for layout in [
            Ok(LayoutOutput {
                regions: vec![],
                elapse: 0.0,
            }),
            Err(EngineError::Engine("unavailable".into())),
        ] {
            let result = enrich_document(
                &image(),
                ocr.clone(),
                layout,
                None,
                true,
                &TableRecoveryPolicy::default(),
            );
            assert_eq!(result.original_ocr, ocr);
            assert_eq!(
                result
                    .blocks
                    .iter()
                    .flat_map(|b| b.source_indices.clone())
                    .collect::<Vec<_>>(),
                [0, 1]
            );
            assert!(result.to_markdown().contains("first"));
            assert!(result.to_markdown().contains("last"));
        }
    }
    #[test]
    fn overlapping_missing_and_invalid_regions_never_drop_or_duplicate_text() {
        let ocr = vec![
            source("outside", 5., 5.),
            source("table", 30., 40.),
            source("tail", 5., 80.),
        ];
        let regions = vec![
            region(LayoutType::Text, 20, 20, 70, 70),
            region(LayoutType::Table, 20, 20, 70, 70),
            region(LayoutType::Table, 80, 80, -10, -10),
            region(LayoutType::Figure, 200, 0, 250, 50),
        ];
        let result = enrich_document(
            &image(),
            ocr.clone(),
            Ok(LayoutOutput {
                regions,
                elapse: 0.0,
            }),
            None,
            true,
            &TableRecoveryPolicy::default(),
        );
        assert_eq!(result.original_ocr, ocr);
        assert_eq!(result.tables.len(), 1);
        assert_eq!(
            result.tables[0].recovery.status,
            TableRecoveryStatus::Disabled
        );
        assert_eq!(result.tables[0].recovery.original_ocr[0].frame.left, 10.0);
        let mut indices: Vec<_> = result
            .blocks
            .iter()
            .flat_map(|b| b.source_indices.clone())
            .collect();
        indices.sort();
        assert_eq!(indices, [0, 1, 2]);
        for text in ["outside", "table", "tail"] {
            assert!(result.to_markdown().contains(text));
        }
    }
    #[test]
    fn runtime_table_failure_returns_readable_text() {
        use crate::engine::InferenceSession;
        use ndarray::ArrayD;
        use std::collections::HashMap;
        struct Broken;
        impl InferenceSession for Broken {
            fn run(&mut self, _: ArrayD<f32>) -> Result<ArrayD<f32>, EngineError> {
                Err(EngineError::Engine("failed".into()))
            }
            fn run_all(
                &mut self,
                _: ArrayD<f32>,
            ) -> Result<HashMap<String, ArrayD<f32>>, EngineError> {
                Err(EngineError::Engine("failed".into()))
            }
            fn run_with_inputs(
                &mut self,
                _: HashMap<String, ArrayD<f32>>,
            ) -> Result<HashMap<String, ArrayD<f32>>, EngineError> {
                Err(EngineError::Engine("failed".into()))
            }
        }
        let mut recognizer =
            TableStructureRecognizer::with_custom_session(TableStructureConfig::default(), Broken);
        let ocr = vec![source("keep", 30., 40.)];
        let layout = Ok(LayoutOutput {
            regions: vec![region(LayoutType::Table, 20, 20, 70, 70)],
            elapse: 0.0,
        });
        let result = enrich_document(
            &image(),
            ocr.clone(),
            layout,
            Some(&mut recognizer),
            true,
            &TableRecoveryPolicy::default(),
        );
        assert_eq!(
            result.tables[0].recovery.status,
            TableRecoveryStatus::ModelError
        );
        assert_eq!(result.original_ocr, ocr);
        assert!(result.blocks[0].html.is_none());
        assert!(result.to_markdown().contains("keep"));
    }
}
