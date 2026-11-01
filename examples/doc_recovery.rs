//! cargo run --release --example doc_recovery -- OCR_DIR LAYOUT.onnx SLANET.onnx IMAGE...
//! No models are downloaded implicitly. JSON includes the preserved source OCR.
use clap::Parser;
use rusto::table::{TableStructureConfig, TableStructureRecognizer};
use rusto::{
    DetectionRunOptions, DocPipeline, InitializeConfig, LayoutDetector, OcrRunOptions, RustO,
};
use serde_json::json;
#[derive(Parser)]
struct Args {
    ocr_dir: PathBuf,
    layout: PathBuf,
    table: PathBuf,
    #[arg(required = true)]
    images: Vec<PathBuf>,
    #[arg(long,default_value="slow",value_parser=["fast","slow"])]
    score_mode: String,
}
use std::path::PathBuf;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let models = args.ocr_dir;
    let ocr = RustO::initialize(
        InitializeConfig::ppv6(
            models.join("det.rten"),
            models.join("rec.rten"),
            models.join("dict.txt"),
        )
        .with_rec_batch_num(1),
    )?;
    let layout = match LayoutDetector::pp_doclayout_s(&args.layout) {
        Ok(m) => Some(m),
        Err(e) => {
            eprintln!("Layout unavailable: {e}");
            None
        }
    };
    let tables = match TableStructureRecognizer::new(TableStructureConfig {
        model_path: args.table,
        ..Default::default()
    }) {
        Ok(m) => Some(m),
        Err(e) => {
            eprintln!("Tables unavailable: {e}");
            None
        }
    };
    let mut pipeline = DocPipeline::with_components(ocr, layout, tables);
    let options = OcrRunOptions {
        postprocess: Some(rusto::PostprocessRunOptions {
            score_mode: Some(args.score_mode),
            ..Default::default()
        }),
        max_side_len: Some(3000.0),
        detection: Some(DetectionRunOptions {
            limit_side_len: Some(2000),
            limit_type: Some("max".into()),
            ..Default::default()
        }),
        ..Default::default()
    };
    for path in &args.images {
        let start = std::time::Instant::now();
        let result = pipeline.run_with_options(path, &options)?;
        let tables:Vec<_>=result.tables.iter().map(|t|json!({"region":t.region,"source_indices":t.source_indices,"recovery":t.recovery})).collect();
        let blocks:Vec<_>=result.blocks.iter().map(|b|json!({"region":b.region,"text":b.text,"html":b.html,"source_indices":b.source_indices})).collect();
        println!(
            "{}",
            json!({"image":path,"seconds":start.elapsed().as_secs_f64(),"raw":result.original_ocr,
            "text":result.plain_text,"markdown":result.to_markdown(),"tables":tables,"blocks":blocks,"warnings":result.warnings})
        );
    }
    Ok(())
}
