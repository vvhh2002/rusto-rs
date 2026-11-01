//! Enrich existing OCR for an explicitly selected table crop. No layout/OCR inference.
//! cargo run --release --example recover_table_crop -- MODEL.onnx CROP.png OCR.json
//! OCR.json contains a Vec<TextResult> in crop-pixel coordinates.
use rusto::table::{recovery::TableRecoveryPolicy, TableStructureConfig, TableStructureRecognizer};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        return Err("usage: recover_table_crop MODEL.onnx CROP.png OCR.json".into());
    }
    let image = rusto::image_impl::imread(&args[1])?;
    let ocr: Vec<rusto::TextResult> = serde_json::from_str(&std::fs::read_to_string(&args[2])?)?;
    let mut model = TableStructureRecognizer::new(TableStructureConfig {
        model_path: args[0].clone().into(),
        ..Default::default()
    })?;
    let result = model.recover(&image, &ocr, &TableRecoveryPolicy::default());
    println!(
        "{}",
        serde_json::json!({"recovery":result,"html":result.to_html()})
    );
    Ok(())
}
