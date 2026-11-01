# Conservative OCR rows and optional table recovery

This branch fixes row grouping and provides **experimental SLANet Plus table
recovery** for the Rust API. It is not a complete PP-Structure pipeline. There are
no implicit model downloads and no changes to the Swift/C/UniFFI API in this work.

## OCR rows

`Lines`, `Words` and `Spatial` now group using local box geometry rather than a
page-wide median height. Rows require strong vertical overlap, compatible heights
and nonoverlapping horizontal extents. Every fragment must agree with every other
fragment in its row, so intermediate boxes cannot bridge neighboring rows.
Nearby fragments can still join; columns with separate boxes are kept apart.
Spatial character width counts Unicode scalar values, not UTF-8 bytes.

This is conservative geometry, not a learned reading-order model. Very slanted
text, superscripts and ambiguous columns may stay separate. No recognized
characters are removed by this step. `DocResult.original_ocr` retains the original
unmerged boxes for search, selection, reprocessing and auditing.

## Supported models

- **PP-DocLayout-S**: use `LayoutDetector::pp_doclayout_s(path)` or the explicit
  `LayoutModelProfile::PPDocLayoutS` with a custom session. The profile supplies
  480×480 RGB normalization, OpenCV-compatible cubic sampling without downsample
  antialiasing, its 23-label mapping, two input tensors and original-pixel output
  coordinates. Confidence filtering and configured IoU suppression are applied.
- **SLANet_plus**: one 488×488 BGR input, aspect-preserving resize, ImageNet
  normalization, then zero padding. Decodes the 50-token structure distribution
  and eight-coordinate cell polygons by tensor shape, independent of output names.
  Coordinates returned by the recognizer are in the original **crop pixels**.
- **SLANeXt / RT-DETR cell detection are not implemented here.** Unsupported
  tensors return errors. The old placeholder 1×1 table and empty-success detector
  have been removed. `TableModelType` remains source-compatible, but is not a
  choice of SLANeXt architecture; both variants use the SLANet Plus contract.

Weights are separate. With Python 3.12+ and the dependencies documented in the
script, run:

```sh
python scripts/prepare_table_models.py --output /path/to/table-models \
  --proxy http://127.0.0.1:1087
```

The script downloads two pinned official archives, verifies SHA-256, safely
extracts them and calls Paddle2ONNX with opset 17. It records the exported hashes
and does not install packages or turn on automatic dependency installation.
RTen 0.27 loads the `.onnx` files directly; renaming Paddle weights to `.rten`
does not convert them. Models, external software and source documents retain
their respective licenses. See the [Paddle model documentation](https://www.paddleocr.ai/latest/en/version3.x/module_usage/table_structure_recognition.html).

## Using the document pipeline

```rust,no_run
use rusto::{DocPipeline, InitializeConfig, LayoutDetector, RustO};
use rusto::table::{TableStructureConfig, TableStructureRecognizer};

let ocr = RustO::initialize(InitializeConfig::ppv6(
    "models/det.rten", "models/rec.rten", "models/dict.txt",
))?;
let layout = LayoutDetector::pp_doclayout_s("models/doclayout-s.onnx").ok();
let tables = TableStructureRecognizer::new(TableStructureConfig {
    model_path: "models/slanet-plus.onnx".into(),
    ..Default::default()
}).ok();
let mut pipeline = DocPipeline::with_components(ocr, layout, tables);
let result = pipeline.run("document.png")?;
// Always keep this source; structure markup is an optional enrichment.
let source = result.original_ocr;
let readable_text = result.plain_text;
# Ok::<(), rusto::EngineError>(())
```

`run_with_options` forwards OCR settings. `set_table_recovery_enabled(false)`
disables structure inference without disabling OCR. `DocPipeline::new` keeps
optional model-loading errors in `DocResult.warnings` and continues with OCR;
initialization failure of the required OCR model still returns an error. Legacy
`table_detector` configuration is not loaded: SLANet Plus predicts its own boxes.
The pipeline runs OCR **once on the full image**, then assigns each source box to
one layout block. Overlapping layout regions do not duplicate sources; uncovered
text stays in ordinary blocks. Tables always have `DocBlock.text`, including on
failure. Successful table HTML takes precedence in `to_markdown()`.

For an already cropped table and OCR from any engine (including Vision), call
`recognizer.recover(&crop, &ocr, &policy)`. All OCR geometry must be in that same
crop's pixel coordinates. `recover_prediction` also accepts a decoded prediction
from a custom backend. Every result carries unchanged `original_ocr` and plain
text. `assignments` maps each accepted cell to its source indices; document-level
results supply another index map back to the full page.

Default checks:

- structure confidence ≥ 0.85;
- valid completed grid, bounded dimensions/spans, no holes or overlapping spans;
- cell boxes inside the crop;
- each nonempty OCR box centered in a cell, with ≥ 80% box coverage;
- best-cell coverage exceeds the second choice by more than 0.10;
- all nonempty source boxes assigned exactly once, or the **entire table falls
  back**. A zero-overlap box is never attached to the nearest cell.

Status values distinguish disabled, bad configuration, invalid region, model
failure, low confidence, invalid grid, no text, unmatched text and ambiguity.
HTML is returned only on success, with escaped cell text. These checks prevent
known destructive results; they do **not** certify semantic table correctness or
recover columns the model never detected. Always retain the source and allow a
plain-text fallback in the consuming app.

## Verification

```sh
cargo test --locked
cargo test --locked --no-default-features
cargo clippy --locked --all-targets -- -D warnings
cargo run --release --no-default-features --example doc_recovery -- \
  /path/to/ocr-models /path/to/doclayout-s.onnx /path/to/slanet-plus.onnx \
  /path/to/document.png
```

The example prints JSON with original OCR, blocks, recovery statuses and HTML.
Its optional `--score-mode fast` reproduces older benchmarks; the default remains
`slow`, matching the current upstream detector. Do not compare different detector
modes as if only row merging changed.

Tests include synthetic failure injection, seven-language text preservation,
OpenCV resize reference pixels, and four captured SLANet outputs from real public
forms/scans (German, Japanese, English). Only token probabilities and geometry
are stored in `tests/fixtures/slanet-plus-real.json`; source document text and
model weights are excluded. [Fixture provenance](../tests/fixtures/table-recovery-sources.md) records the public sources. The German and Japanese captures have incomplete or
out-of-bounds grids and must fail validation. The English capture decodes a
complete 11×5 grid; this is a decoder check, not a claim that the predicted grid
matches its source document.

On the frozen 14-document, seven-language regression set, row fixes improve
several spatial CER scores without introducing new recognized characters. This
small set is not a population accuracy estimate. Korean recognition limitations
of the selected Medium dictionary are unchanged. Table structure does not fix
missing recognition characters. Mobile latency, peak memory and app integration
require separate device validation; this branch's real inference checks ran on
macOS CPU.

For an explicitly selected crop with existing OCR JSON (an array of
`TextResult` in the crop coordinate space), the second example avoids both
layout detection and a second OCR pass:

```sh
cargo run --no-default-features --example recover_table_crop -- \
  models/slanet-plus.onnx table-crop.png existing-ocr.json
```

Verification on 2026-11-02: the four real table regions in the 14-page set all
fell back (three invalid grids and one unmatched OCR box). The official mixed-text
demo restored a 4×4 grid with 13 cells; it is a demonstration image, not evidence
of real-world accuracy. An additional cropped paper-results table was not found
by automatic layout detection; direct crop recovery also fell back because an OCR
header box did not meet the coverage threshold. These failures are retained and
must not be reported as successfully recovered tables. Keep this capability
optional while expanding independently labelled real-table validation.
