# Table recovery test fixture provenance

`slanet-plus-real.json` contains argmax token IDs/probabilities and cell-box
predictions from the official SLANet_plus model (PaddleX 3.7.2 / Paddle 3.3.1).
It does not contain the source images, recognized document text or model weights.
These are regression inputs, not table-structure ground truth. The expected
validity concerns grid topology only.

Source documents:

- `de_val_0-table-0` and `de_val_0-table-1`: `de_val_0.jpg` in
  [XFUND German validation](https://github.com/doc-analysis/XFUND/releases/download/v1.0/de.val.zip),
  original SHA-256 `fd1b96cd7c1e9f2af1b8022e75fc06a211742aa16ee9ec97e16665966e0db91e`.
- `ja_val_0-table-0`: `ja_val_0.jpg` in
  [XFUND Japanese validation](https://github.com/doc-analysis/XFUND/releases/download/v1.0/ja.val.zip),
  original SHA-256 `735863af44c005079919f7ec803f9120e48dfd2291913b6a592fc7462d7dd417`.
- `en-82200067_0069-table-0`: `testing_data/images/82200067_0069.png` in
  [FUNSD](https://guillaumejaume.github.io/FUNSD/), original SHA-256
  `c48d22b3c6692f8b1c08398d66fc9a3f9cc8b5bf0ff6641336228ae3cd117e23`.

Images were capped at a 3000-pixel long edge, and table crops came from
PP-DocLayout-S with a 0.5 confidence threshold, not from labelled cell boundaries.
The source datasets retain their respective terms and attribution.

Model specification and dictionary:
[PaddleOCR SLANet_plus](https://github.com/PaddlePaddle/PaddleOCR/blob/main/configs/table/SLANet_plus.yml),
[PaddleX table processors](https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/inference/models/table_recognition/processors.py)
(Apache-2.0). Rust implementation is independently written against this contract.

`cubic-resize.json` contains reference RGB pixels from OpenCV 4.10.0 `INTER_CUBIC`
for deterministic generated ramps: channel bytes `(index * 37 + 13) % 256`.
It is not document imagery. Dimensions are stored alongside each array.
