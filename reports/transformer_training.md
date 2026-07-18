# DistilBERT Training Workflow

This Week 2 workflow fine-tunes `distilbert/distilbert-base-cased` for binary sequence
classification. Label `0` means human-written and label `1` means AI-generated. Accuracy,
precision, recall, and F1 are reported, with label-1 F1 used to choose the best validation
checkpoint.

## Test-set isolation

`src/train_transformer.py` reads only the `train` and `validation` members of the prepared
DatasetDict. The frozen `test` member is not passed to `Trainer`, evaluated, or used for
checkpoint selection. It remains reserved for the final comparison after the configuration
is frozen.

## Reproducible defaults

- Random, data, NumPy, and PyTorch seeds: 42 through Hugging Face `set_seed` and
  `TrainingArguments`.
- Model: cased DistilBERT with two output labels.
- Per-device train/evaluation batch size: 4.
- Gradient accumulation: 4, for an effective training batch size of 16.
- Learning rate: 2e-5; weight decay: 0.01; warmup ratio: 0.1.
- Epochs: 2 for the eventual full run.
- Mixed precision: FP16.
- Evaluation and checkpoint saving: once per epoch.
- Best model: highest validation F1; at most two saved checkpoints.
- Windows data-loader workers: 0.
- Reporting integrations: disabled.

The saved examples are not statically padded. `DataCollatorWithPadding` pads each batch to
its longest sequence. Before training, the script explicitly retains only `input_ids`,
`attention_mask`, and `labels`; this removes `token_type_ids`, which DistilBERT does not
consume, as well as text and audit metadata that the model does not need.

## Safety and experiment logging

Any stratified subset is labeled `development_smoke` in the experiment log. A max-step run
over the full input is labeled `bounded_full_data`. An unbounded run over all rows is blocked
unless `--confirm-full-run` is supplied, and a non-empty output directory requires an
explicit resume option. These checks reduce the chance that a development result is
mistaken for a full experiment or an existing checkpoint is overwritten.

Successful runs append their configuration, package/GPU versions, dataset sizes, measured
runtime, validation metrics, best checkpoint, peak allocated GPU memory, and a rough scaled
full-run duration estimate to `results/transformer_experiments.csv`. Raw metrics, model
weights, Trainer state, and checkpoints remain under ignored `checkpoints/`.

## Environment verification

On the project Windows environment, CUDA was verified before implementation with:

- NVIDIA GeForce RTX 3060 Laptop GPU, 6,144 MiB VRAM
- NVIDIA driver 581.95
- PyTorch 2.12.1+cu130
- Accelerate 1.14.0
- Transformers 5.14.1
- `torch.cuda.is_available()` returned true
- a CUDA tensor multiplication returned the expected values

The initial measured smoke run and duration estimate will be recorded here after the code
and helper tests pass.
