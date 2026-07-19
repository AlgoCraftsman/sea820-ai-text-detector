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

## Initial smoke-run outcome

The first GPU smoke run completed on a deterministic stratified subset of 256 training and
128 validation examples. It used one epoch, FP16, batch size 4, gradient accumulation 4,
and gradient checkpointing. Dynamic padding was separately verified by collating examples
of 190, 240, 256, and 269 tokens into a batch width of 269 rather than 512.

| Measurement | Result |
| --- | ---: |
| Optimizer steps | 16 |
| Trainer runtime | 15.723 seconds |
| Training throughput | 16.282 examples/second |
| Final validation runtime | 0.889 seconds |
| Validation accuracy | 0.609375 |
| Validation precision (AI class) | 0.000000 |
| Validation recall (AI class) | 0.000000 |
| Validation F1 (AI class) | 0.000000 |
| Peak PyTorch GPU allocation | 1,212.5 MiB |
| Checkpoints retained | 1 |

The tiny smoke subset predicted only the majority human class. Its metrics validate the
execution and logging path; they are not evidence about final model quality and must not be
reported as a full training result. The experiment log explicitly labels the run
`development_smoke` and records `test_split_used=False`.

At the measured throughput, scaling training and validation to all prepared rows gives a
rough estimate of 6.43 hours for one epoch, or about 12.9 hours for the planned two-epoch
run. This is an extrapolation from a small warm-cache run, not a guaranteed duration. A
larger development run would provide a more reliable estimate before committing to the
full experiment.

Transformers 5.14.1 warned that passing `warmup_ratio` to `TrainingArguments` is deprecated.
The workflow therefore retains the user-facing ratio but converts it to an explicit number
of optimizer `warmup_steps` before constructing `TrainingArguments`.

## Larger development-run outcome

A second run used deterministic stratified subsets of 4,096 training and 1,024 validation
examples. It retained the same model and RTX 3060 settings and trained for one epoch. This
run was large enough to confirm that the classifier learns rather than only exercising the
training code.

| Measurement | Result |
| --- | ---: |
| Optimizer steps | 256 |
| Warmup steps | 26 |
| Trainer runtime | 141.383 seconds |
| Training throughput | 28.971 examples/second |
| Training loss | 0.818425 |
| Final validation runtime | 14.417 seconds |
| Validation loss | 0.124723 |
| Validation accuracy | 0.971680 |
| Validation precision (AI class) | 0.948780 |
| Validation recall (AI class) | 0.979849 |
| Validation F1 (AI class) | 0.964064 |
| Peak PyTorch GPU allocation | 1,212.5 MiB |
| Checkpoints retained | 1 |

The best checkpoint was selected at step 256 by validation F1. Its saved configuration
retains the explicit mappings `0 = human` and `1 = AI-generated`, uses single-label
classification, and confirms a maximum of 512 positional embeddings. The experiment log
labels the result `development_smoke` and records `test_split_used=False`.

This development F1 must not be presented as the final Transformer score: both training and
validation were reduced subsets, and the frozen test set remains untouched. It also cannot
be compared directly with the Week 1 baseline scores, which used a different holdout.

The larger run provides a more stable throughput estimate than the initial 256-example
smoke test. Scaling its measured training and validation rates gives approximately 3.74
hours for one full epoch and 7.48 hours for two full epochs. Actual time may vary with
thermal throttling and other GPU use.

The recommended full-run sequence is one confirmed full epoch followed by complete
validation. If that run remains stable, resume the saved checkpoint to a second epoch rather
than committing to both epochs up front. This preserves an intermediate model and makes the
multi-hour decision auditable.
