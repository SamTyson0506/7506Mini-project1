# MP1: A Compact Causal Transformer with Within-Window Repetition Cache

**Status:** Draft. Replace every `[TO FILL]` with measured values before submission. Report the exact full-test BPB from `test_cpu_fp32.json`, not a rounded number. Keep this report at or under ten rendered pages.

## 1. Task and protocol

We model the supplied WikiText-2 raw text with the fixed training-fitted BPE-2048 tokenizer. Evaluation uses independent causal windows with 256 targets, FP32 predictions on CPU, and bits per raw UTF-8 byte (BPB). The validation set was used to select model settings. The test set is reported only for the frozen selected predictor. No external text or pretrained weights were used. We did not modify the supplied data, tokenizer, `common.py`, or `evaluate.py`.

## 2. Method

The selected model has eight Transformer blocks, width 192, six attention heads, tied token/output embeddings, RMSNorm, rotary position embeddings, SwiGLU feed-forward layers, and dropout 0.10 on residual branches during training. The checkpoint contains 3,935,424 learned parameters. AdamW, batch size 32, context length 256, cosine decay with 100-step warmup, and seed 17 were used for 16,000 updates (131,072,000 processed targets). The supplied trainer's exact learning-rate and optimizer settings were retained.

At evaluation, the model forms an additional empirical next-token distribution from earlier occurrences of the current one- or two-token suffix **within the same input window**. A two-token match is preferred; a one-token match is used when no two-token match exists. Previous occurrences vote for their observed successor. This cache is normalized and interpolated with the neural softmax using `alpha = 0.30 * (1 - exp(-mass/2))`. Only strictly earlier positions contribute. The cache is rebuilt for every `predict_log_probs` call; it has no cross-window state or saved evaluation answers.

This mechanism addresses the short context's difficulty with local repetitions, at the cost of quadratic suffix matching and a dense 2048-token distribution. All cache settings were selected using validation.

## 3. Experimental results

All comparisons below use the same validation scoring protocol unless marked test. Fill measured rather than reference values.

| Experiment | Training targets | Validation BPB | CPU FP32 validation seconds | Purpose |
|---|---:|---:|---:|---|
| Supplied GPT, 1,200 steps, seed 17 | 9,830,400 | 2.0711885 | 11.9772 | Initial baseline |
| Selected architecture, 1,200 steps, seed 17, cache on | 9,830,400 | 1.6752045 | 39.9978–43.4088 | Equal-target comparison |
| Selected 16,000-step checkpoint, cache off | 131,072,000 | 1.5828839 | 36.8316 | Cache ablation; same weights |
| Selected 16,000-step checkpoint, cache on | 131,072,000 | 1.5332663 | [TO FILL] | Selected model |

At equal training targets, the student improves validation BPB by 0.3959841 (19.1% relative), while training took 95.40 s versus 31.36 s on the same GPU and validation CPU scoring took 40.00–43.41 s versus 11.98 s. The equal-target comparison changes the architecture and cache together; it does not isolate a single component. The cache ablation holds the learned weights fixed and isolates the inference mechanism: adding the within-window cache reduces validation BPB by 0.0496176 (3.14% relative to no cache). Neither comparison by itself proves that every architectural change is beneficial.

Other validation-only development runs included the 6-layer model (5,000 steps, 1.6633360 without cache; 1.6043829 with a tuned 0.30 cache), the 8-layer model (8,000 steps, 1.5348452 with cache), the 10-layer model (16,000 steps, 1.5243090 with cache but near the CPU time limit), and an 8-layer width-224 candidate (16,000 steps, 1.5304536 with cache). Direct averaging of the 8,000- and 16,000-step checkpoints failed (1.6874678). State the cost and seed of each search run in an appendix or run log. The 10-layer and width-224 runs were not selected because the former had minimal CPU time margin and the latter did not improve validation relative to it. [Revise this sentence if the final selected method changes.]

## 4. Final score and resource limits

| Measure, same frozen predictor | Result | Requirement |
|---|---:|---:|
| Complete test BPB, CPU FP32 | 1.5541534308982003 | Lower is better |
| Complete test CPU scoring time | 50.9821 s | At most 5 times the baseline on the same machine and split |
| Baseline complete test CPU time | 15.4677 s | Ratio denominator |
| CPU scoring ratio | 3.2960 | At most 5.0 |
| Peak evaluation RAM | 1.757 GiB (Windows process-tree peak working set, complete-test CPU FP32 run) | At most 4 GiB |
| Uncompressed model and inference assets | 15.04 MiB (15,767,382 bytes: checkpoint plus `student.py`) | At most 64 MiB |

The 1,200-step baseline took 11.9772 s on validation. A separate 10-step baseline timing was 13.9746 s; timing varied across runs. Therefore, time compliance must be checked against a baseline on the **same split and machine**, with repeat measurements or sufficient margin. The 10-step baseline quality is not a full baseline score. State the CPU model, thread count (4), Python 3.12, PyTorch 2.7.1+cu126, and training GPU (RTX 4060 Laptop GPU). A CUDA validation run is not a substitute for CPU FP32 scoring.

## 5. Limitations and analysis

The 8-layer validation score improved only from 1.5348452 at 8,000 updates to 1.5332663 at 16,000; the 16,000-step trajectory reached 1.5319480 at update 14,000 before rising, but intermediate weights were not saved. More updates alone are unlikely to close the remaining gap to 1.5 validation BPB. The repetition cache helps on repeated local patterns but cannot use repetitions outside the independent window. It also increases CPU computation. Validation-only model search and observed test results must be disclosed; no later method was selected on test performance.

## 6. Reproduction and disclosure

From the unmodified package `code/` directory, install the requirements according to the supplied README. Place the exact submitted `student.py` and `checkpoint.pt` in the package. Then run:

```bash
python evaluate.py --checkpoint PATH/TO/checkpoint.pt --device cpu --precision fp32 --split test --output reproduced_test.json
```

The checkpoint's `implementation` is `student`; `student.py` must match the frozen submitted version. Record SHA-256 for `student.py`, checkpoint, evaluator, and tokenizer from the evaluation JSON: [TO FILL]. Training command: `python train.py --implementation student --steps 16000 --batch-size 32 --seed 17 --eval-every 1000 --device cuda --run-dir runs/student-deep-16000`.

Substantive AI assistance was used to design and write the model and cache code, experiment scripts, and report outline. The student ran and evaluated the experiments, verified the results, and is responsible for the final submission. Cite the supplied course code and WikiText-2 attribution stated in the package README.
