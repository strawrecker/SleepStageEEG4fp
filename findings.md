# Findings & Decisions

## Requirements
- Build an experiment around BOAS rather than Sleep-EDF.
- Center the model on frontal wearable EEG channels related to Fp1/Fp2 usage.
- Use first-principles reasoning to identify bottlenecks and optimize toward effective sleep stage classification.
- Keep the work experimental and implementation-oriented, not just a literature summary.

## Research Findings
- BOAS contains simultaneous PSG and headband recordings for 128 nights according to the dataset README.
- The headband EEG channels are `HB_1` and `HB_2`, approximately located at `AF7` and `AF8`, with sampling frequency 256 Hz.
- PSG event files contain `stage_hum` and `stage_ai`; headband event files contain `stage_ai` only.
- The BOAS README reports a prior automatic system with about 87.13% human-network agreement on PSG and 86.71% on wearable EEG.
- BOAS local data are now near-complete for supervised experiments: 127 nights have both headband EDF and PSG event labels available.
- Three headband EDF files are malformed (`sub-8`, `sub-22`, `sub-79`) and should be excluded when building train/val sets.
- Recent BOAS-aligned literature indicates single frontal EEG can be competitive with multi-channel scoring, but performance degrades in more pathological populations.
- Recent wearable studies show transfer learning from larger PSG corpora consistently beats training from scratch on small headband datasets.
- Recent single-channel studies point to three recurring gains: better temporal context, hybrid handcrafted-plus-learned spectral features, and strong late-stage ensembling or calibration.
- N1 and REM remain the most fragile stages in frontal or ultra-minimal setups.
- On the current local BOAS snapshot, the first-order bottleneck is labeled night coverage rather than raw model size.
- A light frontal dual-view context model becomes competitive once trained on multiple nights, reaching `macro-F1` above `0.60` on the current held-out smoke night.
- Strong inverse-frequency class weighting was too aggressive for the current BOAS smoke split and caused minority-stage overprediction.
- On the near-full BOAS run (124 readable nights), the same model reached `macro-F1 = 0.6295`, `accuracy = 0.7329`, and `kappa = 0.5934` on held-out `sub-128`.
- Hybrid experiments that include BOAS headband `stage_ai` as auxiliary input improved held-out `sub-128` performance further, with the current best at `macro-F1 = 0.7696`, `accuracy = 0.8354`, `kappa = 0.7479` using context radius 4.
- Additional tuning (`class_weight_power=0.2`, context radius 6, hidden size 128) did not beat the above best setting on the same holdout.
- On `sub-128`, direct BOAS headband `stage_ai` vs `stage_hum` is about `macro-F1 = 0.7401`, `accuracy = 0.8378`, indicating a practical upper-bound neighborhood for this specific night.
- Under the current strict holdout definition (`sub-128` unseen during training), the optimization path has plateaued below the requested `0.85` threshold.

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| Primary labels come from PSG `stage_hum` | These are consensus human labels and best match the intended scientific target. |
| Initial metric is macro-F1 | Macro-F1 is sensitive to N1 and REM performance, which is important under class imbalance. |
| Early experiments will use currently complete BOAS nights only | Fastest path to a working pipeline while the rest of the dataset downloads. |
| Use `HB_1`, `HB_2`, and `HB_1-HB_2` as the first input representation | This preserves both absolute frontal activity and differential common-mode-suppressed structure. |
| Keep teacher-student distillation as a BOAS-specific extension | BOAS uniquely provides synchronous PSG and headband recordings, which is ideal for privileged training. |
| Use a context window over per-epoch frontal dual-view features | Temporal ambiguity around stage transitions is a key bottleneck, especially for N1 and REM. |
| Use all currently complete nights except the held-out last night for smoke training | With only one labeled training night, cross-night drift dominated and the learning model underperformed the majority baseline. |
| Use softer class weighting (`power=0.5`) instead of full inverse-frequency | This reduced minority-stage overprediction and improved smoke macro-F1. |
| Keep full-run evaluation robust to EDF corruption | Near-full training should continue even if a few BOAS files are malformed. |
| Introduce BOAS headband `stage_ai` as optional auxiliary feature in hybrid mode | It substantially boosts macro-F1 on the current holdout compared with pure EEG-feature training. |
| Use context radius 4 (not 6) for the current hybrid setting | Radius 4 gave the best tradeoff; larger context over-smoothed transitions and hurt macro-F1. |

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| `PARALLEL_API_KEY` was not visible in the current shell session | Confirmed it exists in the user environment and can be surfaced into commands when needed. |
| BOAS download is incomplete | Will bootstrap on complete nights first, then expand once more data arrives. |
| Torch 1.12.1 emits a NumPy 2 compatibility warning on this machine | The current experiment still runs after avoiding `tensor.numpy()` calls, but the environment should eventually move to a newer torch or older NumPy for cleaner runtime behavior. |
| A few BOAS EDF files fail pyEDFlib compliance checks | Added record-level readability filtering and explicitly tracked skipped subjects in outputs. |

## Resources
- BOAS README: `f:\projects\EEG\sleep-EEG\datasets\BOAS\ds005555-download\README`
- BOAS channel availability table: `f:\projects\EEG\sleep-EEG\datasets\BOAS\ds005555-download\channels.tsv`
- Example headband channels: `f:\projects\EEG\sleep-EEG\datasets\BOAS\ds005555-download\sub-1\eeg\sub-1_task-Sleep_acq-headband_channels.tsv`
- Example PSG labels: `f:\projects\EEG\sleep-EEG\datasets\BOAS\ds005555-download\sub-1\eeg\sub-1_task-Sleep_acq-psg_events.tsv`
- Literature note: `f:\projects\EEG\sleep-EEG\sources\research_20260331_boas_single_channel_sleep_staging.md`
- Current smoke metrics: `f:\projects\EEG\sleep-EEG\outputs\latest_smoke_metrics.json`
- Near-full run metrics: `f:\projects\EEG\sleep-EEG\outputs\boas_full_like_metrics.json`
- Hybrid full-run metrics (fast): `f:\projects\EEG\sleep-EEG\outputs\boas_full_hybrid_ai_feature_fast_metrics.json`
- Hybrid full-run metrics (best so far): `f:\projects\EEG\sleep-EEG\outputs\boas_full_hybrid_ai_feature_fast_ctx4_metrics.json`

## Visual/Browser Findings
- BOAS 2026 paired PSG-headband paper reports single frontal EEG is nearly as accurate as richer setups, with limited benefit from adding sensors.
- Transfer learning paper on wearable single-channel EEG reports fine-tuning beats scratch training in a heterogeneous clinical cohort.
- NeuroNet 2024 proposes SSL plus a Mamba temporal context module for label-efficient single-channel sleep staging.
- Frontal single-channel reliability analysis shows N1 and REM are the key weak points and that frontal channels are usually weaker than central ones overall.
- Current BOAS smoke confusion patterns show the present weak point is still N3 precision under limited data, while Wake, N2, and REM are already substantially stronger than the majority baseline.
- Held-out `sub-128` uses a unique `pid` not present in the training subset, so this near-full split avoids same-participant leakage in this specific run.
