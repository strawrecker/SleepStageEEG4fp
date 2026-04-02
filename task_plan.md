# Task Plan: BOAS Frontal Sleep Staging

## Goal
Build a reproducible BOAS-based sleep staging experiment around frontal wearable EEG channels, then iteratively design and verify a new effective model centered on HB_1/HB_2 and their derived frontal representation.

## Current Phase
Phase 5

## Phases

### Phase 1: Requirements & Discovery
- [x] Understand user intent
- [x] Identify constraints and requirements
- [x] Document findings in findings.md
- **Status:** complete

### Phase 2: Planning & Structure
- [x] Define technical approach
- [x] Create project structure
- [x] Document decisions with rationale
- **Status:** complete

### Phase 3: Implementation
- [x] Build dataset ingestion and labeling pipeline
- [x] Build baseline model and training/evaluation harness
- [ ] Implement new frontal-model hypothesis and ablations
- **Status:** in_progress

### Phase 4: Testing & Verification
- [x] Run smoke experiments on currently complete BOAS nights
- [x] Verify metrics mechanically
- [x] Expand evaluation as more BOAS nights finish downloading
- **Status:** complete

### Phase 5: Delivery
- [x] Summarize the current best model and bottlenecks
- [ ] Leave reproducible commands and artifacts
- [ ] Deliver outcomes and residual risks to user
- **Status:** in_progress

## Key Questions
1. How should we best encode the two frontal-like BOAS headband channels: independent streams, differential channel, or both?
2. Which bottleneck dominates early performance on BOAS: label imbalance, temporal context, cross-night calibration drift, or low channel count?
3. How can we get a robust baseline now while the dataset is still downloading?

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| Use BOAS instead of Sleep-EDF | User redirected the experiment toward frontal wearable EEG closer to Fp1/Fp2 use cases. |
| Use `stage_hum` from PSG event files as supervision | Consensus human labels are the strongest available ground truth in BOAS. |
| Treat `HB_1` and `HB_2` as the core frontal channels and also test `HB_1-HB_2` | BOAS headband exposes two forehead EEG channels around AF7/AF8; a differential derivation may capture robust asymmetry/noise rejection. |
| Start with currently complete BOAS nights and expand later | The dataset is still downloading, so the fastest trustworthy path is to bootstrap the pipeline on the available complete nights. |
| Smoke training should use all currently complete BOAS nights except the held-out last night | The dominant early failure mode was cross-night drift from too little labeled coverage, not insufficient model capacity. |
| Soften class weighting to inverse-square-root strength | Strong inverse-frequency weighting over-predicted minority stages, especially N3, and hurt macro-F1. |
| Skip malformed EDF files during record discovery | BOAS contains a small number of non-compliant headband EDF files that should not block full-run experiments. |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| PowerShell heredoc syntax with `python - <<'PY'` failed | 1 | Switched to PowerShell here-string piped into `python -`. |
| Torch 1.12.1 could not use direct `tensor.numpy()` with NumPy 2.0.2 | 1 | Replaced tensor-to-NumPy conversion with `tolist()` in evaluation. |
| Full BOAS run aborted due malformed EDF Physical Maximum header values | 1 | Added EDF-readability validation and skipped unreadable subjects during record listing. |

## Notes
- This repo is not a git repository, so experiment provenance will rely on autoresearch artifacts plus file-level changes and metric logs.
- Main metric will be macro-F1 on five-stage sleep classification after filtering unsupported labels.
