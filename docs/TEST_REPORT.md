# Test report: DR-XAI

| | |
|---|---|
| Date | 2026-10-10 (before sync point SP0) |
| Commit under test | `494539d` (milestone M0 only) |
| Tester | Claude Opus 5.5, in the same session that wrote the M0 code. The brief assigns a separate Sonnet session, so this pass is **not independent** (ISSUE-008). |
| Scope covered | Phase 0, Phase 1 for the M0 modules, S9 of Phase 2, probes and mutation checks |
| Not covered | Phases 3 to 7 and S1-S8, S10: no dataset, no trained model, no results (ISSUE-007) |

## Sign-off

**Not signed off.** Two S1 defects are open (DEF-001, DEF-002) and eight of ten requirements are NOT
MET because the corresponding code, data or results do not exist yet. No metric value appears in this
report because no run has produced one.

## Phase 0: environment and static checks

| Check | Result | Evidence |
|---|---|---|
| Fresh Python 3.11.15 environment from `requirements.txt` (118 packages), then `pip install -e .`; `uv pip check` reports all packages compatible | PASS | `docs/evidence/phase0_pip_freeze.txt` |
| `ruff check src tests` | PASS, no findings | `docs/evidence/phase0_ruff.txt` |
| `mypy src --ignore-missing-imports` | PASS, no issues in 23 files | `docs/evidence/phase0_mypy.txt` |
| Grep `src/` for `/home/`, `/root/`, `C:\`, `print(`, bare `except:`, `TODO`, `FIXME`, `df.append(`, `verify=False` | PASS, no match | this report |
| Language: only `.py` files under `src/` | PASS | this report |
| `data/` and `checkpoints/` ignored; no tracked file over 1 MB | PASS | `git check-ignore`, `git ls-files` |
| `make` available on the test machine | **No.** `make reproduce` cannot be run here; the README gives the equivalent `python -m xaidr.cli` commands | this report |

Deviations: the environment was built with `uv` (wheels came from its cache, torch is the CPU build).
Linux or GPU installs were not tested.

## Phase 1: unit tests

`python -m pytest tests -m "not slow" -v` in the fresh environment: **58 collected, 56 passed,
2 xfailed, 0 failed, 0 skipped** (`docs/evidence/phase1_pytest.txt`). The two xfails are the
Tester's tests for DEF-001.

| File (brief A2) | Must prove | Status |
|---|---|---|
| `test_manifest.py` | Required columns enforced; wrong dtype rejected | PASS |
| `test_adapters.py` | Correct manifests from fixture data; masks only where `has_masks`; declared column names used | PASS. Fixtures are synthetic trees built in `tests/conftest.py`, not real downloads |
| `test_preprocess.py` | Square within 1 px of its mask after crop, CLAHE, resize; CLAHE never on masks; masks binary | PASS |
| `test_splits.py` | Disjoint; strata within 1 percentage point; same seed reproduces file and hash; resplit without flag raises | PASS |
| `test_no_leakage.py` | External rows never in training loader; training transforms only on training; oversampler draws training indices only | PASS |
| `test_model.py`, `test_loss.py`, `test_metrics.py`, `test_fidelity.py`, `test_explainers.py` | (see brief) | **NOT RUN: files and modules do not exist (M1, M2)** |
| `test_config.py`, `test_cli_data.py` (not in the brief) | Config validation names the bad key; `prepare-data` and `make-splits` end to end | PASS |

### Mutation checks (does the suite notice a planted fault?)

| Planted fault | Result |
|---|---|
| M1 masks shifted 3 px after resize | Detected |
| M2 evaluation loaders given the training augmentation | Detected |
| M3 split made without stratification | Detected |
| M4 oversampling silently disabled in the training loader | **Not detected** by the Engineer's tests (DEF-008); now covered by a Tester test |

Evidence: `docs/evidence/m0_probes_output.txt`, script `docs/evidence/probes/m0_probes.py`.

### Probes

| Probe | Finding | Defect |
|---|---|---|
| P1, P2 seed and augmentation | Same seed gives different augmented tensors | DEF-001 (S1) |
| P5 duplicate images | 32 of 60 planted duplicate pairs ended in two different splits, no warning | DEF-002 (S1) |
| P4 split lock | Manifest split column can be edited without detection; lock file is under an ignored folder | DEF-003 (S2) |
| P3 config change | Processed images silently reused after a CLAHE change | DEF-004 (S2) |
| P6 error handling | Corrupt image and tiny class give tracebacks, not error messages | DEF-005 (S3) |
| P7 aspect ratio | Vertical stretch up to 27.5% on 2 of 10 sample photos | DEF-007 (S4, Researcher) |

## Phase 2: sanity tests

| Test | Status |
|---|---|
| S9 Eval determinism | **PASS** (`tests/test_sanity.py::test_s9_eval_loaders_are_deterministic`, synthetic data) |
| S1, S2, S7, S8 | NOT RUN: `training/metrics.py`, `training/threshold.py` do not exist |
| S3, S10 | NOT RUN: model, loss and trainer do not exist |
| S4, S5, S6 | NOT RUN: `xai/fidelity.py` does not exist |

## Phase 3: data integrity

| Item (brief Section 2) | Expected | Observed | Status |
|---|---|---|---|
| APTOS `sample_submission.csv` | 1,928 ids, labels all 0 | 1,928 rows, 1,928 unique ids, `diagnosis` only 0. SHA-256 `2458B54012288D495D00DADE08D19FE0EEBF22FE349EEDF66F8CFB55B39BAEB1` | CONFIRMED |
| APTOS labelled training images | 3,662 | dataset not on disk | NOT CHECKED |
| IDRiD grading set | 516 | dataset not on disk | NOT CHECKED |
| IDRiD lesion masks | 81 (MA 81, EX 81, HE 80, SE 40) | dataset not on disk | NOT CHECKED |
| Messidor-2 | 1,748 images, drops logged | dataset not on disk | NOT CHECKED |
| `docs/DATA_ACCESS.md` dry run | Download and placement succeed from the text alone | No credentials or registrations in this session; not attempted | NOT RUN |
| No image id in two splits (real data) | none | synthetic only (`test_splits.py`) | NOT CHECKED on real data |

## Phases 4 to 7

NOT RUN. `python -m xaidr.cli all` exits with code 2, "step 'all' is not implemented yet (milestone
M4)" (`docs/evidence/phase4_cli_all.txt`). There is no `test_cli_smoke.py`, no `run.json`, no result
CSV, no notebook and no `v1.0-submission` tag.

## Requirement traceability (R1-R10)

```
R1  Shared backbone, two heads, joint training
    Status:   NOT MET (not implemented: models/, training/ are empty)
    Tests:    none
    Results:  none

R2  Four explainers produce 512x512 float32 maps in [0,1]
    Status:   NOT MET (not implemented: xai/ is empty)
    Tests:    none
    Results:  none

R3  Quantitative fidelity vs expert masks
    Status:   NOT MET (xai/fidelity.py not implemented; IDRiD not on disk)
    Tests:    test_preprocess.py (PASS) covers only the mask/image alignment this depends on
    Results:  none
    Open:     DEF-007 / ISSUE-003 (Researcher)

R4  External sets never used for training or selection; stability status per dataset
    Status:   PARTIAL
    Tests:    test_no_leakage.py (PASS), test_splits.py (PASS), S9 (PASS); all on synthetic data
    Results:  none (no crossdataset_summary.csv)
    Open:     DEF-002 (S1) / ISSUE-002; DEF-003 (S2) / ISSUE-004

R5  Minority oversampling ablation
    Status:   PARTIAL
    Tests:    test_no_leakage.py (PASS), test_sanity.py: oversampling via loader (PASS),
              ablation configs differ only in minority.mode (PASS)
    Results:  none (eval/ablation.py not implemented, no runs)

R6  Replicable protocol
    Status:   NOT MET
    Tests:    test_splits.py reproducibility and lock (PASS); test_sanity.py seed tests (XFAIL)
    Results:  none; `make reproduce` path exits 2; no run.json
    Open:     DEF-001 (S1) / ISSUE-001; DEF-003, DEF-004 (S2)

R7  Datasets: APTOS 2019, IDRiD, Messidor-2
    Status:   NOT MET (adapters pass on synthetic trees; no real dataset on disk; Messidor-2
              label mapping not yet declared)
    Tests:    test_adapters.py (PASS)
    Results:  none
    Open:     ISSUE-007

R8  Literature rows in Table 6 match their sources
    Status:   NOT MET (no table6_comparative.csv)

R9  Python implementation; six notebooks execute
    Status:   NOT MET (language check PASS; notebooks/ is empty)

R10 Datasets and models accessible to the team
    Status:   NOT MET (access dry run not possible; timm lists the pretrained tag
              maxvit_tiny_tf_512.in1k but the weights were not downloaded)
    Open:     ISSUE-007
```

## Open S0-S1 defects

| ID | Severity | Title |
|---|---|---|
| DEF-001 | S1 | Training augmentation is not controlled by the seed |
| DEF-002 | S1 (S0 if confirmed on real data) | Duplicate images not detected; can sit in two splits |

## How to resume

1. Re-run Phase 0-1 in a separate Tester session (ISSUE-008).
2. After the Engineer fixes DEF-001, the two xfail tests in `tests/test_sanity.py` fail as XPASS: remove
   the markers and re-run.
3. When APTOS 2019 is on disk: Phase 3, starting with a duplicate-image count across splits (DEF-002).
4. When M1 exists: `test_model.py`, `test_loss.py`, `test_metrics.py`, then S1-S3, S7, S8, S10.
