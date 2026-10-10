# DR-XAI: Verification & Research Brief (Tester and Researcher, run in parallel)

**Project:** Explainable AI-Driven Deep Learning Model for Automated Diabetic Retinopathy Detection and Severity Grading (U18IT707, Major Project Phase 1)
**Team:** K. Poojitha (B23IT082), K. Reethu Sri (B23IT101), G. Abhiram (B23IT107), I. Deepthi (B23IT119). Supervisor: Dr. K. Praveen Kumar.
**Code deadline:** 14 October 2026
**Replaces:** `TESTING_AND_VERIFICATION_PLAN.md` and `RESEARCHER_CROSS_CHECK_BRIEF.md` (both superseded by this file)
**Reads with:** `docs/DESIGN.md` (the Designer's architecture and engineering brief)
**Version:** 2.0, 10 October 2026

---

## 0. START HERE

### 0.1 What runs and where

Two Claude Code sessions run at the same time on the same repository. Each has one role, one model, and one set of files it owns.

| Session | Role | Model | Owns (may write) | Must never write |
|---|---|---|---|---|
| A | Tester/QA | **Sonnet** | `tests/`, `docs/TEST_REPORT.md`, `docs/DEFECTS.md`, `docs/evidence/` | `src/`, `configs/`, `notebooks/`, `results/`, `docs/SCIENTIFIC_REVIEW.md` |
| B | Researcher | **Sonnet or Opus** (Opus for literature) | `docs/SCIENTIFIC_REVIEW.md`, `docs/evidence/research/` | `src/`, `tests/`, `configs/`, `results/`, `docs/TEST_REPORT.md` |
| Shared | Both | — | `docs/ISSUE_LOG.md` (append-only; see Section 6) | Editing or deleting another role's entries |

Neither session edits code or results. The Software Engineer (Opus, Developer role) is the only role that changes `src/`, `configs/`, `notebooks/` and `results/`.

### 0.2 Paste-in prompts

**Session A (Tester/QA, Sonnet):**
> You are the **Tester/QA** for DR-XAI. Read `docs/VERIFICATION_AND_RESEARCH_BRIEF.md` (Sections 0–2, 4, 6 and 7) and `docs/DESIGN.md`. Your track is Section 4. You may write only the files listed for Session A in Section 0.1. Do not edit code. Work through the phases in order. Log every defect in `docs/DEFECTS.md`, and log every cross-role issue in `docs/ISSUE_LOG.md` (append only). Never mark a requirement as met without evidence. Never invent a metric value. At each sync point in Section 7, check `docs/ISSUE_LOG.md` for entries addressed to you before continuing.

**Session B (Researcher, Sonnet or Opus):**
> You are the **Researcher** for DR-XAI. Read `docs/VERIFICATION_AND_RESEARCH_BRIEF.md` (Sections 0–2, 3, 5 and 8) and `docs/DESIGN.md`. Your track is Section 5. You may write only the files listed for Session B in Section 0.1. Do not edit code, tests or results. Verify literature claims with web search and record every URL used. Quote only short phrases from source papers. Log cross-role issues in `docs/ISSUE_LOG.md` (append only). At each sync point in Section 7, check `docs/ISSUE_LOG.md` for entries addressed to you before continuing. Verdicts must rest on evidence files you have opened.

### 0.3 Token and session warning
- The four supplied research papers total about 840 KB of text. Do not re-read them in full. The Researcher uses the targeted searches in Section 5.7 only.
- Each session should cover one phase or one track section at a time, and save progress to its own file after each step. If a session hits a usage limit, a new session can resume from its report file and the issue log.
- Full training is not run from either session. The Software Engineer runs it; the sessions read the outputs.

---

## 1. The abstract's requirements and claims (one shared standard)

The Tester checks **requirements** (what the code must do). The Researcher checks **claims** (what the paper may say). Both use the same rows below, so a gap found by one role is visible to the other.

| ID | From the abstract | Tester question (code and evidence) | Researcher question (science and wording) |
|---|---|---|---|
| R1 / C1 | Joint binary detection and five-class ICDR grading through a shared-backbone, dual-head model; the joint design is presented as the contribution | One backbone object feeds two heads; both heads get gradients; outputs have correct shapes | Is a separate-task baseline run? If not, the paper may not claim joint training is better |
| R2 / C2 | Four explainers: Grad-CAM++, SHAP, LIME, attention saliency | All four produce 512×512 float32 maps in [0,1] on the same image | Is each method's attribution correct (especially Grad-CAM++ via Captum, F3)? Is attention saliency described as explanation or as an internal signal? |
| R3 / C2 | Quantitative validation vs expert masks with IoU, Dice, pointing game | Metrics match known answers; chance baseline is computed by code; sample sizes reported | Are the metrics appropriate for saliency vs lesion agreement? Is top-k% sensitivity reported? |
| R4 / C3 | Cross-dataset robustness of predictions **and** explanation stability across APTOS 2019, IDRiD, Messidor-2 | External sets never used for training or selection; stability status stated per dataset | Are external sets truly independent? Is stability defined where lesion masks do not exist (F9)? |
| R5 / C4 | Synthetic-data augmentation for Severe and Proliferative classes, with its effect on accuracy and fidelity as a dedicated ablation | Both ablation arms share split hash, seed and selection rule; only `minority.mode` differs; overwrite bug absent | Is the method named correctly (D2)? Is the comparison paired and powered? Is the fidelity claim supported by the CI? |
| R6 | Framework and protocol detailed enough for independent replication | `make reproduce` works on a clean clone; seeds fixed; run records written | Does the paper's protocol match the config that actually ran? |
| R7 | Datasets: APTOS 2019 (5-class), IDRiD (lesion masks), Messidor-2 (independent) | Adapter counts match Section 6 facts; label mapping declared | Is the Messidor-2 label scheme and detection definition stated (F8)? |
| R8 / C5 | Expected performance grounded in the closest published results | Literature rows in Table 6 match their stated source | Are the cited numbers correct, and comparable in dataset, metric and split? (F12) |
| R9 | Python implementation; Jupyter notebooks for visualisation | Language check; six notebooks execute top to bottom | — |
| R10 | Datasets and models accessible to the team | Access dry-run from `docs/DATA_ACCESS.md` succeeds | — |
| C6 | "A step closer to being clinically trustworthy … deployable" | — | Is the language proportionate? Retrospective public data, no clinical validation |
| C7 | No reviewed study combines joint grading, quantitative XAI validation and cross-dataset robustness | — | Is the gap claim supported by a complete, reproducible search? |

**Rule:** a requirement or claim that is NOT MET or BLOCKER at the end means the submission is not complete.

---

## 2. Known facts both roles check first

From the Designer's source audit (`docs/DESIGN.md`, Section 3). These are facts the sessions must confirm, not assume.

| Item | Expected | Note |
|---|---|---|
| APTOS 2019 labelled training images | 3,662 | Kaggle competition page |
| APTOS `sample_submission.csv` | 1,928 IDs, placeholder labels all 0 | Not usable for metrics |
| IDRiD grading set | 516 images | Sub-challenge 2 |
| IDRiD images with pixel-level lesion masks | **81** (MA 81, EX 81, HE 80, SE 40) | The paper says 516 (F1, critical) |
| Messidor-2 images with labels | 1,748 from 874 examinations (drops must be logged) | Label scheme to verify (F8) |
| Paper claims IDRiD used for training **and** only for fidelity | Contradictory (Sec. 4.1 vs 5.3) | Design resolves as external test (D3) |

---

## 3. TRACK A: Tester/QA (Session A)

### A1. Environment and static checks (Phase 0)
- Fresh Python 3.10 or 3.11 environment from `requirements.txt`. Record `pip freeze` in `docs/TEST_REPORT.md`.
- Run `ruff check src tests` and `mypy src --ignore-missing-imports`.
- Grep `src/` for: absolute paths (`/home/`, `/root/`, `C:\`), `print(` outside `cli.py`, bare `except:`, unresolved `TODO`/`FIXME`, `df.append(` (removed in pandas 2.x), `verify=False`.
- **Exit:** all findings logged in `docs/DEFECTS.md`.

### A2. Unit tests (Phase 1)
Run `pytest tests/ -m "not slow" -v`. Each file must prove the following.

| File | Must prove |
|---|---|
| `test_manifest.py` | Required columns enforced; wrong dtype rejected |
| `test_adapters.py` | Correct manifests from fixture CSVs; masks only where `has_masks`; declared column names used |
| `test_preprocess.py` | Synthetic bright square stays within 1 px of its mask after crop, CLAHE and resize; CLAHE never applied to masks; masks binary |
| `test_splits.py` | Disjoint train/val/test; strata within 1 percentage point; same seed reproduces file and hash; resplit without flag raises |
| `test_no_leakage.py` | External rows never in training loader; training transforms attached only to training; oversampler draws training indices only |
| `test_model.py` | Shapes: detection `(B,)`, grade `(B,5)`; both heads receive gradients; backbone object shared |
| `test_loss.py` | Loss equals hand-computed value on a 4-sample batch within 1e-6 |
| `test_metrics.py` | Metrics equal scikit-learn within 1e-9; one-vs-rest AUC is NaN with a warning when a class is absent |
| `test_fidelity.py` | IoU and Dice on hand-made maps; disjoint gives 0; identical gives 1; empty mask gives NaN; pointing boundary case; FOV exclusion; top-k% returns k% of FOV pixels ±1 px |
| `test_explainers.py` | Each explainer returns 512×512 float32 in [0,1]; Grad-CAM++ peaks inside a known patch of a toy model |

### A3. Sanity tests (Phase 2) — Tester writes `tests/test_sanity.py`

| Test | Procedure | Pass condition |
|---|---|---|
| S1 Random-label AUC | Shuffle validation labels, 20 times | Mean detection AUC within 0.5 ± 0.05 |
| S2 Random-label QWK | Shuffle grades | Mean QWK within 0.0 ± 0.05 |
| S3 Overfit one batch | 32 images, 200 steps, no augmentation | Loss falls ≥ 90%; grade accuracy ≥ 0.95 |
| S4 Mask shift | Shift masks by 100 px before fidelity | Good-map IoU drops clearly |
| S5 Random-saliency chance | Uniform random maps in FOV | Mean IoU and pointing match code's chance values within ±0.01 |
| S6 Centred saliency | Gaussian at lesion centroid | Pointing near 1.0; IoU above chance |
| S7 Bootstrap coverage | Known Bernoulli rate, 500 repeats | 95% CI covers the true rate in 90–99% of repeats |
| S8 Threshold | Youden threshold on validation | Sens + spec at threshold ≥ value at 0.5 on validation |
| S9 Eval determinism | Eval loaders | No random transform; two calls give identical outputs |
| S10 Class weights | Recompute from training split | Match inverse square-root weights within 1e-6; none from val or test |

### A4. Data integrity (Phase 3)
- Run `prepare-data` on the real download. Compare counts with Section 2. Log every mismatch.
- Record SHA-256 of each raw label file in `docs/TEST_REPORT.md`.
- Dry-run `docs/DATA_ACCESS.md` on a clean folder, following the text only. Log unclear or wrong steps.
- Confirm per-lesion mask counts for IDRiD.
- Confirm no image ID appears in two splits (direct join).

### A5. Integration and reproducibility (Phase 4)
- `test_cli_smoke.py`: `prepare-data`, `make-splits`, `train` (2 epochs, 32 images, CPU), `evaluate`, `xai` (2 images, all four methods), `crossdataset` (small), `efficiency` (CPU). Under 15 minutes.
- `test_reproducibility.py`: two CPU runs with seed 42 give identical metrics (1e-6) and checkpoint hashes.
- Every run writes `run.json` with config snapshot, git hash, package versions, split hash.

### A6. Full-run output checks (Phase 5, after Engineer's full run)

| Output | Check |
|---|---|
| `table4_detection_grading_aptos_test.csv` | Detection and grading metrics, each with CI; n equals APTOS test size; threshold matches `threshold.json` |
| `table5_xai_fidelity.csv` | Four methods × four lesions + "any lesion"; n per lesion matches mask counts; chance columns present; k=10 primary |
| `table6_comparative.csv` | Literature rows match cited values; proposed row equals Table 4; rows labelled |
| `crossdataset_summary.csv` | Three datasets; detection definition stated per dataset; Messidor-2 not pooled with APTOS; stability status stated |
| `ablation_minority.csv` | Same split hash and seed in both arms; paired CI |
| `efficiency.csv` | Parameter count equals `sum(p.numel())`; median and IQR; device and precision recorded |
| Notebooks 00–05 | Execute top to bottom; three numbers per notebook match CSVs |

### A7. Requirement trace and final regression (Phases 6–7)
- Fill the traceability matrix in `docs/TEST_REPORT.md` for R1–R10: test IDs, result files, status (MET / PARTIAL / NOT MET).
- Clean clone, `make reproduce`, time it. Confirm tag `v1.0-submission` points to the tested commit.
- Confirm no file over 50 MB is committed; `data/` and `checkpoints/` ignored.

### A8. Defect severity

| Severity | Meaning | Action |
|---|---|---|
| S0 Critical | Wrong number reported, leakage, or requirement impossible | Block submission; route to Engineer immediately |
| S1 High | Crash, missing required output, method silently not run | Fix before submission |
| S2 Medium | Incorrect non-fatal behaviour or wrong documentation | Fix before submission |
| S3 Low | Style or naming | Fix if time allows |
| S4 Info | Observation for Researcher | Log in issue log for Session B |

---

## 4. TRACK B: Researcher (Session B)

### B1. Relevance trace (abstract → deliverable)
For each claim C1–C7, trace claim → method → module → test → result file → paper section. Mark each link intact or broken.

| Claim | Module | Test | Result file | Paper section |
|---|---|---|---|---|
| C1 | `models/dual_head.py`, `training/losses.py` | `test_model.py`, `test_loss.py` | Table 4 CSV | Sec. 4.3 |
| C2 | `xai/*`, `xai/fidelity.py` | `test_fidelity.py`, S4, S5 | Table 5 CSV | Sec. 4.4, 4.5, 6.2 |
| C3 | `eval/crossdataset.py`, `data/splits.py` | `test_no_leakage.py` | `crossdataset_summary.csv` | Sec. 4.6, 6.3 |
| C4 | `data/oversample.py`, `eval/ablation.py` | `test_no_leakage.py`, A6 | `ablation_minority.csv` | Sec. 4.2, 6.4 |
| C5 | `eval/evaluate_grading.py` (proposed row) | A7 | `table6_comparative.csv` | Sec. 6.1, 6.6 |
| C6 | none (text) | none | `efficiency.csv` | Sec. 6.5, 9 |
| C7 | none (text) | none | none | Sec. 2.5, Table 1 |

### B2. Methodological validity
Verdict per item: SOUND, WEAK (fixable), or UNSOUND (must change).

- **Study design:** single site, retrospective, public data (stated in limitations). Discussion must not imply clinical readiness (C6).
- **Joint vs separate training:** without a separate-task baseline, the paper cannot claim the joint design is better. Expected verdict: WEAK or UNSOUND depending on wording.
- **Backbone:** the code's backbone must match Table 3. "Hybrid CNN/ViT" with a pure CNN run is UNSOUND as a description (F5).
- **Leakage:**
  - Augmentation or oversampling before splitting inflates accuracy. The Alwakid et al. (2023) paper reports 98.36% after augmentation, so check the ordering in this code. UNSOUND if leaked.
  - IDRiD role contradiction (F2) must be fixed in the text.
  - Messidor-2 scheme and detection definition must be stated (F8).
  - EyePACS: if not used, remove the "held-out EyePACS subset" claim (F10).
- **Patient overlap:** confirm against dataset documentation that APTOS and IDRiD have one image per patient. Do not rely on the paper alone.
- **Selection rule:** the design uses 0.5×QWK + 0.5×det-AUC (D9). The paper must state this. "Best validation loss" would be wrong.
- **Loss weights:** the paper says λ_det and λ_grade were "tuned"; the design uses fixed 0.4/0.6. Either report the tuning or say they were fixed a priori.
- **Seeds:** one seed is weak evidence. Paper must say so and must not call small differences significant.
- **Metrics:** QWK is appropriate for ordinal grades. Macro metrics need justification given rare classes. Every headline metric needs a CI (D8).
- **Ablation (C4):** a paired bootstrap is the correct test. An unpaired comparison of single runs is not valid evidence. If the CI includes zero, the paper must not say augmentation "improves" fidelity.
- **Multiple comparisons:** four methods × four lesions × three k values. State no correction, or apply Holm and report it.
- **Small n:** SE on 40 IDRiD images. Lesion-level results must be labelled exploratory.

### B3. Explainability validity (C2)
1. **Grad-CAM++ (F3):** check whether Captum's `LayerGradCam` is Grad-CAM or Grad-CAM++ using the Captum documentation and the installed source. Record URL and version. Correct the citation. UNSOUND if the attribution is wrong.
2. **SHAP (F4):** the implemented estimator must match "masked-image kernel explainer" in the paper. The inpainting background must be stated; it strongly affects SHAP maps.
3. **LIME:** sample count, segmentation and superpixel settings must be reported.
4. **Attention saliency:** attention weights are not faithful explanations unless a faithfulness test is run. The paper must describe it as a model-internal signal unless tested.
5. **Threshold (F11, D6):** top-10% is a choice. The paper must show results at k = 5, 10 and 20, and say whether conclusions hold.
6. **Chance baseline:** IoU values are uninterpretable without it. Confirm the paper reports chance IoU and chance pointing.
7. **Pointing game:** large lesions are hit by chance more often. Chance pointing must be reported beside it.
8. **Mask coverage:** explanation validity is tested on 81 images only. The paper must say so, and say the 516 grading images have no masks.
9. **Cross-domain stability (F9):** Messidor-2 has no masks. Paper currently implies stability across all three datasets. Approve Proposal A or B (design Section 10.5), or limit the claim to IDRiD.
10. **Clinical meaning:** overlap with a lesion shows the model looked at lesion-like tissue, not that it used the lesion. The paper must not call explanations "clinically trustworthy" on this evidence.

### B4. Literature and citation checks (Table 4, Table 6, references)

**Numbers:**
- **Chilukoti et al. [48]:** Table 4 gives Messidor QWK 0.944; Section 6.3 gives 0.854 (EyePACS-pretrained on Messidor-2) and 0.967 (APTOS). Confirm which number belongs to which model.
- **Dinpajhouh & Seyyedsalehi [50]:** 98.44% detection accuracy on APTOS. Confirm test protocol and split.
- **Akhtar et al. [49]:** "multi-center incl. APTOS". Confirm datasets and subset.
- **Van Craenendonck et al. [51]:** ECS 0.51 used as ceiling for IoU/Dice. ECS is not IoU or Dice. Paper must say they are not comparable.

**References:**
- Refs [4] and [5] share DOI `10.1038/s41598-026-58292-6`. Confirm which paper each belongs to.
- Ref [5] names Graefe's Archive with a Sci. Rep.-style DOI. Confirm venue.
- Incomplete: [3] (volume), [21] (journal), [37] (volume and pages).
- Check [46] (SHAP) and [47] (LIME) cite the method papers actually used.
- Check [43] (ViT) supports the hybrid-architecture sentence; it is a pure ViT paper.

Resolve each via `https://doi.org/<DOI>` (WebFetch), and record the URL and title.

**Literature gap (C7):** run at least three targeted searches (for example: "diabetic retinopathy" with "Grad-CAM" and "IDRiD" and "pixel-level"; "diabetic retinopathy" with "cross-dataset" and "explainable"; "diabetic retinopathy" with "joint detection and grading" and "multi-task"). Log queries and candidate counts. If any paper meets all three criteria, soften C7.

### B5. Targeted searches in supplied papers (token-safe)
- Alwakid et al. (2023): search "augment" and "split" to confirm the ordering of augmentation and splitting.
- Alavee et al. (2024): search "Grad-CAM", "Messidor", "accuracy" to see how their XAI was validated (supports the novelty framing in C2).
- Bashir et al. (2023) and Ishtiaq et al. (2023): search "dataset" and "EyePACS" to check for test-set overlap with APTOS or IDRiD.

### B6. Verdicts
Each claim C1–C7 gets READY, FIX, or BLOCKER. Overall: PUBLISHABLE AS-IS, PUBLISHABLE AFTER FIXES, or NOT YET PUBLISHABLE, with at most three reasons.

---

## 5. JOINT CHECKS (both roles, overlapping evidence)

These items sit where the Tester's evidence feeds the Researcher's verdict. Each has one owner for evidence and one for judgement.

| Joint item | Tester provides (evidence) | Researcher decides (judgement) |
|---|---|---|
| J1 Fidelity protocol | S4, S5, S6 outputs; `test_fidelity.py` | Whether k-sensitivity and chance baseline satisfy C2 |
| J2 Leakage | `test_no_leakage.py`; split hashes; ID-overlap join | Whether any leakage invalidates a claim |
| J3 Ablation | Config diff (exactly one key differs); shared split hash | Whether paired CI supports "improves fidelity" |
| J4 Cross-dataset | `crossdataset_summary.csv` structure; no pooled labels | Whether external datasets and label definitions support C3 |
| J5 Stability | Status field per dataset; code for Proposal A | Approval of stability definition (F9) |
| J6 Number traceability | Each paper number maps to a CSV row | Whether the paper's narrative matches the number's meaning |
| J7 Literature rows | Table 6 literature rows match DOIs (Tester checks the numbers copied) | Whether cited comparisons are fair (same dataset, metric, split) |
| J8 Protocol vs paper | Config snapshot values | Whether paper text describes the same protocol (LR, batch, selection rule, backbone) |

**Rule:** if the Tester finds that a number traces to the wrong row, the Tester logs it (defect). The Researcher then decides whether the claim built on it still holds.

---

## 6. Shared issue log and routing

### 6.1 `docs/ISSUE_LOG.md` (append-only)
Each entry is one block. Only the author edits their own entry's status line.

```
## ISSUE-<nnn>  [<severity>]  <short title>
From:        Tester | Researcher
To:          Tester | Researcher | Software Engineer | Authors | Editor
Refs:        DEF-xxx / claim C# / requirement R# / file:line / paper section
Finding:     one or two sentences
Evidence:    file path or URL
Needed:      what must happen to close it
Status:      OPEN | IN-PROGRESS (by <role>) | CLOSED (commit or text change) | WONTFIX (reason)
```

### 6.2 Routing rules (from the project instructions)
- **Code defect** found by Tester: `DEFECTS.md` plus an issue to the **Software Engineer**. The user sends it to the Engineer session.
- **Scientific or protocol issue** found by either role: issue to **Researcher** for judgement, and to the **Software Engineer** if code changes are needed. The Tester re-tests after the fix.
- **Paper-only text issue**: issue to **Authors**, and to the **Editor** when that role begins.
- **Inconsistency between code and paper or abstract**: Tester checks the evidence first, then the Researcher decides which side changes. The project instructions require checking with Tester and Software Engineer, so the Researcher must name both in the issue.
- The Editor does not draft results text until Researcher verdicts for C1–C5 are READY or FIX-complete.

### 6.3 Closing loop
After the Engineer reports a fix, the Tester re-runs the affected tests and marks the defect CLOSED with the commit hash. The Researcher re-reads the affected claim and updates its verdict. Both update only their own files and their own issue entries.

---

## 7. Sync points (the two sessions run in parallel)

| Sync | When | Tester does | Researcher does |
|---|---|---|---|
| SP0 | Sun 11 Oct, after scaffold | Phase 0 and unit tests start | Read claims; start literature checks B4 |
| SP1 | Mon 12 Oct, after first training run | Sanity tests S1–S10; Phase 3 data checks | Check C1 and B2 data and leakage items |
| SP2 | Tue 13 Oct, after Table 5 and ablation exist | Phase 5 output checks | B3 explainability; B2 statistics |
| SP3 | Wed 14 Oct, 11:00 IST | Traceability matrix; list of open S0–S1 defects | Finalise verdicts; list BLOCKERs |
| SP4 | Wed 14 Oct, 13:00 IST | Final regression on clean clone | Confirm each BLOCKER is closed or claim removed |

Before each sync point, each session reads `docs/ISSUE_LOG.md` for entries addressed to it. If the other session has posted something that changes its work, it finishes the current step and then acts on the entry.

---

## 8. Output files

| File | Owner | Content |
|---|---|---|
| `docs/TEST_REPORT.md` | Tester | Environment, phase results, data integrity table, R1–R10 trace matrix, sign-off or blockers |
| `docs/DEFECTS.md` | Tester | Defect records with severity, route, status |
| `docs/evidence/` | Tester | Logs and output excerpts for each PASS and FAIL |
| `docs/SCIENTIFIC_REVIEW.md` | Researcher | Claim verdicts C1–C7, methodology and explainability findings, literature log, overall publishability verdict |
| `docs/evidence/research/` | Researcher | Saved literature pages, search queries, candidate lists |
| `docs/ISSUE_LOG.md` | Shared (append-only) | Cross-role issues |

### 8.1 `SCIENTIFIC_REVIEW.md` template
```
# Scientific Review: DR-XAI
Date / Researcher model / inputs read (files and sections)

## Summary
Overall verdict: PUBLISHABLE AS-IS | AFTER FIXES | NOT YET
Blockers: n   Fixes: n   Ready: n

## Claim verdicts
| Claim | Verdict | Key reason | Route to |

## Findings
### F-001 [BLOCKER|FIX|NOTE] Title
Where: paper Sec. X / file:line / results file
What is wrong:
Why it matters for the abstract or peer review:
Evidence checked (file, line, URL):
Route to:
Suggested fix:

## Relevance table (Section B1 completed)
## Literature verification log (item | source | URL | matches? | note)
## Open questions for the team
```

### 8.2 `TEST_REPORT.md` requirement line
```
R3  Quantitative fidelity vs expert masks
    Status:   PARTIAL
    Tests:    test_fidelity.py (PASS 9/9), S4 (PASS), S5 (FAIL: chance IoU off by 0.03)
    Results:  results/table5_xai_fidelity.csv (4 methods; n per lesion as expected)
    Open:     DEF-004 (S1); ISSUE-007 (Researcher, claim C2)
```

### 8.3 `DEFECTS.md` record
```
ID: DEF-001   Severity: S0|S1|S2|S3|S4   Requirement: R# or none
File: src/xaidr/xai/fidelity.py:42   Found in: Phase 1, test_fidelity.py::test_topk_fraction
Observed: top-k returns 9.8% of FOV pixels for k=10
Expected: 10% ±1 px
Steps: python -m pytest tests/test_fidelity.py::test_topk_fraction -v
Evidence: docs/evidence/...
Route to: Software Engineer
Status: OPEN | CLOSED (commit)
```

---

## 9. Timing

| Day | Tester (Session A) | Researcher (Session B) |
|---|---|---|
| Sun 11 Oct | Phase 0, Phase 1 on first modules | B1 trace; B4 numbers and references start |
| Mon 12 Oct | Phase 2 sanity tests; Phase 3 data | B2 study design, leakage, data protocol |
| Tue 13 Oct | Phase 4 integration and reproducibility; Phase 5 as outputs arrive | B3 explainability; B5 targeted searches; B4 gap search |
| Wed 14 Oct, morning | Phase 6–7; sign-off by 11:00 IST | Verdicts by 11:00 IST; BLOCKERs closed or claims removed by 13:00 IST |

If Phase 5 outputs are not ready by Tuesday evening, the Tester records NOT MET for those rows rather than waiting. The Researcher then writes verdicts as "cannot assess" for those claims, which is an honest result.

---

## 10. Limits of each role
- **Tester:** does not fix code, judge the science, or alter results.
- **Researcher:** does not run training, change code or tests, or decide clinical use. Clinical validation is outside this project.
- **Both:** neither changes the abstract. If the abstract overstates the work, the team revises it with the supervisor.
- **Both:** neither reports a number that is not in a saved result file.
