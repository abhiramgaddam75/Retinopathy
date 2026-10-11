# Scientific Review: DR-XAI

| | |
|---|---|
| Date | 2026-10-11 (sync point SP0) |
| Researcher | Claude Opus 5.5, in the same session that acted as Software Engineer and Tester. The brief assigns a separate session, so this review is **not independent** of the code it discusses. |
| Inputs read | Paper draft `Explainable AI_.docx` (all sections, tables, 51 references); abstract form `U18IT707 ... Abstract`; `docs/DESIGN.md`; `docs/VERIFICATION_AND_RESEARCH_BRIEF.md`; `docs/TEST_REPORT.md`, `docs/DEFECTS.md`, `docs/ISSUE_LOG.md`, `docs/DECISIONS_LOG.md`; repository at commit `f0ab1e4`; targeted pattern searches in the four supplied papers; the web sources in `docs/evidence/research/literature_log.md` (L1-L13, S1-S9, G1-G3) |
| Not done | Refs [6]-[18], [25]-[36], [38]-[47] were not opened. Dinpajhouh & Seyyedsalehi [50] could not be read (paywall). No dataset is on disk and no model has been trained, so nothing about results could be assessed. |

## Summary

**Overall verdict: NOT YET PUBLISHABLE.**

Blockers: 3 claims   Fixes: 4 claims   Ready: 0 claims

Reasons (at most three):

1. **There are no results.** The paper's Section 6 says its own numbers are "to be measured", and the
   repository implements only the data layer (milestone M0). Every claim in the abstract is a plan.
2. **The benchmark tables contain citation errors.** Two of the three rows in Tables 4 and 6 are
   attributed to the wrong paper or misread from the source, and one confidence interval is wrong.
3. **Statements about the datasets and the protocol contradict the facts and the design.** IDRiD
   masks (516 vs 81), the role of IDRiD, EyePACS, the Grad-CAM++ library, the backbone, and
   "explanation stability" on a dataset without lesion masks.

All three are fixable. None requires abandoning the project.

## Claim verdicts

| Claim | Verdict | Key reason | Route to |
|---|---|---|---|
| C1 Joint detection and grading as the contribution | FIX | Not implemented yet; no separate-task baseline is planned, so "joint is better" cannot be claimed; the detection label is a function of the grade (F-010) | Authors, Software Engineer |
| C2 Four explainers, validated quantitatively against lesion masks | BLOCKER | 516-vs-81 mask error, Grad-CAM++ attributed to a library that does not provide it, protocol gaps (F-005, F-006, F-012); no results | Authors, Software Engineer |
| C3 Cross-dataset robustness of predictions and of explanations | BLOCKER | Explanation stability is undefined where no masks exist; EyePACS and IDRiD statements contradict the build; the cited "domain shift" figure is misread (F-002, F-007, F-008) | Authors, Software Engineer |
| C4 Synthetic augmentation ablation | FIX | The method is oversampling, not synthetic data; supporting citations are mischaracterised and one is retracted (F-009) | Authors |
| C5 Expected performance grounded in published results | BLOCKER | Akhtar et al. row is not that paper's result; Chilukoti comparison is invalid; one source unverified (F-001 to F-004) | Authors |
| C6 "Clinically trustworthy ... deployable" | FIX | Disproportionate for retrospective public data with no clinical validation (F-016) | Authors |
| C7 No reviewed study combines the three dimensions | FIX | True of the 16 listed studies, but at least one 2026 preprint combines two of the three with the same metrics; the search is not reproducible (F-015) | Authors |

## Findings

### F-001 [BLOCKER] Tables 4 and 6 attribute another paper's results to Akhtar et al. [49]
Where: paper Table 4 row 2, Table 6 row 2, Section 7.
What is wrong: The row reads "Akhtar et al. [49], 2025 | Multi-center incl. APTOS 2019 | 98.50% / (99.46% sens.) | 89.60% / 93.00%". Akhtar et al. is a Messidor-1-only study whose own results are 99.36% (4-class) and 99.37% (binary) accuracy, with no kappa. The four numbers in the row appear in that paper only in its related-work section, where they describe Shakibania et al., "Dual Branch Deep Learning Network for Detection and Stage Grading of Diabetic Retinopathy", evaluated on APTOS 2019. That study is the draft's own ref [27].
Why it matters: A reviewer who opens [49] will not find these numbers. It also changes the argument: Shakibania et al. is a joint detection-and-grading model on APTOS 2019, which is direct prior art for C1.
Evidence checked: L3, L4, L5.
Route to: Authors.
Suggested fix: Attribute the row to Shakibania et al. [27], with dataset "APTOS 2019", after reading that paper for its split protocol. Remove "multi-center". Describe [27] correctly in Section 2.3 (it is currently called a "biomedical-signal" approach) and add it to Table 1 as a joint model.

### F-002 [BLOCKER] Section 6.3 misreads Chilukoti et al. [48]; it is not cross-dataset evidence
Where: paper Section 6.3, Fig. 4, Table 4 row 3, Table 6 row 3.
What is wrong: The draft says a model "pretrained only on EyePACS reached a QWK of 0.854" on Messidor-2, "compared with 0.967 ... on APTOS 2019 — a gap of roughly 0.11 QWK attributable to domain shift". In the source, 0.854 is Messidor-2 **without** DR pretraining and without the ensemble; 0.944 is Messidor-2 **with** EyePACS pretraining and the ensemble. The draft compares two different datasets (Messidor-2 vs APTOS) and has the pretraining condition reversed. Every model in that table is fine-tuned on the dataset it is scored on; nothing is tested zero-shot. The source also does not report the APTOS or Messidor-2 splits, or whether its Table 6 is validation or test.
Why it matters: The paragraph is the paper's only quantitative motivation for C3, and it does not show domain shift. The 0.967 in Table 6 cannot be compared with a held-out 15% test split.
Evidence checked: L1, L2.
Route to: Authors.
Suggested fix: Rewrite as: on Messidor-2, EyePACS pretraining plus ensembling raised QWK from 0.854 to 0.944 in [48]. State that [48] reports fine-tuned, not zero-shot, performance and does not describe its APTOS split. Cite a genuinely zero-shot cross-dataset study for the domain-shift magnitude, or state that no such figure is given.

### F-003 [FIX] Van Craenendonck et al. [51]: wrong confidence interval, and ECS used as an IoU/Dice ceiling
Where: paper Section 6.2, Table 5, Section 7.
What is wrong: (a) The draft gives "ECS of 0.51 (95% CI: 0.46-0.51)". The source gives [0.46; 0.55]. (b) ECS is the overlap of the top-15 cells of a 10x10 grid; the source argues pixel-overlap scores are unsuitable. Section 7 nevertheless says fidelity scores that "meet or exceed the 0.51 ECS ceiling" would be an improvement. IoU against lesions covering 1-2% of the image cannot approach 0.5 for reasons unrelated to explanation quality (see F-012). (c) The 0.51 is on 54 tuning images; the 27-image test score was 0.48. (d) Their models were trained on EyePACS and Grad-CAM++ was not among the ten methods.
Why it matters: The stated success criterion is not achievable by construction and would make good results look like failure.
Evidence checked: L6.
Route to: Authors.
Suggested fix: Correct the interval. Say plainly that ECS and IoU/Dice are not comparable and remove the "ceiling" sentence. To compare with [51], compute ECS as well (it is cheap once the maps exist; route to Software Engineer if wanted).

### F-004 [FIX] Dinpajhouh & Seyyedsalehi [50] numbers could not be verified
Where: paper Table 4 row 1, Table 6 row 1.
What is wrong: Bibliographic details match Crossref. The values 98.44%, 99.55%, 83.69% and 89.26% could not be checked: the article is paywalled and two searches found no page containing 98.44 or 83.69.
Why it matters: After F-001 and F-002, every number in these tables needs a source check before submission.
Evidence checked: L11, S3, S4.
Route to: Authors.
Suggested fix: Open the article through the institution's access, record the page and table for each number, and record the test protocol (split, whether augmentation preceded the split).

### F-005 [BLOCKER] IDRiD is described incorrectly in four places
Where: paper Section 4.1, Table 2, Section 4.5, Section 5.3, Section 8.
What is wrong: (a) "516 fundus images with ... pixel-level expert segmentation masks": pixel-level masks exist for 81 images; 516 is the grading set. (b) Section 4.1 says IDRiD is used "exclusively" for fidelity; Section 5.3 says it is split 70/15/15 for grading; the build uses it only as an external test set (D3). (c) Section 4.5 says "every IDRiD test image". (d) Section 8 gives 516 as the sample size limiting the fidelity conclusions; the limit is 81, and 40 for soft exudates. (e) Section 5.3 states that APTOS 2019 and IDRiD "provide at most one fundus image per patient". I found no documentation supporting this for either dataset. APTOS 2019 is reported by competition participants to contain duplicate images (one count: 262).
Why it matters: (a) and (d) overstate the evidence base for the central claim by a factor of six. (e) is used to justify not de-duplicating.
Evidence checked: L6 (81 images, 54/27), S7, S8; `docs/DESIGN.md` F1, F2; the per-lesion counts 81/81/80/40 are from the design brief and are **not yet confirmed** by me or by the Tester.
Route to: Authors; Tester (confirm lesion counts in Phase 3).
Suggested fix: State 516 grading images and 81 lesion-annotated images, with per-lesion counts from the download. Describe IDRiD as external for grading and as the fidelity set. Delete the one-image-per-patient sentence and describe the duplicate handling actually used (F-014).

### F-006 [BLOCKER] Grad-CAM++ and attention saliency are attributed to Captum
Where: paper Section 5.1.
What is wrong: The draft says the methods use "the Captum library for Grad-CAM++ and attention-based saliency ... so that each method follows its original authors' reference implementation". Captum's documented CAM class is `LayerGradCam`, the original Grad-CAM. The build uses `pytorch-grad-cam` (`GradCAMPlusPlus`), which is a third-party implementation, not the original authors'. Captum has no attention-saliency method for timm models; the build plans a forward hook on MaxViT's last attention block.
Why it matters: A wrong implementation statement for the headline method is a factual error a reviewer can check in one minute.
Evidence checked: L7 (first 100,000 characters of the page; the remainder not read); `docs/VERIFICATION.md` V3, V5, V8.
Route to: Authors.
Suggested fix: Name `pytorch-grad-cam` with its version, drop "original authors' reference implementation", and describe the attention extraction as implemented once it exists.

### F-007 [BLOCKER] "Explanation stability across three datasets" is not defined
Where: abstract; paper Sections 1, 4.6, 6.3; `docs/DESIGN.md` 10.5.
What is wrong: Section 4.6 says explanation consistency is measured "via the fidelity metrics of Section 4.5, computed against the IDRiD-trained saliency baselines". Lesion masks exist only for 81 IDRiD images, so the fidelity metrics cannot be computed on APTOS or Messidor-2. No model is trained on IDRiD in this design, so "IDRiD-trained saliency baselines" do not exist.
Decision (requested by the design, F9, and by joint check J5):
- **Proposal B is rejected.** It needs a model trained on IDRiD, which contradicts D3 and would leave too few images.
- **Proposal A is approved with two conditions.** (1) Three seeds are trained (D12). (2) The Spearman agreement is reported for the same method on all three test sets (APTOS test, IDRiD, Messidor-2), so that the comparison is in-domain versus external. Agreement between seeds on one dataset is seed stability, not cross-dataset stability.
- The quantity must be named for what it is: "between-seed explanation agreement". It says nothing about correctness.
- **If three seeds cannot be trained by the deadline,** the abstract and Sections 1, 4.6 and 6.3 must drop "explanation stability across three datasets" and claim lesion-level fidelity on IDRiD only.
Why it matters: It is one of the three contributions listed in the abstract.
Evidence checked: paper text; `docs/DESIGN.md` Sections 5 and 10.5.
Route to: Authors, Software Engineer.

### F-008 [BLOCKER] EyePACS is described as used; it is out of scope
Where: paper Sections 4.1, 4.6, 5.2, 5.3, 6.3, Tables 2 and 3.
What is wrong: The draft describes optional EyePACS pretraining and evaluation on "a held-out EyePACS subset". The build excludes EyePACS (D4). If both were done without a fixed split, the test would overlap the pretraining data.
Route to: Authors.
Suggested fix: Move EyePACS to future work everywhere, including Table 2 and the backbone row of Table 3.

### F-009 [FIX] "Synthetic-data augmentation" is minority oversampling; supporting citations are weak
Where: abstract; paper Sections 1, 4.2, 6.4, Table 3.
What is wrong: (a) The implemented method repeats Severe and Proliferative training images three times with random flips, rotation and brightness jitter. No image is synthesised. Section 4.2 says the module "generates additional minority-class samples, following ... Vasireddi et al. [5]", which describes generative synthesis. (b) Section 6.4 says [5], [19] and [20] report that "GAN/ESRGAN-based augmentation improves classification". ESRGAN in [19] is super-resolution enhancement, not class-balancing augmentation. (c) Ref [20] (Alwakid et al., Digital Health 2023) appears to have been retracted for redundant publication with [19]; the copy in the team's folder is itself named "retracted". (d) In [19], 80% of the data is given as 9,360 images although APTOS 2019 has 3,662, which implies the split was made after augmentation; its 98.36% is therefore not a clean test estimate.
Why it matters: The abstract names synthetic data as a component. Citing a retracted paper as support is avoidable.
Evidence checked: `src/xaidr/data/oversample.py`, `src/xaidr/data/augment.py`; S9 (mirror of the notice; confirm on the publisher's site); literature log section D.
Route to: Authors (the abstract wording needs the supervisor, per brief Section 10).
Suggested fix: Rename to "minority-class oversampling with augmentation" throughout, or add a generative method. Remove [20] or cite it as retracted. Use [19] as an example of the leakage this design avoids, not as support.

### F-010 [FIX] The joint design is presented as a contribution without a baseline
Where: abstract (i); paper Sections 1, 4.3, 7.
What is wrong: (a) No separate-task or single-head baseline is planned, so the draft cannot say the joint design helps. (b) The detection label is defined as grade >= 1, so the detection target is a deterministic function of the grading target. A grading-only model already yields a detection score as 1 - P(grade 0). The second head adds a loss term, not new supervision. (c) Joint detection-and-grading on APTOS 2019 already exists in the draft's own references ([27], and the title of [50]).
Why it matters: A reviewer will ask what the second head buys.
Evidence checked: `src/xaidr/data/adapters/aptos2019.py:38` (dr_label from grade); L4, L5, L11.
Route to: Authors; Software Engineer if the baseline is added.
Suggested fix: Either add one ablation (grading head only, detection derived from it, same seed and split, paired bootstrap on detection AUC and QWK), or reword to "we adopt a dual-head design" and remove any comparative language. The ablation costs one extra training run.

### F-011 [FIX] The paper's protocol does not match the configuration that will run
Where: paper Sections 4.3, 5.2, 5.3, Table 3 versus `configs/default.yaml`.

| Item | Paper | Build | Change |
|---|---|---|---|
| Backbone | "CNN / ViT hybrid (pretrained on ImageNet / EyePACS)" | `maxvit_tiny_tf_512`, ImageNet-1k | Name the model; drop EyePACS |
| Loss weights | "tuned on the validation split" | Fixed 0.4 / 0.6 | Say fixed a priori, or report the tuning |
| Loss form | "class-weighted categorical cross-entropy (or ordinal)" | Class-weighted CE, inverse square-root frequency | Remove "(or ordinal)"; state the weighting |
| Batch size | 32 | 16 with accumulation 2 | State both |
| Model selection | not stated | 0.5 x QWK + 0.5 x detection AUC on validation, patience 8 | State it |
| Detection threshold | not stated | Youden's J on validation, frozen | State it |
| Split | APTOS and IDRiD each 70/15/15 | APTOS only; duplicates grouped | Rewrite Section 5.3 |
| Saliency threshold | "we threshold each method's saliency map" | Top 10% of field-of-view pixels; 5% and 20% as sensitivity | State it and the chance baseline |
| Input | "resized to a fixed input resolution" | Border crop, CLAHE on L channel, direct resize to 512 x 512 | See F-013 |
| Seeds | "fixed and logged" | One seed (42) | Say one seed; do not call small differences significant |
| Compressed model | "quantized/pruned variant" will be benchmarked | Listed as SHOULD, not planned | Remove unless done |

Route to: Authors. This is joint check J8; the Tester supplies the config snapshot once a run exists.

### F-012 [FIX] The fidelity protocol has four design problems to settle before Table 5 is computed
Where: `docs/DESIGN.md` Sections 7.21, 10.3, D6, D11; paper Sections 4.5, 5.4.
1. **IoU has a low ceiling that depends on lesion size.** With the top k% of pixels selected and a lesion covering a fraction a of the field of view, IoU cannot exceed a/k when a < k. A lesion covering 2% of the field gives a maximum IoU of 0.20 at k = 10; microaneurysms are far smaller. Chance alone does not make this interpretable. Report, for every row, the maximum attainable IoU and Dice beside the chance value, and add a size-independent measure: the fraction of lesion pixels inside the top-k region (lesion recall at k) with its chance value k/100.
2. **The explained class can be "No DR".** D11 explains the predicted grade. If the model predicts grade 0 for a lesion image, the map shows evidence for the absence of disease, and overlap with lesions is meaningless. Explain a fixed disease target for the fidelity study (for example the summed logits or probability of grades 1-4), and report how many of the 81 images were predicted as grade 0.
3. **Resolution.** Grad-CAM++ on the last MaxViT stage and SHAP on a 16 x 16 grid both resolve about 32 x 32 pixels at 512 px. Microaneurysm results will be dominated by this, not by the model. Label MA results exploratory for these two methods and say why.
4. **Multiple comparisons and small n.** Four methods x five lesion targets x three k values is sixty comparisons with chance. Declare k = 10 as the single primary analysis, treat k = 5 and 20 as sensitivity only, and apply Holm correction across the 20 primary comparisons, or state that none is applied. Rows with n < 20 are exploratory (soft exudates has about 40; the pointing game on small n is coarse).
Also: overlap shows where the saliency falls, not that the model used the lesion. The paper must not present it as evidence of clinical reasoning (brief B3 item 10). Attention maps are a model-internal signal and should be described as such unless a faithfulness test (deletion or insertion) is run.
Route to: Software Engineer and Designer (items 1 and 2 change `xai/fidelity.py` and D11 before M2); Authors (wording).

### F-013 [FIX] Judgement on ISSUE-003: the aspect-ratio stretch should be removed
Where: `docs/DESIGN.md` 7.5; `src/xaidr/data/preprocess.py` (resize_image); DEF-007.
Judgement: **WEAK, change it.** The Tester measured a vertical stretch of 0-0.5% on six sample photos, about 4.4% on two and about 27% on two whose retina is cut off at the top and bottom. The amount depends on the camera and crop of each source, so it differs between APTOS, IDRiD and Messidor-2. That adds an acquisition-dependent shape distortion to a study whose purpose is to measure cross-dataset behaviour, and it changes lesion shapes that the fidelity study measures. Masks stay aligned, so nothing is invalid, but the distortion is avoidable.
Suggested fix: pad the field-of-view crop to a square with black before resizing, record the padding in the manifest, apply the same to masks. Do this before any real data is preprocessed. Requires a Designer note because it changes the transform record in Section 7.3.
Route to: Software Engineer, Designer.

### F-014 [FIX] Judgement on ISSUE-002: duplicate images
Where: `src/xaidr/data/splits.py`; DEF-002.
Judgement: grouping byte-identical copies into one split, as now implemented, removes the leakage. Two decisions were left to the Researcher:
1. **Copies with the same grade:** keep one copy only. Extra copies give that image extra weight in training and are counted twice in test metrics.
2. **Copies with conflicting grades:** exclude the whole group from validation and test, because the reference label is not defined; they may stay in training only if the group falls in the training split, otherwise drop them. Report both counts in the paper.
Only byte-identical files are detected. Say so in the limitations. The actual counts are unknown until APTOS is on disk; severity for the paper depends on them.
Route to: Software Engineer (implement), Tester (re-test in Phase 3), Authors (report counts).

### F-015 [FIX] The gap claim (C7) should be narrowed
Where: paper Sections 2.5, 3, 7.
What is wrong: The sentence is limited to "the works we reviewed", which is defensible for the 16 studies in Table 1. Three things weaken it. (a) After F-001, [27] should be in Table 1 as a joint model. (b) A 2026 preprint (arXiv 2608.28207) scores Grad-CAM and HiResCAM against IDRiD lesion masks with Dice, IoU and a pointing game and adds external evaluation; that is two of the three dimensions, with the same three metrics this paper proposes. Van Craenendonck et al. [51] is a second quantitative precedent. (c) The literature search is not described, so "none" cannot be reproduced.
My three searches found no work meeting all three criteria, which supports a narrow claim. They are not a systematic search.
Evidence checked: G1-G3, L12, L13, L6.
Route to: Authors.
Suggested fix: State the search (databases, strings, dates, counts). Cite arXiv 2608.28207 and position this work against it. Claim: "to our knowledge, no study combines all three", not that quantitative validation is new.

### F-016 [FIX] Clinical language is disproportionate (C6)
Where: abstract last sentence; keywords; Sections 1, 7, 9.
What is wrong: "clinically trustworthy ... deployable in real-world settings" rests on retrospective public images, one training site, no reader study and no prospective data. Section 8 already says deployment claims are feasibility evidence only; the abstract should agree with it.
Route to: Authors (abstract change needs the supervisor).
Suggested fix: "a step toward explanations that can be checked quantitatively against expert lesion annotations".

### F-017 [FIX] References and table presentation
- Ref [5]: wrong DOI (it duplicates [4]). Correct DOI 10.1007/s00417-026-07284-3; the venue, Graefe's Archive, is right (L9).
- Refs [21]-[28]: no titles. Titles for [21]-[24] are in the literature log, table C. [3]: add volume 16, article 27354. [37]: add pages 1395-1405. [24]: add volume 12, pages 73950-73969.
- Ref [23] is a survey, but Section 2.3 lists it as a classification study.
- Ref [43] (Dosovitskiy et al.) is a pure vision-transformer paper. It does not support "hybrid convolutional/vision-transformer design"; cite the MaxViT paper for the backbone actually used. (Judged from the citation; [43] was not re-opened.)
- Ref [46] introduces Kernel SHAP. If the build uses the partition explainer with an inpainting masker, say so; the inpainting background strongly affects the maps and must be reported (brief B3 item 2).
- QWK is written as a decimal in Table 4 (0.901, 0.967, 0.944) and as a percentage in Tables 4 and 6 (89.26%, 93.00%, 96.70%). Use one form.
- The draft lists only the supervisor as author.
Route to: Authors, Editor.

### F-018 [NOTE] Statistical points to carry into the results text
- One seed: the paired bootstrap in the ablation captures test-set sampling error only, not training variability. A difference whose interval excludes zero still comes from one training run per arm. Say so, and do not write "improves" if the interval includes zero.
- Messidor-2 has two images per examination (874 examinations, 1,748 images). Bootstrap by examination, not by image, or the intervals will be too narrow.
- Macro-averaged sensitivity and macro AUC give Severe and Proliferative the same weight as the large classes; with about 29 and 44 test images these estimates are noisy. Report per-class values with intervals.
- Messidor-2 labels come from a third party (the commonly used set is the Krause et al. adjudicated grades, five levels, with a gradability flag; four images ungradable). The release used, its scale and the detection definition must be stated; the build's default of "referable" differs from the "any DR" definition used for APTOS and IDRiD, and the two must not share a table column.

### F-019 [NOTE] Leakage in the cited comparison literature
Three of the papers the draft leans on split after augmenting, or test on data that includes APTOS: Alwakid et al. [19] (9,360 "training" images from a 3,662-image dataset), Akhtar et al. [49] ("the augmented dataset is partitioned"), Bashir et al. [21] (merged set containing 2,690 APTOS images, 70/30 split). Their accuracies are not comparable with a model tested on unaugmented, de-duplicated held-out images. The paper should say this in Section 6 and present its own split-first protocol as a design choice, which the code enforces (`tests/test_no_leakage.py`, split lock).

## Methodological validity (brief B2)

| Item | Verdict | Note |
|---|---|---|
| Study design | SOUND as a retrospective benchmark study | Wording in abstract must match (F-016) |
| Joint vs separate training | WEAK | No baseline (F-010) |
| Backbone description | UNSOUND as written | "Hybrid" is right for MaxViT, but EyePACS pretraining and ref [43] are wrong (F-011, F-017) |
| Leakage: augmentation and oversampling order | SOUND in the code | Split first; sampler accepts training rows only; loaders verify the split lock. Synthetic data only so far |
| Leakage: duplicates | WEAK until measured | F-014 |
| IDRiD role | UNSOUND in the paper | F-005 |
| Messidor-2 scheme | WEAK | Not yet declared in config or paper (F-018) |
| EyePACS | UNSOUND in the paper | F-008 |
| Patient overlap statement | UNSOUND | Unsupported sentence in Section 5.3 (F-005) |
| Selection rule, loss weights | WEAK | Not stated or misstated (F-011) |
| Seeds | WEAK | One seed (F-018) |
| Metrics | SOUND | QWK suits ordinal grades; intervals are planned for every metric |
| Ablation | SOUND in design | Paired bootstrap, one config key differs (Tester test); see F-018 for its limit |
| Multiple comparisons | WEAK | Not addressed (F-012) |
| Small n | WEAK | Must be labelled exploratory (F-012) |

## Relevance table (brief B1)

| Claim | Module | Test | Result file | Paper section | Chain |
|---|---|---|---|---|---|
| C1 | `models/dual_head.py`, `training/losses.py`: **absent** | absent | absent | Sec. 4.3 | Broken at the module |
| C2 | `xai/*`, `xai/fidelity.py`: **absent**. Mask alignment present (`data/preprocess.py`) | `test_preprocess.py` only | absent | Sec. 4.4, 4.5, 6.2 | Broken at the module |
| C3 | `data/splits.py` present; `eval/crossdataset.py` **absent** | `test_no_leakage.py`, `test_splits.py` pass | absent | Sec. 4.6, 6.3 | Intact to the test for leakage; broken after |
| C4 | `data/oversample.py` present; `eval/ablation.py` **absent** | `test_no_leakage.py`, `test_sanity.py` pass | absent | Sec. 4.2, 6.4 | Intact to the test for the sampler; broken after |
| C5 | `eval/evaluate_grading.py` **absent** | absent | absent | Sec. 6.1, 6.6 | Broken; literature rows also wrong (F-001, F-002) |
| C6 | none (text) | none | absent | Abstract, Sec. 6.5, 9 | Text only; overstated |
| C7 | none (text) | none | none | Sec. 2.5, Table 1 | Text only; needs narrowing |

## Literature verification log

Full URLs and search strings are in `docs/evidence/research/literature_log.md`.

| Item | Source | Matches the paper? | Note |
|---|---|---|---|
| Chilukoti [48] QWK 0.901 / 0.967 / 0.944 | L1, L2 | Yes | Fine-tuned models; split not reported |
| Chilukoti [48] 0.854 as "EyePACS-pretrained on Messidor-2" | L2 | **No** | It is the model without DR pretraining (F-002) |
| Akhtar [49] 98.50 / 99.46 / 89.60 / 93.00 | L3, L4, L5 | **No** | Belong to Shakibania et al., the paper's ref [27] (F-001) |
| Akhtar [49] "multi-center incl. APTOS 2019" | L4 | **No** | Messidor-1 only |
| Dinpajhouh [50] 98.44 / 99.55 / 83.69 / 89.26 | L11, S3, S4 | **Not verified** | Paywalled (F-004) |
| Van Craenendonck [51] ECS 0.21-0.51, best VGG16 + Grad-CAM | L6 | Yes | |
| Van Craenendonck [51] 95% CI 0.46-0.51 | L6 | **No** | Source: [0.46; 0.55] (F-003) |
| Captum provides Grad-CAM++ | L7 | **No** | Only `LayerGradCam` seen |
| Ref [4] and [5] share a DOI | L8, L9 | Confirmed error | DOI belongs to [4]; [5] is 10.1007/s00417-026-07284-3 |
| Ref [5] venue | L9 | Yes | Graefe's Archive |
| Refs [3], [21], [24], [37] incomplete | L10 | Confirmed | Details in the log |
| Ref [20] | S9 | **Retracted** (to confirm on publisher site) | F-009 |
| IDRiD 516 images, 81 with masks | L6, S8 | Paper wrong on masks | Per-lesion counts not confirmed |
| Messidor-2 1,748 images, 874 examinations | L2, S5 | Yes | Labels are third-party |
| APTOS one image per patient | S6, S7 | **Not supported** | Duplicates reported by users |

## Open questions for the team

1. Will three seeds be trained? This decides whether "explanation stability" stays in the abstract (F-007).
2. Will the grading-only baseline be run (F-010)? One extra training run.
3. Which Messidor-2 label release will be used, and is detection there "referable" or "any DR"?
4. Can someone with library access open [50] and [27] and record the numbers and protocols (F-001, F-004)?
5. Does the supervisor accept renaming "synthetic-data augmentation" and softening the last sentence of the abstract (F-009, F-016)?
6. Padding to a square (F-013) and the fidelity changes (F-012 items 1 and 2) alter the design brief. Who signs off for the Designer?

## Status against the sync points

At SP0 the Researcher's B1 and B4 are complete as far as sources allow. B2 is complete for the design
and the M0 code. B3 items 1, 8 and 9 are decided; the rest wait for the explainers. Verdicts for C1 to
C5 will be re-issued when result files exist; until then each is also "cannot assess" on results.
