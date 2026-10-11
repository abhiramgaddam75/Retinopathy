# Issue log

Append-only. Format and routing: VERIFICATION_AND_RESEARCH_BRIEF.md, Section 6. Only the author
edits the status line of their own entry.

## ISSUE-001  [S1]  Training augmentation is not controlled by the seed
From:        Tester
To:          Software Engineer
Refs:        DEF-001 / requirement R6 / src/xaidr/data/augment.py:24, src/xaidr/seed.py:13
Finding:     Two runs with seed 42 produce different augmented training tensors, so training cannot be reproduced and the A5 reproducibility test cannot pass.
Evidence:    docs/evidence/m0_probes_output.txt (P1, P2); tests/test_sanity.py (2 xfail tests)
Needed:      Seed the augmentation pipeline (including DataLoader workers); the two xfail tests must then pass and their markers be removed by the Tester.
Status:      OPEN

## ISSUE-002  [S1]  Duplicate images are not detected and can sit in two splits
From:        Tester
To:          Software Engineer, Researcher
Refs:        DEF-002 / requirement R4 / joint check J2 / claim C3 / src/xaidr/data/splits.py
Finding:     Byte-identical images with different ids are split independently; in a synthetic set 32 of 60 planted pairs straddled two splits. Whether APTOS 2019 contains such pairs is not yet measured (no data on disk).
Evidence:    docs/evidence/m0_probes_output.txt (P5)
Needed:      Engineer: detect duplicates at manifest build and keep each group in one split, logging counts. Researcher: decide how duplicates with conflicting grades are handled and reported. Tester re-tests on the real data in Phase 3; severity becomes S0 if cross-split duplicates exist there.
Status:      OPEN

## ISSUE-003  [S4]  Preprocessing stretches cut-off fundus photos by up to 27.5%
From:        Tester
To:          Researcher
Refs:        DEF-007 / requirements R2, R3 / claim C2 / DESIGN 7.5 / src/xaidr/data/preprocess.py:77
Finding:     The field-of-view crop is resized to 512 x 512 without keeping its aspect ratio; two of the ten sample photos are stretched vertically by about 27%. Masks get the same transform, so alignment holds.
Evidence:    docs/evidence/m0_probes_output.txt (P7)
Needed:      Researcher judges whether this is acceptable or the design should pad to a square; if a change is wanted, route to the Software Engineer and the Designer.
Status:      OPEN

## ISSUE-004  [S2]  Split lock is not enforced when data is loaded, and cannot be committed
From:        Tester
To:          Software Engineer
Refs:        DEF-003 / requirement R6 / joint check J2 / DESIGN 8.5 rule 5 vs DESIGN 6 layout
Finding:     Editing the split column of manifest.csv moves locked test images into training without error, and the lock file lives under the git-ignored data/ folder.
Evidence:    docs/evidence/m0_probes_output.txt (P4)
Needed:      Verify the manifest's split against the lock before any loader is built; store the lock where it can be committed.
Status:      OPEN

## ISSUE-005  [S2]  Processed images are silently reused after a preprocessing-config change
From:        Tester
To:          Software Engineer
Refs:        DEF-004 / requirement R6 / src/xaidr/data/preprocess.py:159
Finding:     Changing CLAHE settings and rerunning prepare-data without --force leaves all processed images unchanged, with no warning.
Evidence:    docs/evidence/m0_probes_output.txt (P3)
Needed:      Record the preprocessing parameters with the processed data and rebuild or stop when they differ.
Status:      OPEN

## ISSUE-006  [S3]  Section numbers in the paste-in prompts of the verification brief are wrong
From:        Tester
To:          Authors
Refs:        docs/VERIFICATION_AND_RESEARCH_BRIEF.md Sections 0.2, 0.3, 1 (row R7)
Finding:     The Tester prompt names Section 4 as its track (Track A is Section 3) and the Researcher prompt names Section 5 (Track B is Section 4); Section 0.3 cites "Section 5.7" (it is B5) and row R7 cites "Section 6 facts" (they are in Section 2).
Evidence:    docs/VERIFICATION_AND_RESEARCH_BRIEF.md
Needed:      Correct the references so a new session reads the right sections.
Status:      OPEN

## ISSUE-007  [S2]  No dataset and no GPU: Phases 3 to 7 cannot start
From:        Tester
To:          Authors, Software Engineer
Refs:        requirements R1-R10 / docs/DATA_ACCESS.md / brief Section 7 sync points
Finding:     APTOS 2019, IDRiD and Messidor-2 are not on disk and the only machine has no CUDA GPU, so there is no model, no results file and no notebook to test. Eight of ten requirements are NOT MET today.
Evidence:    docs/TEST_REPORT.md (traceability matrix); docs/evidence/phase4_cli_all.txt
Needed:      Team supplies the datasets and a GPU; Engineer delivers M1-M4; Tester then runs Phases 2-7.
Status:      OPEN

## ISSUE-008  [S2]  Tester independence: this Tester session also wrote the code under test
From:        Tester
To:          Authors
Refs:        brief Section 0.1 (Tester is a separate Sonnet session)
Finding:     The Phase 0-1 pass on 2026-10-10 was run in the same session that implemented M0, on a different model from the one the brief assigns. Its findings are real, but it is not an independent review.
Evidence:    docs/TEST_REPORT.md (header)
Needed:      Repeat Phase 0-1 in a separate Tester session before sign-off, starting from docs/TEST_REPORT.md.
Status:      OPEN

## ISSUE-009  [BLOCKER]  Tables 4 and 6 attribute the results of Shakibania et al. to Akhtar et al. [49]
From:        Researcher
To:          Authors, Tester
Refs:        claim C5 / requirement R8 / joint check J7 / paper Table 4, Table 6, Sec. 7 / SCIENTIFIC_REVIEW F-001
Finding:     98.50%, 99.46%, 89.60% and QWK 93.00% are not results of Akhtar et al. (a Messidor-1 study); they describe Shakibania et al., the paper's own ref [27], on APTOS 2019. "Multi-center incl. APTOS 2019" is also wrong.
Evidence:    docs/evidence/research/literature_log.md (L3, L4, L5)
Needed:      Authors re-attribute the row after reading [27]. Tester: when results/table6_comparative.csv exists, check its literature rows against this finding, not against the draft.
Status:      OPEN

## ISSUE-010  [BLOCKER]  Section 6.3 misreads Chilukoti et al. [48] and it is not cross-dataset evidence
From:        Researcher
To:          Authors
Refs:        claims C3, C5 / paper Sec. 6.3, Fig. 4, Table 6 / SCIENTIFIC_REVIEW F-002
Finding:     0.854 is Messidor-2 without DR pretraining; 0.944 is Messidor-2 with EyePACS pretraining. The draft reverses this and compares it with the APTOS figure. All models in the source are fine-tuned on the dataset they are scored on.
Evidence:    docs/evidence/research/literature_log.md (L1, L2)
Needed:      Rewrite the paragraph; remove the "0.11 QWK domain shift" statement.
Status:      OPEN

## ISSUE-011  [FIX]  Van Craenendonck et al. [51]: confidence interval is wrong and ECS is used as an IoU ceiling
From:        Researcher
To:          Authors
Refs:        claims C2, C5 / paper Sec. 6.2, Table 5, Sec. 7 / SCIENTIFIC_REVIEW F-003
Finding:     The source gives a 95% CI of [0.46; 0.55], not 0.46-0.51. ECS is a top-15-of-100-grid-cells overlap, not IoU or Dice, so "meet or exceed the 0.51 ECS ceiling" is not a valid target.
Evidence:    docs/evidence/research/literature_log.md (L6)
Needed:      Correct the interval and remove the ceiling sentence.
Status:      OPEN

## ISSUE-012  [BLOCKER]  Paper statements about IDRiD, EyePACS and the Grad-CAM++ library contradict the facts and the build
From:        Researcher
To:          Authors, Tester, Software Engineer
Refs:        claims C2, C3 / requirement R7 / joint check J8 / paper Sec. 4.1, 4.5, 4.6, 5.1, 5.3, 8, Tables 2 and 3 / SCIENTIFIC_REVIEW F-005, F-006, F-008, F-011
Finding:     The draft says 516 IDRiD images have lesion masks (81 do), gives IDRiD two incompatible roles, describes EyePACS pretraining and a held-out EyePACS test that are out of scope, attributes Grad-CAM++ to Captum (which documents only the original Grad-CAM), and says APTOS and IDRiD have one image per patient without support. Protocol values (loss weights, batch size, selection rule, threshold) differ from configs/default.yaml.
Evidence:    docs/evidence/research/literature_log.md (L6, L7, S7, S8); docs/SCIENTIFIC_REVIEW.md table in F-011
Needed:      Authors rewrite these sections to match the build. Tester confirms the IDRiD per-lesion counts in Phase 3 and supplies the config snapshot for J8. Named per the brief's rule for code-versus-paper inconsistencies: Tester and Software Engineer.
Status:      OPEN

## ISSUE-013  [BLOCKER]  Decision on explanation stability (F9, J5): Proposal A approved with conditions, Proposal B rejected
From:        Researcher
To:          Software Engineer, Authors, Tester
Refs:        claim C3 / requirement R4 / joint check J5 / DESIGN 10.5, D12 / SCIENTIFIC_REVIEW F-007
Finding:     Lesion fidelity can only be computed on 81 IDRiD images, and no IDRiD-trained model exists, so the paper's definition is not computable. Proposal A (between-seed Spearman agreement) is approved only if three seeds are trained and the agreement is reported on APTOS test, IDRiD and Messidor-2 alike.
Evidence:    docs/SCIENTIFIC_REVIEW.md F-007
Needed:      Engineer: train seeds 42, 43, 44 and report agreement per dataset, or mark stability "not computed". Authors: if not computed, remove "explanation stability across three datasets" from the abstract and Sections 1, 4.6, 6.3.
Status:      OPEN

## ISSUE-014  [FIX]  Fidelity protocol needs two changes before Table 5 is computed
From:        Researcher
To:          Software Engineer, Tester
Refs:        claim C2 / requirement R3 / joint check J1 / DESIGN 7.21, 10.3, D6, D11 / SCIENTIFIC_REVIEW F-012
Finding:     (1) With top-k% thresholding, IoU cannot exceed lesion-area/k, so small lesions have a ceiling near zero and the chance value alone does not make scores interpretable. (2) Explaining the predicted grade yields a "No DR" map whenever a lesion image is predicted as grade 0.
Evidence:    docs/SCIENTIFIC_REVIEW.md F-012
Needed:      Engineer (with a Designer note): add maximum attainable IoU and Dice and lesion recall at k, with chance, to every Table 5 row; explain a fixed disease target for the fidelity study and report how many images are predicted grade 0. Declare k = 10 primary and apply Holm correction across the 20 primary comparisons, or state none. Tester: extend S5 and test_fidelity.py to cover the new columns.
Status:      OPEN

## ISSUE-015  [FIX]  Reply to ISSUE-003: remove the aspect-ratio stretch by padding to a square
From:        Researcher
To:          Software Engineer, Tester
Refs:        ISSUE-003 / DEF-007 / claims C2, C3 / DESIGN 7.3, 7.5 / SCIENTIFIC_REVIEW F-013
Finding:     The stretch depends on each source's camera and crop (0% to about 27% on the sample photos), so it adds a dataset-dependent shape distortion to a cross-dataset study. It is avoidable.
Evidence:    docs/evidence/m0_probes_output.txt (P7); docs/SCIENTIFIC_REVIEW.md F-013
Needed:      Engineer: pad the field-of-view crop to a square before resizing, record the padding, apply the same to masks, before any real data is preprocessed; log the design change. Tester: extend the alignment test to a non-square crop.
Status:      OPEN

## ISSUE-016  [FIX]  Reply to ISSUE-002: rule for duplicate images
From:        Researcher
To:          Software Engineer, Tester, Authors
Refs:        ISSUE-002 / DEF-002 / claim C3 / joint check J2 / SCIENTIFIC_REVIEW F-014
Finding:     Grouping removes the leakage. Remaining rule: keep one copy of same-grade duplicates; exclude groups with conflicting grades from validation and test (training only, or drop). Only byte-identical copies are detected.
Evidence:    docs/SCIENTIFIC_REVIEW.md F-014
Needed:      Engineer implements the rule and logs both counts. Tester measures the counts on real APTOS data in Phase 3. Authors report them and delete the one-image-per-patient sentence.
Status:      OPEN

## ISSUE-017  [FIX]  Joint design has no baseline; the detection label is a function of the grade
From:        Researcher
To:          Authors, Software Engineer
Refs:        claim C1 / requirement R1 / SCIENTIFIC_REVIEW F-010
Finding:     dr_label is grade >= 1, so a grading-only model already gives a detection score. Without a grading-only baseline the paper cannot claim the dual head helps.
Evidence:    src/xaidr/data/adapters/aptos2019.py:38; docs/SCIENTIFIC_REVIEW.md F-010
Needed:      Either one extra run (grading head only, detection derived as 1 - P(grade 0), same seed and split, paired bootstrap), or wording that makes no comparative claim.
Status:      OPEN

## ISSUE-018  [FIX]  Abstract and text: "synthetic-data augmentation", clinical wording, gap claim, references
From:        Researcher
To:          Authors, Editor
Refs:        claims C4, C6, C7 / SCIENTIFIC_REVIEW F-009, F-015, F-016, F-017
Finding:     The method is minority oversampling, not synthetic data. Ref [20] appears to be retracted and ref [19] splits after augmenting. "Clinically trustworthy ... deployable" is not supported. The gap claim should cite arXiv 2608.28207 and describe the search. Ref [5] carries the DOI of ref [4]; refs [21]-[28] lack titles; QWK units are mixed.
Evidence:    docs/evidence/research/literature_log.md (L8-L10, L12, S9, section D)
Needed:      Text changes; abstract changes go through the supervisor. Editor does not draft results text until C1-C5 are READY or FIX-complete.
Status:      OPEN

## ISSUE-019  [NOTE]  Researcher independence and unverified items
From:        Researcher
To:          Authors
Refs:        brief Section 0.1 / SCIENTIFIC_REVIEW header
Finding:     This review was written in the same session that acted as Engineer and Tester. Dinpajhouh and Seyyedsalehi [50] could not be read (paywall), refs [6]-[18], [25]-[36], [38]-[47] were not opened, and the IDRiD per-lesion counts are not confirmed.
Evidence:    docs/evidence/research/literature_log.md
Needed:      A team member with library access checks [50] and [27]; a separate Researcher session re-reads the verdicts once results exist.
Status:      OPEN
