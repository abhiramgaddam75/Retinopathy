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
