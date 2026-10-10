# Defects

Tester/QA record (VERIFICATION_AND_RESEARCH_BRIEF.md, Sections A8 and 8.3). Commit under test:
`494539d`. Evidence paths are relative to the repository root. Line numbers refer to that commit.

Summary on 2026-10-10: 2 x S1, 2 x S2, 2 x S3, 1 x S4 open; 1 closed.

---

```
ID: DEF-001   Severity: S1   Requirement: R6
File: src/xaidr/data/augment.py:24 (train_transform), src/xaidr/seed.py:13 (set_seed)
Found in: Phase 1 probe P1/P2; tests/test_sanity.py (2 strict xfail tests)
Observed: With set_seed(42) called before each run, two runs of train_transform on the same image
          give different tensors. Two fresh training loaders with the same seed draw the same image
          ids in the same order but return different image tensors. set_seed seeds python, numpy and
          torch; the albumentations 2.0.8 pipeline does not follow any of them.
Expected: Same seed, same config -> identical augmented training inputs (DESIGN 4.2 rule 3, 7.2;
          brief A5 requires two seed-42 runs to give identical metrics and checkpoint hashes).
Steps: python docs/evidence/probes/m0_probes.py   (sections P1 and P2)
       python -m pytest tests/test_sanity.py -rx
Evidence: docs/evidence/m0_probes_output.txt; docs/evidence/phase1_pytest.txt
Route to: Software Engineer
Status: OPEN
Note: when fixed, the two xfail tests XPASS and fail the suite on purpose; the Tester then removes
      the markers. Worker processes (train.num_workers > 0) need the same check.
```

```
ID: DEF-002   Severity: S1 (becomes S0 if confirmed on the real data)   Requirement: R4, J2
File: src/xaidr/data/adapters/aptos2019.py:32, src/xaidr/data/splits.py (make_splits)
Found in: Phase 1 probe P5
Observed: Images that are byte-identical but listed under different ids are treated as independent.
          With 60 planted duplicate pairs in a 320-image synthetic APTOS set, 32 pairs ended with
          one copy in one split and the other copy in a different split. No warning is raised.
Expected: Duplicate images are detected when the manifest is built and either kept in one split
          (grouped as one unit) or removed, with the count logged (DESIGN 4.2 rule 1, "no leakage").
Steps: python docs/evidence/probes/m0_probes.py   (section P5)
Evidence: docs/evidence/m0_probes_output.txt
Route to: Software Engineer; Researcher (ISSUE-002)
Status: OPEN
Note: whether APTOS 2019 really contains duplicates, and how many carry conflicting grades, has NOT
      been measured here because the dataset is not on disk. It is commonly reported for this
      dataset, so it must be measured in Phase 3 before any test metric is trusted.
```

```
ID: DEF-003   Severity: S2   Requirement: R6, J2
File: src/xaidr/data/splits.py:79; .gitignore:2; src/xaidr/data/dataset.py:75
Found in: Phase 1 probe P4
Observed: (a) The split lock is only compared when make-splits runs. Loaders read the split column
          of manifest.csv and never check it against the lock: after moving all 15 test rows to
          "train" in the manifest, make_train_loader built a training set containing them, with no
          error. (b) The lock is written to data/splits/, and /data/ is git-ignored, so the lock
          file cannot be committed.
Expected: DESIGN 8.5 rule 5: "Splits are stratified and locked by SHA-256; the lock file is
          committed." A manifest whose split column disagrees with the lock should be refused.
Steps: python docs/evidence/probes/m0_probes.py   (section P4)
Evidence: docs/evidence/m0_probes_output.txt
Route to: Software Engineer (the design's own layout, Section 6, puts data/splits under an ignored
          folder, so the Designer's rule and layout disagree)
Status: OPEN
```

```
ID: DEF-004   Severity: S2   Requirement: R6
File: src/xaidr/data/preprocess.py:159
Found in: Phase 1 probe P3
Observed: After changing data.clahe.clip_limit from 2.0 to 8.0, prepare-data (without --force) exits
          0 and leaves every processed image unchanged, with no warning. Nothing records which
          preprocessing parameters produced the files in data/processed/.
Expected: A rerun with different preprocessing parameters either rebuilds the images or stops with a
          message. The behaviour matches the wording of DESIGN 7.5 ("skips files that exist with
          matching size unless --force"), but a model could then be trained on images that do not
          match its config snapshot.
Steps: python docs/evidence/probes/m0_probes.py   (section P3)
Evidence: docs/evidence/m0_probes_output.txt
Route to: Software Engineer
Status: OPEN
```

```
ID: DEF-005   Severity: S3   Requirement: none
File: src/xaidr/cli.py:117
Found in: Phase 1 probe P6
Observed: An unreadable image makes prepare-data end with an uncaught PreprocessError traceback. A
          grade with only 2 images makes make-splits end with an uncaught scikit-learn
          ValueError traceback. In both cases no manifest or split is written (good).
Expected: A one-line error and exit code 1, as for the other handled errors.
Steps: python docs/evidence/probes/m0_probes.py   (section P6)
Evidence: docs/evidence/m0_probes_output.txt
Route to: Software Engineer
Status: OPEN
```

```
ID: DEF-006   Severity: S3   Requirement: none
File: src/xaidr/config.py:103 and :140; src/xaidr/cli.py:107
Found in: Phase 0 review
Observed: The config has two keys, top-level `deterministic` and `train.deterministic`. Only the
          top-level one is read; `train.deterministic` has no effect.
Expected: One key, or both honoured. (Both keys come from DESIGN 9.1.)
Steps: grep -n deterministic src/xaidr/*.py
Evidence: this record
Route to: Software Engineer
Status: OPEN
```

```
ID: DEF-007   Severity: S4   Requirement: R2, R3 (observation for the Researcher)
File: src/xaidr/data/preprocess.py:77 (resize_image), as specified by DESIGN 7.5
Found in: Phase 1 probe P7
Observed: The cropped field of view is resized to 512 x 512 without keeping its aspect ratio. On the
          ten supplied sample photos the vertical stretch is 0.0-0.5% for six, about 4.4% for two,
          and 26.5-27.5% for two (the photos whose retina is cut off at the top and bottom).
Expected: Not a code defect: the code follows the design, and masks receive the same transform. The
          Researcher should judge whether a stretch of this size matters for grading and for
          lesion-level fidelity.
Steps: python docs/evidence/probes/m0_probes.py   (section P7)
Evidence: docs/evidence/m0_probes_output.txt
Route to: Researcher (ISSUE-003)
Status: OPEN
```

```
ID: DEF-008   Severity: S2   Requirement: R5
File: tests/test_no_leakage.py (test gap)
Found in: Phase 1 mutation check M4
Observed: With oversampling silently disabled inside make_train_loader, all 9 tests in
          tests/test_no_leakage.py still passed. The existing test exercised minority_sampler
          directly, not the loader that training uses.
Expected: The suite fails if the training loader ignores minority.mode.
Steps: python docs/evidence/probes/m0_probes.py   (section P8, M4)
Evidence: docs/evidence/m0_probes_output.txt
Route to: Tester (tests/ is Tester-owned)
Status: CLOSED by the Tester: tests/test_sanity.py::test_training_loader_applies_the_configured_
        oversampling (both modes) now covers it. M1-M3 were detected by the existing tests.
```
