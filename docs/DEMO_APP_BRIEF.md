# DR-XAI demo web app: architecture and plan of action

**From:** Architect role, 12 October 2026
**To:** Developer role
**Deadline:** panel demonstration tomorrow morning (13 October 2026)
**Goal:** a web app in which a person uploads a photo of the back of the eye (a fundus photo) and
gets back (1) whether diabetic retinopathy is present, (2) the severity grade, and (3) a picture
showing where the model looked.

This brief replaces the publication pipeline as tonight's priority. Nothing in the existing
repository is removed; the app is added on top of it.

---

## 1. Feasibility verdict

**Feasible, with time to spare, because no training is needed.**

| Question | Answer |
|---|---|
| Is a new training run required? | **No.** A trained model already exists: run `20261011-222646_convnext_tiny_oversample_seed42` (ConvNeXt-Tiny, 256 px, 20 epochs). Its file is `checkpoints/<run>/best.pt`. |
| Does the system need a GPU? | **No.** Training was done on this laptop's CPU. Predicting one image takes about 0.1 s on the CPU (measured: median 98 ms). |
| How long did training take? | 5.6 hours for the 20-epoch model; 1.7 hours for a 5-epoch model. Neither needs repeating. |
| How long to build the app? | **About 2 to 3 hours** of developer work, including tests and a rehearsal. |
| Is the current architecture suitable? | Yes. The model, the preprocessing and the Grad-CAM++ explainer are already written and tested. The app is a thin layer that calls them. |

Measured quality of the model the app will use (526 held-out APTOS images it never saw):
detection accuracy 98.9%, grading accuracy 86.7%, QWK 0.91. Known weakness: it recognises only
13 of 40 Proliferative images and often calls them Moderate. The presenter should know this.

**Retraining is optional and not recommended tonight.** If the team insists on a fresh run that
fits in 3 to 4 hours, `train.epochs=5` takes about 1.7 hours and `train.epochs=10` about 3 hours
on this laptop, plugged in and awake. The existing 20-epoch model is better than either.

---

## 2. Architecture

```
 Browser (same laptop)                    Python process on the laptop (CPU only)
 ─────────────────────                    ────────────────────────────────────────
 upload fundus photo  ──────────────►  Streamlit app  (app/demo_app.py)
                                              │
                                              ▼
                                   xaidr.inference.predict_image()
                                              │
        ┌─────────────────────────────────────┼─────────────────────────────────────┐
        ▼                                     ▼                                     ▼
 1. Preprocess (existing code)        2. Model (existing code)            3. Explanation (existing code)
    crop black border                    DRDualHeadModel, best.pt            Grad-CAM++ on the
    CLAHE contrast                       detection score + 5 grade            "any DR" score
    pad to square, resize 256            probabilities                        (about 0.3 s)
        │                                     │                                     │
        └─────────────────────────────────────┼─────────────────────────────────────┘
                                              ▼
 result page  ◄──────────────  DR present / absent (threshold 0.79 from threshold.json)
                               severity grade 0-4 with a probability bar per grade
                               original image | preprocessed image | saliency overlay
                               "research prototype, not a medical device" notice
```

### Design decisions

| # | Decision | Reason |
|---|---|---|
| A1 | **Streamlit** for the web app | One Python file, image upload and result layout built in, runs locally in the browser. Fastest route to a presentable page. |
| A2 | **Fallback if Streamlit cannot be installed** (slow network): a single-page app on Python's built-in `http.server` | Needs no download. Decide within 15 minutes of starting; do not lose an hour to a failing install. |
| A3 | Reuse the trained fallback model; **no retraining** | Deadline. It is the only trained 20-epoch model and its test results are known. |
| A4 | All model logic goes in **one new module, `src/xaidr/inference.py`**, with a function `predict_image(image) -> result` | The web page stays thin; the logic can be unit-tested without a browser. |
| A5 | Preprocessing in the app is **the same code path as training** (`preprocess_arrays`, `eval_transform`) | A different preprocessing at demo time is the most common cause of wrong predictions. |
| A6 | Detection uses the **frozen threshold from `threshold.json`** (0.79), not 0.5 | It is the threshold the reported accuracy was measured with. |
| A7 | Explanation shown by default: **Grad-CAM++ only** | Instant. SHAP and LIME take one to two minutes per image on this CPU; offer them behind a button labelled with the wait, or leave them out. |
| A8 | The model is loaded **once** and cached | Otherwise every upload reloads 113 MB of weights. |
| A9 | The app must work **with no internet** | Panel rooms often have none. The checkpoint supplies all weights, so nothing is downloaded at start-up. Verify by starting the app with Wi-Fi off. |
| A10 | A folder of **ready-made sample images** with their true grades (`demo_samples/`), taken from the APTOS **test** split | The presenter needs images that are guaranteed to load, and showing the true grade next to the prediction is honest. Include at least one case the model gets wrong. |
| A11 | A visible notice: **research prototype, not for diagnosis** | Required for any health-related demo; also protects the team in questions. |
| A12 | A basic input check: warn when the upload does not look like a fundus photo (no dark border and round bright field) | The model returns a grade for any picture, including a selfie. A warning avoids an embarrassing confident answer. |

### Out of scope for tonight
The IDRiD fidelity study (Table 5), Messidor-2, extra seeds, paper corrections, pushing to GitHub,
user accounts, a database, deployment to the internet, and a mobile layout. The app runs on the
presenter's laptop only.

---

## 3. Instructions to the Developer

Work in this order. Stop after each step if it fails and report; do not skip ahead.

| Step | Task | Time |
|---|---|---|
| 1 | Install Streamlit into `.venv` and confirm `streamlit hello` starts. If the install has not finished in 15 minutes, switch to the built-in `http.server` fallback (A2). Record the version in `requirements.txt`. | 15 min |
| 2 | Write `src/xaidr/inference.py`: `load_predictor(run_dir)` and `predict_image(predictor, image_rgb)`. The result holds: DR present (bool), detection score, threshold, predicted grade, grade name, the five probabilities, the preprocessed 256 px image, the Grad-CAM++ map, and a "looks like a fundus photo" flag. | 40 min |
| 3 | Write `tests/test_inference.py`: result fields and ranges; probabilities sum to 1; the same image twice gives the same result; a processed APTOS test image gives the same grade as in `predictions_test.csv`; a plain non-fundus image raises the warning flag. | 25 min |
| 4 | Write `app/demo_app.py`: title and notice, file uploader (jpg, jpeg, png), a sample-image picker, the result block (large DR present / absent line, grade with name, probability bars), and three images side by side (uploaded, preprocessed, saliency overlay). Show the model name and its measured test accuracy in a sidebar. | 45 min |
| 5 | Build `demo_samples/`: two test images per grade 0-4 with a small CSV of true grades. Choose by image id, not by cherry-picking only correct ones; include one known miss. | 10 min |
| 6 | Add `run_demo.bat` (one double-click start) and `docs/DEMO_GUIDE.md` (how to start, what to say, known limits, what to do if it fails). | 15 min |
| 7 | Rehearse: start with Wi-Fi off, upload each sample image and two of the supplied sample photos, check every result renders in under 3 seconds. Fix what breaks. | 20 min |
| 8 | Run the existing test suite (`-m "not slow"`), commit. | 10 min |

**Total: about 3 hours.** Steps 1 to 4 give a working demo in about 2 hours.

### Rules for the Developer
1. Do not retrain. Do not change the model, the preprocessing or the threshold.
2. Do not invent accuracy figures in the app; show only numbers from `results/table4_detection_grading_aptos_test.csv`.
3. Label the app "research prototype" and the model "ConvNeXt-Tiny, 256 px".
4. Keep the page to one screen; the panel should see the answer without scrolling.
5. If anything in this brief conflicts with a faster working demo, tell the user and take the faster route.

### Acceptance criteria
- Double-clicking `run_demo.bat` opens the app in the browser within 30 seconds, with Wi-Fi off.
- Uploading a fundus photo shows detection, grade, probabilities and the saliency overlay in under 3 seconds.
- A non-fundus picture produces a warning, not a silent diagnosis.
- The ten sample images load from a picker and show the true grade beside the prediction.
- `tests/test_inference.py` passes.

---

## 4. Risks for tomorrow morning

| Risk | Mitigation |
|---|---|
| Laptop asleep, battery flat, or app not started when the panel arrives | Start the app 10 minutes early, charger connected, lid open. `run_demo.bat` on the desktop. |
| No internet in the room | The app is offline by design (A9); rehearse with Wi-Fi off. |
| The panel uploads their own image and the grade is wrong | Say so plainly: 87% grading accuracy, weakest on Proliferative. Have the confusion matrix figure open. |
| Streamlit install fails tonight | Fallback A2. |
| A question about the "hybrid CNN/ViT" or the four explainers in the abstract | Honest answer: the demo runs the CPU-trained ConvNeXt baseline with Grad-CAM++; the larger model needs a GPU. |
| App crashes mid-demo | Keep notebook `04_xai_gallery` and the result figures open in another window as backup. |

---

## 5. What the team must decide before the Developer starts

1. **Go or no-go on this plan.**
2. Whether IDRiD work (the plan given earlier, 3 to 5 hours of CPU) should run **after** the app is
   finished, overnight, or be dropped. It must not run while the app is being built or rehearsed:
   it would slow the laptop.
