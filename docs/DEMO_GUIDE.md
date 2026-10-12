# Demo guide: DR-XAI web app

## Start it

1. Plug in the charger. Keep the lid open.
2. Double-click **`run_demo.bat`** in the project folder
   (`C:\Users\abhir\Downloads\MJ\github\Retinopathy`).
3. A black window opens and, after about 30 seconds, the browser shows the app at
   <http://localhost:8501>. If the browser does not open, type that address into it.
4. Leave the black window open. Closing it stops the app.

No internet connection is needed. Start it 10 minutes before the panel arrives: the first start
is the slowest.

## Show it

1. **Pick "Sample 9"** from the list. The model detects DR and grades it 4 · Proliferative, which
   matches the reference grade. Point at the heat-map: it marks where the model looked.
2. **Pick "Sample 1"**: a healthy eye, "not detected", grade 0.
3. **Pick "Sample 10"**: a miss. The reference grade is 4 · Proliferative and the model says
   Moderate. Show this on purpose; it leads into the known limits in the left panel.
4. **Upload a photo**: click *Upload* and choose any fundus photograph (jpg or png). The sample
   photos supplied with the project are in `tests\fixtures\sample_fundus\`.
5. **Upload something that is not an eye** (any ordinary photo): the app warns that it may not be a
   fundus photograph.

The ten samples come from the test split: the model never saw them in training. They were chosen by
rule (the first two test images of each grade), not hand-picked. The app grades six of the ten
correctly (checked on 12 October 2026):

| Sample | Reference grade | App result | |
|---|---|---|---|
| 1 | 0 · No DR | not detected, grade 0 | correct |
| 2 | 0 · No DR | not detected, grade 0 | correct |
| 3 | 1 · Mild | grade 1, but "not detected" (score 0.77, threshold 0.79) | grade correct; the two outputs disagree and the app says so |
| 4 | 1 · Mild | detected, grade 2 | one grade too high |
| 5 | 2 · Moderate | detected, grade 2 | correct |
| 6 | 2 · Moderate | detected, grade 2 | correct |
| 7 | 3 · Severe | detected, grade 2 | one grade too low |
| 8 | 3 · Severe | detected, grade 2 | one grade too low |
| 9 | 4 · Proliferative | detected, grade 4 | correct |
| 10 | 4 · Proliferative | detected, grade 2 | two grades too low |

DR itself is detected in every diseased sample except the borderline Sample 3. Each image takes
about half a second.

Of the ten supplied sample photos (no reference grades), the app reports DR in three
(`sample_7`, `sample_8`, `sample_9`, all graded Moderate) and none in the other seven. `sample_3`
is very dark and triggers the "may not be a fundus photograph" warning.

## What to say

- **What it is.** One neural network (ConvNeXt-Tiny) with two outputs: DR present or absent, and the
  severity grade 0 to 4 on the international scale. It was trained on 2,452 photographs from the
  public APTOS 2019 dataset, on this laptop's CPU, in under six hours.
- **How good it is.** On 526 photographs it had never seen: 98.9% accuracy at detecting DR, 86.7% at
  grading, quadratic weighted kappa 0.91.
- **How it explains itself.** Grad-CAM++ highlights the regions that pushed the model toward "DR".
- **What happens to the photo.** The black border is cropped, contrast is equalised (CLAHE), and the
  image is resized to 256 x 256 before the model sees it. The middle picture shows that version.

## Known limits (say them before the panel finds them)

- It is weakest on Proliferative DR: of 40 such test images it graded 13 correctly and called 20
  Moderate.
- It has been tested on one public dataset. Photographs from other cameras may be graded worse.
- The heat-map shows where the model looked. It has not yet been checked against expert lesion
  outlines (that study needs the IDRiD dataset and is not finished).
- The model shown is the CPU-trained ConvNeXt-Tiny at 256 px with one explanation method. The larger
  hybrid model and the other explainers named in the abstract need a GPU and were not trained.
- It is a research prototype, not a medical device.

## If something goes wrong

| Problem | What to do |
|---|---|
| Browser shows "This site can't be reached" | The app is still starting. Wait 30 seconds and refresh. |
| Black window closed or shows an error | Double-click `run_demo.bat` again. |
| "Port 8501 is already in use" | Another copy is running. Close every black window and start again. |
| A photo takes long | The first image after starting takes a few seconds; later ones under a second. |
| The app will not start at all | Open `notebooks\04_xai_gallery.ipynb` and the pictures in `results\figures\` and present from those. |
