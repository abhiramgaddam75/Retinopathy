# Training the specified model on a GPU

The model in DESIGN.md (`maxvit_tiny_tf_512`, 512 px, 30 epochs) needs a CUDA GPU. Measured on the
project laptop (Intel i7-1165G7, 4 cores, no GPU) with real APTOS batches:

| Model | Input | Seconds per image | One epoch | 30 epochs |
|---|---|---|---|---|
| `maxvit_tiny_tf_512` (specified) | 512 px | 3.31 | 2.8 h | 3.5 days |
| `convnext_tiny` (baseline) | 512 px | 1.18 | 1.0 h | 1.3 days |

These steps have **not been run by the Engineer** (no GPU was available); they follow the same CLI
that is tested on CPU. Report anything that fails.

## Google Colab (recommended: no GitHub or Kaggle account needed)

Everything Colab needs is in one file, `dr_xai_colab.zip` (about 1.3 GB): the code, the 3,662
preprocessed 512 px images, the manifest and the locked split. Build it on the laptop with

```
python tools/make_colab_bundle.py C:\Users\<you>\Downloads\dr_xai_colab.zip
```

1. Upload `dr_xai_colab.zip` to the top level of your Google Drive ("My Drive").
2. Go to <https://colab.research.google.com>, choose **Upload**, and upload
   `colab/DR_XAI_train_colab.ipynb` from this repository.
3. **Runtime -> Change runtime type -> T4 GPU -> Save.**
4. Run cells 1 and 2. Cell 1 asks for permission to read your Drive. Cell 2 must print a GPU name
   (for example "Tesla T4"). If it raises an error instead, the GPU is not switched on (step 3).
5. Run cell 3. It trains the specified model and prints one line per epoch. When it ends, the
   last line is `run id: <something>`. Copy that id.
6. Paste the id into cell 4 in place of `PASTE_RUN_ID_HERE` and run it. It scores the model once
   on the held-out test split.
7. In Google Drive, open `dr_xai_checkpoints`. Download the run folder and the `results_<run id>`
   folder. Put the run folder into `checkpoints\` in the project on the laptop.

If Colab disconnects during cell 3, reconnect, run cells 1 and 2 again, put the run id (the name of
the newest folder in `My Drive/dr_xai_checkpoints`) into cell 3b, remove the `#` and run it.

Cells 5 and 6 are optional further runs (no-oversampling ablation arm; seeds 43 and 44). Each one is
another full training, so check the remaining GPU time first.

Not verified: Colab's Python is newer than the 3.11 this project is tested on, and its
preinstalled package versions differ. The notebook installs the four packages that matter and runs
the code from source. If a cell fails, copy the full error text back to the Engineer.

## Kaggle Notebooks (free GPU)

1. Push this repository to GitHub first (the notebook clones it), including `data/splits/`.
2. On kaggle.com create a notebook. In the right-hand panel: **Add input** -> the competition
   "APTOS 2019 Blindness Detection"; **Accelerator** -> GPU T4; **Internet** -> On.
3. Run these cells in order.

```python
# cell 1: code and the few packages Kaggle does not ship at the right version
!git clone https://github.com/abhiramgaddam75/Retinopathy.git
%cd Retinopathy
!pip install -q timm==1.0.30 albumentations==2.0.8 grad-cam==1.5.7
import os, sys, torch
os.environ["PYTHONPATH"] = "src"
os.environ["PYTHONHASHSEED"] = "42"
print(sys.version, torch.__version__, torch.cuda.get_device_name(0))
```

```python
# cell 2: point the project at Kaggle's read-only copy of the data
!mkdir -p data/raw
!ln -s /kaggle/input/aptos2019-blindness-detection data/raw/aptos2019
!ls data/raw/aptos2019 | head
```

```python
# cell 3: preprocess (about 10 minutes) and rebuild the split
!python -m xaidr.cli prepare-data --datasets aptos2019
!python -m xaidr.cli make-splits
```

`make-splits` must finish without a "differs from the locked split" error. That error means the
Kaggle environment produced a different split from the one committed in `data/splits/`; stop and
tell the Engineer, do not pass `--resplit`.

```python
# cell 4: train; the last log line prints the run id
!python -m xaidr.cli train
```

If the GPU runs out of memory, restart cell 4 with `train.batch_size=8 train.grad_accum=4` appended
(the effective batch size stays 32) and note it; the override is saved in the run's `config.yaml`.
If the session is interrupted, continue with `python -m xaidr.cli train --resume <run_id>`.

```python
# cell 5: evaluate on the held-out test split and pack everything worth keeping
RUN = "<run id printed by cell 4>"
!python -m xaidr.cli evaluate --run $RUN
!zip -r -q /kaggle/working/dr_xai_run.zip checkpoints/$RUN results data/splits data/manifests
```

Download `dr_xai_run.zip` from the notebook's Output panel and unpack it into the repository root
on the laptop. `checkpoints/<run id>/best.pt` is then available for the explainability steps.

Kaggle's Python may be newer than 3.11. The cells above set `PYTHONPATH=src` instead of installing
the package, so the `requires-python` limit in `pyproject.toml` does not block them; if an import
fails on a newer Python, report the traceback.

## Any other CUDA machine

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   # match the driver
pip install -r requirements.txt
pip install -e .
python -m xaidr.cli prepare-data --datasets aptos2019
python -m xaidr.cli make-splits
python -m xaidr.cli train
python -m xaidr.cli evaluate --run <run id>
```

## CPU fallback (what the laptop can do)

`configs/cpu_fallback.yaml` trains `convnext_tiny` at 256 px for at most 20 epochs. It exists to run
the whole pipeline on real data without a GPU. Its results are a fallback baseline and must be
labelled as such; they are not the proposed framework's headline numbers.

```
python -m xaidr.cli prepare-data --config configs/cpu_fallback.yaml --datasets aptos2019
python -m xaidr.cli make-splits  --config configs/cpu_fallback.yaml
python -m xaidr.cli train        --config configs/cpu_fallback.yaml
```
