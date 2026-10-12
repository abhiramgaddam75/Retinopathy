"""Write colab/DR_XAI_train_colab.ipynb, the notebook to upload to Google Colab.

Run from the repository root:  python tools/make_colab_notebook.py
"""
from __future__ import annotations

from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]

CELLS: list[tuple[str, str]] = [
    ("markdown", "# DR-XAI: train the specified model on a Colab GPU\n\n"
     "Before running: **Runtime -> Change runtime type -> T4 GPU**, and upload `dr_xai_colab.zip` "
     "to the top level of your Google Drive (My Drive). Run the cells in order. "
     "Checkpoints are written to `My Drive/dr_xai_checkpoints`, so a disconnect loses at most one epoch."),
    ("code", "# 1. Mount Drive and unpack the bundle (code + preprocessed images + locked split)\n"
     "from google.colab import drive\n"
     "drive.mount('/content/drive')\n"
     "!rm -rf /content/dr-xai && mkdir -p /content/dr-xai\n"
     "!unzip -q -o /content/drive/MyDrive/dr_xai_colab.zip -d /content/dr-xai\n"
     "%cd /content/dr-xai\n"
     "!ls && ls data/processed/dr512/aptos2019/images | wc -l   # expect 3662"),
    ("code", "# 2. Packages and GPU check\n"
     "!pip install -q timm==1.0.30 albumentations==2.0.8 pandas==3.0.6 grad-cam==1.5.7\n"
     "import os, sys, torch\n"
     "os.environ['PYTHONPATH'] = 'src'\n"
     "os.environ['PYTHONHASHSEED'] = '42'\n"
     "CKPT = '/content/drive/MyDrive/dr_xai_checkpoints'\n"
     "os.makedirs(CKPT, exist_ok=True)\n"
     "print(sys.version)\n"
     "print(torch.__version__, torch.cuda.get_device_name(0))   # must print a GPU name"),
    ("code", "# 3. Train the specified model: maxvit_tiny_tf_512, 512 px, up to 30 epochs, seed 42\n"
     "#    The last line printed is 'run id: <RUN_ID>'. Copy it into the next cell.\n"
     "#    Out of memory? add:  train.batch_size=8 train.grad_accum=4\n"
     "!python -m xaidr.cli train paths.checkpoints=$CKPT train.num_workers=2"),
    ("code", "# 3b. ONLY if the session disconnected during cell 3: rerun cells 1-2, then this one\n"
     "RUN = 'PASTE_RUN_ID_HERE'\n"
     "# !python -m xaidr.cli train paths.checkpoints=$CKPT train.num_workers=2 --resume $RUN"),
    ("code", "# 4. Evaluate once on the held-out test split (writes Table 4 and Table 6)\n"
     "RUN = 'PASTE_RUN_ID_HERE'\n"
     "!python -m xaidr.cli evaluate --run $RUN paths.checkpoints=$CKPT train.num_workers=2\n"
     "!mkdir -p $CKPT/results_$RUN && cp -r results/* $CKPT/results_$RUN/\n"
     "!ls -la $CKPT/$RUN $CKPT/results_$RUN"),
    ("markdown", "## Optional extra runs (each is another full training)\n\n"
     "Run these only if there is GPU time left. Each prints its own run id.\n\n"
     "- **Ablation arm, no oversampling:** needed for the Section 6.4 comparison.\n"
     "- **Seeds 43 and 44:** needed if the paper keeps the 'explanation stability' claim."),
    ("code", "# 5. Optional: ablation arm without minority oversampling\n"
     "# !python -m xaidr.cli train paths.checkpoints=$CKPT train.num_workers=2 minority.mode=none"),
    ("code", "# 6. Optional: two more seeds\n"
     "# !python -m xaidr.cli train paths.checkpoints=$CKPT train.num_workers=2 seed=43\n"
     "# !python -m xaidr.cli train paths.checkpoints=$CKPT train.num_workers=2 seed=44"),
    ("markdown", "## When finished\n\n"
     "In Google Drive open `dr_xai_checkpoints`. Download each run folder (it contains `best.pt`, "
     "`run.json`, `threshold.json`, `metrics.jsonl`, `config.yaml`, prediction CSVs) and the matching "
     "`results_<RUN_ID>` folder. Put the run folders into the project's `checkpoints/` folder on the "
     "laptop, then tell the Engineer the run ids."),
]


def main() -> None:
    notebook = nbformat.v4.new_notebook()
    for kind, source in CELLS:
        make = nbformat.v4.new_markdown_cell if kind == "markdown" else nbformat.v4.new_code_cell
        notebook.cells.append(make(source))
    notebook.metadata["accelerator"] = "GPU"
    notebook.metadata["colab"] = {"gpuType": "T4", "provenance": []}
    notebook.metadata["kernelspec"] = {"display_name": "Python 3", "name": "python3"}
    target = ROOT / "colab" / "DR_XAI_train_colab.ipynb"
    target.parent.mkdir(exist_ok=True)
    nbformat.write(notebook, target)
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
