"""DR-XAI demo: upload a fundus photo, get DR detection, severity grade and a saliency map.

Start it with run_demo.bat, or:  python -m streamlit run app/demo_app.py
Everything runs on this computer's CPU; no internet connection is used.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")  # no version check: the app is offline
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

from xaidr.inference import GRADE_NAMES, load_predictor, overlay, predict_image  # noqa: E402

TABLE4 = ROOT / "results" / "table4_detection_grading_aptos_test.csv"
SAMPLES = ROOT / "data" / "demo_samples"
GRADE_HELP = {
    0: "No signs of diabetic retinopathy found.",
    1: "Mild non-proliferative DR: a few microaneurysms.",
    2: "Moderate non-proliferative DR: more than mild, less than severe.",
    3: "Severe non-proliferative DR: extensive haemorrhages or vessel changes.",
    4: "Proliferative DR: new abnormal vessels; the most advanced stage.",
}

st.set_page_config(page_title="DR-XAI demo", page_icon="👁", layout="wide")


def model_of_record() -> str:
    """Run id of the model whose test results are in results/ (or XAIDR_RUN to override)."""
    if os.environ.get("XAIDR_RUN"):
        return os.environ["XAIDR_RUN"]
    return str(pd.read_csv(TABLE4)["run_id"].iloc[0])


@st.cache_resource(show_spinner="Loading the model…")
def predictor():
    return load_predictor(ROOT / "checkpoints" / model_of_record())


@st.cache_data(show_spinner="Analysing the image…")
def analyse(image_bytes: bytes):
    import io

    rgb = np.asarray(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    return rgb, predict_image(predictor(), rgb)


def sidebar() -> None:
    st.sidebar.header("About this model")
    model = predictor()
    st.sidebar.write(model.label.split(": ")[-1].replace("convnext_tiny", "ConvNeXt-Tiny"))
    st.sidebar.caption("Trained on APTOS 2019 on a laptop CPU. One shared network with two outputs: "
                       "DR present / absent, and the severity grade 0-4.")
    if TABLE4.is_file():
        table = pd.read_csv(TABLE4).set_index(["task", "metric"])
        n = int(table["n"].iloc[0])
        st.sidebar.subheader(f"Measured on {n} unseen test images")
        st.sidebar.metric("Detection accuracy", f"{table.loc[('detection', 'accuracy'), 'value']:.1%}")
        st.sidebar.metric("Grading accuracy", f"{table.loc[('grading', 'accuracy'), 'value']:.1%}")
        st.sidebar.metric("Grading agreement (QWK)", f"{table.loc[('grading', 'qwk'), 'value']:.2f}")
    st.sidebar.subheader("Known limits")
    st.sidebar.write("- Weakest on Proliferative DR, which it often grades as Moderate.\n"
                     "- Tested on one public dataset only.\n"
                     "- The heat-map shows where the model looked, not a confirmed lesion.")
    st.sidebar.warning("Research prototype. Not a medical device and not for diagnosis.")


def choose_image() -> tuple[bytes | None, str, int | None]:
    """Uploaded file, or a sample with its true grade."""
    upload = st.file_uploader("Upload a fundus photograph", type=["jpg", "jpeg", "png"])
    if upload is not None:
        return upload.getvalue(), upload.name, None
    listing = SAMPLES / "samples.csv"
    if not listing.is_file():
        return None, "", None
    samples = pd.read_csv(listing, dtype={"image_id": str})
    labels = ["—"] + [f"Sample {i + 1}" for i in range(len(samples))]
    pick = st.selectbox("…or try one of the test-set samples (never used in training)", labels)
    if pick == "—":
        return None, "", None
    row = samples.iloc[labels.index(pick) - 1]
    return (SAMPLES / row["file"]).read_bytes(), pick, int(row["true_grade"])


def show_result(rgb: np.ndarray, result, name: str, true_grade: int | None) -> None:
    if not result.looks_like_fundus:
        st.warning(f"This may not be a fundus photograph: {result.fundus_note}. "
                   "The result below is not meaningful for other kinds of picture.")
    left, right = st.columns([1, 1])
    with left:
        if result.dr_present:
            st.error(f"### Diabetic retinopathy: **detected**\n"
                     f"Detection score {result.detection_score:.2f} (threshold {result.threshold:.2f})")
        else:
            st.success(f"### Diabetic retinopathy: **not detected**\n"
                       f"Detection score {result.detection_score:.2f} (threshold {result.threshold:.2f})")
        st.markdown(f"### Severity grade: **{result.grade} · {result.grade_name}**")
        st.caption(GRADE_HELP[result.grade])
        if result.dr_present != (result.grade >= 1):
            st.info("The two outputs disagree for this image (detection and grade come from separate "
                    "heads of the network). Treat the result as uncertain.")
        if true_grade is not None:
            verdict = "matches" if true_grade == result.grade else "does **not** match"
            st.markdown(f"Reference grade from the dataset: **{true_grade} · {GRADE_NAMES[true_grade]}** "
                        f"— the prediction {verdict}.")
    with right:
        st.markdown("**Probability of each grade**")
        for grade, probability in enumerate(result.probabilities):
            st.progress(float(probability), text=f"{grade} · {GRADE_NAMES[grade]}  —  {probability:.0%}")

    st.divider()
    _, first, second, third, _ = st.columns([0.4, 1, 1, 1, 0.4])
    first.image(rgb, caption=f"Uploaded image ({name})", use_container_width=True)
    second.image(result.processed_rgb, caption="After preprocessing (crop, contrast, resize)",
                 use_container_width=True)
    healthy = not result.dr_present and result.grade == 0
    # the map is always scaled to its own maximum, so on a healthy eye it is shown faintly and
    # labelled as attention, not as a finding
    caption = ("No DR found: faint map of where the model looked, not a lesion" if healthy
               else "Where the model looked (Grad-CAM++); red = strongest")
    third.image(overlay(result.processed_rgb, result.saliency, 0.25 if healthy else 0.6),
                caption=caption, use_container_width=True)


def main() -> None:
    # tighter spacing so that the verdict and the three images fit on one laptop screen
    st.markdown(
        "<style>.block-container{padding-top:2.2rem;padding-bottom:1rem;max-width:1250px}"
        "h1{font-size:1.9rem !important;padding-bottom:0.2rem}"
        "h3{padding-top:0.3rem;padding-bottom:0.2rem}hr{margin:0.6rem 0}</style>",
        unsafe_allow_html=True,
    )
    st.title("Diabetic Retinopathy Detection & Severity Grading")
    st.caption("Explainable AI demo · upload a photograph of the back of the eye")
    sidebar()
    image_bytes, name, true_grade = choose_image()
    if image_bytes is None:
        st.info("Upload a fundus photograph or pick a sample to begin.")
        return
    try:
        rgb, result = analyse(image_bytes)
    except Exception as exc:  # noqa: BLE001  (show the problem instead of a blank page)
        st.error(f"This file could not be analysed: {exc}")
        return
    show_result(rgb, result, name, true_grade)


main()
