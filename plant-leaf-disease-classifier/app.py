from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from plant_disease.gradcam import overlay
from plant_disease.predictor import Predictor, pretty_label

MODEL_PATH = Path("models/best_plant_disease_model.pth")
SAMPLES_DIR = Path("assets/samples")
LOW_CONFIDENCE = 0.70

NOTES = {
    "Bacterial spot": "Small dark, water-soaked spots, often with a yellow halo. Caused by Xanthomonas bacteria.",
    "Early blight": "Brown lesions with concentric rings ('target' pattern), usually on older leaves first. Caused by Alternaria.",
    "Late blight": "Large, irregular grey-green to brown patches that spread quickly in cool, wet weather. Caused by Phytophthora infestans.",
    "Healthy": "No visible signs of the diseases this model was trained on.",
}

st.set_page_config(page_title="Tomato Leaf Disease Classifier", page_icon="🍅", layout="wide")


@st.cache_resource
def get_predictor():
    return Predictor(MODEL_PATH)


predictor = get_predictor()

with st.sidebar:
    st.header("About")
    st.write(f"`{predictor.arch}` fine-tuned on tomato leaf images from the PlantVillage dataset.")
    st.write("Classes: " + ", ".join(pretty_label(c) for c in predictor.classes))
    if predictor.metrics:
        st.write(f"Validation accuracy: {predictor.metrics['val_acc']:.1%}")
    show_cam = st.toggle("Show Grad-CAM", value=True, help="Highlights the regions that drove the prediction.")
    st.caption(
        "Trained on lab photos with plain backgrounds. Expect lower accuracy on field photos, "
        "and treat the output as a hint, not a diagnosis."
    )

st.title("Tomato Leaf Disease Classifier")
st.write("Upload a photo of a single tomato leaf, or pick one of the sample images.")

source = st.radio("Image source", ["Upload", "Sample"], horizontal=True, label_visibility="collapsed")
image = None
if source == "Upload":
    uploaded = st.file_uploader("Leaf image", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
else:
    samples = sorted(SAMPLES_DIR.glob("*.jpg")) if SAMPLES_DIR.exists() else []
    if samples:
        choice = st.selectbox("Sample", samples, format_func=lambda p: p.stem.replace("_", " "))
        image = Image.open(choice).convert("RGB")
    else:
        st.info("No sample images found in assets/samples.")

if image is not None:
    with st.spinner("Running model..."):
        result = predictor.predict(image, explain=show_cam)

    left, right = st.columns(2)
    left.image(image, caption="Input", width="stretch")
    if show_cam:
        right.image(overlay(image, result.heatmap), caption="Grad-CAM", width="stretch")

    st.subheader(f"{result.label}  ·  {result.confidence:.1%}")
    if result.confidence < LOW_CONFIDENCE:
        st.warning(
            "Low confidence. The image may be blurry, show more than one leaf, "
            "or not be a tomato leaf at all."
        )
    if result.label in NOTES:
        st.write(NOTES[result.label])

    probs = (
        pd.DataFrame({"class": list(result.probabilities), "probability": list(result.probabilities.values())})
        .sort_values("probability", ascending=False)
        .set_index("class")
    )
    st.bar_chart(probs, horizontal=True)
