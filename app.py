from pathlib import Path

import streamlit as st
from PIL import Image
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best.pt"


@st.cache_resource
def load_model() -> YOLO:
    return YOLO(str(MODEL_PATH))


def predict(image: Image.Image) -> tuple[str, float, list[tuple[str, float]]]:
    model = load_model()
    result = model.predict(source=image, verbose=False)[0]
    if result.probs is None:
        raise RuntimeError("This model did not return classification probabilities.")

    probabilities = result.probs.data.tolist()
    class_names = model.names
    ranked = sorted(
        ((class_names[index], float(score)) for index, score in enumerate(probabilities)),
        key=lambda item: item[1],
        reverse=True,
    )
    return ranked[0][0], ranked[0][1], ranked


st.set_page_config(page_title="Neuro-Somatic AI", page_icon=":brain:", layout="centered")
st.title("Neuro-Somatic AI")
st.caption("Medical image classification powered by trained YOLO model")

if not MODEL_PATH.is_file():
    st.error(f"Model file not found at {MODEL_PATH}")
    st.stop()

uploaded_file = st.file_uploader("Upload a medical image", type=["jpg", "jpeg", "png", "bmp", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Run prediction", type="primary", use_container_width=True):
        with st.spinner("Analyzing image..."):
            try:
                predicted_class, confidence, ranked_predictions = predict(image)
            except Exception as error:
                st.error(f"Prediction failed: {error}")
            else:
                st.success(f"Prediction: {predicted_class}")
                st.metric("Confidence", f"{confidence:.2%}")
                st.subheader("Class probabilities")
                for class_name, probability in ranked_predictions:
                    st.write(f"{class_name}: {probability:.2%}")
                    st.progress(probability)
else:
    st.info("Upload an image to begin.")