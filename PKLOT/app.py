import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError

# -------------------- Page setup --------------------
st.set_page_config(
    page_title="PKLot Parking Spot Classifier",
    page_icon="🅿️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Dark Themed Look
st.markdown(
    """
    <style>
    :root { color-scheme: dark; }

    /* App shell */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #0e1117 !important;
        color: #fafafa !important;
    }
    [data-testid="stSidebar"] {
        background-color: #0b0f14 !important;
    }

    /* Main predict button */
    .stButton > button {
        background-color: #1f77b4 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
    }
    .stButton > button:hover {
        background-color: #16689b !important;
    }

    /* Uploader dropzone dark */
    [data-testid="stFileUploaderDropzone"] {
        background-color: #111827 !important;
        border: 1px solid #2c3440 !important;
        border-radius: 10px !important;
    }

    /* "Choose a parking spot image..." -> white */
    [data-testid="stFileUploader"] label {
        color: #ffffff !important;
    }

    /* Helper text in uploader -> white */
    [data-testid="stFileUploaderDropzone"] small,
    [data-testid="stFileUploaderDropzoneInstructions"] span {
        color: #d1d5db !important;
    }

    /* "Browse files" button -> dark scheme */
    [data-testid="stFileUploaderDropzone"] button {
        background-color: #1f2937 !important;
        color: #ffffff !important;
        border: 1px solid #374151 !important;
        border-radius: 8px !important;
    }
    [data-testid="stFileUploaderDropzone"] button:hover {
        background-color: #111827 !important;
        border-color: #4b5563 !important;
    }

    /* Uploaded file name chip -> dark scheme */
    [data-testid="stFileUploaderFile"] {
        background-color: #1f2937 !important;
        border: 1px solid #374151 !important;
        border-radius: 8px !important;
    }
    [data-testid="stFileUploaderFile"] span,
    [data-testid="stFileUploaderFileName"] {
        color: #f9fafb !important;
    }

    /* Caption under st.image */
    [data-testid="stImageCaption"] {
        color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------- Constants --------------------
MODEL_PATH = "best_model.keras"
IMG_SIZE = (96, 96)
THRESHOLD = 0.5

# -------------------- Model loading --------------------
@st.cache_resource
def load_model(model_path: str):
    """Load and cache model once per app session."""
    return tf.keras.models.load_model(model_path)

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize + normalize image to match training input."""
    img = image.convert("RGB").resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

# -------------------- UI --------------------
st.title("🅿️ PKLot Parking Spot Classifier")
st.write("Upload a parking spot image to predict whether it is **Empty** or **Occupied**.")

with st.expander("Model Settings", expanded=False):
    st.write(f"- Input size: `{IMG_SIZE[0]}x{IMG_SIZE[1]}`")
    st.write(f"- Decision threshold: `{THRESHOLD}`")
    st.write(f"- Model file: `{MODEL_PATH}`")

# Loading the Model
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error("Model could not be loaded.")
    st.exception(e)
    st.info("Make sure `best_model.keras` is in the same folder as this app.")
    st.stop()

uploaded_file = st.file_uploader(
    "Choose a parking spot image...",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG",
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Parking Spot", use_container_width=True)
    except UnidentifiedImageError:
        st.error("Invalid image file. Please upload a valid JPG/PNG image.")
        st.stop()

    if st.button("Predict", use_container_width=True):
        input_tensor = preprocess_image(image)
        pred = float(model.predict(input_tensor, verbose=0)[0][0])

        if pred < THRESHOLD:
            label = "Empty"
            confidence = (1.0 - pred) * 100.0
            st.success(f"Prediction: **{label}**")
        else:
            label = "Occupied"
            confidence = pred * 100.0
            st.warning(f"Prediction: **{label}**")

        st.progress(min(max(confidence / 100.0, 0.0), 1.0))
        st.write(f"Confidence: **{confidence:.2f}%**")
        st.caption(f"Raw model score (occupied probability): `{pred:.4f}`")
