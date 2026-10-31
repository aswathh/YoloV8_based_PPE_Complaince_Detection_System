from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from src.compliance import check_workers, dashboard
from src.inference import load_model

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best.pt"

st.title("PPE Compliance Detection")

if not MODEL_PATH.exists():
    st.error(f"Model not found: {MODEL_PATH}")
    st.stop()


@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)


model = get_model()

f = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if f:
    img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
    r = model.predict(img, conf=0.25, verbose=False)[0]
    statuses = check_workers(r)

    st.image(cv2.cvtColor(r.plot(), cv2.COLOR_BGR2RGB), caption="Result")

    st.subheader("Worker Status")
    for i, s in enumerate(statuses, 1):
        st.write(f"Worker {i}: {s}")

    st.subheader("Summary")
    st.write(dashboard(statuses))