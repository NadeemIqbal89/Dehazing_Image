import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
import tempfile

# ---------------------------
# Dehazing Functions
# ---------------------------
def dark_channel(img, window_size=15):
    min_channel = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
    dark = cv2.erode(min_channel, kernel)
    return dark

def estimate_atmospheric_light(img, dark, top_percent=0.1):
    img_size = img.shape[0] * img.shape[1]
    num_pixels = int(max(1, img_size * top_percent))

    flat_img = img.reshape(img_size, 3)
    flat_dark = dark.ravel()

    indices = np.argpartition(flat_dark, -num_pixels)[-num_pixels:]
    atmospheric_light = np.max(flat_img[indices], axis=0)

    return atmospheric_light

def estimate_transmission(img, atmospheric_light, window_size=15, omega=0.95):
    normalized_img = img / atmospheric_light
    transmission = 1 - omega * dark_channel(normalized_img, window_size)
    return transmission

def refine_transmission(transmission, img, r=60, eps=1e-3):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = gray_img / 255.0
    transmission = cv2.ximgproc.guidedFilter(gray_img, transmission.astype(np.float32), r, eps)
    return transmission

def recover_scene(img, transmission, atmospheric_light, t0=0.1):
    transmission = np.clip(transmission, t0, 1.0)
    scene_radiance = np.empty_like(img)
    for i in range(3):
        scene_radiance[:, :, i] = (img[:, :, i] - atmospheric_light[i]) / transmission + atmospheric_light[i]
    scene_radiance = np.clip(scene_radiance, 0, 255)
    return scene_radiance.astype(np.uint8)

def remove_haze(img):
    img = img.astype(np.float32)
    dark = dark_channel(img)
    atmospheric_light = estimate_atmospheric_light(img, dark)
    transmission = estimate_transmission(img, atmospheric_light)
    transmission_refined = refine_transmission(transmission, img)
    dehazed = recover_scene(img, transmission_refined, atmospheric_light)
    return dehazed

# ---------------------------
# Streamlit App
# ---------------------------
st.set_page_config(page_title="Image Dehazing", layout="wide")
st.title("🌀 **HazeLifter:** Lifts the haze and reveals the true image beneath.")

uploaded_file = st.file_uploader("Upload a hazy image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Read uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    try:
        # Dehaze the image
        dehazed_bgr = remove_haze(image_bgr)
        dehazed_rgb = cv2.cvtColor(dehazed_bgr, cv2.COLOR_BGR2RGB)

        # Display side-by-side
        st.subheader("Comparison of Hazy and Dehazed Images")
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Original Image", use_container_width=True)

        with col2:
            st.image(dehazed_rgb, caption="Dehazed Image", use_container_width=True)

        # Save dehazed image
        os.makedirs("Dehazed", exist_ok=True)
        file_name = os.path.splitext(uploaded_file.name)[0]
        extension = os.path.splitext(uploaded_file.name)[1]
        output_path = f"Dehazed/{file_name}d{extension}"
        cv2.imwrite(output_path, dehazed_bgr)

        st.success(f"Dehazed image saved as: {output_path}")

    except Exception as e:
        st.error(f"Error during processing: {e}")
