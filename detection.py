import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os

# Set Page Config
st.set_page_config(page_title="GMRT RFI Detector", layout="wide")

st.title("📡 GMRT RFI Detection Dashboard")
st.markdown("Visualize raw radio signals and detect interference (RFI) using robust statistics.")

# =========================
# DATA LOADING (CACHED)
# =========================

@st.cache_data
def load_and_process_fits(file_path):
    """Loads FITS and extracts power to avoid re-processing on every interaction."""
    if not os.path.exists(file_path):
        return None
    
    with fits.open(file_path) as hdul:
        # Based on your logic: remove singleton axes and extract complex power
        sig = hdul[0].data['DATA'][:, 0, 0, 0, :, :, :]
        
        real = sig[:, :, 0]
        imag = sig[:, :, 1]
        
        power = real**2 + imag**2
        power = np.nan_to_num(power)
        # Assuming we take the first polarization/index as per your Cell 6
        power2d = power[:, :, 0].astype(np.float64)
        
    return power2d

# =========================
# RFI DETECTION LOGIC
# =========================

def detect_rfi_robust(power2d, sigma=7):
    med = np.median(power2d, axis=0)
    mad = np.median(np.abs(power2d - med), axis=0)
    robust_std = 1.4826 * mad
    threshold = med + sigma * robust_std
    rfi_mask = power2d > threshold
    return rfi_mask.astype(np.uint8)

# =========================
# SIDEBAR / INPUTS
# =========================

st.sidebar.header("Settings")
# Update this path to your local file or use st.file_uploader
default_path = "/Users/aryamadiwale/Documents/gmrt data rfi project/dataset/3C338_150.LTA_RRLL.RRLLFITS.fits"
file_path = st.sidebar.text_input("FITS File Path", value=default_path)

sigma_val = st.sidebar.slider("Sigma Threshold", min_value=1, max_value=20, value=7)
n_time_view = st.sidebar.number_input("Samples to Visualize", value=5000)

# =========================
# MAIN APP FLOW
# =========================

if os.path.exists(file_path):
    power2d = load_and_process_fits(file_path)
    
    if power2d is not None:
        # Run Detection
        rfi_mask = detect_rfi_robust(power2d, sigma=sigma_val)
        rfi_perc = np.mean(rfi_mask) * 100

        # Stats Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Mean Power", f"{np.mean(power2d):.2f}")
        col2.metric("Median Power", f"{np.median(power2d):.2f}")
        col3.metric("RFI Detected", f"{rfi_perc:.4f}%")
        col4.metric("Data Shape", f"{power2d.shape}")

        # --- Plotting ---
        
        st.subheader("1. Raw GMRT Waterfall")
        fig1, ax1 = plt.subplots(figsize=(12, 5))
        img_slice = power2d[:n_time_view, :]
        
        im1 = ax1.imshow(
            img_slice,
            aspect='auto',
            origin='lower',
            vmin=np.percentile(img_slice, 1),
            vmax=np.percentile(img_slice, 99),
            cmap='viridis'
        )
        fig1.colorbar(im1, ax=ax1, label="Power")
        ax1.set_xlabel("Frequency Channel")
        ax1.set_ylabel("Time (samples)")
        st.pyplot(fig1)

        st.subheader("2. Binary RFI Mask")
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        mask_slice = rfi_mask[:n_time_view, :]
        
        ax2.imshow(
            mask_slice,
            aspect='auto',
            origin='lower',
            cmap='gray'
        )
        ax2.set_xlabel("Frequency Channel")
        ax2.set_ylabel("Time")
        st.pyplot(fig2)

else:
    st.error(f"File not found at: {file_path}. Please check the path in the sidebar.")

