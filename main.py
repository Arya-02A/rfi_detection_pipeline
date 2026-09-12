import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
import os
from dataset_builder import process_ml_dataset
import joblib

from model_training import build_overlay_map, train_rfi_model, extract_features
from analysis import run_full_analysis

# Set Page Config
st.set_page_config(page_title="GMRT RFI Detector", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

/* ══ PAGE BACKGROUND ══════════════════════════════════════════ */
.stApp, .main, [data-testid="stAppViewContainer"] {
    background-color: #f1f5f9 !important;
}
.block-container {
    padding-top: 3.5rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}
html, body, [class*="css"] {
    font-family: "Inter", sans-serif !important;
    color: #1e293b !important;
}

/* ══ HEADER BANNER ════════════════════════════════════════════ */
.header-banner {
    background: linear-gradient(135deg, #0c1e3c 0%, #0f3460 50%, #1a1a5e 100%);
    padding: 60px 48px 28px 48px;
    margin: 0 -4rem 32px -4rem;
    text-align: center;
    border-bottom: 4px solid #0ea5e9;
}
.main-title {
    font-family: "Inter", sans-serif;
    font-weight: 900;
    font-size: 2.8rem;
    color: #ffffff;
    letter-spacing: 0.01em;
    line-height: 1.15;
    margin: 0 0 10px 0;
    text-shadow: 0 2px 16px rgba(14,165,233,0.4);
}
.sub-title {
    font-family: "IBM Plex Mono", monospace;
    font-size: 0.9rem;
    font-weight: 500;
    color: #bae6fd;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin: 0;
}

/* ══ TABS ══════════════════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
    background: #ffffff;
    border-bottom: 2px solid #e2e8f0;
    gap: 0;
    padding: 0 8px;
    border-radius: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: "Inter", sans-serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    color: #64748b !important;
    background: transparent !important;
    border-radius: 0 !important;
    padding: 14px 32px !important;
    border-bottom: 3px solid transparent !important;
    transition: all 0.15s ease;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #0284c7 !important;
    background: #f0f9ff !important;
}
.stTabs [aria-selected="true"] {
    color: #0284c7 !important;
    background: #f0f9ff !important;
    border-bottom: 3px solid #0284c7 !important;
}
.stTabs [data-baseweb="tab-panel"] {
    padding-top: 32px !important;
    background: transparent !important;
}

/* ══ METRIC CARDS ══════════════════════════════════════════════ */
[data-testid="metric-container"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    padding: 20px 24px !important;
    box-shadow: 0 1px 6px rgba(0,0,0,0.08) !important;
    border-top: 4px solid #0ea5e9 !important;
}
[data-testid="metric-container"] label {
    font-family: "IBM Plex Mono", monospace !important;
    color: #64748b !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}
[data-testid="metric-container"] [data-testid="metric-value"] {
    font-family: "Inter", sans-serif !important;
    font-weight: 800 !important;
    font-size: 2rem !important;
    color: #0f172a !important;
    line-height: 1.2 !important;
}

/* ══ SECTION HEADERS ═══════════════════════════════════════════ */
.section-hdr {
    font-family: "Inter", sans-serif;
    font-weight: 800;
    color: #0f172a;
    font-size: 1.4rem;
    padding: 12px 18px;
    margin: 40px 0 20px 0;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-left: 5px solid #0284c7;
    border-radius: 0 10px 10px 0;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}

/* ══ INSIGHT CARDS ═════════════════════════════════════════════ */
.insight-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #0284c7;
    border-radius: 0 10px 10px 0;
    padding: 18px 24px;
    margin: 8px 0 28px 0;
    font-family: "Inter", sans-serif;
    font-size: 1rem;
    color: #334155;
    line-height: 1.8;
}
.insight-card b {
    color: #0284c7;
    font-weight: 700;
}

/* ══ BADGES ════════════════════════════════════════════════════ */
.badge {
    display: inline-block;
    font-family: "IBM Plex Mono", monospace;
    font-size: 0.82rem;
    font-weight: 600;
    padding: 6px 16px;
    border-radius: 6px;
    margin: 4px 5px;
    letter-spacing: 0.04em;
}
.badge-good   { background: #dcfce7; border: 1.5px solid #16a34a; color: #15803d; }
.badge-warn   { background: #fef9c3; border: 1.5px solid #ca8a04; color: #92400e; }
.badge-danger { background: #fee2e2; border: 1.5px solid #dc2626; color: #991b1b; }

/* ══ BUTTONS ═══════════════════════════════════════════════════ */
.stButton > button {
    font-family: "Inter", sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    background: #0284c7 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 32px !important;
    transition: all 0.18s ease !important;
    box-shadow: 0 2px 8px rgba(2,132,199,0.35) !important;
    letter-spacing: 0.02em !important;
}
.stButton > button:hover {
    background: #0369a1 !important;
    box-shadow: 0 4px 16px rgba(2,132,199,0.5) !important;
}

/* ══ EXPANDER ══════════════════════════════════════════════════ */
.streamlit-expanderHeader {
    font-family: "Inter", sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    color: #334155 !important;
    background: #f8fafc !important;
}

/* ══ SIDEBAR ═══════════════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
}
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #0f172a !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
}
[data-testid="stSidebar"] label {
    color: #374151 !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {
    color: #374151 !important;
}

/* ══ STREAMLIT NATIVE TEXT ═════════════════════════════════════ */
h1, h2, h3 { color: #0f172a !important; font-weight: 700 !important; }
p, .stMarkdown p, li { color: #334155 !important; font-size: 1rem !important; }
.stAlert > div { color: #1e293b !important; font-size: 0.97rem !important; }
hr { border-color: #e2e8f0 !important; }

/* ══ PROGRESS ══════════════════════════════════════════════════ */
.stProgress > div > div {
    background: linear-gradient(90deg, #0284c7, #7c3aed);
    border-radius: 4px;
}

/* ══ SUCCESS / ERROR / INFO ════════════════════════════════════ */
[data-testid="stAlert"] {
    border-radius: 8px !important;
    font-size: 0.97rem !important;
}

/* ══ MATPLOTLIB CHARTS (light bg) ═════════════════════════════ */
[data-testid="stImage"] img {
    border-radius: 8px;
    border: 1px solid #e2e8f0;
}

/* ══ PLOTLY CHARTS ═════════════════════════════════════════════ */
.js-plotly-plot {
    border-radius: 10px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 1px 6px rgba(0,0,0,0.06);
}

/* ══ STREAMLIT INPUTS ══════════════════════════════════════════ */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {
    background: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 6px !important;
    color: #0f172a !important;
    font-size: 0.9rem !important;
}
[data-testid="stSlider"] {
    color: #0f172a !important;
}

/* ══ SELECTBOX / DROPDOWN ══════════════════════════════════════ */
[data-testid="stSelectbox"] select {
    background: #ffffff !important;
    color: #0f172a !important;
    border: 1px solid #cbd5e1 !important;
}

/* ══ SPINNER ═══════════════════════════════════════════════════ */
.stSpinner > div { color: #0284c7 !important; }

/* ══ DIVIDER ═══════════════════════════════════════════════════ */
[data-testid="stDivider"] { border-color: #e2e8f0 !important; }

/* ══ st.write / st.info / st.success ══════════════════════════ */
[data-testid="stAlert"][data-type="info"] {
    background: #f0f9ff !important;
    border-left: 4px solid #0284c7 !important;
    color: #0c4a6e !important;
}
[data-testid="stAlert"][data-type="success"] {
    background: #f0fdf4 !important;
    border-left: 4px solid #16a34a !important;
    color: #14532d !important;
}
[data-testid="stAlert"][data-type="error"] {
    background: #fef2f2 !important;
    border-left: 4px solid #dc2626 !important;
    color: #7f1d1d !important;
}

/* ══ CODE BLOCKS ═══════════════════════════════════════════════ */
code, pre {
    background: #f1f5f9 !important;
    color: #0f172a !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 6px !important;
}

/* ══ TOAST ═════════════════════════════════════════════════════ */
[data-testid="toastContainer"] {
    font-family: "Inter", sans-serif !important;
    color: #0f172a !important;
}
</style>

<div class="header-banner">
    <div class="main-title">&#128225; GMRT RFI Detection Dashboard</div>
    <div class="sub-title">Giant Metrewave Radio Telescope &nbsp;&bull;&nbsp; Radio Frequency Interference Analysis Pipeline</div>
</div>
""", unsafe_allow_html=True)

tab_main, tab_analysis = st.tabs(["\U0001f6f0\ufe0f  RFI PIPELINE", "\U0001f52d  FITS ANALYSIS"])

with tab_main:
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
    default_path = "/rfi_detection_pipeline/dataset/3C338_150.LTA_RRLL.RRLLFITS.fits"
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
            im1 = ax1.imshow(img_slice, aspect='auto', origin='lower', 
                            vmin=np.percentile(img_slice, 1), vmax=np.percentile(img_slice, 99), cmap='viridis')
            fig1.colorbar(im1, ax=ax1, label="Power")
            ax1.set_xlabel("Frequency Channel")
            ax1.set_ylabel("Time (samples)")
            st.pyplot(fig1)

            with st.expander("🔍 Interpret this Plot"):
                st.markdown("""
                **What does this show?** A 'Waterfall' plot (spectrogram) representing radio intensity over time (Y-axis) and frequency (X-axis). Bright streaks or spots indicate high power levels.
            
                **How did we come up with it?** By extracting the `DATA` column from the FITS file, calculating the squared magnitude ($Real^2 + Imag^2$) of the complex voltage, and slicing the first 5000 time samples for performance.
            
                **Significance:** This is the "raw" view of the universe. It allows astronomers to visually identify patterns, such as constant frequency interference (vertical lines) or transient bursts (horizontal lines).
                """)

            st.subheader("2. Binary RFI Mask")
            fig2, ax2 = plt.subplots(figsize=(12, 5))
            mask_slice = rfi_mask[:n_time_view, :]
            ax2.imshow(mask_slice, aspect='auto', origin='lower', cmap='gray')
            ax2.set_xlabel("Frequency Channel")
            ax2.set_ylabel("Time")
            st.pyplot(fig2)

            with st.expander("🔍 Interpret this Plot"):
                st.markdown("""
                **What does this show?** A binary map where **white** pixels represent detected RFI (noise) and **black** pixels represent clean data or background sky.
            
                **How did we come up with it?** Using **Robust Statistics**. We calculate the Median Absolute Deviation (MAD) to find a threshold. Any signal exceeding the median by more than $7\sigma$ (standard deviations) is flagged as interference.
            
                **Significance:** This acts as a "truth mask." It highlights exactly which parts of the data are corrupted and need to be removed before scientific analysis begins.
                """)

    else:
        st.error(f"File not found at: {file_path}. Please check the path in the sidebar. Enter your dataset path there.")

    # --- Dataset_builder.py file code ---

    st.divider()
    st.header("🛠 ML Dataset Builder")

    if st.button("Build Dataset"):
        with st.spinner("Slicing waterfall and normalizing data..."):
            # We use the power2d and rfi_mask generated in the previous steps
            try:
                # Running the processing function
                xtr, xte, ytr, yte = process_ml_dataset(
                    power2d, 
                    rfi_mask, 
                    patch_size=64, 
                    stride=32
                )

                # Display Results
                st.success("Dataset successfully built!")
            
                c1, c2 = st.columns(2)
                with c1:
                    st.write("**Training Set**")
                    st.info(f"X: {xtr.shape} | y: {ytr.shape}")
                
                    # Show Class Distribution
                    clean_count = np.sum(ytr == 0)
                    rfi_count = np.sum(ytr == 1)
                    st.write(f"✅ Clean: {clean_count} | ⚠️ RFI: {rfi_count}")

                with c2:
                    st.write("**Testing Set**")
                    st.info(f"X: {xte.shape} | y: {yte.shape}")

                # Optional: Save to disk
                np.save("dataset_ml/X_train.npy", xtr)
                np.save("dataset_ml/y_train.npy", ytr)
                st.toast("Files saved to dataset_ml/ folder")

            except Exception as e:
                st.error(f"Error building dataset: {e}")

    # --- Modal Training ---

    st.divider()
    st.header("🤖 Model Training")

    if st.button("Train Model"):
        if 'X_train' not in locals() and not os.path.exists("dataset_ml/X_train.npy"):
            st.error("Please build the dataset first!")
        else:
            with st.spinner("Extracting features and training Random Forest..."):
                # Load the data we saved in the previous step
                X_data = np.load("dataset_ml/X_train.npy")
                y_data = np.load("dataset_ml/y_train.npy")
            
                # Train
                model, acc, feat_cols = train_rfi_model(X_data, y_data)
            
                # Save for future use
                joblib.dump(model, "models/rfi_model.pkl")
            
                st.success(f"Model Trained! Accuracy: {acc:.2%}")
            
                # Display Feature Importance
                st.subheader("Feature Importance")
                importances = model.named_steps['rf'].feature_importances_
                feat_imp = pd.Series(importances, index=feat_cols).sort_values()
                st.bar_chart(feat_imp)

                with st.expander("🔍 Interpret Feature Importance"):
                    st.markdown("""
                    **What does this show?** This bar chart ranks the "influence" of different statistical features (Mean, Std Dev, Skewness, Kurtosis) on the AI's classification.
                    
                    **How did we come up with it?** The Random Forest algorithm tracks how much the 'Gini Impurity' (prediction error) drops every time it uses a specific feature to split the data.
                    
                    **Significance:** It provides **Explainability**. If 'Kurtosis' is the top feature, it confirms the AI is correctly identifying RFI based on its "peakiness." This builds trust with researchers that the model is making decisions based on physics, not random noise.
                    """)

    # --- AI Overlay Map ---
    st.divider()
    st.header("🗺 AI Classification Overlay")

    # Initialize session state for predictions if it doesn't exist
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None

    if os.path.exists("models/rfi_model.pkl"):
        if st.button("Generate AI Overlay Map"):
            with st.spinner("Classifying full waterfall..."):
                # 1. Load Model
                model_pkg = joblib.load("models/rfi_model.pkl")
                model = model_pkg if not isinstance(model_pkg, dict) else model_pkg['model']
            
                # 2. Extract features and PREDICT
                X_data = np.load("dataset_ml/X_train.npy")
                full_features = extract_features(X_data)
            
                # STORE IN SESSION STATE so it persists
                st.session_state.predictions = model.predict(full_features)
            
                # 3. Build and Plot
                overlay = build_overlay_map(
                    st.session_state.predictions, 
                    power2d.shape, 
                    patch_t=64, patch_f=64, stride_t=32, stride_f=32
                )
            
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.imshow(power2d[:5000, :], aspect='auto', origin='lower', cmap='viridis')
                im = ax.imshow(overlay[:5000, :], aspect='auto', origin='lower', 
                               cmap='coolwarm', alpha=0.4, vmin=0, vmax=2)
                plt.colorbar(im, ax=ax, label="0=Clean, 1=Narrow, 2=Broad")
                st.pyplot(fig)

                with st.expander("🔍 Interpret this Plot"):
                    st.markdown("""
                    **What does this show?** A semantic segmentation map. The colors represent different *types* of RFI: 
                    * **Blue/Cool**: Narrowband (Specific frequencies)
                    * **Red/Warm**: Broadband (Burst/Impulsive noise)
                
                    **How did we come up with it?** We trained a **Random Forest Classifier** on features like mean, variance, and "peakiness" (kurtosis) extracted from small 64x64 patches of the waterfall. The model then predicts the category for every pixel.
                
                    **Significance:** Unlike the binary mask, this AI view tells us the **nature** of the noise. Identifying the type of RFI helps in tracing its physical source (e.g., a nearby cell tower vs. a faulty power line).
                    """)

    # --- RFI Analysis Section ---
    st.divider()
    st.header("📊 Detailed RFI Analysis")

    if st.button("Analyse RFI"):
        # Now we check the Session State instead of local variables
        if st.session_state.predictions is None:
            st.error("Please run the 'Generate AI Overlay Map' step first!")
        else:
            preds = st.session_state.predictions
            total_patches = len(preds)
        
            # Calculate stats
            p_clean = (np.sum(preds == 0) / total_patches) * 100
            p_narrow = (np.sum(preds == 1) / total_patches) * 100
            p_broad = (np.sum(preds == 2) / total_patches) * 100

            # Metrics display
            c1, c2, c3 = st.columns(3)
            c1.metric("Clean Data", f"{p_clean:.1f}%")
            c2.metric("Narrowband (Satellites/FM)", f"{p_narrow:.1f}%")
            c3.metric("Broadband (Power/Sparking)", f"{p_broad:.1f}%")

            # Visual Pie Chart
            fig_pie, ax_pie = plt.subplots(figsize=(4, 4))
            ax_pie.pie([p_clean, p_narrow, p_broad], 
                       labels=['Clean', 'Narrowband', 'Broadband'], 
                       autopct='%1.1f%%', colors=['#2ecc71', '#3498db', '#e74c3c'])
            st.pyplot(fig_pie, use_container_width=False)

            with st.expander("🔍 Interpret this Plot"):
                st.markdown("""
                **What does this show?** The total distribution of data quality across the entire observation.
            
                **How did we come up with it?** By aggregating the predictions from the AI model across all processed patches and calculating the percentage of each class.
            
                **Significance:** This is the "Decision Support" view. It allows an astronomer to decide if an observation is "healthy" enough to keep or if the interference is so high (e.g., >20%) that the data should be discarded.
                """)

            st.info(f"""
            **Analysis Summary:**
            * **Narrowband ({p_narrow:.1f}%)**: Likely terrestrial transmitters or satellite downlinks.
            * **Broadband ({p_broad:.1f}%)**: Likely impulsive noise from power lines or local electronics.
            * **Scientific Yield**: {p_clean:.1f}% of the data is suitable for imaging.
            """)


# --- Assistant / FAQ in Sidebar ---
st.sidebar.divider()
st.sidebar.subheader("🙋 Researcher Help Center")

faq = {
    "What is RFI?": "Radio Frequency Interference (RFI) is unwanted signal noise from human electronics (cell towers, satellites, etc.) that masks faint cosmic signals.",
    "What is MAD?": "Median Absolute Deviation (MAD) is a 'robust' measure of variability. Unlike standard deviation, it isn't skewed by the very RFI we are trying to detect.",
    "Why Random Forest?": "It effectively handles non-linear patterns in waterfall data, such as distinguishing between a constant 'leak' (Narrowband) and a sudden 'spark' (Broadband).",
    "What is the 'Overlay'?": "It maps AI predictions back onto the original time-frequency grid to show exactly where and what type of interference was identified.",
    "How to improve accuracy?": "Increase the 'Sigma Threshold' if the model is too sensitive, or retrain the model with more diverse FITS files.",
    "What is the Gaussian Noise Floor?": "It is the baseline signal level created by the telescope's electronics and the natural cosmic background radiation.",
    "Is 7 sigma too high?": "For very weak RFI, yes. If you suspect low-level interference is leaking through, try lowering the Sigma to 5 or 4 in the settings."
}

query = st.sidebar.selectbox("Common Technical Queries:", ["Select a question..."] + list(faq.keys()))
if query != "Select a question...":
    st.sidebar.info(faq[query])

# --- Final Explainability Summary ---

st.divider()
st.header("🧠 AI Explainability Summary")

if st.session_state.get('predictions') is not None:
    st.success("Analysis Complete")
    st.write("Based on the classification, this observation contains a mix of terrestrial interference and celestial signals.")
    
    # Summary of logic
    st.info("""
    **Executive Summary:**
    1. **Detection:** We used Robust MAD statistics to isolate outliers from the Gaussian noise floor.
    2. **Classification:** A Random Forest model categorized patches into Narrowband (horizontal/vertical lines) or Broadband (bursts).
    3. **Conclusion:** The assistant and analysis plots allow researchers to validate the 'Scientific Yield' of the observation before proceeding to high-resolution imaging.
    """)
else:
    st.info("Run the 'Analyse RFI' button above to generate the final explainability report.")

# ══════════════════════════════════════════════════════════════
# AUTO ANALYSIS TAB
# ══════════════════════════════════════════════════════════════

with tab_analysis:

    st.markdown("""
    <div style="background:#f0f9ff;border:1px solid #bae6fd;border-left:5px solid #0284c7;
                border-radius:0 10px 10px 0;padding:20px 26px;margin-bottom:32px;
                font-family:Inter,sans-serif;font-size:1.05rem;color:#0c4a6e;line-height:1.85;">
        A <b style="color:#0284c7;">simple, plain-language overview</b> of what's inside the FITS file —
        its structure, columns, header metadata, and how the signal looks across records and channels.<br>
        Uses the same FITS file path set in the sidebar.
    </div>
    """, unsafe_allow_html=True)

    col_run, _ = st.columns([1, 3])
    with col_run:
        run_btn = st.button("🚀  EXPLORE FITS FILE", use_container_width=True)

    if run_btn:
        if not os.path.exists(file_path):
            st.error(f"File not found: {file_path}")
        else:
            prog = st.progress(0, text="Opening FITS file...")

            try:
                import time
                prog.progress(20, text="Reading metadata and sampling data...")
                results = run_full_analysis(file_path)
                prog.progress(80, text="Building charts...")
                time.sleep(0.2)
                prog.progress(100, text="Done!")
                time.sleep(0.3)
                prog.empty()

                info  = results["info"]
                stats = results["stats"]

                # ── TOP METRIC ROW ──────────────────────────────────
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Records",     f"{stats['Total Records in File']:,}")
                c2.metric("Freq. Channels",    f"{stats['Frequency Channels']}")
                c3.metric("Data Shape (1 rec)",stats['Visibility Data Shape'])

                # ══════════════════════════════════════════════════
                # SECTION 1 — FILE STRUCTURE
                # ══════════════════════════════════════════════════
                st.markdown('<div class="section-hdr">01 · File Structure — What\'s Inside the FITS File?</div>', unsafe_allow_html=True)
                st.plotly_chart(results["fig_hdu"], use_container_width=True)
                st.markdown("""<div class="insight-card">
                    A FITS file is organised into <b>HDUs (Header Data Units)</b> — think of them as
                    separate sheets in a workbook. The chart shows each block and how many data elements
                    it contains. The primary HDU holds the main observation data (visibilities); additional
                    HDUs often store antenna tables, source catalogues, or calibration info.
                </div>""", unsafe_allow_html=True)

                # ══════════════════════════════════════════════════
                # SECTION 2 — COLUMNS / PARAMETERS
                # ══════════════════════════════════════════════════
                st.markdown('<div class="section-hdr">02 · Columns in the Data (Parameter Names)</div>', unsafe_allow_html=True)
                parnames = info.get("parnames", [])
                if parnames:
                    param_descriptions = {
                        "UU": "U baseline coordinate (antenna separation in the East-West direction)",
                        "VV": "V baseline coordinate (antenna separation in the North-South direction)",
                        "WW": "W baseline coordinate (line-of-sight separation)",
                        "DATE": "Observation date/time of the record",
                        "BASELINE": "Pair of antennas used for this measurement",
                        "SOURCE": "Source ID being observed",
                        "FREQSEL": "Frequency setup/band selector",
                        "INTTIM": "Integration time (how long data was averaged)",
                        "WEIGHT": "Statistical weight of this data point",
                        "FLUX": "Flux density / brightness of the source",
                    }
                    rows_html = ""
                    for p in parnames:
                        desc = next((v for k, v in param_descriptions.items() if k in p.upper()), "Data parameter stored per visibility record")
                        rows_html += f"""
                        <tr style="border-bottom:1px solid #f1f5f9;">
                          <td style="color:#0284c7;font-family:'IBM Plex Mono',monospace;font-size:0.95rem;
                                     font-weight:600;padding:12px 0;">{p}</td>
                          <td style="color:#334155;font-size:1rem;padding:12px 0;">{desc}</td>
                        </tr>"""
                    st.markdown(f"""
<div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;
            padding:28px 32px;margin-bottom:24px;box-shadow:0 1px 6px rgba(0,0,0,0.07);">
  <table style="width:100%;border-collapse:collapse;font-family:Inter,sans-serif;">
    <thead>
      <tr style="border-bottom:2px solid #e2e8f0;">
        <th style="color:#64748b;font-size:0.78rem;letter-spacing:.09em;text-transform:uppercase;
                   font-weight:700;padding:0 0 14px 0;text-align:left;width:180px;">Parameter</th>
        <th style="color:#64748b;font-size:0.78rem;letter-spacing:.09em;text-transform:uppercase;
                   font-weight:700;padding:0 0 14px 0;text-align:left;">What it Means</th>
      </tr>
    </thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("No named parameter columns found in the primary HDU.")

                # ══════════════════════════════════════════════════
                # SECTION 3 — HEADER METADATA
                # ══════════════════════════════════════════════════
                st.markdown('<div class="section-hdr">03 · Header Metadata — Key Facts About the Observation</div>', unsafe_allow_html=True)
                st.markdown("""<div class="insight-card">
                    The FITS header stores everything about <b>how</b> the observation was taken —
                    the telescope, frequency, date, number of antennas, etc.
                    Below are the keywords stored in the file's primary header.
                </div>""", unsafe_allow_html=True)

                header_items = info.get("header_items", [])
                if header_items:
                    # Show first 40 most informative header entries
                    shown = header_items[:60]
                    import pandas as pd
                    df_hdr = pd.DataFrame(shown)[["keyword", "value", "comment"]]
                    df_hdr.columns = ["Keyword", "Value", "Description"]
                    st.dataframe(df_hdr, use_container_width=True, hide_index=True)
                else:
                    st.info("No header keywords found.")

                st.success("✅ FITS file explored successfully!")

            except Exception as e:
                import traceback
                prog.empty()
                st.error(f"Analysis failed: {e}")
                st.code(traceback.format_exc())
