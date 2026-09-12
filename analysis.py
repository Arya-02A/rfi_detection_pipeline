"""
analysis.py — GMRT FITS Data Explorer
Simple, easy-to-understand overview of what's inside the FITS file:
data shape, columns, header info, and basic data distribution.
"""

import numpy as np
from astropy.io import fits
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ─────────────────────────────────────────────────────────────
# SHARED THEME
# ─────────────────────────────────────────────────────────────

LAYOUT = dict(
    paper_bgcolor="#ffffff",
    plot_bgcolor="#f8fafc",
    font=dict(family="Inter, sans-serif", color="#334155", size=13),
    title_font=dict(family="Inter, sans-serif", color="#0f172a", size=17),
    margin=dict(l=70, r=40, t=65, b=60),
)

AXIS = dict(
    gridcolor="#e2e8f0",
    zerolinecolor="#cbd5e1",
    linecolor="#cbd5e1",
    tickfont=dict(color="#475569", size=12),
    title_font=dict(color="#1e293b", size=13),
    title_standoff=14,
)


def _apply(fig, title=""):
    fig.update_layout(**LAYOUT, title=title)
    fig.update_xaxes(**AXIS)
    fig.update_yaxes(**AXIS)
    return fig


# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────

def load_fits_overview(file_path):
    """Extract high-level metadata and a small sample of the data."""
    info = {}
    with fits.open(file_path) as hdul:
        # HDU structure
        info["n_hdus"] = len(hdul)
        info["hdu_list"] = []
        for i, hdu in enumerate(hdul):
            entry = {
                "index": i,
                "name": hdu.name or f"HDU {i}",
                "type": type(hdu).__name__,
                "shape": hdu.data.shape if hdu.data is not None else None,
            }
            info["hdu_list"].append(entry)

        primary = hdul[0]
        data = primary.data

        # Header key-value pairs (filter to useful ones)
        header = primary.header
        header_items = []
        skip_keys = {"", "COMMENT", "HISTORY", "END"}
        for key, val in header.items():
            if key not in skip_keys and val != "":
                header_items.append({"keyword": key, "value": str(val), "comment": header.comments[key]})
        info["header_items"] = header_items

        # Column / parameter names in the binary table
        if hasattr(data, "parnames"):
            info["parnames"] = list(data.parnames)
        else:
            info["parnames"] = []

        # Data shape
        n_records = len(data)
        sample_rec = data[0]
        vis = sample_rec.data  # visibility array for one record
        info["n_records"] = n_records
        info["vis_shape"] = vis.shape  # shape of one visibility record
        info["data_ndim"] = vis.ndim

        # Sample a small set of records for distribution plots
        n_sample = min(200, n_records)
        sample_indices = np.linspace(0, n_records - 1, n_sample, dtype=int)
        sample_values = []
        for idx in sample_indices:
            rec = data[idx]
            v = rec.data
            real = v[..., 0].flatten()
            imag = v[..., 1].flatten()
            sample_values.append(np.sqrt(real**2 + imag**2).mean())

        info["sample_means"] = np.array(sample_values)
        info["sample_indices"] = sample_indices

        # Per-channel mean across sample records (first 100)
        n_ch_sample = min(100, n_records)
        channel_means = []
        for i in range(n_ch_sample):
            rec = data[i]
            v = rec.data
            real = v[..., 0]
            imag = v[..., 1]
            amp = np.sqrt(real**2 + imag**2)
            # collapse all axes except last spatial/channel dimension
            amp_flat = amp.reshape(-1, amp.shape[-2]) if amp.ndim >= 2 else amp.reshape(1, -1)
            channel_means.append(amp_flat.mean(axis=0))

        info["channel_means"] = np.mean(channel_means, axis=0) if channel_means else np.array([])
        info["n_channels"] = len(info["channel_means"])

    return info


# ─────────────────────────────────────────────────────────────
# PLOT A — HDU Structure bar chart
# ─────────────────────────────────────────────────────────────

def plot_hdu_structure(info):
    """Show what HDUs (data blocks) exist in the file."""
    hdus = info["hdu_list"]
    names = [f"[{h['index']}] {h['name']}" for h in hdus]
    sizes = []
    for h in hdus:
        if h["shape"] is not None:
            total = 1
            for s in h["shape"]:
                total *= s
            sizes.append(total)
        else:
            sizes.append(0)

    fig = go.Figure(go.Bar(
        x=names,
        y=sizes,
        marker_color="#0284c7",
        text=[f"{s:,}" for s in sizes],
        textposition="outside",
        hovertemplate="%{x}<br>Total elements: %{y:,}<extra></extra>",
    ))
    fig.update_xaxes(title="HDU Block")
    fig.update_yaxes(title="Total Data Elements")
    return _apply(fig, "FITS File Structure — Data Blocks (HDUs)")


# ─────────────────────────────────────────────────────────────
# PLOT B — Record-by-record mean signal across the observation
# ─────────────────────────────────────────────────────────────

def plot_record_means(info):
    """Show how signal strength varies record by record (like a timeline)."""
    sample_means = info["sample_means"]
    sample_indices = info["sample_indices"]

    overall_mean = float(np.mean(sample_means))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(sample_indices),
        y=list(sample_means),
        mode="lines+markers",
        line=dict(color="#0284c7", width=1.5),
        marker=dict(size=4, color="#0284c7", opacity=0.6),
        name="Mean Signal",
        hovertemplate="Record #%{x}<br>Mean Signal: %{y:.4f}<extra></extra>",
    ))
    fig.add_hline(
        y=overall_mean,
        line_dash="dash", line_color="#16a34a",
        annotation_text=f"Overall mean: {overall_mean:.4f}",
        annotation_font=dict(color="#16a34a"),
    )
    fig.update_xaxes(title="Record Index (observation timeline)")
    fig.update_yaxes(title="Mean Signal Strength")
    return _apply(fig, "Signal Strength Across the Observation")


# ─────────────────────────────────────────────────────────────
# PLOT C — Per-channel average signal (which channels carry data)
# ─────────────────────────────────────────────────────────────

def plot_channel_overview(info):
    """Show average signal per frequency channel — a simple spectral overview."""
    ch_means = info["channel_means"]
    if len(ch_means) == 0:
        return None

    channels = np.arange(len(ch_means))
    fig = go.Figure(go.Bar(
        x=list(channels),
        y=list(ch_means),
        marker_color="#7c3aed",
        hovertemplate="Channel %{x}<br>Avg Signal: %{y:.4f}<extra></extra>",
    ))
    fig.update_xaxes(title="Frequency Channel Index")
    fig.update_yaxes(title="Average Signal Strength")
    return _apply(fig, f"Average Signal per Frequency Channel ({len(ch_means)} channels)")


# ─────────────────────────────────────────────────────────────
# PLOT D — Distribution of signal values (histogram)
# ─────────────────────────────────────────────────────────────

def plot_signal_distribution(info):
    """Histogram of per-record mean signal values."""
    sample_means = info["sample_means"]

    fig = go.Figure(go.Histogram(
        x=list(sample_means),
        nbinsx=40,
        marker_color="#0ea5e9",
        opacity=0.85,
        hovertemplate="Signal range: %{x}<br>Count: %{y}<extra></extra>",
    ))
    fig.update_xaxes(title="Mean Signal Value per Record")
    fig.update_yaxes(title="Number of Records")
    return _apply(fig, "Distribution of Signal Values Across Records")


# ─────────────────────────────────────────────────────────────
# SUMMARY STATS (simple)
# ─────────────────────────────────────────────────────────────

def simple_summary(info):
    means = info["sample_means"]
    return {
        "Total Records in File":    info["n_records"],
        "Frequency Channels":       info["n_channels"],
        "Visibility Data Shape":    str(info["vis_shape"]),
        "Parameters (Columns)":     len(info["parnames"]),
        "Header Keywords":          len(info["header_items"]),
        "Signal Min (sample)":      float(np.min(means)),
        "Signal Max (sample)":      float(np.max(means)),
        "Signal Mean (sample)":     float(np.mean(means)),
        "Signal Std (sample)":      float(np.std(means)),
    }


# ─────────────────────────────────────────────────────────────
# MASTER RUNNER
# ─────────────────────────────────────────────────────────────

def run_full_analysis(file_path, **kwargs):
    """Load FITS file and produce simple overview results."""
    info = load_fits_overview(file_path)
    results = {}

    results["info"]            = info
    results["stats"]           = simple_summary(info)
    results["fig_hdu"]         = plot_hdu_structure(info)
    results["fig_records"]     = plot_record_means(info)
    results["fig_channels"]    = plot_channel_overview(info)
    results["fig_distribution"]= plot_signal_distribution(info)

    return results