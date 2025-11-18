import rasterio
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from rasterio.mask import mask
from scipy.stats import skew, kurtosis, gaussian_kde
from collections import Counter

# ================= GLOBAL SETTINGS =================
# Disable interactive mode and cursor globally
plt.ioff()
matplotlib.rcParams['toolbar'] = 'none'

# ================================================================
# ================== CORE UTILITY FUNCTIONS ======================
# ================================================================

def load_and_clip(raster_path, vector_path):
    """Load raster, load AOI, fix CRS if needed, and clip raster."""
    
    aoi = gpd.read_file(vector_path)

    with rasterio.open(raster_path) as src:

        # Reproject AOI if CRS mismatch
        if aoi.crs != src.crs:
            aoi = aoi.to_crs(src.crs)

        out_image, _ = mask(src, aoi.geometry.values, crop=True)
        data = out_image[0]

        return data, src.nodata, src.res, src.crs


def clean_valid_pixels(data, nodata_val):
    """Flatten raster to 1D and filter NoData + NaN pixels."""

    flat = data.flatten()
    mask = np.ones(flat.shape, dtype=bool)

    # Remove NoData
    if nodata_val is not None:
        mask &= (flat != nodata_val)

    # Remove NaN (floats)
    if np.issubdtype(flat.dtype, np.floating):
        mask &= ~np.isnan(flat)

    valid = flat[mask]

    return valid, flat.size, flat.size - valid.size


# ================================================================
# ===================== STATISTICS SECTION =======================
# ================================================================

def compute_statistics(valid_data):
    """Compute all statistical QC metrics."""

    stats = {}

    # Basic
    stats["min"] = float(np.min(valid_data))
    stats["max"] = float(np.max(valid_data))
    stats["mean"] = float(np.mean(valid_data))
    stats["std"] = float(np.std(valid_data))

    # Percentiles
    p = np.percentile(valid_data, [1, 5, 50, 95, 99])
    stats["p1"], stats["p5"], stats["p50"], stats["p95"], stats["p99"] = p

    # Distribution shape
    stats["skewness"] = float(skew(valid_data))
    stats["kurtosis"] = float(kurtosis(valid_data))

    # Mode + unique counts
    counts = Counter(valid_data)
    mode_val, mode_freq = counts.most_common(1)[0]
    stats["mode"] = float(mode_val)
    stats["unique_values"] = len(counts)

    # Outliers beyond P1-P99
    stats["outlier_count"] = int(
        np.sum((valid_data < stats["p1"]) | (valid_data > stats["p99"]))
    )
    stats["outlier_percent"] = float(
        stats["outlier_count"] / len(valid_data) * 100
    )

    return stats


# ================================================================
# ======================= PLOT FUNCTIONS =========================
# ================================================================

def plot_quicklook_raster(clipped_data, stats, title):
    """Quick-look raster image scaled to 1st-99th percentile."""
    safe_data = np.nan_to_num(clipped_data, nan=stats["min"], posinf=stats["max"], neginf=stats["min"])

    vmin, vmax = stats["p1"], stats["p99"]
    plt.figure(figsize=(8, 6))
    im = plt.imshow(safe_data, cmap="viridis", vmin=vmin, vmax=vmax)
    plt.colorbar(im, label="Pixel Value")
    plt.title(f"Quick-Look Raster – {title}")
    plt.tight_layout()
    plt.show()


def plot_histogram(valid_data, stats, title):
    """Histogram with KDE + percentile markers."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(valid_data, bins=60, alpha=0.75, edgecolor='white')
    kde = gaussian_kde(valid_data)
    xs = np.linspace(stats["min"], stats["max"], 300)
    ax.plot(xs, kde(xs), linewidth=2)
    for p in ["p1", "p5", "p50", "p95", "p99"]:
        ax.axvline(stats[p], linestyle="--", linewidth=1.2)
    ax.set_title(f"Histogram – {title}")
    ax.set_xlabel("Pixel Value")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def plot_boxplot(valid_data, title):
    """Boxplot of pixel distribution."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.boxplot(valid_data, vert=True, patch_artist=True)
    ax.set_title(f"Boxplot – {title}")
    ax.set_ylabel("Pixel Value")
    plt.tight_layout()
    plt.show()


# ================================================================
# ===================== MAIN QC FUNCTION =========================
# ================================================================

def raster_qc_statistics(raster_path, vector_path, aoi_name="AOI"):
    """
    Full QC script producing statistical outputs + visuals:
    - Quick-look raster image
    - Histogram + KDE
    - Boxplot
    - Full descriptive statistics
    """

    print(f"\n=== Raster QC (Statistics + Visuals) — {aoi_name} ===\n")

    # 1. Load + clip raster
    clipped_data, nodata_val, resolution, crs = load_and_clip(raster_path, vector_path)

    # 2. Clean valid pixels
    valid_data, total_pixels, nodata_pixels = clean_valid_pixels(clipped_data, nodata_val)

    if valid_data.size == 0:
        print("No valid pixel values found inside the AOI.")
        return

    # 3. Compute statistics
    stats = compute_statistics(valid_data)

    # 4. Structured Output
    print("--- General Info ---")
    print(f"CRS:                 {crs}")
    print(f"Resolution:          {resolution[0]} x {resolution[1]}")
    print(f"Total Pixels:        {total_pixels}")
    print(f"NoData Pixels:       {nodata_pixels} ({nodata_pixels/total_pixels*100:.2f}%)")

    print("\n--- Basic Stats ---")
    print(f"Min:                 {stats['min']:.3f}")
    print(f"Max:                 {stats['max']:.3f}")
    print(f"Mean:                {stats['mean']:.3f}")
    print(f"StdDev:              {stats['std']:.3f}")

    print("\n--- Percentiles ---")
    print(f"P1:                  {stats['p1']:.3f}")
    print(f"P5:                  {stats['p5']:.3f}")
    print(f"P50 (Median):        {stats['p50']:.3f}")
    print(f"P95:                 {stats['p95']:.3f}")
    print(f"P99:                 {stats['p99']:.3f}")

    print("\n--- Distribution Shape ---")
    print(f"Skewness:            {stats['skewness']:.3f}")
    print(f"Kurtosis:            {stats['kurtosis']:.3f}")

    print("\n--- Additional Metrics ---")
    print(f"Mode:                {stats['mode']:.3f}")
    print(f"Unique Values:       {stats['unique_values']}")
    print(f"Outliers (P1-P99):   {stats['outlier_count']} ({stats['outlier_percent']:.2f}%)")

    # 5. Visuals
    plot_quicklook_raster(clipped_data, stats, aoi_name)
    plot_histogram(valid_data, stats, aoi_name)
    plot_boxplot(valid_data, aoi_name)


# ================================================================
# ======================== RUN SCRIPT ============================
# ================================================================

if __name__ == "__main__":
    RASTER_FILE = r"C:\Users\udgir\Documents\GitHub\Rastor-QC\ARUP_PILOT_DATA\Rasters\DEM\DEM_UTM42.tif"
    VECTOR_FILE = r"C:\Users\udgir\Documents\GitHub\Rastor-QC\ARUP_PILOT_DATA\Area_of_Interest\Arup_AOI.shp"

    raster_qc_statistics(RASTER_FILE, VECTOR_FILE, "Arup")
