````markdown
# Raster Quality Control (QC) Toolkit

This toolkit provides a professional-grade automated workflow for auditing geospatial raster datasets against a Vector Area of Interest (AOI). It goes beyond basic statistics to perform robust data cleaning and advanced statistical analysis.

## 🚀 How to Run

### 1. Environment Setup
It is recommended to use a Conda environment to handle geospatial dependencies (GDAL/Rasterio) correctly.

```bash
# Create a new environment (Python 3.10 recommended)
conda create -n geo_qc python=3.10
conda activate geo_qc

# Install required libraries
conda install -c conda-forge rasterio geopandas numpy matplotlib scipy
````

### 2\. Configure the Script

Open the python script and look for the **Run Configuration** section at the bottom. Update these paths to match your local files:

```python
RASTER_FILE = r"path/to/your/data.tif"
VECTOR_FILE = r"path/to/your/data.shp"
AOI_NAME = "My_Study_Area"
```

### 3\. Execute

Run the script from your terminal:

```bash
python qc.py
```

-----

## ⚙️ What It Does

The script performs a sequential health check on your data:

1.  **CRS Alignment:** Checks if the AOI and Raster are in the same Coordinate Reference System. If metadata mismatch is found, it auto-corrects the AOI to match the Raster to ensure proper clipping.
2.  **Clipping:** Cuts the raster "cookie dough" using the AOI "cookie cutter."
3.  **Robust Cleaning:** Filters out invalid data using a dual-check system:
      * Removes Metadata `NoData` values (e.g., -9999).
      * Removes Floating-point `NaN` values (common in hazard rasters).
4.  **Statistical Calculation:** Computes descriptive statistics (Min, Max, Mean, Std) and structural statistics (Skewness, Kurtosis, Percentiles).
5.  **Visualization:** Generates three diagnostic plots:
      * **Quick-Look Map:** Scaled to the 1st-99th percentile to ignore outliers.
      * **Histogram:** Shows data distribution with a KDE curve.
      * **Boxplot:** Visually identifies statistical outliers.

-----

## 🧠 Understanding the Output

When interpreting the results, use this guide to understand the "story" your data is telling:

  * **High % NoData (e.g., \>50%):**
      * *In Elevation (DEM):* Usually a critical error. Your AOI might be hanging off the edge of the raster file.
      * *In Hazard (Flood):* Likely correct. It often implies "Dry Land" (0 depth) where no data exists.
  * **Empty Data Warning:**
      * If the script reports "No valid data," it means the AOI physically overlaps the file, but every single pixel inside is empty.
  * **Distribution Shape:**
      * *Normal (Bell Curve):* Typical for natural terrain.
      * *Skewed:* Typical for flood maps (lots of shallow water, very little deep water).

-----

## 📚 Learning: Advanced Metrics & Why They Matter

We moved beyond basic `Min/Max` checks to include advanced metrics. Here is why these additions are useful for professional QC:

### 1\. Robust Data Cleaning (`NaN` Filtering)

  * **What it is:** A filter that removes both official `NoData` values and computer `NaN` (Not a Number) codes.
  * **Why it's useful:** Standard tools crash when they hit a `NaN`. This ensures the script runs smoothly on complex Float32 rasters (like flood depths) without reporting meaningless "nan" statistics.

### 2\. Percentiles (P1, P5, P95, P99)

  * **What it is:** Metrics that look at the data excluding the bottom 1% and top 1%.
  * **Why it's useful:** `Min` and `Max` are often just sensor noise or glitches. Percentiles show the **"True Range"** of the physical data, allowing you to ignore one-off errors.

### 3\. Skewness & Kurtosis

  * **What it is:** Measures of the "shape" of the data distribution (symmetry and tail heaviness).
  * **Why it's useful:** They validate physical reality. For example, if a flood model has "Negative Skew" (mostly deep water, little shallow water), the model physics might be wrong.

### 4\. Mode & Unique Counts

  * **What it is:** Counts how often specific values appear.
  * **Why it's useful:** Detects processing artifacts. If a DEM has 50% of its pixels at *exactly* the same height, it suggests a "flat-filling" error during data processing.

### 5\. Boxplot Visualization

  * **What it is:** A chart showing the median and quartiles.
  * **Why it's useful:** It provides an instant visual check for **Outliers**. Any dots outside the "whiskers" represent suspicious data points (spikes or sinks) that need investigation.

### 6\. Percentile-Scaled Quick-Look

  * **What it is:** A map colored based on the P1-P99 range rather than Min-Max.
  * **Why it's useful:** Prevents the map from looking "washed out" or solid-colored due to a single bright outlier pixel, ensuring you can actually see the terrain features.

<!-- end list -->

```
```