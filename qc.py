import rasterio
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from rasterio.mask import mask
from rasterio.plot import show as rio_show

def qc_raster(raster_path, vector_path, aoi_name="AOI"):
    """
    Performs QC on a raster file clipped to a vector AOI.
    
    Computes statistics, plots a quick-look image with the AOI overlay,
    and plots a histogram of valid pixel values.
    """
    
    print(f"--- Starting QC for {aoi_name} ---")
    
    try:
        # Load the AOI vector
        aoi_poly = gpd.read_file(vector_path)
        
        # Open the raster file
        with rasterio.open(raster_path) as raster:
            # Ensure AOI CRS matches raster CRS
            if aoi_poly.crs != raster.crs:
                print(f"Warning: Reprojecting AOI from {aoi_poly.crs} to {raster.crs}")
                aoi_poly = aoi_poly.to_crs(raster.crs)
            
            # Get the geometry for masking
            geoms = aoi_poly.geometry.values
            
            # Clip the raster to the AOI
            # This returns the clipped image data as a numpy array
            # and the transform (spatial metadata) for the new array
            out_image, out_transform = mask(raster, geoms, crop=True)
            
            # Get the NoData value from the raster's metadata
            nodata_val = raster.nodata
            
            # --- Statistics Computation ---
            
            # Select the first band (assuming single-band DEM/hazard)
            data = out_image[0]
            
            # Get total number of pixels in the *clipped* extent
            total_pixels = data.size
            
            # Create a "valid data" array by filtering out NoData values
            # This is what we'll use for stats
            if nodata_val is not None:
                valid_data = data[data != nodata_val]
            else:
                # If NoData is not defined, assume all data is valid
                valid_data = data.flatten()

            # Calculate statistics
            if valid_data.size == 0:
                print("Error: No valid data found within the AOI.")
                print("Please check for an empty raster or AOI misalignment.")
                return

            min_val = valid_data.min()
            max_val = valid_data.max()
            mean_val = valid_data.mean()
            std_val = valid_data.std()
            nodata_pixels = total_pixels - valid_data.size
            percent_nodata = (nodata_pixels / total_pixels) * 100
            
            # Print the QC statistics
            print("\n📊 QC Statistics:")
            print(f"  AOI:            {aoi_name}")
            print(f"  CRS:            {raster.crs.to_string()}")
            print(f"  Min:            {min_val:.2f}")
            print(f"  Max:            {max_val:.2f}")
            print(f"  Mean:           {mean_val:.2f}")
            print(f"  Std. Dev:       {std_val:.2f}")
            print(f"  % NoData:       {percent_nodata:.2f}%")
            print(f"  (Total Pixels:  {total_pixels}, NoData Pixels: {nodata_pixels})")

            
            # --- Plotting ---
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
            fig.suptitle(f"Raster QC Report: {aoi_name} on {raster.name.split('/')[-1]}", fontsize=16)
            
            # Plot 1: Quick-Look Image
            # Create a masked array to properly hide NoData values in the plot
            masked_data = np.ma.masked_where(data == nodata_val, data)
            
            ax1.set_title("Quick-Look Image (Clipped to AOI)")
            # Use imshow to plot the numpy array
            img = ax1.imshow(masked_data, cmap='terrain', vmin=min_val, vmax=max_val)
            # Overlay the AOI vector for context
            aoi_poly.plot(ax=ax1, facecolor='none', edgecolor='red', linewidth=1)
            ax1.axis('off')
            plt.colorbar(img, ax=ax1, orientation='horizontal', pad=0.05, label='Value')
            
            # Plot 2: Histogram
            ax2.set_title("Histogram of Valid Pixel Values")
            ax2.hist(valid_data, bins=50, color='blue', alpha=0.7)
            ax2.set_xlabel("Pixel Value")
            ax2.set_ylabel("Frequency (Count)")
            ax2.grid(True, linestyle='--', alpha=0.5)
            # Add vertical lines for mean, min, max
            ax2.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.2f}')
            ax2.axvline(min_val, color='black', linestyle='dotted', linewidth=2, label=f'Min: {min_val:.2f}')
            ax2.axvline(max_val, color='black', linestyle='dotted', linewidth=2, label=f'Max: {max_val:.2f}')
            ax2.legend()
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

    except FileNotFoundError as e:
        print(f"Error: File not found. {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

# --- --- --- --- --- --- --- --- ---
# --- ⚙️ RUN THE QC ---
# --- --- --- --- --- --- --- --- ---

# 1. DEFINE YOUR FILE PATHS
#    (Replace with your actual file paths)
RASTER_FILE = r"C:\Users\udgir\Documents\GitHub\Rastor-QC\ARUP_PILOT_DATA\Rasters\Population\Population_UTM42.tif"
VECTOR_FILE = r"C:\Users\udgir\Documents\GitHub\Rastor-QC\ARUP_PILOT_DATA\Area_of_Interest\Arup_AOI.shp"  # e.g., "Kutch.shp" or "Lodha.shp"
AOI_NAME = "Arup" # Friendly name for titles

# 2. RUN THE FUNCTION
qc_raster(RASTER_FILE, VECTOR_FILE, AOI_NAME)