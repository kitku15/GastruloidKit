<p align="left">
  <img src="README_images\GastruloidKit_logo.png" alt="Example gastruloid radial bins" width="500"/>
</p>

# GastruloidKit 
A simple Python library for analyzing widefield microscopy images of 2D gastruloids grown on a 26 × 26 chip. The main focus is on quantifying marker expression localization using bin-based regions (“donuts”). The workflow combines **ImageJ** for preprocessing with **Python** for data analysis and visualization. 

## Author Notes
An example Jupyter Notebook is provided in the `Examples` folder, which you can adapt to your own experiments. The step-by-step guide below will walk you through the process, so no prior Python expertise is required. I recommend using **Visual Studio Code** to run and customize the analysis. 

An example of how I used GastruloidKit presented [here](https://docs.google.com/presentation/d/1hZn3ObTQKTqmRE0uhgRw3Acb85YmJmBYopF3wSZANaU/edit?usp=sharing). 

## Installation
```bash
python3 -m venv gastruloid-env

# On linux/macOS
source gastruloid-env/bin/activate
# On windows
gastruloid-env\Scripts\activate

cd GastruloidKit
pip install .
```

## Preprocessing
The preprocessing section of the workflow involves using both **ImageJ** and GastruloidKit in **Python**. **ImageJ** is used to manually crop, adjust and scale your images as well as generating binary masks for each channel. 

In summary, this is what you will do step by step:
1. **Convert your images with GastruloidKit**: All `.czi` files must first be converted into .tiff, since GastruloidKit only accepts .tiff input. GastruloidKit has a function for this
2. **Crop, adjust and scale images in ImageJ**: a tutorial on how to do that will be here [ImageJ_GastruloidKit_tutorial](https://docs.google.com/document/d/1LvXgfY4J6XFhC1LLktmw8N-1fBrI3O6e-Ck8_1cskTw/edit?usp=sharing).
3. **Split tiff into channels with GastruloidKit**: Split your adjusted image with multiple channels into multiple images with 1 channel each.
4. **Make masks in ImageJ**: Make a binary mask for each channel split previously. Tutorial for this also in the link above. 


Lets start **step 1** by placing all your `.czi` images into a dedicated directory following this structure:

```
CHIP_REPEATS
├── 1 
│   ├── WT.czi
│   └── ND6.czi
├── 2
│   ├── WT.czi
│   └── ND6.czi
├── 3
│   ├── WT.czi
│   └── ND6.czi
```
In the example setup, **WT** and **mutant (ND6)** samples are stored in separate numbered folders, with each folder corresponding to a biological repeat. The top-level directory is named `CHIP_REPEATS` in this example, but you can name it anything you like.

You define the structure and parameters for your analysis by editing the configuration settings:

|variable|details|  
|-------------------|---|  
|**wt**|Label for your wild-type sample|
|**mutant**|Label for your mutant sample|
|**repeats**|List of repeats to include in the analysis|
|**markers**|List of imaged markers| 
|**ref_marker**|Nuclear/DNA stain used as reference| 
|**channel_folders**|Dictionary mapping how markers are assigned to channels in your 3D image| 
|**marker_colors**|Dictionary specifying colors for each marker in plots and figures| 
|**directory**|Path to the main folder where all analysis will take place| 


```python
# 1. EXPERIMENT DETAILS SET UP ---------------------------------
wt = "WT" # this should be WildType 
mutant = "ND6" # change this to your mutant name 
conditions = [wt, mutant]

repeats = [1,2,3] # set this to whichever repeat you want to focus on, it could be [1] only or [1,3] only, etc. any combination you want. 

markers = ["DAPI", "SOX2", "BRA", "GATA3"] # change this to your markers 
ref_marker = "DAPI" # your reference marker / marker to mark cells in general

# change this to your markers and how theyre arranged on the CZI channels - look at ur images on ImageJ when assigning this
channel_folders = { 
        0: 'DAPI',
        1: 'SOX2',
        2: 'GATA3',
        3: 'BRA'
    }
# in the example above, channel 0 is DAPI, channel 1 is SOX2 and so on..

# -------------------------------------------------------------
# 2. DATA ANALYSIS SET UP -------------------------------------
directory = "CHIP_REPEATS" # directory in which you saved your .czi images in the structure shown above.  

marker_colors = { # This dictionary will determine what marker is plotted in what color. This is entirely up to you!
    'SOX2': '#00bcd4', # cyan
    'BRA': '#ffeb3b', # yellow
    'GATA3': '#9c27b0', # magenta
    'DAPI': "#ffffffff", # white
    'psmad159': "#ff0000ff" # red
}

num_bins = 15 # set how many bins you want; in this example I set 15
gastruloid_radius = 110 # the radius of the gastruloid (will be the outermost circle); in this example I set 110 pixels
```
Now we're all set up, lets import all the modules in GastruloidKit.
```python
from GastruloidKit.bins import *
from GastruloidKit.distributions import *
from GastruloidKit.detection import *
from GastruloidKit.intensity_bins import *
from GastruloidKit.preprocessing import *
from GastruloidKit.visualizechannels import *
```
Convert your `.czi` images with GastruloidKit into `.tiff`
```python
czi_to_tiff(directory)
```

Now we move on to **step 2**, Open **ImageJ/Fiji** to preprocess your images with the following steps:
1. **Crop and align**: Manually crop the image and adjust the orientation so that the chip is square and aligned. This ensures that when divided into a 26 × 26 grid, each grid cell contains exactly one gastruloid.
2. **Scale**: Resize the cropped image to **7800 × 7800 pixels**.
3. **Save**: Export the cropped and scaled image into your project directory, organized as follows:

if still unsure, link to tutorial is presented above. 
```python
    CHIP_REPEATS
    ├── 1 
    │   ├── {condition}_scaled.tiff 
    │   ├── WT_scaled.tiff
    │   └── ND6_scaled.tiff
    ├── 2
    │   ├── ...
    │   └── ...
    ├── 3
    │   ├──...
    │   └── ...
```
Now we move on to **step 3** which includes:
1. **Verify image dimensions**: Ensure that all scaled images are **7800 × 7800 pixels**.
2. **Split channels**: Separate the `.tiff` images into individual channels based on the channel_folders configuration defined earlier.
```python
check_dimensions(directory) 
# it should show your scaled images have dimensions ({number of channels}, 7800, 7800)

# split tiffs into channels 
split_into_channels(directory, repeats, conditions, channel_folders)
```
This process will generate **{number of channels} separate** `.tiff` files for each image, saved in the following format:
```python
    CHIP_REPEATS
    ├── 1 
    │   ├── WT_scaled.tiff
    │   ├── ND6_scaled.tiff
    │   ├── WT_scaled_{marker}.tiff # tiffs split by channel/marker
    │   ├── ND6_scaled_{marker}.tiff
    │   ├── WT_scaled_DAPI.tiff # example for DAPI
    │   ├── ND6_scaled_DAPI.tiff
    │   ├── WT_scaled_SOX2.tiff # example for SOX2
    │   ├── ND6_scaled_SOX2.tiff
    │   └── ...
    ├── 2
    │   ├── ...
    │   └── ...
    ├── 3
    │   ├──...
    │   └── ...
```
Lastly, lets move on to **step 4**. Open **ImageJ/Fiji** to manually create masks for each marker in every condition. These masks are essential for filtering out background noise and isolating the signal of interest.

Save the masks in the same directory as your scaled images, using the following format:
```python
    CHIP_REPEATS
    ├── 1 
    │   ├── WT_{marker}_mask.tiff
    │   ├── ND6_{marker}_mask.tiff
    │   ├── WT_DAPI_mask.tiff # example mask for DAPI
    │   ├── ND6_DAPI_mask.tiff
    │   ├── WT_SOX2_mask.tiff # example mask for SOX2
    │   ├── ND6_SOX2_mask.tiff
```
## Gastruloid Detection
From this section onwards, you dont have to use ImageJ and will only be using GastruloidKit in Python.

Once all images and masks are aligned and scaled to **7800 × 7800 pixels**, the next step is to split them into **676 individual boxes** using a 26 × 26 grid. This step may take some time depending on the number of markers and repeats. For reference, processing **4 markers across 2 repeats** typically takes around **10 minutes**.

You only need to run this step **once**, as the resulting outputs are saved for later analysis.

```python
grid_split(directory, markers, conditions, repeats)
```
The output of the grid-splitting step will be organized into the following folders:

- `{directory}/{repeat}/boxes_npz`: Contains zipped NumPy arrays of the boxes for downstream analysis.
- `{directory}/{repeat}/boxes_tiff`: Contains `.tiff` images of each individual box.

The next step is to **manually select which boxes contain gastruloids** that you want to include in your analysis. Boxes you wish to keep will be copied into a new folder called `boxes_tiff_selected`, properly indexed.

Selection is based on the **model/reference marker** (usually the nuclear/DAPI stain). When you run the selection function, a pop-up window will appear allowing you to click **“Yes”** to include or **“No”** to discard each box. You only need to do this once, as your decisions are automatically saved in an Excel (.csv) file inside a folder called **selection**.

This process can be time-consuming and somewhat tedious, but it is the most accurate way to ensure that all false positives are excluded from your analysis.

```python
select_gastruloids(directory, ref_marker, markers, conditions, repeats) # opens pop-up for selection, makes the new folder containing only selected boxes. 
```
## Radial Bin Analysis

Now we begin the radial bin analysis of the gastruloids. Simply put, this involves drawing multiple concentric circles centered on each gastruloid, effectively dividing it into **“donuts”** or rings. We then quantify the expression of each marker within these rings to identify patterns or phenotypes that may differ between your **mutant** and **WT** gastruloids.

Below is an example of radial bin analysis with 15 bins:
<p align="center">
  <img src="README_images\1.png" alt="Example gastruloid radial bins" width="200"/>
</p>

You can choose how many equally sized bins you want your gastruloid to be divided into—anywhere from 5 to 50. This is set in your configuration settings above. Lower values provide coarser measurements, while very high values may introduce noise.

Since the gastruloid radius varies (typically between 110–130 pixels), it needs to be adjusted manually. By “adjusting manually,” we mean setting the radius to a specific value and visually checking that the outermost circle aligns approximately with the gastruloid boundary in images like the example above.

To start this process, run the block below with *adjusting = True* and *loading = False*. This will create a new folder called **adjusting** in your project directory containing the images used for visual checks. Other folders created include:

- **coordinates**: Stores the coordinates of the gastruloid center.
- **binarymasks**: Stores binary masks of the detected gastruloid (white regions = detected gastruloid).

After running the `bin_setting` function for the first time, the coordinates and binary masks are saved. You don’t need to recreate them for future adjustments. To reload the saved data instead of generating new ones (faster), set: *loading = True*. 

When you are not adjusting the gastruloid radius, set: *adjusting = False*. This prevents saving images in the adjusting folder and makes the process run faster. 

**Tip**: I usually adjust on one of my repeats and use the same settings for my future repeats! This way I only need to adjust once. 

**Note**: Occasionally, the gastruloid center may be detected incorrectly, typically due to noisy images where background intensity is inconsistent with the gastruloid. This is rare and generally does not affect the analysis, but it’s worth being aware of.

```python
# First time running it: run it with adjusting True and Loading False 
bin_setting(directory, repeats, conditions, markers, gastruloid_radius, num_bins, adjusting=True, loading=False)
```

```python
# When adjusting for gastruloid radius set adjusting True and Loading True
bin_setting(directory, repeats, conditions, markers, gastruloid_radius, num_bins, adjusting=True, loading=True)
```

```python
# when you are not adjusting for gastruloid radius set adjusting False and loading True. 
bin_setting(directory, repeats, conditions, markers, gastruloid_radius, num_bins, adjusting=False, loading=True)

# Remember loading is always False the first time you run it. Always True the next time you do to make sure it runs faster as making binary masks take quite awhile and its much faster to load them!
```
Once the radial bins are set and the gastruloid center coordinates are obtained, the next step is to **measure the raw intensities** of each marker within each bin. These values are then **normalized** using your chosen reference marker (e.g., DAPI).

**Special case – Nail polish artifact:**
If you used pink nail polish to stick the chip onto the slides, you may notice that the **GATA3 signal becomes noisy**. In this case, a GATA3 noise filter can be applied. This filter detects blurriness using the variance of the Laplacian to remove noise, as illustrated in the image below.

<p align="center">
  <img src="README_images\2.png" alt="GATA3 Filter" width="300"/>
</p>

```python 
make_GATA3_filter(directory, repeats, conditions) # makes the GATA3 filter and saves it in a folder called GATA3filter 
```
Now it’s time to measure the intensities!

```python 
# measure raw intensities of all markers 
get_rawintensities(directory, repeats, conditions, markers, gastruloid_radius, num_bins)

# normalize intensities of all markers using your reference marker (DAPI in my case)
normalize_intensities(directory, repeats, conditions, markers, ref_marker, num_bins)
```

To inspect the results, we can visualize the average gastruloid profile for each condition and repeat. These data are stored in an Excel (.csv) file called: `{num_bins}_meta_intensities.csv` in your directory. In this file, higher bin numbers correspond to regions closer to the gastruloid periphery, while lower numbers are near the center.

Below are examples of the profile plots GastruloidKit can generate from this data:

Gastruloid Profile Line plots: 
<p align="center">
  <img src="README_images\3.png" alt="Gastruloid Profile example WT" width="300"/>
  <img src="README_images\4.png" alt="Gastruloid Profile example ND6" width="300"/>

Gastruloid Profile Representation plots per marker:
<p align="center">
  <img src="README_images\5.png" alt="Gastruloid radialheatmap example WT-SOX2" width="200"/>
  <img src="README_images\6.png" alt="Gastruloid radialheatmap example WT-BRA" width="200"/>
  <img src="README_images\7.png" alt="Gastruloid radialheatmap example WT-GATA3" width="200"/>

Gastruloid Profile Overlap Graphs per marker: 
<p align="center">
  <img src="README_images\8.png" alt="Gastruloid Overlap example BRA" width="300"/>
  <img src="README_images\9.png" alt="Gastruloid Overlap example SOX2" width="300"/>


You can find all of these under the **plots** folder in your directory!
</p>

```python
# make plots 
plot_gastruloidprofiles(directory, repeats, conditions, markers, ref_marker, num_bins, marker_colors)
```
If your gastruloids appear consistent within each chip, the average profile plots alone are usually sufficient to visualize and quantify differences in marker expression between WT and mutant gastruloids.

However, if the gastruloids are not consistent within a chip, you may want to explore the data in more detail by plotting individual gastruloid profiles, grouped by a specific parameter.

For example:

- Comparing gastruloids with low vs. high nuclear counterstain intensity to see if relative cell number affects marker distribution.
- Comparing gastruloids with a smaller vs. larger high-density center to examine differences in profile patterns.

Here, the “high-density center” refers to the region in the center of the gastruloid with a high concentration of cells as shown below by DAPI with smallest on the left and largest on the right. 

<p align="center">
  <img src="README_images\14.png" alt="Gastruloid Overlap example WT" width="150"/>
  <img src="README_images\15.png" alt="Gastruloid Overlap example ND6" width="150"/>
  <img src="README_images\16.png" alt="Gastruloid Overlap example ND6" width="150"/>
</p>


In the second scenario (grouping gastruloids by features such as high-density center size) here is an example of the plots you can generate:
<p align="center">
  <img src="README_images\10.png" alt="Gastruloid Overlap example WT" width="250"/>
  <img src="README_images\11.png" alt="Gastruloid Overlap example ND6" width="250"/>
  <img src="README_images\12.png" alt="Gastruloid Overlap example ND6" width="250"/>
  
  - Left: Gastruloid profile with a smaller high-density center. 
  - Middle: Gastruloid profile with a medium high-density center. 
  - Right: Gastruloid profile with a larger high-density center. 
</p>

Currently, the code allows grouping gastruloids by:

1. **High-density DAPI center size** – ideal for my dataset, but may not be the best parameter for yours. Threshold set between 0-1, larger value = strict intensity tolerance, smaller value, less strict intensity tolerance for center detection. 
2. **Overall DAPI intensity** – generally less reliable, as raw intensity can be misleading without proper normalization.

```python
get_distributions(directory, repeats, conditions, markers) # get raw whole intensity distirbutions of all markers including DAPI
DAPIintensity_split_profiles(directory, repeats, conditions, num_bins, marker_colors) # split profiles by DAPI Intensity

# to run the one below, you need to run get_distributions first above as getting DAPI center size relies on DAPI intensity distirbution
get_DAPIcenter_distributions(directory, repeats, conditions, num_bins, threshold=0.8) # get DAPI center size distribution
DAPIcenter_split_profiles(directory, repeats, conditions, num_bins, marker_colors) # split profiles by DAPI center size
```
The distributions themselves are also useful for assessing how consistent your gastruloids are within the same chip.

Plots generated by running the block below are saved in the plots folder, organized into the following sub-folders:
- **DAPI_profile**: Shows DAPI intensity across all bins, including the distribution of bins where intensity drops below a threshold (default = 0.8; adjustable).
- **DAPIcenter_profiles**: Shows how marker expression varies with the size of the gastruloid’s high-density center.
- **DAPIintensity_profiles**: Shows how marker expression varies with overall DAPI intensity.

You can also select a specific gastruloid to generate a figure like the one shown below!

<p align="center">
  <img src="README_images\13.png" alt="channel split example" width="1000"/>
</p>

You can choose whether to include your reference marker in the merged image:
- `include_ref_marker=False` = Reference marker is excluded from the merge
- `include_ref_marker=True` = Reference marker is included in the merge

The merged image personally looks better to me when the reference marker is excluded. 

```python
# Set the details of which gastruloid you want to plot like above -----------
chosen_condition = 'WT'
chosen_repeat = 1
chosen_id = 275
# ---------------------------------------------------------------------------

channels_plot_any(chosen_id, directory, chosen_repeat, chosen_condition, markers, marker_colors, ref_marker, include_ref_marker=False) # makes a plot like the one above
# choose whether to include your nuclear counterstain / DAPI from the merged image. 

markers_pair=("DAPI", "BRA") # 2 chosen markers for one that does pairs (because why not)
channels_plot_pair(chosen_id, directory, chosen_repeat, chosen_condition, markers_pair, marker_colors) # same as above but only for 2 chosen markers 
```
All generated figures can be viewed in the `channels` subfolder within the `plots` folder. 

## 🏁 Outputs Overview
### 1. CSV Files
Below is an overview of the CSV files (Excel) created during analysis. These files can be used to continue analysis in Excel or create additional plots:
|Directory|details|  
|-------|-----------------------------------|  
|**{repeat}/DAPI_profile/{condition}_drop.csv**|Furthest bin where normalized DAPI intensity is above 0.8 (or your chosen threshold) for each gastruloid|
|**{repeat}/distribution/{condition}_{marker}.csv**|Total intensity of a given marker for each gastruloid. Raw intensity for DAPI; normalized intensity for other markers|
|**{repeat}/intensities/{num_bins}\_meta\_individual\_{condition}.csv**|Normalized intensity of each marker in every bin for each gastruloid|
|**{num_bins}\_meta\_intensities.csv**|Average normalized intensity of markers per bin for each condition and repeat| 
### 2. Plots
|Directory|details|  
|-------|-----------------------------------|  
|**{repeat}/plots/channels**|Merged images of markers, optionally including the reference marker (controlled by include_ref_marker)|
|**{repeat}/plots/DAPI_profile**|DAPI intensity across bins, including distribution of bins where intensity drops below threshold (default 0.8; adjustable)|
|**{repeat}/plots/DAPIcenter_profiles**|Gastruloid marker expression profiles grouped by high-density center size|
|**{repeat}/plots/DAPIintensity_profiles**|Gastruloid marker expression profiles grouped by overall DAPI intensity |
|**{repeat}/plots/gastruloid_profiles**|Average gastruloid marker expression profiles and graphs|
|**{repeat}/plots/overlapdensity**|Overlap Density plots of average gastruloid marker expression profiles, WT vs mutant|


## Contact

If you have questions, encounter issues, or want to provide feedback, you can reach me at:

📩 Email: bungatiasyaira@outlook.com (Syaii)