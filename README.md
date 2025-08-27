<p align="left">
  <img src="README_images\GastruloidKit_logo.png" alt="Example gastruloid radial bins" width="500"/>
</p>

# GastruloidKit 
A simple Python library for analyzing widefield microscopy images of 2D gastruloids on a 26 x 26 chip. Focuses on marker expression localization in form of bins (donuts).

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
Before this library is used, the user will have to manually crop, adjust and create binary masks for all channels in the image using ImageJ. First you will have to convert all your .czi images into .tiff as GastruloidKit takes images in .tiff format. Arrange all of your czi files in a specific directory in this format:
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
In the example above we have WT and mutant (ND6) arranged in their respective numbered folders which represent which repeat they are a part of. In the example above, the directory name is 'CHIP_REPEATS' but this can be changed to anything you like. You set this up by configuring the settings for the analysis:

|variable|details|  
|-------------------|---|  
|**wt**|what you label your wild type sample|
|**mutant**|what you label your mutant sample|
|**repeats**|which of your repeats to include in the analysis|
|**markers**|List of markers imaged| 
|**ref_marker**|Your nuclear stain / DNA stain| 
|**channel_folders**|Dictionary of how markers are arranged in channels in your 3 dimensional image.| 
|**marker_colors**|Dictionary of how you want your markers to be colored in plots and figures.| 
|**directory**|A folder where all the analysis will take place| 


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
    'DAPI': "#0004ffff", # blue
    'psmad159': "#ff0000ff" # red
}

num_bins = 15 # set how many bins you want; in this example I set 15
gastruloid_radius = 110 # the radius of the gastruloid (will be the outermost circle); in this example I set 110 pixels
```
Now we're all set up, lets convert CZI into Tiff.
```python

from GastruloidKit.f_preprocessing import czi_to_tiff

czi_to_tiff(directory)

```

After the step above step, you should use ImageJ/ Fiji to do these steps:
1. Manually crop and adjust angles so that the CHIP is straight and a square. So that when the image is divided by a 26x26 grid, each grid will have 1 gastruloid in it. 
2. Scale the cropped image to 7800 x 7800 pixels
3. Save the cropped scaled image in directory like this:
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
Once youre done, Run the block below which:
1. Checks that your scaled images have the right 7800 x 7800 dimensions
2. Split the Tiffs into channels according to the channel folders set above 
```python
from GastruloidKit.f_preprocessing import check_dimensions, split_into_channels

check_dimensions(directory) 
# it should show your scaled images have dimensions ({number of channels}, 7800, 7800)

# split tiffs into channels 
split_into_channels(directory, repeats, conditions, channel_folders)
```
This will create {number of channels} tiffs in this format:
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
Now you will open ImageJ and start manually making masks for each of marker in each condition. We need this mask as it lets us filter out noise and focus specifically on signal. You will put the masks in the directory you set along with the scaled images but in this format:
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
Now all Images and Masks are aligned and the same size (7800 x 7800), we will split them into 676 boxes using a 26 x 26 grid. This process might take awhile and depend on how many channels and repeats you have. But you only have to run this process once as outputs are saved. 
```python
from GastruloidKit.f_modelDetection import grid_split

grid_split(directory, markers, conditions, repeats)
```
Outputs are these folders

- {directory}/{repeat}/boxes_npz: contains zipped numpy array of the boxes for downstream analysis
- {directory}/{repeat}/boxes_tiff: contains .tiff images of each individual box

Next step is to manually select which boxes contain gastruloids that you want to include in  your analsyis and which you want to discard. We will also create a new folder called **boxes_tiff_selected** that contains your chosen boxes properly indexed. The model marker below is what you will base your decision on whether to include or exclude a gastruloid. Usually the reference marker (DAPI in this case). When you run the function below, a pop-window will show and you can click on the buttons 'yes' or 'no' to decide. You only have to do this once as an excel (.csv) file containing your decisions are saved in a folder called **selection**. 
```python
from GastruloidKit.f_modelDetection import select_gastruloids, boxes_tiff_selected

select_gastruloids(directory, ref_marker, conditions, repeats) # opens pop-up for selection, makes the new folder containing only selected boxes. 
```
and you're done for preprocessing!! 
## Radial Bin Analysis

Now we begin Radial bin analysis of the gastruloids. In simple words, when you draw multiple circles of different sizes thats aligned to the center of the gastruloid, you basically divide the gastruloid into multiple donuts/ rings. We will quantify the expresion of your marker within these rings to look at/ find interesting patterns / phenotypes that your mutant gastruloid might have compared to the WT. Below is what I mean by radial bin analysis with an example of 15 bins. 

<p align="center">
  <img src="README_images\1.png" alt="Example gastruloid radial bins" width="200"/>
</p>

You can choose how many bins of equal sizes you want your gastruloid to be divided into (yes any, 3-50 it can take it). This was set in your settings above. 

Since we dont know what the gastruloid radius is (somewhere between 110-130 pixels usually) We will need to adjust it manually. What I mean by adjusting the radius manually is by setting the radius at a particular size and going over images like the one above to check that the outermost circle is approximately the same size as the gastruloid itself. 

Run the block below to start adjustment with *adjusting = True* and *loading = False*. A new folder called **adjusting** will be made in your directory which will contain the images I mentioned above. Other folder made include:

- **coordinates**: contains coordinates of the gastruloid center
- **binarymasks**: contains binary masks that shows the detected gastruloid (white regions (1) indicate detected gastruloid region)

Once you've run the *bin_setting* function for the first time. It will save coordinates and binary masks into the folders I mentioned above and you dont have to make them again. Hence, for adjusting the gastruloid radius after that, you want to set *loading = True* so that it just reloads the binary masks and coordinates already made instead of making new ones (which take time).

When youre not adjusting for gastruloid radius, you dont want to save images into adjusting so you can set *adjusting=False* which means no images will be saved (makes it run faster).

Sometimes the coordinates / center of the gastruloid is not detected correctly. This is usually due to a really noisy image where the background intensity is not consistent/ similar to the gastruloid's intensity. However,  this does't tend to happen a lot and could be ignored. Just stating it here so you are aware that this may happen. 

```python
from GastruloidKit.f_coordFinder import bin_setting

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
Now we've set the radial bins and got coordinates of the gastruloid centers, its time to measure raw intensities of each marker within each bin and normalize it by whatever reference marker you have (DAPI in my case) 

In the fortunate case that you did not use pink nail polish to stick the chip onto the slides, you can skip this: Using said nail polish on GATA3 seems to make the signal rather noisy and I had to manually create a GATA3 filter that filters out noise by detecting bluriness (variance of laplacian) described by the image below.

<p align="center">
  <img src="README_images\2.png" alt="GATA3 Filter" width="300"/>
</p>

```python 
from GastruloidKit.f_intensityMeasurement import make_GATA3_filter

make_GATA3_filter(directory, repeats, conditions) # makes the GATA3 filter and saves it in a folder called GATA3filter 
```
Time to measure intensities!!

```python 
from GastruloidKit.f_intensityMeasurement import get_rawintensities, normalize_intensities

# measure raw intensities of all markers 
get_rawintensities(directory, repeats, conditions, markers, gastruloid_radius, num_bins)

# normalize intensities of all markers using your reference marker (DAPI in my case)
normalize_intensities(directory, repeats, conditions, markers, ref_marker, num_bins)
```

To visualize what we have, lets visualize the average gastruloid profile for each condition in each repeat which is stored in an excel (.csv) file called **{num_bins}_meta_intensities.csv** in your directory. The higher the bin number, the closer it is to the peripheral of the gastruloid. These are examples of the profile plots we will make:
<p align="center">
  <img src="README_images\3.png" alt="Gastruloid Profile example WT" width="300"/>
  <img src="README_images\4.png" alt="Gastruloid Profile example ND6" width="300"/>

These are examples of the Gastruloid Representation plots we will make:
<p align="center">
  <img src="README_images\5.png" alt="Gastruloid radialheatmap example WT-SOX2" width="200"/>
  <img src="README_images\6.png" alt="Gastruloid radialheatmap example WT-BRA" width="200"/>
  <img src="README_images\7.png" alt="Gastruloid radialheatmap example WT-GATA3" width="200"/>

These are examples of the overlap profile plots we will make which is done per marker. 
<p align="center">
  <img src="README_images\8.png" alt="Gastruloid Overlap example BRA" width="300"/>
  <img src="README_images\9.png" alt="Gastruloid Overlap example SOX2" width="300"/>


You can find all of these under the **plots** folder in your directory!
</p>

```python
from GastruloidKit.f_intensityMeasurement import plot_gastruloidprofiles

# make plots 
plot_gastruloidprofiles(directory, repeats, conditions, markers, ref_marker, num_bins, marker_colors)
```
If by eye you can see your gastruloids are rather consistent per chip, these plots itself should be enough to let you visualize and quantify the difference in marker expression between your WT and mutant gastruloids. Whats interesting is when your gastruloids dont seem consistent in each CHIP and you want to explore the images even further. More specifically, you want to plot several gastruloid profiles split by a certain parameter. 

For example, you want to see whether gastruloids with low DAPI intensity have a different profile compared to those with high DAPI intensity. Or.. if you want to see whether gastruloids with a smaller high density center have a different profile to those with a larger high density center. This is what I mean by high density center. 
<p align="center">
  <img src="README_images\14.png" alt="Gastruloid Overlap example WT" width="150"/>
  <img src="README_images\15.png" alt="Gastruloid Overlap example ND6" width="150"/>
  <img src="README_images\16.png" alt="Gastruloid Overlap example ND6" width="150"/>
  
  On the left is the gastruloid profile when the high density center is lower, on the right is when its higher. 
</p>


The second case is what I did and this is an example of the plots I get:
<p align="center">
  <img src="README_images\10.png" alt="Gastruloid Overlap example WT" width="250"/>
  <img src="README_images\11.png" alt="Gastruloid Overlap example ND6" width="250"/>
  <img src="README_images\12.png" alt="Gastruloid Overlap example ND6" width="250"/>
  
  On the left is the gastruloid profile when the high density center is lower, on the right is when its higher. 
</p>

I currently have this set up for:

1. High Density DAPI center size (ideal for my data but may not be for yours!)
2. Overall DAPI Intensity (not the best from my experience as intensity is a questionable parameter when not normalized)
```python
from GastruloidKit.f_intensityMeasurement import DAPIintensity_split_profiles, DAPIcenter_split_profiles
from GastruloidKit.f_distributions import get_distributions, get_DAPIcenter_distributions

get_distributions(directory, repeats, conditions, markers) # get raw whole intensity distirbutions of all markers including DAPI
DAPIintensity_split_profiles(directory, repeats, conditions, num_bins, marker_colors) # split profiles by DAPI Intensity

# to run the one below, you need to run get_distributions first above as getting DAPI center size relies on DAPI intensity distirbution
get_DAPIcenter_distributions(directory, repeats, conditions, num_bins) # get DAPI center size distribution
DAPIcenter_split_profiles(directory, repeats, conditions, num_bins, marker_colors) # split profiles by DAPI center size

```
The distributions themself are also to look at to see how consistent your gastruloids are on the same chip! Plots made from running the block below are in the plots folder under sub-folders named:
- DAPI_profile: DAPI Intensity in all bins along with the distribution of when they drop below 0.8 (you can adjust this drop threshold.)
- DAPIcenter_profiles: how marker expression varies depending on the size of high density center of the gastruloid. 
- DAPIintensity_profiles: how marker expression varies depending on DAPI Intensity 


Now, I'm sure it will also be useful to pick a specific gastruloid and get a figure that looks like the one below!
<p align="center">
  <img src="README_images\13.png" alt="channel split example" width="1000"/>
  
  (merge doesn't include DAPI!)
</p>

```python
from GastruloidKit.f_visualizechannels import channels_plot_any,

# Set the details of which gastruloid you want to plot like above -----------
chosen_condition = 'WT'
chosen_repeat = 1
chosen_id = 275
# ---------------------------------------------------------------------------

channels_plot_any(chosen_id, directory, chosen_repeat, chosen_condition, marker_colors) # makes a plot like the one above

markers_pair=("DAPI", "BRA") # 2 chosen markers for one that does pairs (because why not)
channels_plot_pair(chosen_id, directory, chosen_repeat, chosen_condition, markers_pair, marker_colors) # same as above but only for 2 chosen markers 
```
Now I'll wrap up with what excel (csv) files we have created incase you want to continue your analysis in excel and make more graphs! 
|Directory|details|  
|-------|-----------------------------------|  
|**{repeat}/DAPI_profile/{condition}_drop.csv**|furthest bin in where normalized DAPI intensity is above 0.8 (or any threshold you set previously) for each gastruloid|
|**{repeat}/distribution/{condition}_{marker}.csv**| The total intensity of a particular marker for each gastruloid. Raw intensity for DAPI, normalized intensity for other markers.|
|**{repeat}/intensities/{num_bins}\_meta\_individual\_{condition}.csv**|Normalized intensity of markers in each bin for each gastruloid.|
|**{num_bins}\_meta\_intensities.csv**|Average Normalized intensity of markers in each bin for each condition and repeat.| 
