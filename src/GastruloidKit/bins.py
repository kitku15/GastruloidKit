import os                        
import numpy as np                 
from skimage import color        
from skimage.filters import gaussian
from scipy.ndimage import gaussian_filter1d
from skimage import img_as_float
from skimage.measure import label, regionprops

from skimage import measure
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from concurrent.futures import ThreadPoolExecutor
from GastruloidKit.detection import load_boxes, load_allowed_ids
import itertools


def get_binary_mask(area, sigma):

    # If RGBA, drop alpha and convert to grayscale
    if area.ndim == 3 and area.shape[2] == 4:
            area = area[:, :, :3]  # Drop alpha
            area = color.rgb2gray(area)

    # normalize before applying Gaussian
    normalized = (area - area.min()) / (area.max() - area.min())

    # Apply Gaussian blur
    blurred = gaussian(normalized, sigma=sigma)

    def regions_touch_border(binary_mask):
        '''
        function that, for a threshold, generates binary mask and finds regions touching border.

        '''
        labeled = measure.label(binary_mask)
        regions = measure.regionprops(labeled)
        height, width = binary_mask.shape

        if not regions:
            return False  # no regions at all
        
        # Find largest region by area
        largest_region = max(regions, key=lambda r: r.area)
        minr, minc, maxr, maxc = largest_region.bbox
        
        height, width = binary_mask.shape
        # Check if bounding box touches border
        if minr == 0 or minc == 0 or maxr == height or maxc == width:
            return True
        return False

    def find_best_slope_tol(image, slope_tol_range, sigma=2, spike_fraction=0.2):
        '''
        Iterate over a range of slope_tol, find threshold, make binary mask, test border touching
        '''
        best_slope_tol = None
        best_threshold = None
        
        # Normalize image as in your processing
        min_intensity = image.min()
        max_intensity = image.max()
        norm_image = (image - min_intensity) / (max_intensity - min_intensity)
        blurred = gaussian(norm_image, sigma=sigma)
        
        for slope_tol in slope_tol_range:
            threshold = find_curve_base_threshold(image, slope_tol=slope_tol, spike_fraction=spike_fraction)
            
            binary_mask = blurred > threshold
            if not regions_touch_border(binary_mask):
                best_slope_tol = slope_tol
                best_threshold = threshold
            else:
                # regions touch border - slope_tol too big probably, skip
                pass
        
        if best_slope_tol is None:
            # print("No slope_tol found without border-touching regions")
            # fallback - smallest slope_tol's threshold
            best_threshold = find_curve_base_threshold(image, slope_tol=slope_tol_range[0], spike_fraction=spike_fraction)
        
        return best_slope_tol, best_threshold
    
    def find_curve_base_threshold(image, slope_tol, spike_fraction=0.2):
        
        smooth_sigma = 2
        min_intensity = image.min()

        hist, bin_edges = np.histogram(image.ravel(), bins=400, range=(min_intensity, 1))
        
        hist_smooth = gaussian_filter1d(hist.astype(float), sigma=smooth_sigma)
        deriv = np.gradient(hist_smooth)


        max_count = hist_smooth.max()

        # Find index where histogram falls below a fraction of the max count
        spike_end_idx = 0
        for i, count in enumerate(hist_smooth):
            if count < max_count * spike_fraction:
                spike_end_idx = i
                break
        
        # Start searching after the spike ends
        start_idx = spike_end_idx if spike_end_idx > 0 else 10


        for idx in range(start_idx, len(deriv)):
            # Find first index where slope magnitude is below tolerance (flattening)
            if abs(deriv[idx]) < slope_tol:
                return bin_edges[idx]
        
        # Fallback if no flattening found
        # print("no flattening found")
        return 0.1
    
    def multi_stage_slope_tol_search(image, sigma, spike_fraction):

        max_slope = 500

        # Stage 0: Wide search with big steps (50)---------------------------------------------
        stage0_vals = np.arange(1, max_slope+1, 50)
        best_slope, best_thresh = find_best_slope_tol(image, stage0_vals, sigma=sigma, spike_fraction=spike_fraction)
        
        if best_slope is None:
            # print("No suitable slope_tol found in stage 1, falling back to default threshold")
            return 10, 0.2  # or any reasonable default slope_tol and threshold

        # Stage 1: Wide search with big steps (5)---------------------------------------------
        low = max(1, best_slope - 50)
        high = min(max_slope, best_slope + 50)
        stage1_vals = np.arange(low, high + 5, 5)
        best_slope, best_thresh = find_best_slope_tol(image, stage1_vals, sigma=sigma, spike_fraction=spike_fraction)
        
        if best_slope is None:
            # print("No suitable slope_tol found in stage 1, falling back to default threshold")
            return 10, 0.2  # or any reasonable default slope_tol and threshold
        
        # Stage 2: Narrow search +/- 5 around best from stage 1, steps of 1-------------------
        low = max(1, best_slope - 5)
        high = min(max_slope, best_slope + 5)
        stage2_vals = np.arange(low, high + 1, 1)

        best_slope, best_thresh = find_best_slope_tol(image, stage2_vals, sigma=sigma, spike_fraction=spike_fraction)
        
        if best_slope is None:
            # print("No suitable slope_tol found in stage 2, falling back to default threshold")
            return 10, 0.2
        
        # Stage 3: Narrow search +/- 1 around best from stage 2, steps of 0.2----------------------
        low = max(1, best_slope - 1)
        high = min(max_slope, best_slope + 1)
        stage3_vals = np.arange(low, high + 0.2, 0.2)
        best_slope, best_thresh = find_best_slope_tol(image, stage3_vals, sigma=sigma, spike_fraction=spike_fraction)
        
        if best_slope is None:
            # print("No suitable slope_tol found in stage 3, falling back to default threshold")
            return 10, 0.2
        
        # Stage 4: Narrow search +/- 0.5 around best from stage 2, steps of 0.05----------------------
        low = max(1, best_slope - 0.5)
        high = min(max_slope, best_slope + 0.5)
        stage4_vals = np.arange(low, high + 0.05, 0.05)
        best_slope, best_thresh = find_best_slope_tol(image, stage4_vals, sigma=sigma, spike_fraction=spike_fraction)
        
        if best_slope is None:
            # print("No suitable slope_tol found in stage 4, falling back to default threshold")
            return 10, 0.2
        
        return best_slope, best_thresh

    # get the best slope tol value (maximum with no regions touching border)
    best_slope, binary_thresh = multi_stage_slope_tol_search(blurred, sigma=2, spike_fraction=0.3)

    # Convert to binary mask
    binary_mask = blurred > binary_thresh # can adjust

    labeled = measure.label(binary_mask) # Label connected components
    regions = measure.regionprops(labeled) # Measure properties
    if not regions:
        raise Exception("No regions found for binary mask")

    largest_region = max(regions, key=lambda r: r.area)

    return largest_region, binary_mask, blurred, binary_thresh


def get_coordinates(img_boxes, selection_csv):

    coordinates = []
    coordinates_ids = []
    largest_region_list = []
    binary_mask_list = []

    # get the selected boxes and sort the list 
    selected_boxes_ids = load_allowed_ids(selection_csv)
    selected_boxes_ids.sort()

    # Get coordinates 
    print("Making Binary masks----------------------")
    # Filter only selected boxes for processing
    box_args = [(i, img_box) for i, img_box in enumerate(img_boxes) if i+1 in selected_boxes_ids]

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_binarymask, box_args))  # preserves original order of box_args

    # Now unpack results in order
    for idx, coordinate, largest_region, binary_mask in results:
        coordinates.append(coordinate)
        coordinates_ids.append(idx)
        largest_region_list.append(largest_region)
        binary_mask_list.append(binary_mask)

    print("Saved all Binary masks----------------------")
    return coordinates, coordinates_ids, largest_region_list, binary_mask_list

def process_binarymask(args):
    i, img_box = args
    img_box = img_as_float(img_box)
    largest_region, binary_mask, _, _ = get_binary_mask(img_box, 10)
    coordinate = largest_region.centroid
    return (i+1, coordinate, largest_region, binary_mask)


def get_bin_radii(num_bins, outer_radius):
    """
    Returns a list of outer radii for bins of equal thickness.
    
    :param num_bins: Number of bins (int)
    :param outer_radius: Radius of the full circle (float)
    :return: List of radii (float) from inner to outer
    """
    step = outer_radius / num_bins  # thickness of each bin
    radii = [(i + 1) * step for i in range(num_bins)]
    return radii

def bin_setting(directory, repeats, conditions, markers, gastruloid_radius, num_bins, adjusting=False, loading=True):

    print("Numbers of bins set:", num_bins)

    # code that calculates the radius of all bins based on gastruloid_radius and bins_no
    # radiis is a list of radiuses of length num_bins 
    radiis = get_bin_radii(num_bins, gastruloid_radius)

    print("Radius list:", radiis)


    for repeat in repeats:
        for condition in conditions:

            print(f"Repeat: {repeat}, Condition: {condition}")

            # LOADING DAPI---------------------------------------------------------------------------------------------
            # DAPI is used to adjust for coordinates. 
            image_boxes_path = f"{directory}/{repeat}/boxes_npz/img_DAPI_{condition}.npz"
            img_boxes = load_boxes(image_boxes_path)

            # selected boxes csv 
            selection_output_dir = f"{directory}/{repeat}/selection"
            selection_csv = f"{selection_output_dir}/img_DAPI_{condition}.csv"
            # get selected ids:
            selected_boxes_ids = load_allowed_ids(selection_csv)
            selected_boxes_ids.sort()

            if not loading:
        
                # Refine coordinates 
                coordinates, coordinates_ids, largest_region_list, binary_mask_list = get_coordinates(img_boxes, selection_csv)

                # print what we have to reconfirm, all should be the same length 
                print("All these below should be the same length:-----------------------")
                print("coor len", len(coordinates))
                print("coor id len", len(coordinates_ids))

                # convert coordinates (in tuple format) into numpy array format
                converted_coordinates = [np.array([[float(y), float(x)]]) for x, y in coordinates]


            if loading:
                # Load coordinates 
                coor_output_dir = f"{directory}/{repeat}/coordinates"
                coordinates_path = f"{coor_output_dir}/{condition}.npz"

                data = np.load(coordinates_path)
                coordinates = data["coords"]

            def adjust(marker_adjusting, radiis):

                marker_boxes_path = f"{directory}/{repeat}/boxes_npz/img_{marker_adjusting}_{condition}.npz"
                marker_boxes = load_boxes(marker_boxes_path)


                # Get coordinates 
                counter = 0
                for i, marker_box in enumerate(marker_boxes):
                    if i+1 in selected_boxes_ids:
                        
                        marker_box = img_as_float(marker_box)
                    
                        # Visualize binary mask ---------------------------------
                        fig, ax = plt.subplots(figsize=(8, 4))   

                        # Use ax[0] for the original image and annotations
                        ax.imshow(marker_box, cmap='gray')
                        ax.set_title(f"{marker_adjusting}_{num_bins}_{gastruloid_radius}")

                        if loading:
                            cx, cy = coordinates[counter][0]
                        else:
                            cy, cx = coordinates[counter]

                        ax.scatter(cx, cy, s=50, edgecolors='red', facecolors='none', linewidth=2, label='Centroid')

                        # make circles
                        for radii in radiis:
                            circle = patches.Circle((cx, cy), radius=radii, edgecolor="#FF93BC", facecolor='none', linewidth=1)
                            ax.add_patch(circle)

                        ax.axis('off')

                        output_dir = f'{directory}/{repeat}/adjusting/{condition}_{marker_adjusting}'
                        os.makedirs(output_dir, exist_ok=True)

                        filename = f'{i+1}.png'
                        filepath = os.path.join(output_dir, filename)
                        plt.savefig(filepath, bbox_inches='tight')
                        plt.close(fig)
                        #----------------------------------------------------------------
                        counter = counter + 1


                print(f"Finished adjusting for {marker_adjusting}")


            if adjusting:
                for marker in markers:
                    if marker == "DAPI":
                        adjust(marker, radiis)

            # if not loading stuff (as in raw identifying coordinates etc. no need to save this. )
            if not loading:
            
                # save coordinates
                coor_output_dir = f"{directory}/{repeat}/coordinates"
                os.makedirs(coor_output_dir, exist_ok=True)

                coordinates_path = f"{coor_output_dir}/{condition}.npz"
                np.savez(coordinates_path, coords=converted_coordinates)

                # save region (gastruloid area)
                regions_output_dir = f"{directory}/{repeat}/regions"
                os.makedirs(regions_output_dir, exist_ok=True)

                regions_path = f"{regions_output_dir}/{condition}.npz"
                np.savez(regions_path, regions=largest_region_list)
                
                # save binary mask 
                binarymasks_output_dir = f"{directory}/{repeat}/binarymasks"
                os.makedirs(binarymasks_output_dir, exist_ok=True)

                binarymasks_path = f"{binarymasks_output_dir}/{condition}.npz"
                
                print(binary_mask_list)
                # Flatten if nested

                if any(isinstance(i, list) for i in binary_mask_list):
                    binary_mask_list = list(itertools.chain.from_iterable(binary_mask_list))

                # Convert all to numpy arrays
                binary_mask_list = [np.array(mask) if not isinstance(mask, np.ndarray) else mask for mask in binary_mask_list]
                np.savez(binarymasks_path, *binary_mask_list)


            
