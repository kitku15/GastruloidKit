
import os
import csv
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import make_interp_spline
from csbdeep.utils import normalize
from GastruloidKit.bins import get_bin_radii
from GastruloidKit.detection import load_allowed_ids, load_boxes
from GastruloidKit.profiling_utils import get_avg_df, make_profile_plot, plot_radial_bin_heatmap


def normalize_tiff(tiff):
    '''
    Normalizes a TIFF image based on percentile scaling to enhance contrast.

    Parameters:
        tiff (numpy.ndarray): The input TIFF image as a NumPy array, typically 2D or 3D.

    Returns:
        numpy.ndarray: The normalized image with intensity values scaled between 0 and 1 
        using the 1st and 99.8th percentiles.

    Notes:
        - This is useful for preparing images for intensity-based measurements or visualization.
        - Percentile normalization reduces the impact of extreme outlier values.
        - Normalization is applied along the spatial axes (axis=(0, 1)).
    '''

    img_norm = normalize(tiff, pmin=1, pmax=99.8, axis=(0, 1))


    return img_norm



def measure_blob_intensity_zones(idx, marker, image, mask, center, radiis, GATA3_mask=None, print_info=False):
    """
    Measures intensity in three non-overlapping radial zones: inner, mid ring, outer ring.

    Parameters:
        marker (str): Name of the channel/marker (e.g., "DAPI", "MarkerX").
        image (2D ndarray): Grayscale cropped image.
        mask (2D ndarray): Binary mask (same size as image), where 1 indicates valid signal.
        center (tuple): (x, y) center of the circles.
        inner_r (float): Radius of inner circle.
        mid_r (float): Radius of middle circle.
        outer_r (float): Radius of outer circle.
        print_info (bool): If True, prints debug info.

    Returns:
        tuple: (inner_mean, mid_bin_mean, outer_bin_mean)
    """

    h, w = image.shape
    cx, cy = center[0]

    yy, xx = np.ogrid[:h, :w]
    dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2

    signal_means = [] # the length of this list will be the length of radiis (num_bins)

    for i, radii in enumerate(radiis):

        # MAKE BIN MASK
        if i == 0:
            # special case for innermost bin (circle not donut)
            bin_mask = dist_sq <= radii**2

        else:
            # for all other cases we need to also make a mask of the smaller bin (donut)
            current_radii = radii
            inner_radii = radiis[i-1]

            inner_bin_mask = dist_sq <= inner_radii**2
            bin_mask = (dist_sq <= current_radii**2) & (~inner_bin_mask)

        # count pixels in area selected
        bin_area = np.count_nonzero(bin_mask)


        # if marker is not DAPI, apply binary intensity mask (made from ImageJ)
        if marker != "DAPI":
            if image.shape != mask.shape:
                raise ValueError(f"Image and mask must have the same shape, got {image.shape} and {mask.shape}")
            
            

            # Apply signal mask: only count values where signal exists
            if marker == "GATA3":
                # add special mask for GATA3 only
                signal = image * ((mask > 0) & bin_mask & (GATA3_mask > 0))
            else:
                signal = image * ((mask > 0) & bin_mask)


            # Sum signal only (not mean over signal area)
            signal_sum = np.sum(signal)

            # Normalize to total area of radial bin
            signal_mean = signal_sum / bin_area if bin_area > 0 else 0
        else:
            # this is for DAPI: no masking — use raw pixel intensities
            signal_mean = np.sum(image[bin_mask]) / bin_area if bin_area > 0 else 0

        # add to signal means list
        signal_means.append(signal_mean)

    return signal_means




def measure_all_blob_intensities_zones(marker, img_boxes, mask_boxes, coordinates, radiis, GATA3_masks):
    """
    Measures intensities for all blobs using constant inner, mid, and outer radii.

    Parameters:
        img_boxes (list of 2D ndarrays): Cropped grayscale images of blobs.
        coordinates (list of arrays): Each is a (1, 2) array for the blob center.
        inner_radius (float): Radius for the inner zone.
        mid_radius (float): Radius for the mid zone.
        outer_radius (float): Radius for the outer zone.

    Returns:
        tuple of 3 lists: (inner_means, mid_means, outer_means)
    """
    bin_means = []

    if GATA3_masks == None:
        for idx, (img, mask, center) in enumerate(zip(img_boxes, mask_boxes, coordinates)):
            means = measure_blob_intensity_zones(idx, marker, img, mask, center, radiis)
            bin_means.append(means)
    else:
        for idx, (img, mask, center, GATA3_mask) in enumerate(zip(img_boxes, mask_boxes, coordinates, GATA3_masks)):
            means = measure_blob_intensity_zones(idx, marker, img, mask, center, radiis, GATA3_mask)
            bin_means.append(means)



    return bin_means



def get_radius(csv_path, current_repeat, current_condition):

    df = pd.read_csv(csv_path)
    filtered_row = df[(df['repeat'] == current_repeat) & (df['condition'] == current_condition)]

    if not filtered_row.empty:
        outer_r = float(filtered_row['outer_r'].values[0])
        mid_r = float(filtered_row['mid_r'].values[0])
        inner_r = float(filtered_row['inner_r'].values[0])
        return outer_r, mid_r, inner_r

    raise ValueError(f"No radius data found for repeat={current_repeat}, condition={current_condition}")



def get_rawintensities(directory, repeats, conditions, markers, gastruloid_radius, num_bins):


    print("Numbers of bins set:", num_bins)
    radiis = get_bin_radii(num_bins, gastruloid_radius)
    print("Radius list:", radiis)

    for repeat in repeats:
        for condition in conditions:
            # Load coordinates 
            coor_output_dir = f"{directory}/{repeat}/coordinates"
            coordinates_path = f"{coor_output_dir}/{condition}.npz"

            data = np.load(coordinates_path)
            coordinates = data["coords"]

            for marker in markers:
                print(f"------------Repeat: {repeat}, Condition: {condition}, marker: {marker}")
                    
                # get selected ids, masks and boxes:
                selected_boxes_ids = load_allowed_ids(f'{directory}/{repeat}/selection/img_DAPI_{condition}.csv')
                selected_boxes_ids.sort()

                mask_boxes_path = f"{directory}/{repeat}/boxes_npz/mask_{marker}_{condition}.npz"
                image_boxes_path = f"{directory}/{repeat}/boxes_npz/img_{marker}_{condition}.npz"
                    
                img_boxes = load_boxes(image_boxes_path)
                mask_boxes = load_boxes(mask_boxes_path)

                # FILTER IMG_BOX to only contain selected ones
                filtered_img_boxes = [img_box for i, img_box in enumerate(img_boxes) if i+1 in selected_boxes_ids]
                filtered_mask_boxes = [mask_box for i, mask_box in enumerate(mask_boxes) if i+1 in selected_boxes_ids]

                # GET GATA 3 MASKS
                if "GATA3" in markers:
                    loaded_GATA3masks = np.load(f"{directory}/{repeat}/GATA3filter/masks/{condition}.npz")
                    GATA3_masks = [loaded_GATA3masks[key] for key in loaded_GATA3masks]
                else:
                    GATA3_masks = None


                # function to measure intensity and save 
                def measure_intensity_and_save(radiis):

                    # get intensities 
                    bin_means = measure_all_blob_intensities_zones(marker, filtered_img_boxes, filtered_mask_boxes, coordinates, radiis, GATA3_masks)

                    # Save
                    bin_means_dir = f"{directory}/{repeat}/intensities"
                    os.makedirs(bin_means_dir, exist_ok=True)

                    bin_means_output_path = f"{bin_means_dir}/{num_bins}_{marker}_{condition}"
                    np.save(bin_means_output_path, bin_means)

                    print("saved to:", bin_means_output_path)

                # run function above
                measure_intensity_and_save(radiis)
            

def meta_intensities_save(directory, repeat, condition, marker, bins, num_bins):
    """
    Saves metadata and bin intensity values to meta_intensities.csv.

    Parameters:
        directory (str): Folder to save the CSV.
        repeat (int/str): Repeat number or label.
        condition (str): Experimental condition.
        marker (str): Marker name.
        bins (list): List of bin values (length can vary).
    """

    # Create row: metadata + bin values
    new_row = [repeat, condition, marker] + bins

    csv_file = os.path.join(directory, f'{num_bins}_meta_intensities.csv')

    # Check if file exists
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)

        # Write header only if file doesn't exist
        if not file_exists:
            header = ['repeat', 'condition', 'marker'] + [f'bin_{i+1}' for i in range(len(bins))]
            writer.writerow(header)

        writer.writerow(new_row)


def meta_intensities_save_individual(directory, ID, repeat, condition, marker, bins, num_bins):
    """
    Saves individual metadata and bin intensity values to meta_individual CSV.

    Parameters:
        directory (str): Base folder to save the CSV.
        ID (str/int): Identifier for the individual.
        repeat (str/int): Repeat number or label (used for folder path).
        condition (str): Experimental condition.
        marker (str): Marker name.
        bins (list): List of bin values (any length).
    """

    # Create row: ID + marker + bins
    new_row = [ID, marker] + bins

    # Ensure the folder exists
    save_dir = os.path.join(directory, str(repeat), 'intensities')
    os.makedirs(save_dir, exist_ok=True)

    csv_file = os.path.join(save_dir, f'{num_bins}_meta_individual_{condition}.csv')

    # Check if file exists
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)

        # Write header only if file doesn't exist
        if not file_exists:
            header = ['ID', 'marker'] + [f'bin_{i+1}' for i in range(len(bins))]
            writer.writerow(header)

        writer.writerow(new_row)


def normalize_intensities(directory, repeats, conditions, markers, ref_marker, num_bins):

    for repeat in repeats:
        for condition in conditions:
            for marker in markers:
                if marker != ref_marker: 

                    # load bin means 
                    # load intensities 
                    print(f"loading intensities for {marker} and {ref_marker}")

                    bin_means_path = f"{directory}/{repeat}/intensities/{num_bins}_{marker}_{condition}.npy"
                    DAPI_bin_means_path = f"{directory}/{repeat}/intensities/{num_bins}_{ref_marker}_{condition}.npy"

                    bin_means = np.load(bin_means_path, allow_pickle=True)
                    DAPI_bin_means = np.load(DAPI_bin_means_path, allow_pickle=True)

                    normalized_bin_means = [] # average normalized values of all gastruloids in all bins (length is num of bins) [x, x]
                    avg_normalized_bin_means = [] # normalized values of all gastruloids in all bins (length is num of gastruloids)  [[list length of 7], [list length of 7]] 
                    meta_ready_means = []
                    # normalize per bin 
                    for i, bin_mean in enumerate(bin_means):

                        # length of bin mean is length of selected box ids
                        DAPI_bin_mean = DAPI_bin_means[i] #  values of DAPI gastruloid i in the all bins

                        normalized_bin_mean = [] # normalized values of gastruloid i in the all bins (length 7)

                        for j in range(len(bin_mean)): # len of bin mean is 7

                            m = bin_mean[j]
                            d = DAPI_bin_mean[j]

                            n = m/d

                            # add the normalized value of one gastruloid in one bin to the normalized_bin_mean list
                            normalized_bin_mean.append(n)
                        
                        # append to big list for individual 
                        normalized_bin_means.append(normalized_bin_mean) # append the list of 7
                        # print(f"Got values for gastruloid {i}:, {normalized_bin_mean}")

                    
                    averaged_list = [sum(x)/len(x) for x in zip(*normalized_bin_means)]
                    # save to main meta 
                    print(f"saving to meta csv for {marker} for {len(averaged_list)} bins")
                    print("should be bin len", len(averaged_list))
                    
                    meta_intensities_save(directory, repeat, condition, marker, averaged_list, num_bins)

                    # save to individual meta 
                    for i in range(1, len(normalized_bin_means)+1):
                        print("should be bin len", len(normalized_bin_means[i-1]))
                        
                        meta_intensities_save_individual(directory, i, repeat, condition, marker, normalized_bin_means[i-1], num_bins)

def plot_gastruloidprofiles(directory, repeats, conditions, markers, ref_marker, num_bins, marker_colors):

    meta_path = f'{directory}/{num_bins}_meta_intensities.csv'

    for repeat in repeats:
        save_dir = f'{directory}/{repeat}/plots/gastruloid_profiles'
        bins_plot(meta_path, repeat, marker_colors, plot_type='line', save_dir=save_dir)
        for condition in conditions:
            plot_radial_bin_heatmap(meta_path, repeat, condition, marker_colors, save_dir)


    for repeat in repeats:
        save_dir = f'{directory}/{repeat}/plots/gastruloid_profiles'
        for marker in markers:
            if marker != ref_marker:
                plot_marker_condition_overlap(meta_path, repeat, marker, save_dir=save_dir)
                
def white_to_color(color_name):
                return LinearSegmentedColormap.from_list("", ["#FFFFFF00", color_name])

def bins_plot(csv_path, repeat, marker_colors, plot_type='line', save_dir='plots'):


    df = pd.read_csv(csv_path)

    # Remove DAPI
    df = df[df['marker'].str.upper() != 'DAPI']

    # Focus on specified repeat
    df = df[df['repeat'] == repeat]


    # Unique (repeat, condition) pairs
    groups = df.groupby(['repeat', 'condition'])

    # Dynamically detect regions (columns starting with 'bin_')
    regions = [col for col in df.columns if col.startswith('bin_')]
    x = np.arange(len(regions))

    # Define marker colors
    marker_colors = {
        'SOX2': '#00bcd4',
        'BRA': '#ffeb3b',
        'GATA3': '#9c27b0',
        'psmad159': "#1013b1"
    }

    os.makedirs(save_dir, exist_ok=True)

    for (repeat, condition), group in groups:
        fig, ax = plt.subplots(figsize=(6,5))

        markers = group['marker'].tolist()
        n_markers = len(markers)
        bar_width = 0.25  # width of bars

        # For line plot smoothing
        x_smooth = np.linspace(x.min(), x.max(), 300)

        for i, (_, row) in enumerate(group.iterrows()):
            marker = row['marker']
            color = marker_colors.get(marker, 'gray')
            y = row[regions].values.astype(float)
            y_norm = (y - y.min()) / (y.max() - y.min())
            

            if plot_type == 'line':
                spline = make_interp_spline(x, y_norm, k=2)
                y_smooth = spline(x_smooth)
                ax.plot(x_smooth, y_smooth, label=marker, color=color, linewidth=2)
                ax.scatter(x, y_norm, color=color, edgecolor='k', zorder=5)

            elif plot_type == 'bar':
                # Shift bars by i * bar_width
                ax.bar(x + i * bar_width, y, width=bar_width, label=marker, color=color, edgecolor='black')

        if plot_type == 'bar':
            # Center x-ticks under group of bars
            ax.set_xticks(x + bar_width * (n_markers - 1) / 2)
        else:
            ax.set_xticks(x)

        # Use the detected regions as x-tick labels
        ax.set_xticklabels([str(i+1) for i in range(len(regions))])
        ax.set_ylabel('Average Normalized Intensity')
        ax.set_xlabel('Bin')
        ax.set_title(f'Marker Intensities - Repeat {repeat}, Condition {condition}')
        ax.legend(title='Marker')
        plt.tight_layout()
        plt.ylim(0, 1.1)

        output_path = os.path.join(save_dir, f'intensities_plot_{condition}.png')
        plt.savefig(output_path)
        plt.close()
        print(f"Saved: {output_path}")


def DAPIintensity_split_profiles(directory, repeats, conditions, num_bins, marker_colors):
    """
        Analyze marker profiles by splitting gastruloids into DAPI intensity bins.

        Parameters
        ----------
        directory : str
            Base directory containing input CSV files.
        repeats : list of str
            Biological repeat identifiers.
        conditions : list of str
            Experimental condition identifiers.
        num_bins : int
            Number of bins in the meta_individual CSVs.
        marker_colors : dict, optional
            Dictionary mapping markers to hex/RGB colors.
    """
    for repeat in repeats:
        for condition in conditions:
            save_dir = f"{directory}/{repeat}/plots/DAPIintensity_profiles"

            meta_indiv_path = f"{directory}/{repeat}/intensities/{num_bins}_meta_individual_{condition}.csv"
            meta_indiv_df = pd.read_csv(meta_indiv_path)

            dapi_path = f"{directory}/{repeat}/distribution/{condition}_DAPI.csv"
            dapi_df = pd.read_csv(dapi_path)

            # Create equal bins
            bin_edges = np.linspace(dapi_df["intensity"].min(), dapi_df["intensity"].max(), 6)
            bin_categories = pd.cut(dapi_df["intensity"], bins=bin_edges, include_lowest=True)

            print(f"\nDAPI intensity ranges for repeat {repeat}, condition {condition}:")
            for i, interval in enumerate(bin_categories.cat.categories, start=1):
                print(f"  Bin {i}: {interval.left:.2f} - {interval.right:.2f}")

            dapi_df["intensity_bin"] = pd.cut(
                dapi_df["intensity"], bins=bin_edges, labels=[1,2,3,4,5], include_lowest=True
            )

            for bin_idx in [1,2,3,4,5]:
                print(dapi_df.columns)
                print(meta_indiv_df.columns)
                avg_df, n_points = get_avg_df(dapi_df, "intensity_bin", [bin_idx], meta_indiv_df, id_col="ID")
                make_profile_plot(avg_df, bin_idx, n_points, bin_categories, repeat, condition,
                                  save_dir, bin_label="DAPI", marker_colors=marker_colors)
                plot_radial_bin_heatmap(avg_df, bin_idx, repeat, condition, save_dir, "DAPI", marker_colors)

def DAPIcenter_split_profiles(directory, repeats, conditions, num_bins, marker_colors):
    """
        Analyze marker profiles by splitting gastruloids into bins based on DAPI center size (furthest_bin).

        Parameters
        ----------
        directory : str
            Base directory containing input CSV files.
        repeats : list of str
            Biological repeat identifiers.
        conditions : list of str
            Experimental condition identifiers.
        num_bins : int
            Number of bins in the meta_individual CSVs.
        marker_colors : dict
            Dictionary mapping markers to hex/RGB colors.
    """
    for repeat in repeats:
        for condition in conditions:
            save_dir = f"{directory}/{repeat}/plots/DAPIcenter_profiles"

            meta_indiv_path = f"{directory}/{repeat}/intensities/{num_bins}_meta_individual_{condition}.csv"
            meta_indiv_df = pd.read_csv(meta_indiv_path)

            furthest_path = f"{directory}/{repeat}/DAPI_profile/{condition}_drop.csv"
            furthest_df = pd.read_csv(furthest_path)

            bin_edges = np.linspace(furthest_df["furthest_bin"].min(), furthest_df["furthest_bin"].max(), 6)
            bin_categories = pd.cut(furthest_df["furthest_bin"], bins=bin_edges, include_lowest=True)

            print(f"\nFurthest_bin ranges for repeat {repeat}, condition {condition}:")
            for i, interval in enumerate(bin_categories.cat.categories, start=1):
                print(f"  Bin {i}: {interval.left:.2f} - {interval.right:.2f}")

            furthest_df["furthest_bin_cat"] = pd.cut(
                furthest_df["furthest_bin"], bins=bin_edges, labels=[1,2,3,4,5], include_lowest=True
            )

            for bin_idx in [1,2,3,4,5]:
                avg_df, n_points = get_avg_df(furthest_df, "furthest_bin_cat", [bin_idx], meta_indiv_df)
                make_profile_plot(avg_df, bin_idx, n_points, bin_categories, repeat, condition,
                                  save_dir, bin_label="Furthest_bin", marker_colors=marker_colors)
                plot_radial_bin_heatmap(avg_df, bin_idx, repeat, condition, save_dir, "Furthest_bin", marker_colors)



def plot_marker_condition_overlap(csv_path, repeat, marker, save_dir='plots'):
    df = pd.read_csv(csv_path)

    # Remove DAPI
    df = df[df['marker'].str.upper() != 'DAPI']
    
    # Focus only on specified marker
    df = df[df['marker'].str.upper() == marker.upper()]

    # Focus on specified repeat
    df = df[df['repeat'] == repeat]
    
    if df.empty or len(df['condition'].unique()) < 2:
        print(f"Not enough data for marker '{marker}' with both conditions.")
        return

    # Dynamically detect regions (columns starting with 'bin_')
    regions = [col for col in df.columns if col.startswith('bin_')]
    x = np.arange(len(regions))
    x_smooth = np.linspace(x.min(), x.max(), 300)

    # Set colors
    condition_colors = {
        'WT': "#00da1d",
        'ND6': "#ff320e"
    }

    os.makedirs(save_dir, exist_ok=True)

    # Get average intensity per region for each condition
    avg_df = df.groupby('condition')[regions].mean()

    fig, ax = plt.subplots(figsize=(6,5))

    curves = {}
    for condition in ['WT', 'ND6']:
        if condition not in avg_df.index:
            continue

        y = avg_df.loc[condition].values.astype(float)
        y_norm = (y - y.min()) / (y.max() - y.min())
        spline = make_interp_spline(x, y_norm, k=2)
        y_smooth = spline(x_smooth)
        curves[condition] = y_smooth

        ax.plot(x_smooth, y_smooth, label=condition, color=condition_colors[condition], linewidth=2)
        ax.scatter(x, y_norm, color=condition_colors[condition], edgecolor='k', zorder=5)

    # Highlight overlap region
    y_min = np.minimum(curves['WT'], curves['ND6'])
    y_max = np.maximum(curves['WT'], curves['ND6'])

    ax.fill_between(x_smooth, y_min, where=(curves['WT'] > 0) & (curves['ND6'] > 0), 
                    interpolate=True, color="#ff8f0e", alpha=0.3, label='Overlap')

    # AUCs
    auc_wt = np.trapz(curves['WT'], x_smooth)
    auc_nd6 = np.trapz(curves['ND6'], x_smooth)
    auc_overlap = np.trapz(y_min, x_smooth)
    total_auc = auc_wt + auc_nd6

    overlap_percentage = 100 * (2 * auc_overlap) / total_auc
    overlap_text = f"Overlap: {overlap_percentage:.2f}%"

    # Add text to plot
    ax.text(0.95, 0.05, overlap_text,
            horizontalalignment='right',
            verticalalignment='bottom',
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

    ax.set_xticks(x)
    # Optional: make labels prettier, e.g., 'Bin 1', 'Bin 2', ...
    ax.set_xticklabels([str(i+1) for i in range(len(regions))])
    ax.set_ylabel('Average Normalized Intensity')
    ax.set_xlabel('Bin')
    ax.set_title(f'{marker.upper()} Intensities (WT vs ND6)')
    ax.legend()
    plt.tight_layout()
    plt.ylim(0, 1.2)

    output_path = os.path.join(save_dir, f'overlap_plot_{marker.upper()}.png')
    plt.savefig(output_path)
    plt.close()

    print(f"Saved plot: {output_path}")
    print(f"AUC WT: {auc_wt:.3f}")
    print(f"AUC ND6: {auc_nd6:.3f}")
    print(f"Overlap AUC: {auc_overlap:.3f}")
    print(f"Overlap %: {overlap_percentage:.2f}%")


def variance_of_laplacian(image):
    """Compute the Laplacian of the image and return the variance."""
    return cv2.Laplacian(image, cv2.CV_64F).var()

def detect_blurriness_in_parts(image, grid_size, blur_threshold):

    # Ensure grayscale
    if len(image.shape) == 2:
        gray = image
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    
    h, w = gray.shape
    num_rows, num_cols = grid_size
    cell_h, cell_w = h // num_rows, w // num_cols
    mask = np.zeros((h, w), dtype=np.uint8)

    for row in range(num_rows):
        for col in range(num_cols):
            y1, y2 = row * cell_h, (row + 1) * cell_h
            x1, x2 = col * cell_w, (col + 1) * cell_w
            cell = gray[y1:y2, x1:x2]

            score = variance_of_laplacian(cell)
            is_sharp = score >= blur_threshold
            if is_sharp:
                mask[y1:y2, x1:x2] = 1  # sharp = 1

    # Convert mask to 3-channel so we can stack it with original image
    mask_rgb = (mask * 255).astype(np.uint8)  # 0 or 255
    mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_GRAY2BGR)

    # Normalize original grayscale image
    image_bgr_float = image_bgr.astype(np.float32)
    image_bgr_float /= image_bgr_float.max()

    # Convert mask to float 0–1
    mask_rgb_float = mask_rgb.astype(np.float32) / 255.0

    # Stack them
    combined_float = np.hstack((image_bgr_float, mask_rgb_float))

    return combined_float, mask


def make_GATA3_filter(directory, repeats, conditions):
    marker = "GATA3"
    for repeat in repeats:
        for condition in conditions:
            print(f"------------Repeat: {repeat}, Condition: {condition}, marker: {marker}")

            # output directort for images (to visualize the mask)
            img_output_dir = f"{directory}/{repeat}/GATA3filter/images_{condition}"
            os.makedirs(img_output_dir, exist_ok=True)

            # get selected boxes of GATA3
            selected_boxes_ids = load_allowed_ids(f'{directory}/{repeat}/selection/img_DAPI_{condition}.csv')
            selected_boxes_ids.sort()

            image_boxes_path = f"{directory}/{repeat}/boxes_npz/img_{marker}_{condition}.npz"  
            img_boxes = load_boxes(image_boxes_path)
            filtered_img_boxes_with_index = [(i, img_box) for i, img_box in enumerate(img_boxes) if i+1 in selected_boxes_ids]

            masks = []
            all_box_bluriness = []
            for i, image in filtered_img_boxes_with_index:
                # calculate the total blurriness of the image
                box_blurriness = variance_of_laplacian(image)
                all_box_bluriness.append(box_blurriness)

            bluriness_average = sum(all_box_bluriness) / len(all_box_bluriness)
            blur_threshold = bluriness_average * 0.75

            for i, image in filtered_img_boxes_with_index:
                # detect blurriness and make mask
                grid_size = (30, 30)
                output_image, mask = detect_blurriness_in_parts(image, grid_size, blur_threshold)

                # save image (for visual validation side by side with image)
                savepath = os.path.join(img_output_dir, f"{i+1}.png")
                plt.imsave(savepath, output_image, cmap='gray')

                masks.append(mask)


            # save GATA3 masks
            mask_output_dir =  f"{directory}/{repeat}/GATA3filter/masks"
            os.makedirs(mask_output_dir, exist_ok=True)

            masks_path = f"{mask_output_dir}/{condition}.npz"
            np.savez_compressed(masks_path, *masks)
            print(f"Saved GATA3 filter mask for Repeat: {repeat}, Condition: {condition}")

            
            print(f"Bluriness for this set is {bluriness_average}")