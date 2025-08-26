
import os                              
import csv                            
import numpy as np                     
import pandas as pd                   
import matplotlib.pyplot as plt        
from scipy.stats import gaussian_kde    
from scipy.interpolate import make_interp_spline  
from GastruloidKit.detection import load_boxes, load_allowed_ids

def get_intensities(idx, image, center, diameter):

    raw_radii = diameter/2

    h, w = image.shape
    cx, cy = center[0]

    yy, xx = np.ogrid[:h, :w]
    dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2

    mask = dist_sq <= raw_radii ** 2

    # Apply mask and compute mean intensity
    mean_intensity = image[mask].mean() if np.any(mask) else 0.0

    return mean_intensity


def get_all_intensities(img_boxes, coordinates, diameters):
    
    intensity_means = []

    for idx, (img, center, diameter) in enumerate(zip(img_boxes, coordinates, diameters)):
        mean_intensity = get_intensities(idx, img, center, diameter)
        intensity_means.append(mean_intensity)


    return intensity_means


def get_all_diameter(largest_region_list):
    
    diameters = []

    for idx, larg_region in enumerate(largest_region_list):
        diameter = larg_region.equivalent_diameter  # CALCULATE DIAMETER
        diameters.append(diameter)

    return diameters


def save_histograms(csv_path):
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    # Define the features and corresponding filenames
    features = ['intensity']
    
    for feature in features:
        plt.figure()
        plt.hist(df[feature], bins=20, edgecolor='black')
        plt.title(f'Histogram of {feature.capitalize()}')
        plt.xlabel(feature.capitalize())
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.tight_layout()
        
        # Save plot
        plot_path = csv_path.replace(".csv", "")

        plt.savefig(f'{plot_path}_{feature}_histogram.png')
        plt.close()


def plot_wt_mutant_overlap(marker, repeat, wt_csv, mutant_csv, save_dir='plots', features=None):
    os.makedirs(save_dir, exist_ok=True)

    # Load CSVs
    df_wt = pd.read_csv(wt_csv)
    df_mut = pd.read_csv(mutant_csv)

    condition_colors = {
        'WT': "#00da1d",
        'Mutant': "#ff320e"
    }

    for feature in features:
        data_wt = df_wt[feature].dropna().values
        data_mut = df_mut[feature].dropna().values

        # KDE estimation (smoother histogram)
        kde_wt = gaussian_kde(data_wt)
        kde_mut = gaussian_kde(data_mut)

        x_min = min(data_wt.min(), data_mut.min())
        x_max = max(data_wt.max(), data_mut.max())
        x_range = np.linspace(x_min, x_max, 1000)

        y_wt = kde_wt(x_range)
        y_mut = kde_mut(x_range)

        # Calculate AUCs
        auc_wt = np.trapz(y_wt, x_range)
        auc_mut = np.trapz(y_mut, x_range)
        y_overlap = np.minimum(y_wt, y_mut)
        auc_overlap = np.trapz(y_overlap, x_range)

        # Overlap % (normalized to average area)
        overlap_percentage = 100 * (2 * auc_overlap) / (auc_wt + auc_mut)

        # Plotting
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(x_range, y_wt, label='WT', color=condition_colors['WT'], linewidth=2)
        ax.plot(x_range, y_mut, label='Mutant', color=condition_colors['Mutant'], linewidth=2)
        ax.fill_between(x_range, y_overlap, color='orange', alpha=0.3, label='Overlap')

        ax.set_title(f"Repeat {repeat} {feature.capitalize()} Distribution (WT vs Mutant)")
        ax.set_xlabel(f"{feature.capitalize()}")
        ax.set_ylabel("Frequency (Density)")
        ax.legend()

        ax.text(0.95, 0.95, f"Overlap: {overlap_percentage:.2f}%",
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                fontsize=10,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

        plt.tight_layout()
        output_path = os.path.join(save_dir, f'{marker}_overlap_{feature}.png')
        plt.savefig(output_path)
        plt.close()

        print(f"Saved plot: {output_path}")
        print(f"AUC WT: {auc_wt:.3f}, AUC Mutant: {auc_mut:.3f}, Overlap AUC: {auc_overlap:.3f}, Overlap %: {overlap_percentage:.2f}%")


def normalize_by_dapi(marker_csv, dapi_csv, output_path):
    # Read both CSVs
    marker_df = pd.read_csv(marker_csv)
    dapi_df = pd.read_csv(dapi_csv)

    # Check that the Index columns match
    if not marker_df['Index'].equals(dapi_df['Index']):
        raise ValueError("Indexes do not match between marker and DAPI CSVs.")

    # Add a new column with normalized values
    marker_df['normalized_intensity'] = marker_df['intensity'] / dapi_df['intensity']

    # Save to the same marker CSV or to a new file
    marker_df.to_csv(output_path, index=False)
    print(f"Normalized data added and saved to: {output_path}")



def combined_plot_intensity_vs_DAPIintensity(
    repeats_data,  # List of tuples: (repeat_label, csv1_path, csv2_path, marker)
    marker_loc_dict,  # e.g., {'SOX2': 'inner', 'BRA': 'mid', 'GATA3': 'outer'}
):
    plt.figure(figsize=(10, 7))

    for directory, repeat, condition, csv1_path, csv2_path, marker in repeats_data:
        location = marker_loc_dict.get(marker)
        if location is None:
            print(f"Skipping marker {marker}: location not found in marker_loc_dict.")
            continue

        # Load CSVs
        df_intensity = pd.read_csv(csv1_path)
        df_diameter = pd.read_csv(csv2_path)
        df_intensity['aligned_index'] = df_intensity['ID'] - 1

        # Filter and merge
        df_marker = df_intensity[df_intensity['marker'] == marker]
        merged = df_marker.merge(df_diameter, left_on='aligned_index', right_on='Index')
        num_points = merged.shape[0]

        # Scatter plot for this group
        plt.scatter(
            merged['intensity'],
            merged[location],
            alpha=0.7,
            label=f'{condition}_{marker} (Repeat {repeat}, N={num_points})'
        )

    # Final plot settings
    plt.ylim(0, 1)
    plt.xlabel('DAPI Intensity')
    plt.ylabel('Marker Intensity (region-specific)')
    plt.title('Combined Marker Intensity vs DAPI Intensity')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    output_path = f'{directory}/{repeat}/plots/{marker}_{location}_DAPIIntensityDis.png'

    plt.savefig(output_path)
    plt.close()

    print(f"Combined plot saved to: {output_path}")


def get_distributions(directory, repeats, conditions, markers):

    for repeat in repeats:
        for condition in conditions:
            print(f"Starting Repeat: {repeat}, condiiton: {condition}")
            # load coordinates, regions and binary masks
            print("loading coordinates...")
            coor_output_dir = f"{directory}/{repeat}/coordinates"
            coordinates_path = f"{coor_output_dir}/{condition}.npz"
            print("Trying to load:", coordinates_path)
            coord_data = np.load(coordinates_path)
            coordinates = coord_data["coords"]

            print("loading regions...")
            regions_output_dir = f"{directory}/{repeat}/regions"
            regions_path = f"{regions_output_dir}/{condition}.npz"
            print("Trying to load:", regions_path)
            regions_data = np.load(regions_path, allow_pickle=True)
            regions = regions_data["regions"]
            
            print("loading binary masks...")
            binarymasks_output_dir = f"{directory}/{repeat}/binarymasks"
            binarymasks_path = f"{binarymasks_output_dir}/{condition}.npz"
            print("Trying to load:", binarymasks_path)
            binarymasks_data = np.load(binarymasks_path, allow_pickle=True)
            binary_masks = binarymasks_data["binarymasks"]

            # load selected IDs
            selection_output_dir = f"{directory}/{repeat}/selection"
            selection_csv = f"{selection_output_dir}/img_DAPI_{condition}.csv"
            selected_boxes_ids = load_allowed_ids(selection_csv)
            selected_boxes_ids.sort()
            
            def get_raw_distributions():
                for marker in markers:
                    print(f"Geting Raw Intensity Distribution for Marker: {marker}")

                    # SET SAVING DIRECTORIES 
                    intensity_means_output_path = f"{directory}/{repeat}/distribution/{condition}_{marker}.csv"
                    intensity_directory = os.path.dirname(intensity_means_output_path)
                    os.makedirs(intensity_directory, exist_ok=True)

                    # load boxes 
                    image_boxes_path = f"{directory}/{repeat}/boxes_npz/img_{marker}_{condition}.npz"
                    img_boxes = load_boxes(image_boxes_path)

                    # FILTER IMG_BOX to only contain selected ones
                    filtered_img_boxes = [img_box for i, img_box in enumerate(img_boxes) if i+1 in selected_boxes_ids]

                    # CALCULATE DIAMETER AND CIRCULARITY OF MODEL, USE TO GET INTENSITY
                    diameters = get_all_diameter(regions)
                    intensity_means = get_all_intensities(filtered_img_boxes, coordinates, diameters)

                    # SAVE TO CSV
                    with open(intensity_means_output_path, 'w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(['Index', 'intensity']) 
                        for i in range(len(intensity_means)):
                            writer.writerow([i+1, intensity_means[i]])

                    # PLOT DATA ON CSV AND SAVE 
                    save_histograms(intensity_means_output_path) 

            def normalize_distributions():
                for marker in markers:                    
                    if marker != "DAPI":
                        print(f"Normalizing data for Marker: {marker}")
                        intensity_means_output_path = f"{directory}/{repeat}/distribution/{condition}_{marker}.csv"

                        # normalize intensity values by DAPI 
                        DAPI_intensity_path = f"{directory}/{repeat}/distribution/{condition}_DAPI.csv"
                        normalize_by_dapi(intensity_means_output_path, DAPI_intensity_path, intensity_means_output_path)

            get_raw_distributions()
            print(f"Finished getting raw distributions for Repeat: {repeat}, condiiton: {condition}")
            normalize_distributions()
            print(f"Finished Normalizing distributions for Repeat: {repeat}, condiiton: {condition}")


        def overlapdensity_plot():
                
            # whats to be plotted is different depending on marker 
            DAPI_features = ['intensity']
            marker_features = ['normalized_intensity']

            for marker in markers:
                print(f"Plotting data for Marker: {marker}")
                wt_csv = f"{directory}/{repeat}/distribution/{conditions[0]}_{marker}.csv"
                mutant_csv = f"{directory}/{repeat}/distribution/{conditions[1]}_{marker}.csv"
                if marker == "DAPI":
                    plot_wt_mutant_overlap(marker, repeat, wt_csv, mutant_csv, save_dir=f'{directory}/{repeat}/plots/overlapdensity', features=DAPI_features)
                else:
                    plot_wt_mutant_overlap(marker, repeat, wt_csv, mutant_csv, save_dir=f'{directory}/{repeat}/plots/overlapdensity', features=marker_features)

        overlapdensity_plot()

def get_DAPIcenter_distributions(directory, repeats, conditions, num_bins):
    for repeat in repeats:
        for condition in conditions:

            path = f"{directory}/{repeat}/intensities/{num_bins}_DAPI_{condition}.npy"
            data = np.load(path)

            x = np.arange(1, 16)  # bins 1..15
            x_smooth = np.linspace(1, 15, 200)  # smooth curve along bins
            plt.figure(figsize=(10, 6))


            results = []  # store ID + furthest_bin
            threshold = 0.8


            for i in range(data.shape[0]):  # 252 rows
                y = data[i, :]

                # normalize to [0,1]
                y_norm = (y - y.min()) / (y.max() - y.min())

                # spline interpolation
                spline = make_interp_spline(x, y_norm, k=2)
                y_smooth = spline(x_smooth)

                # plot curve
                plt.plot(x_smooth, y_smooth, color="#ff6fb2", alpha=0.3, linewidth=1)

                # find indices where y_smooth > threshold
                below = np.where(y_smooth > threshold)[0]

                if len(below) > 0:
                    furthest_x = x_smooth[below[-1]]
                    results.append([i+1, furthest_x])   # store ID and value
                else:
                    results.append([i+1, None])         # None if never drops below

            plt.title(f"Normalized DAPI Intensity Profiles of {len(data)} Gastruloids, Repeat {repeat}, Condition {condition}")
            plt.xlabel("Bin")
            plt.ylabel("Normalized DAPI Intensity")
            plt.ylim(0, 1.08)

            out_dir = f"{directory}/{repeat}/plots/DAPI_profile"
            out_name = f"{condition}_all.png"
            out_path = os.path.join(out_dir, out_name)

            os.makedirs(out_dir, exist_ok=True)
            plt.savefig(out_path)
            plt.close()

            # save to CSV
            df = pd.DataFrame(results, columns=["ID", "furthest_bin"])

            csv_out_dir = f"{directory}/{repeat}/DAPI_profile"
            os.makedirs(csv_out_dir, exist_ok=True)
            csv_out_name = f"{condition}_drop.csv"
            csv_out_path = os.path.join(csv_out_dir, csv_out_name)
            df.to_csv(csv_out_path, index=False)       

            # Drop rows with missing/None values
            df = df.dropna(subset=["furthest_bin"])

            # Convert to float just in case
            df["furthest_bin"] = df["furthest_bin"].astype(float)

            # Define common bins (edges)
            bin_edges = np.linspace(1, 15, 15)  # 14 bins between 1 and 15

            plt.figure(figsize=(8,5))
            plt.hist(
                df["furthest_bin"],
                bins=bin_edges,  # use fixed edges
                color="#ff6fb2",
                weights=np.ones(len(df)) / len(df) * 100,  # percentage
                edgecolor="black",
                alpha=0.7
            )
            plt.title(f"Distribution of Furthest Bin Below Threshold, Repeat {repeat}, Condition: {condition}")
            plt.xlabel("Furthest Bin")
            plt.xlim(1, 15)
            plt.ylabel("Percentage of Gastruloids (%)")

            dropdis_out_dir = f"{directory}/{repeat}/plots/DAPI_profile"
            os.makedirs(dropdis_out_dir, exist_ok=True)
            dropdis_out_name = f"{condition}_dropdistribution.png"
            dropdis_out_path = os.path.join(dropdis_out_dir, dropdis_out_name)

            plt.savefig(dropdis_out_path)
            plt.close()

              


                




