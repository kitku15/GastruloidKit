import pandas as pd
import numpy as np
import os
from scipy.interpolate import make_interp_spline
from GastruloidKit.src.GastruloidPy.gastruloid_detection import load_allowed_ids, load_boxes
from PIL import Image
import matplotlib.pyplot as plt
from f_visualizechannels import overlay_channels


def score_gastruloid_similarity(directory, repeat, marker, condition, num_bins):

    meta_individual = f"{directory}/{repeat}/intensities/{num_bins}_meta_individual_{condition}.csv"
    meta_intensities = f'{directory}/{num_bins}_meta_intensities.csv'

    output_csv_path=f'{directory}/{repeat}/{condition}_overlap_scores_{marker}_{num_bins}bins.csv'

    # Load data
    interest_df = pd.read_csv(meta_individual) # contains intensity measurements for each individual gastruloid
    wt_df = pd.read_csv(meta_intensities) # contains AVERAGE intensity measurements for each repeat and condition 

    # Preprocess WT AVERAGE intensity data
    wt_df = wt_df[wt_df['marker'].str.upper() != 'DAPI']
    wt_df = wt_df[wt_df['marker'].str.upper() == marker.upper()]
    wt_df = wt_df[wt_df['repeat'] == repeat]
    wt_df = wt_df[wt_df['condition'] == "WT"]

    if wt_df.empty:
        print("No WT data found for given marker/repeat.")
        return

    # Preprocess gastruloid of interest individual data (called interest bcs it could be ND6 or WT)
    interest_df = interest_df[interest_df['marker'].str.upper() == marker.upper()]
    if interest_df.empty:
        print("No ND6 individual data found for given marker.")
        return

    # Get all bin columns dynamically (those starting with 'bin_')
    bin_cols = [col for col in wt_df.columns if col.startswith('bin_')]

    # Regions (x positions) will just be the bin indices
    x = np.arange(len(bin_cols))
    x_smooth = np.linspace(x.min(), x.max(), 300)

    # Compute WT average curve
    wt_means = wt_df[bin_cols].mean().values.astype(float)
    spline_wt = make_interp_spline(x, wt_means, k=2)
    wt_smooth = spline_wt(x_smooth)

    # Compute scores for each ND6 gastruloid
    results = []
    for _, row in interest_df.iterrows():
        nd6_vals = row[bin_cols].values.astype(float)
        spline_nd6 = make_interp_spline(x, nd6_vals, k=2)
        nd6_smooth = spline_nd6(x_smooth)

        # Overlap calculation (same as plot logic)
        y_min = np.minimum(wt_smooth, nd6_smooth)
        auc_wt = np.trapz(wt_smooth, x_smooth)
        auc_nd6 = np.trapz(nd6_smooth, x_smooth)
        auc_overlap = np.trapz(y_min, x_smooth)
        total_auc = auc_wt + auc_nd6

        overlap_percentage = 100 * (2 * auc_overlap) / total_auc

        results.append({
            'ID': row['ID'],
            'marker': row['marker'].upper(),
            'overlap_score': overlap_percentage
        })

    # Save to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv_path, index=False)
    print(f"Saved scores to {output_csv_path}")
    return results_df, output_csv_path


def sort_overlap_scores(input_csv, output_csv=None):
    # Read CSV
    df = pd.read_csv(input_csv)
    
    # Sort by overlap_score descending
    df_sorted = df.sort_values(by='overlap_score', ascending=False)
    
    # Save or return
    if output_csv:
        df_sorted.to_csv(output_csv, index=False)
        print(f"Sorted CSV saved to {output_csv}")
    return df_sorted

def score_gastruloid(directory, repeats, markers, conditions, num_bins):
    for repeat in repeats:
        for marker in markers:
            for condition in conditions:
                if marker != "DAPI": # filter out dapi
                    results, output_csv_path = score_gastruloid_similarity(directory, repeat, marker, condition, num_bins)
                    results_sorted = results.sort_values(by='overlap_score', ascending=False)
                    results_sorted.to_csv(output_csv_path, index=False)

def final_score_gastruloid(directory, repeats, conditions, num_bins):

    for repeat in repeats:
        for condition in conditions:
            # Paths to your CSVs

            gata3_csv = f"{directory}/{repeat}/{condition}_overlap_scores_GATA3_{num_bins}bins.csv"
            sox2_csv = f"{directory}/{repeat}//{condition}_overlap_scores_SOX2_{num_bins}bins.csv"
            bra_csv = f"{directory}/{repeat}//{condition}_overlap_scores_BRA_{num_bins}bins.csv"

            # Load and rename overlap_score columns
            gata3_df = pd.read_csv(gata3_csv)[['ID', 'overlap_score']].rename(columns={'overlap_score': 'GATA3_score'})
            sox2_df = pd.read_csv(sox2_csv)[['ID', 'overlap_score']].rename(columns={'overlap_score': 'SOX2_score'})
            bra_df   = pd.read_csv(bra_csv)[['ID', 'overlap_score']].rename(columns={'overlap_score': 'BRA_score'})

            # Merge all on ID
            merged_df = gata3_df.merge(sox2_df, on='ID').merge(bra_df, on='ID')

            # Calculate final average score
            merged_df['final_score'] = merged_df[['GATA3_score', 'SOX2_score', 'BRA_score']].mean(axis=1)

            # rank based on final score 
            merged_df_sorted = merged_df.sort_values(by='final_score', ascending=False)

            # Save to CSV
            save_path = f"{directory}/{repeat}/{condition}_overlap_scores_compiled_{num_bins}bins.csv"
            merged_df_sorted.to_csv(save_path, index=False)

            print(f"Saved {save_path}")




def plot_top_ids(condition, csv_path, images_dir, output_dir, top_n=5, selection='top'):
    df = pd.read_csv(csv_path)

    # Choose IDs
    if selection == 'top':
        selected = df.sort_values('final_score', ascending=False).head(top_n)
    elif selection == 'bottom':
        selected = df.sort_values('final_score', ascending=True).head(top_n)
    elif selection == 'middle':
        sorted_df = df.sort_values('final_score', ascending=False)
        mid_idx = len(sorted_df)//2
        start = max(0, mid_idx - top_n//2)
        selected = sorted_df.iloc[start:start+top_n]
    else:
        raise ValueError("selection must be 'top', 'bottom', or 'middle'")

    os.makedirs(output_dir, exist_ok=True)

    for _, row in selected.iterrows():
        ID = row['ID']
        # Build image paths
        gata3_img_path = f"{images_dir}/img_GATA3_{condition}/{ID:.0f}.tiff"
        sox2_img_path  = f"{images_dir}/img_SOX2_{condition}/{ID:.0f}.tiff"
        bra_img_path   = f"{images_dir}/img_BRA_{condition}/{ID:.0f}.tiff"

        sox2_rgb, bra_rgb, gata3_rgb, merge_rgb = overlay_channels(gata3_img_path, sox2_img_path, bra_img_path)

        # Create a single figure with 4 panels in a row
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        images = [sox2_rgb, bra_rgb, gata3_rgb, merge_rgb]
        titles = ['SOX2', 'BRA', 'GATA3', 'Merge']

        for ax, img, title in zip(axes, images, titles):
            ax.imshow(img)
            ax.set_title(title)
            ax.axis('off')

        plt.suptitle(f"ID: {ID}  |  Score: {row['final_score']:.2f}")
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)

        save_path = os.path.join(output_dir, f"{condition}_{selection}_{int(ID)}.png")
        plt.savefig(save_path)
        plt.close(fig)


def plot_gastruloid_scoring(directory, repeats, selection, conditions, num_bins):
    for repeat in repeats:
        for condition in conditions:
            csv_path = f"{directory}/{repeat}/{condition}_overlap_scores_compiled_{num_bins}bins.csv"
            images_dir = f"{directory}/{repeat}/boxes_tiff_selected"  # adjust to your actual image folder
            output_dir = f"{directory}/{repeat}/plots"
            plot_top_ids(condition, csv_path, images_dir, output_dir, top_n=5, selection=selection)

        

def DAPIIntensity_vs_score_scatterplot(directory, repeats, conditions, num_bins, score="final"):
    '''
    The score argument lets you pick what type of score to plot along with DAPI Intensity. The options are:
    - "final" (average between GATA3, SOX2, BRA). Its final on default. 
    -  "GATA3": GATA3 Score
    -  "SOX2": SOX2 Score
    -  "BRA": BRA Score
    '''

    chosen_score = f"{score}_score" # either GATA3_score, SOX2_score, BRA_score, final_score

    point_colors = {
        "WT": "#FF93BC",
        "ND6": "#819EFF"
    }

    for repeat in repeats:
        for condition in conditions:

            scores_path = f"{directory}/{repeat}/{condition}_overlap_scores_compiled_{num_bins}bins.csv"
            dapi_info_path = f"{directory}/{repeat}/distribution/{condition}_DAPI.csv"
            save_path = f"{directory}/{repeat}/plots/DAPIIntensity_vs_{score}score_scatterplot_{condition}_{num_bins}bins.png"

            # Load CSVs
            scores_df = pd.read_csv(scores_path)  
            dapi_df = pd.read_csv(dapi_info_path)

            # Match IDs: ID 1 -> Index 0
            scores_df['DAPI_intensity'] = scores_df['ID'].apply(lambda x: dapi_df.loc[x-1, 'intensity'])

            # Scatter plot
            plt.figure(figsize=(6,5))
            plt.scatter(scores_df[chosen_score], scores_df['DAPI_intensity'], color = point_colors.get(condition), label='Data points')

            # Add trendline
            z = np.polyfit(scores_df[chosen_score], scores_df['DAPI_intensity'], 1)  # linear fit
            p = np.poly1d(z)
            plt.plot(scores_df[chosen_score], p(scores_df[chosen_score]), color="#7B2296", linestyle='--', label=f'Trendline (slope={z[0]:.2f})')

            plt.xlabel(chosen_score)
            plt.xlim(0, 100)
            plt.ylabel("DAPI Intensity")
            plt.title(f"{condition}, {score} score vs DAPI Intensity (R:{repeat})")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()

def DAPIIntensity_vs_score_scatterplot_combined(directory, repeats, conditions, num_bins, score="final"):
    """
    Plots DAPI intensity vs final score for each repeat.
    - Multiple conditions shown with different point colors.
    - One combined trendline for all conditions in the repeat.
    
    Args:
        directory (str): Base directory path.
        repeats (list): List of repeats to plot.
        conditions (list): List of conditions per repeat.
        point_colors (dict): Optional mapping {condition: color}.
        trendline_color (str): Color for the combined trendline.
    """

    point_colors = {
        "WT": "#FF93BC",
        "ND6": "#819EFF"
    }

    trendline_color="#7B2296"

    chosen_score = f"{score}_score" # either GATA3_score, SOX2_score, BRA_score, final_score
    
    for repeat in repeats:
        plt.figure(figsize=(6,5))
        
        all_scores = []
        all_dapi = []

        for condition in conditions:
            scores_path = f"{directory}/{repeat}/{condition}_overlap_scores_compiled_{num_bins}bins.csv"
            dapi_info_path = f"{directory}/{repeat}/distribution/{condition}_DAPI.csv"

            # Load CSVs
            scores_df = pd.read_csv(scores_path)  
            dapi_df = pd.read_csv(dapi_info_path)

            # Match IDs
            scores_df['DAPI_intensity'] = scores_df['ID'].apply(lambda x: dapi_df.loc[x-1, 'intensity'])

            # Store for combined trendline
            all_scores.extend(scores_df[chosen_score])
            all_dapi.extend(scores_df['DAPI_intensity'])

            # Scatter plot for this condition
            plt.scatter(
                scores_df[chosen_score], 
                scores_df['DAPI_intensity'], 
                color=point_colors.get(condition, None) if point_colors else None,
                label=condition
            )

        # Combined trendline
        z = np.polyfit(all_scores, all_dapi, 1)
        p = np.poly1d(z)
        x_range = np.linspace(min(all_scores), max(all_scores), 100)
        plt.plot(
            x_range, 
            p(x_range), 
            color=trendline_color, 
            linestyle='--', 
            linewidth=2,
            label=f"Combined trend (slope={z[0]:.2f})"
        )

        plt.xlim(0, 100)
        plt.xlabel(f"Gastruloid {score} Score")
        plt.ylabel("DAPI Intensity")
        plt.title(f"Gastruloid {score} Score vs DAPI Intensity (Repeat: {repeat})")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        save_path = f"{directory}/{repeat}/plots/DAPIIntensity_vs_{score}score_scatterplot_combined_{num_bins}bins.png"
        plt.savefig(save_path)
        plt.close()

def DAPI_average_intensity(directory, repeats, conditions):
    """
    Calculates the average intensity from multiple CSV files and logs them into one output CSV.

    Parameters:
        directory (str): Base directory containing the data.
        repeats (list): List of repeat folder names.
        conditions (list): List of condition names.
        output_file (str): Path to save the output CSV.
    """
    results = []
    output_file=f"{directory}/meta_DAPIintensity.csv"

    for repeat in repeats:
        for condition in conditions:
            csv_path = f"{directory}/{repeat}/distribution/{condition}_DAPI.csv"
            
            df = pd.read_csv(csv_path)
            if 'intensity' not in df.columns:
                raise ValueError(f"CSV at {csv_path} must contain an 'intensity' column.")
            
            avg_intensity = df['intensity'].mean()
            results.append({"repeat": repeat, "condition": condition, "average_intensity": avg_intensity})

    # Save to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Saved average intensities to {output_file}")


def final_score_distribution(directory, repeats, conditions, num_bins, score='final'):
    # Define fixed bins from 0 to 100, step of 10
    bins = list(range(0, 101, 10))

    chosen_score = f"{score}_score" # either GATA3_score, SOX2_score, BRA_score, final_score


    for repeat in repeats:
        for condition in conditions:
            
            csv_path = f"{directory}/{repeat}/{condition}_overlap_scores_compiled_{num_bins}bins.csv"
            save_path = f"{directory}/{repeat}/plots/{condition}_{score}_reproducibility_{num_bins}bins.png"

            if not os.path.exists(csv_path):
                print(f"Skipping missing file: {csv_path}")
                continue

            df = pd.read_csv(csv_path)

            colors_dict = { "BRA": "#FFEE00", "SOX2": "#00FFFF", "GATA3": "#FF00FF", "final": "#FF81CA"}  # custom colors

            plt.hist(df[chosen_score], bins=bins, color=colors_dict.get(score), edgecolor='black')
            plt.xlabel(f'Gastruloid {score} Score')
            plt.ylabel('Frequency')
            plt.title(f'Distribution of {condition} Gastruloid {score} Scores (R: {repeat})')

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()



def final_score_distribution_combined(directory, repeats, conditions, num_bins, score='final'):
    """
    Plots combined histograms for all repeats per condition.
    Histograms are plotted in ascending order of mean final_score so that
    lower scoring repeats appear in front.
    """
    bins = list(range(0, 101, 10))  # fixed bins 0-100
    chosen_score = f"{score}_score" # either GATA3_score, SOX2_score, BRA_score, final_score
    colors = ["#FF81CA", "#6EC1E4", "#FFD700"]  # custom colors

    for condition in conditions:
        plt.figure(figsize=(8, 6))

        # Load all repeats into a list with their mean final_score
        repeat_data = []
        for repeat in repeats:
            csv_path = f"{directory}/{repeat}/{condition}_overlap_scores_compiled_{num_bins}bins.csv"

            if not os.path.exists(csv_path):
                print(f"Skipping missing file: {csv_path}")
                continue
            df = pd.read_csv(csv_path)
            mean_score = df[chosen_score].mean()
            repeat_data.append((mean_score, repeat, df))

        # Sort repeats by mean_score ascending (lowest first)
        repeat_data.sort(key=lambda x: x[0])

        # Plot each repeat histogram in order
        for i, (_, repeat, df) in enumerate(repeat_data):
            plt.hist(
                df[chosen_score],
                bins=bins,
                alpha=0.4,
                color=colors[i % len(colors)],
                edgecolor='black',
                label=f"Repeat {repeat}"
            )

        plt.xlabel(f'Gastruloid {score} Score')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {condition} Gastruloid {score} Scores (All Repeats)')
        plt.legend()

        save_path = f"{directory}/plots/{condition}_{score}_reproducibility_{num_bins}bins_combined.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()


def final_score_distribution_markercombined(directory, repeats, conditions, chosen_markers, num_bins):
    """
    Plots combined histograms for all repeats per condition.
    Histograms are plotted in ascending order of mean final_score so that
    lower scoring repeats appear in front.
    """
    bins = list(range(0, 101, 10))  # fixed bins 0-100

    colors_dict = { "BRA": "#FFEE00", "SOX2": "#00FFFF", "GATA3": "#FF00FF"}  # custom colors

    for repeat in repeats:
        for condition in conditions:
            plt.figure(figsize=(8, 6))

            # Load all repeats into a list with their mean final_score
            marker_data = []
            for marker in chosen_markers:
                csv_path = f"{directory}/{repeat}/{condition}_overlap_scores_compiled_{num_bins}bins.csv"

                if not os.path.exists(csv_path):
                    print(f"Skipping missing file: {csv_path}")
                    continue

                df = pd.read_csv(csv_path)
                chosen_score = f"{marker}_score"
                mean_score = df[chosen_score].mean()
                marker_data.append((mean_score, chosen_score, marker, df))

            # Sort repeats by mean_score ascending (lowest first)
            marker_data.sort(key=lambda x: x[0])

            # Plot each repeat histogram in order
            for i, (_, chosen_score, marker, df) in enumerate(marker_data):
                plt.hist(
                    df[chosen_score],
                    bins=bins,
                    alpha=0.4,
                    color= colors_dict.get(marker),
                    edgecolor='black',
                    label=f"{marker}"
                )

            plt.xlabel(f'Gastruloid Markers Score')
            plt.ylabel('Frequency')
            plt.title(f'Distribution of {condition} Gastruloid Markers Scores (Repeat {repeat})')
            plt.legend()

            save_path = f"{directory}/{repeat}/plots/{condition}_markers_reproducibility_{num_bins}bins_combined.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()
