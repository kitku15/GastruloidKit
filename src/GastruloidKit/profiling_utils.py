import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from matplotlib.patches import Wedge
from matplotlib.colors import LinearSegmentedColormap


def white_to_color(color_name):
    return LinearSegmentedColormap.from_list("", ["#FFFFFF00", color_name])

def get_avg_df(bin_df, bin_col, bins_of_interest, meta_indiv_df, id_col="ID"):
    """
    Compute average marker intensity profiles for gastruloids in specific bins.

    Parameters
    ----------
    bin_df : pandas.DataFrame
        DataFrame containing bin assignments for each gastruloid.
    bin_col : str
        Name of the column containing the categorical bin assignments.
    bins_of_interest : list of int
        List of bin indices (1-5) to include.
    meta_indiv_df : pandas.DataFrame
        Marker intensity profiles per gastruloid.
    id_col : str, optional (default="ID")
        Column used to match gastruloid IDs across DataFrames.

    Returns
    -------
    avg_df : pandas.DataFrame
        DataFrame containing average marker profiles per bin column,
        grouped by marker.
    n_points : int
        Total number of data points included before grouping.
    """
    matching_ids = bin_df.loc[bin_df[bin_col].isin(bins_of_interest), id_col]
    filtered_df = meta_indiv_df[meta_indiv_df[id_col].isin(matching_ids)]

    bin_cols = [col for col in filtered_df.columns if col.startswith("bin_")]
    avg_df = filtered_df.groupby("marker")[bin_cols].mean().reset_index()
    return avg_df, filtered_df.shape[0]


def make_profile_plot(avg_df, bins_of_interest, n_points, bin_categories,
                      repeat, condition, save_dir, bin_label="DAPI", marker_colors=None):
    """
    Generate and save a normalized intensity profile plot.

    Parameters
    ----------
    avg_df : pandas.DataFrame
        Average marker intensity profiles.
    bins_of_interest : int
        Bin index (1-5) being plotted.
    n_points : int
        Number of data points used to compute the averages.
    bin_categories : pandas.Categorical
        Categories used for binning (to get numeric ranges).
    repeat, condition : str
        Identifiers for labeling.
    save_dir : str
        Directory where plots are saved.
    bin_label : str, optional
        Label to use in the plot title and filename ("DAPI" or "Furthest_bin").
    marker_colors : dict, optional
        Dictionary mapping markers to colors.
    """
    regions = [col for col in avg_df.columns if col.startswith('bin_')]
    x = np.arange(len(regions))
    x_smooth = np.linspace(x.min(), x.max(), 300)

    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7,5))

    for _, row in avg_df.iterrows():
        marker = row['marker']
        y = row[regions].values.astype(float)
        y_norm = (y - y.min()) / (y.max() - y.min() + 1e-6)
        spline = make_interp_spline(x, y_norm, k=2)
        y_smooth = spline(x_smooth)
        color = marker_colors.get(marker, "gray") if marker_colors else "gray"
        ax.plot(x_smooth, y_smooth, label=marker, linewidth=2, color=color)
        ax.scatter(x, y_norm, color=color, edgecolor='k', zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels([str(i+1) for i in range(len(regions))])
    ax.set_ylabel('Normalized Intensity (0–1)')
    ax.set_xlabel('Bin')

    r = bin_categories.cat.categories[bins_of_interest-1]
    range_text = f"{r.left:.2f} - {r.right:.2f}"

    ax.set_title(
        f'Normalized Patterning\nRepeat {repeat}, Condition {condition}\n'
        f'{bin_label} Bin {bins_of_interest} (Range: {range_text})\n'
        f'N = {(n_points/3):.0f} gastruloids'
    )

    ax.legend(title='Marker')
    plt.tight_layout()

    out_dir = f"{save_dir}/lineplot/{condition}"
    out_name = f"{bin_label.lower()}_bin{bins_of_interest}.png"
    output_path = os.path.join(out_dir, out_name)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_radial_bin_heatmap(avg_df, bins_of_interest, repeat, condition, save_dir, bin_label, marker_colors):
    """
    Make radial donut-style heatmaps for each marker and a combined overlay.

    Parameters
    ----------
    avg_df : pandas.DataFrame
        Averaged marker intensity profiles.
    bins_of_interest : int
        Bin index (1–5) being plotted.
    repeat, condition : str
        Identifiers for labeling.
    save_dir : str
        Directory where plots are saved.
    bin_label : str
        Label to use in plot titles/filenames (e.g. "Furthest_bin").
    marker_colors : dict
        Dictionary mapping markers to hex/RGB colors.
    """
    regions = [col for col in avg_df.columns if col.startswith("bin_")]
    num_bins = len(regions)

    # Normalize per marker
    normed_values = {}
    for _, row in avg_df.iterrows():
        marker = row["marker"]
        y = row[regions].values.astype(float)
        normed_values[marker] = (y - y.min()) / (y.max() - y.min() + 1e-6)

    # ---- One heatmap per marker ----
    for marker, values in normed_values.items():
        fig, ax = plt.subplots(figsize=(5,5))
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)

        cmap = plt.cm.get_cmap("Blues")
        if marker in marker_colors:
            cmap = white_to_color(marker_colors[marker])  # assume this function is defined elsewhere

        for j, val in enumerate(values):
            r_inner = j / num_bins
            r_outer = (j + 1) / num_bins
            color = cmap(val)

            wedge = Wedge(center=(0, 0), r=r_outer,
                          theta1=0, theta2=360,
                          width=r_outer - r_inner,
                          facecolor=color, edgecolor="none")
            ax.add_patch(wedge)

        outer_circle = plt.Circle((0, 0), radius=1.0, fill=False, edgecolor="k", linewidth=1.5)
        ax.add_patch(outer_circle)

        ax.set_title(f"{marker} Bin {bins_of_interest}, {condition} repeat: {repeat}", fontsize=12)
        plt.tight_layout()
        out_dir =  f"{save_dir}/radialheatmap/{condition}"
        out_name = f"{marker}_{bin_label.lower()}_bin{bins_of_interest}.png"
        out_path = os.path.join(out_dir, out_name)
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(out_path, dpi=300)
        plt.close()

    # ---- Combined overlay heatmap ----
    fig, ax = plt.subplots(figsize=(5,5))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)

    for marker, values in normed_values.items():
        cmap = white_to_color(marker_colors.get(marker, "gray"))
        for j, val in enumerate(values):
            r_inner = j / num_bins
            r_outer = (j + 1) / num_bins
            color = cmap(val)

            wedge = Wedge(center=(0, 0), r=r_outer,
                          theta1=0, theta2=360,
                          width=(r_outer - r_inner),
                          facecolor=color, edgecolor="none")
            ax.add_patch(wedge)

    outer_circle = plt.Circle((0, 0), radius=1.0, fill=False, edgecolor="k", linewidth=1.5)
    ax.add_patch(outer_circle)
    ax.set_title(f"Combined Bin {bins_of_interest}, {condition} repeat: {repeat}", fontsize=12)

    out_dir = f"{save_dir}/radialheatmap/{condition}"
    out_name = f"combined_{bin_label.lower()}_bin{bins_of_interest}.png"
    out_path = os.path.join(out_dir, out_name)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()