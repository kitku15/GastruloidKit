import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

def hex_to_rgb01(color):
    # If it's already an (r,g,b) tuple in 0–1 range
    if isinstance(color, tuple):
        return color
    
    # Remove leading "#"
    hex_color = color.lstrip('#')
    
    # Handle 8-digit hex (#RRGGBBAA)
    if len(hex_color) == 8:
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        a = int(hex_color[6:8], 16) / 255.0
        return (r, g, b, a)
    
    # Handle 6-digit hex (#RRGGBB)
    elif len(hex_color) == 6:
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        return (r, g, b)
    
    else:
        raise ValueError(f"Invalid hex color format: {color}")


def color_image(gray_img, color):
    """Convert a grayscale image to an RGB image with a specific color."""
    return np.stack([gray_img]*3, axis=-1) * color

def overlay_channels(gata3_img_path, sox2_img_path, bra_img_path, dapi_img_path, marker_colors):

    for key in marker_colors:
        if isinstance(marker_colors[key], str):
            marker_colors[key] = hex_to_rgb01(marker_colors[key])

    # Load grayscale images
    dapi = np.array(Image.open(dapi_img_path).convert('L'), dtype=float)/255.0
    gata3 = np.array(Image.open(gata3_img_path).convert('L'), dtype=float)/255.0
    sox2  = np.array(Image.open(sox2_img_path).convert('L'), dtype=float)/255.0
    bra   = np.array(Image.open(bra_img_path).convert('L'), dtype=float)/255.0

    # Create colored images
    dapi_rgb = color_image(gata3, marker_colors['DAPI'])
    gata3_rgb = color_image(gata3, marker_colors['GATA3'])
    sox2_rgb  = color_image(sox2, marker_colors['SOX2'])
    bra_rgb   = color_image(bra, marker_colors['BRA'])

    # Merge: pixel-wise max
    merge_rgb = np.maximum.reduce([gata3_rgb, sox2_rgb, bra_rgb])
    merge_rgb = np.clip(merge_rgb, 0, 1)

    return sox2_rgb, bra_rgb, gata3_rgb, dapi_rgb, merge_rgb

def channels_plot_any(ID, directory, repeat, condition, marker_colors):

    images_dir = f"{directory}/{repeat}/boxes_tiff_selected"
    output_dir = f"{directory}/{repeat}/plots"
    
    dapi_img_path = f"{images_dir}/img_DAPI_{condition}/{ID:.0f}.tiff"
    gata3_img_path = f"{images_dir}/img_GATA3_{condition}/{ID:.0f}.tiff"
    sox2_img_path  = f"{images_dir}/img_SOX2_{condition}/{ID:.0f}.tiff"
    bra_img_path   = f"{images_dir}/img_BRA_{condition}/{ID:.0f}.tiff"

    sox2_rgb, bra_rgb, gata3_rgb, dapi_rgb, merge_rgb = overlay_channels(gata3_img_path, sox2_img_path, bra_img_path, dapi_img_path, marker_colors)

    # Create a single figure with 4 panels in a row
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    images = [dapi_rgb, sox2_rgb, bra_rgb, gata3_rgb, merge_rgb]
    titles = ['DAPI','SOX2', 'BRA', 'GATA3', 'Merge']

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')

    plt.suptitle(f"ID: {ID}, {condition}, R: {repeat}")
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    out_dir = f"{output_dir}/channels/"
    out_name =  f"{condition}_{int(ID)}.png"
    out_path = os.path.join(out_dir, out_name)

    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)

def channels_plot_pair(ID, directory, repeat, condition, markers, marker_colors):
    """
    Plot two chosen markers and their merge for a given gastruloid ID, condition, and repeat.

    Parameters
    ----------
    ID : int or float
        Gastruloid ID (can be float but will be cast to int in filenames).
    directory : str
        Base directory path.
    repeat : str or int
        Repeat identifier.
    condition : str
        Condition name.
    markers : tuple of str
        Two markers chosen from ("SOX2", "BRA", "GATA3").
    """


    images_dir = f"{directory}/{repeat}/boxes_tiff_selected"
    output_dir = f"{directory}/{repeat}/plots"
    os.makedirs(output_dir, exist_ok=True)


    # Build image paths
    paths = {
        "SOX2": f"{images_dir}/img_SOX2_{condition}/{ID:.0f}.tiff",
        "BRA": f"{images_dir}/img_BRA_{condition}/{ID:.0f}.tiff",
        "GATA3": f"{images_dir}/img_GATA3_{condition}/{ID:.0f}.tiff",
        "DAPI": f"{images_dir}/img_DAPI_{condition}/{ID:.0f}.tiff",
        "psmad159": f"{images_dir}/img_psmad159_{condition}/{ID:.0f}.tiff",
    }

    # Overlay just the selected markers
    imgs_rgb = {}

    for marker in markers:
        gray = np.array(Image.open(paths[marker]).convert("L"), dtype=float)/255.0
        imgs_rgb[marker] = color_image(gray, hex_to_rgb01(marker_colors[marker]))

    # Merge by pixel-wise max
    merge_rgb = np.maximum(imgs_rgb[markers[0]], imgs_rgb[markers[1]])

    # Plot: marker1, marker2, merge
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, marker in zip(axes, [markers[0], markers[1]]):
        ax.imshow(imgs_rgb[marker])
        ax.set_title(marker)
        ax.axis("off")

    axes[2].imshow(merge_rgb)
    axes[2].set_title("Merge")
    axes[2].axis("off")

    plt.suptitle(f"ID: {ID}, {condition}, R: {repeat}")
    plt.tight_layout()
    plt.subplots_adjust(top=0.8)

    out_dir = f"{output_dir}/channels/pairs/"
    out_name =  f"{condition}_{int(ID)}_{markers[0]}_{markers[1]}.png"
    out_path = os.path.join(out_dir, out_name)

    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)
