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
    """
    Multiply grayscale image (H, W) by RGB color (3,) or RGBA (4,).
    Result is an RGB image where intensity is modulated by color.

    """

    rgb_img = np.stack([gray_img]*3, axis=-1)  # shape (H, W, 3)
    color = np.array(color[:3])  # ignore alpha if present
    return rgb_img * color



def overlay_channels(images_dir, condition, ID, markers, marker_colors):
    for key in marker_colors:
        if isinstance(marker_colors[key], str):
            marker_colors[key] = hex_to_rgb01(marker_colors[key])

    colored_images = []
    blended_image = None

    for marker in markers:
        marker_path = f"{images_dir}/img_{marker}_{condition}/{ID:.0f}.tiff"

        if not os.path.exists(marker_path):
            print(f"Warning: file not found for {marker}: {marker_path}")
            continue

        gray = np.array(Image.open(marker_path).convert('L'), dtype=float) / 255.0
        rgb = color_image(gray, marker_colors[marker])
        colored_images.append(rgb)

        if blended_image is None:
            blended_image = rgb.copy()
        else:
            blended_image += rgb

    # Normalize: clip to [0,1]
    blended_image = np.clip(blended_image, 0, 1)

    return colored_images, blended_image

def channels_plot_any(ID, directory, repeat, condition, markers, marker_colors):

    images_dir = f"{directory}/{repeat}/boxes_tiff_selected"
    output_dir = f"{directory}/{repeat}/plots"

    
    colored_images, merge_rgb = overlay_channels(images_dir, condition, ID, markers, marker_colors)

    # Create a single figure with 4 panels in a row
    num_plots = len(markers)+1
    len_plot = 4*num_plots
    fig, axes = plt.subplots(1, num_plots, figsize=(len_plot, 4))

    images = []
    for image in colored_images:
        images.append(image)
    images.append(merge_rgb)

    titles = []
    for marker in markers:
        titles.append(marker)
    titles.append('Merge')

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
    Plot two chosen markers and their merged overlay for a given gastruloid ID, condition, and repeat.

    Parameters
    ----------
    ID : int or float
        Gastruloid ID (cast to int for filename).
    directory : str
        Base directory path.
    repeat : str or int
        Repeat identifier.
    condition : str
        Condition name.
    markers : tuple of str
        Two markers to plot (e.g. ("SOX2", "BRA")).
    marker_colors : dict
        Mapping marker names to hex colors or RGB tuples.
    """

    images_dir = f"{directory}/{repeat}/boxes_tiff_selected"
    output_dir = f"{directory}/{repeat}/plots"
    os.makedirs(output_dir, exist_ok=True)

    imgs_rgb = {}

    # Convert colors upfront (handle hex -> rgb)
    for key in marker_colors:
        if isinstance(marker_colors[key], str):
            marker_colors[key] = hex_to_rgb01(marker_colors[key])

    # Load and color images
    for marker in markers:
        marker_path = f"{images_dir}/img_{marker}_{condition}/{int(ID)}.tiff"
        if not os.path.exists(marker_path):
            raise FileNotFoundError(f"Marker image not found: {marker_path}")

        gray = np.array(Image.open(marker_path).convert("L"), dtype=float) / 255.0
        imgs_rgb[marker] = color_image(gray, marker_colors[marker])

    # Blend the two images by simple addition (clipped)
    merge_rgb = np.clip(imgs_rgb[markers[0]] + imgs_rgb[markers[1]], 0, 1)

    # Plot: individual markers and merged overlay
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, marker in zip(axes[:2], markers):
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
    os.makedirs(out_dir, exist_ok=True)
    out_name = f"{condition}_{int(ID)}_{markers[0]}_{markers[1]}.png"
    out_path = os.path.join(out_dir, out_name)

    plt.savefig(out_path)
    plt.close(fig)

