import os
import csv
import numpy as np
import pandas as pd
import tifffile
from czifile import CziFile
import czifile
from PIL import Image
from skimage.transform import resize, rotate


def czi_to_tiff(folder_path):
    """
    Takes a folder containing all CZI files,
    converts them into multi-channel TIFFs without splitting channels.
    Saves them in the same folder.
    """

    czi_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.czi'):
                file_path = os.path.join(root, file)
                print(f"Processing: {file_path}")
                czi_paths.append(file_path)

    for czi_path in czi_paths:
        try:
            img = czifile.imread(czi_path)
            czi = CziFile(czi_path)

            print(f"Original CZI shape: {img.shape}")
            print("Metadata:", czi.metadata())

            img_squeezed = img.squeeze()  # Remove singleton dimensions
            print(f"Squeezed shape: {img_squeezed.shape}")

            # Save the entire image as a multi-channel TIFF
            base_path, _ = os.path.splitext(czi_path)
            output_path = f"{base_path}.tiff"
            tifffile.imwrite(output_path, img_squeezed, photometric='minisblack', metadata={'axes': 'CZYX'})
            print(f"Saved multi-channel TIFF: {output_path}")

        except Exception as e:
            print(f"Error processing {czi_path}: {e}")


    
def split_into_channels(directory, repeats, conditions, channel_folders):
    """
    Splits multi-channel TIFF files into separate single-channel TIFFs for multiple experimental conditions.

    For each combination of repeat and condition, this function reads the corresponding multi-channel
    TIFF file (expected at path: directory/repeat/condition.tiff), splits it into individual channels, 
    and saves each channel as a separate TIFF file with a suffix indicating the channel name.

    Parameters:
        directory (str): Base directory containing all TIFF files organized by repeat and condition.
        repeats (list of str): List of repeat identifiers (e.g., experimental repeats or sample batches).
        conditions (list of str): List of condition identifiers corresponding to TIFF filenames.
        channel_folders (dict): Mapping of channel indices (int) to channel names (str), 
                                used to name output files (e.g., {0: 'DAPI', 1: 'SOX2'}).

    Returns:
        list of str: Paths to the saved single-channel TIFF files for the last processed condition.
                     (Note: earlier results are not retained if multiple files are processed.)
    """
    no_of_markers = len(channel_folders)-1

    for repeat in repeats:
        for condition in conditions:
            print(f"Processing repeat: {repeat}, condition: {condition}---------")

            tiff_path = f"{directory}/{repeat}/{condition}_scaled.tiff"

            try:
                img = tifffile.imread(tiff_path)

                if img.ndim < no_of_markers:
                    raise ValueError("Input TIFF does not have enough dimensions for channels set.")

                num_channels = img.shape[0]
                print(f"TIFF shape: {img.shape}")
                print(f"Detected channels: {num_channels}")

                base_path, _ = os.path.splitext(tiff_path)

                for channel_no in range(num_channels):
                    if channel_no not in channel_folders:
                        print(f"Skipping unknown channel: {channel_no}")
                        continue

                    channel_data = img[channel_no]  # (Z, Y, X)
                    output_path = f"{base_path}_{channel_folders[channel_no]}.tiff"
                    tifffile.imwrite(output_path, channel_data)
                    print(f"Saved channel {channel_no} TIFF: {output_path}")

            except Exception as e:
                print(f"Error splitting {tiff_path} into channels: {e}")
                return []



def crop_tiff(tiff_path, ymin, ymax, xmin, xmax, rotate_angle):
    '''
    Crops a TIFF image to a specified rectangular region and returns the result.

    Parameters:
        tiff_path (str): Path to the TIFF file to be cropped.
        ymin (int): Starting Y-coordinate (row) of the crop.
        ymax (int): Ending Y-coordinate (row) of the crop (exclusive).
        xmin (int): Starting X-coordinate (column) of the crop.
        xmax (int): Ending X-coordinate (column) of the crop (exclusive).
        rotate_angle (int): How much to rotate the image if needed 

    Returns:
        numpy.ndarray: The cropped image region as a NumPy array.

    Notes:
        - Automatically squeezes singleton dimensions from the image (e.g., shape (1, Y, X) → (Y, X)).
        - Supports cropping 2D or 3D TIFF images (only spatial dimensions Y and X are cropped).
    '''

    img = tifffile.imread(tiff_path)
    img = np.squeeze(img)
    # original_dimensions = image_squeezed.shape
    # print("original dimensions:", original_dimensions)

    print(f"\nProcessing {tiff_path}")
    print(ymin, ymax, xmin, xmax, rotate_angle)
    
    # rotateing if needed
    if rotate_angle != 0:
        print(f"rotating this much: {rotate_angle}")
        img = rotate(img, angle=rotate_angle, reshape=True)

    # cropping
    print("Cropping...")
    img = img[ymin:ymax, xmin:xmax]

    print(f"Finished cropping {tiff_path}")

    # resize 
    size=(7800, 7800)
    img = resize(img, size, anti_aliasing=True, preserve_range=True).astype(img.dtype)

    return img

def resize_tiff(input_path, output_path, size=(7800, 7800)):
    # Open the TIFF image
    with Image.open(input_path) as img:
        # Resize to the target size
        size=(7800, 7800)
        resized_img = img.resize(size, Image.LANCZOS)  # LANCZOS gives high-quality resampling
        
      

def crop_all_tiffs_in_repeat(directory, repeat, condition, markers, ymin, ymax, xmin, xmax, rotate_angle):
    """
    Crops and saves TIFF mask images for a given experimental repeat and condition.

    This function loads mask TIFF files for each specified marker, applies cropping
    and rotation using the provided coordinates and angle, and saves the cropped
    versions into a subdirectory named 'CROPPED_MASKS'.

    Parameters:
        directory (str): Base directory containing the experimental repeats.
        repeat (str): Identifier for the repeat (subdirectory name).
        condition (str): Experimental condition (used to build the filename).
        markers (list of str): List of marker names whose masks should be processed.
        ymin (int): Minimum Y-coordinate for cropping.
        ymax (int): Maximum Y-coordinate for cropping.
        xmin (int): Minimum X-coordinate for cropping.
        xmax (int): Maximum X-coordinate for cropping.
        rotate_angle (float): Angle (in degrees) to rotate the cropped image.

    Returns:
        None
    """

    
    img_dict = {}
    for marker in markers:

        # Directory of the mask itself thats to be cropped 
        full_directory = f"{directory}/{repeat}"
        mask_path = f"{full_directory}/{condition}_{marker}_mask.tiff"
        
        # crop the mask and resize 
        mask = crop_tiff(mask_path, ymin, ymax, xmin, xmax, rotate_angle)
        img_dict[f"mask_{marker}"] = mask

        # mask will replace the uncropped mask 
        tifffile.imwrite(mask_path, mask)
    
def compile_crop_coordinates(directory, repeats, conditions, cropping_csv_path):
    """
    Compiles and saves cropping coordinates from a CSV file of X, Y points.

    This function reads X and Y coordinates from a given CSV file, calculates the 
    bounding box (min/max values), and appends the result along with the rotation 
    angle, repeat, and condition to a cropping configuration CSV. If the output 
    file does not exist, a header is written first.

    Parameters:
        repeat (str): Identifier for the repeat (e.g., experiment batch or sample).
        condition (str): Experimental condition associated with the coordinates.
        coords_path (str): Path to the input CSV file containing 'X' and 'Y' columns.
        cropping_csv_path (str): Path to the output CSV file to save crop metadata.
        angle (float): Rotation angle to be associated with the crop coordinates.

    Returns:
        None
    """
    print("Compiling cropping Coordinates----------")

    for repeat in repeats:
        for condition in conditions:
            x_vals = []
            y_vals = []

            coords_path = f"{directory}/{repeat}/{condition}_coords.csv"

            # Read the input CSV with X,Y columns
            with open(coords_path, 'r', newline='') as infile:
                reader = csv.DictReader(infile)
                for row in reader:
                    x_vals.append(int(row['X']))
                    y_vals.append(int(row['Y']))

            # Calculate bounding box values
            xmin = min(x_vals)
            xmax = max(x_vals)
            ymin = min(y_vals)
            ymax = max(y_vals)

            # Prepare the new row 
            new_row = {
                'repeat': repeat,
                'condition': condition,
                'ymin': ymin,
                'ymax': ymax,
                'xmin': xmin,
                'xmax': xmax,
                'rotate_angle': 0.0
            }

            # Check if output file exists and has a header
            try:
                with open(cropping_csv_path, 'r', newline='') as outfile:
                    has_header = csv.Sniffer().has_header(outfile.read(1024))
            except FileNotFoundError:
                has_header = False

            # Append the new row
            with open(cropping_csv_path, 'a', newline='') as outfile:
                fieldnames = ['repeat', 'condition', 'ymin', 'ymax', 'xmin', 'xmax', 'rotate_angle']
                writer = csv.DictWriter(outfile, fieldnames=fieldnames)

                if not has_header:
                    writer.writeheader()

                writer.writerow(new_row)
                print(f"Saved Coordinates from repeat: {repeat}, condition: {condition}")


def get_crop_coordinates(csv_path, repeat, condition):
    """
    Fetches cropping coordinates for a given repeat and condition from a CSV file.

    Parameters:
        csv_path (str): Path to the CSV file.
        repeat (int): Repeat number (e.g., 1, 2, 3).
        condition (str): Experimental condition (e.g., "WT", "ND6").

    Returns:
        list: [ymin, ymax, xmin, xmax, rotate_angle]

    Raises:
        ValueError: If the combination of repeat and condition is not found.
    """
    df = pd.read_csv(csv_path)

    # Filter based on repeat and condition
    row = df[(df['repeat'] == repeat) & (df['condition'] == condition)]

    if row.empty:
        raise ValueError(f"No entry found for repeat {repeat} and condition '{condition}'.")

    # Return as list
    return row.iloc[0][['ymin', 'ymax', 'xmin', 'xmax', 'rotate_angle']].tolist()

def load_image_and_mask(directory, repeat, condition, markers):
    images = []
    masks = []

    for marker in markers:
        img = tifffile.imread(f'{directory}/{repeat}/{condition}_scaled_{marker}.tiff')
        images.append(img)
        mask = tifffile.imread(f'{directory}/{repeat}/{condition}_{marker}_mask.tiff')
        masks.append(mask)

    return images, masks


def crop_masks(directory, cropping_csv_path, conditions, repeats, markers):

    for condition in conditions:
        for i in repeats:
            ymin, ymax, xmin, xmax, rotate_angle = get_crop_coordinates(cropping_csv_path, repeat=i, condition=condition)
            crop_all_tiffs_in_repeat(directory=directory, repeat=i, condition=condition, markers=markers, ymin=ymin, ymax=ymax, xmin=xmin, xmax=xmax, rotate_angle=rotate_angle)

def check_dimensions(directory):
    # Walk through all subdirectories
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith(('.tif', '.tiff')):
                filepath = os.path.join(root, filename)
                try:
                    img = tifffile.imread(filepath)
                    shape = img.shape
                    print(f"{filepath}: {shape}")
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")
                    