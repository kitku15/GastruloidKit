import os
import tifffile
from czifile import CziFile
import czifile


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

def load_image_and_mask(directory, repeat, condition, markers):
    images = []
    masks = []

    for marker in markers:
        img = tifffile.imread(f'{directory}/{repeat}/{condition}_scaled_{marker}.tiff')
        images.append(img)
        mask = tifffile.imread(f'{directory}/{repeat}/{condition}_{marker}_mask.tiff')
        masks.append(mask)

    return images, masks