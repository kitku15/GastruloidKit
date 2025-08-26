import os                              
import csv                            
import shutil                       
import numpy as np                    
import matplotlib.pyplot as plt        
from skimage.transform import rescale, resize  
from skimage.filters import gaussian            
from skimage import img_as_float                 
from skimage.feature import blob_log
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor 
from tkinter import Tk, Label, Button           
from PIL import Image, ImageTk                 
from GastruloidKit.preprocessing import load_image_and_mask


def draw_grid(img):
    '''
    Divides an input image into a 26x26 grid and assigns a unique integer label to 
    each grid cell.

    Parameters:
        img (numpy.ndarray): The input image as a 2D (grayscale) or 3D (color) NumPy array.
    Returns:
        numpy.ndarray: A 2D NumPy array (grid_mask) of the same height and width as the input 
        image, where each element contains a unique label corresponding to its grid cell 
        (ranging from 1 to 676).
    
    Notes:
        - The image is divided evenly into 26 rows and 26 columns.
        - If the image dimensions are not perfectly divisible by 26, the last row and column 
        will absorb the remainder to ensure full coverage.
    '''
    
    image_shape_yx = img.shape

    # Define grid dimensions
    grid_rows, grid_cols = 26, 26
    
    # Define the height and width of each box
    box_height = image_shape_yx[0] // grid_rows
    box_width = image_shape_yx[1] // grid_cols

    # Create coordinate grids
    grid_mask = np.zeros(image_shape_yx, dtype=np.uint16)

    label = 1  # Start labeling from 1

    for i in range(grid_rows): 
        for j in range(grid_cols):
            # Define box boundaries
            start_y = i * box_height
            end_y = (i + 1) * box_height if i < grid_rows - 1 else image_shape_yx[0]

            start_x = j * box_width
            end_x = (j + 1) * box_width if j < grid_cols - 1 else image_shape_yx[1]

            # Label the region
            grid_mask[start_y:end_y, start_x:end_x] = label
            label += 1

    print(f"\nMade grid")

    return grid_mask


def save_blobs(filename, blobs, coordinates, radii):
    '''
    Saves blob detection results to a compressed `.npz` file.

    Parameters:
        filename (str): Path to the output `.npz` file (should end with `.npz`).
        blobs (numpy.ndarray): Array containing blob detection results (e.g., combined data or metadata).
        coordinates (numpy.ndarray): Array of (y, x) or (z, y, x) positions of detected blobs.
        radii (numpy.ndarray): Array of radii corresponding to each blob.

    Returns:
        None

    Notes:
        - The data can be loaded later using `np.load(filename)`.
        - Useful for saving intermediate or final results of blob detection workflows.
    '''
    np.savez(filename, blobs=blobs, coordinates=coordinates, radii=radii)



def load_blobs(filename):
    '''
    Loads blob detection data from a `.npz` file saved using `save_blobs`.

    Parameters:
        filename (str): Path to the `.npz` file containing blob data.

    Returns:
        tuple:
            - blob (numpy.ndarray): The original blob data array.
            - coordinates (numpy.ndarray): Array of blob positions (e.g., (y, x) or (z, y, x)).
            - radii (numpy.ndarray): Array of radii corresponding to each blob.

    Notes:
        - The `.npz` file must have been saved using the `save_blobs` function format.
        - `allow_pickle=True` is used to support loading any object arrays, if present.
    '''
    data = np.load(filename, allow_pickle=True)
    blob = data['blobs']
    coordinates = data['coordinates']
    radii = data['radii']
    
    return blob, coordinates, radii

def load_allowed_ids(csv_path):
    allowed_ids = []
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 2 and row[1].strip().lower() == 'yes':
                allowed_ids.append(int(row[0]))
    return allowed_ids


def manual_selection(image_folder, output_csv):

    # === Load images ===
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))]
    image_files.sort()  # Optional: sort images for consistency
    current_index = 0

    # === Load existing labels if continuing ===
    labels = {}
    if os.path.exists(output_csv):
        with open(output_csv, newline='') as f:
            reader = csv.reader(f)
            labels = {rows[0]: rows[1] for rows in reader}
        # Remove already-labeled images
        image_files = [img for img in image_files if img not in labels]

    # === Tkinter Setup ===
    root = Tk()
    root.title("Image Labeler")

    # === Image Display ===
    label = Label(root)
    label.pack()

    def load_image():
        if current_index >= len(image_files):
            print("All images labeled!")
            root.quit()
            return
        img_path = os.path.normpath(os.path.join(image_folder, image_files[current_index]))
        img = Image.open(img_path)
        img.thumbnail((600, 600))  # Resize for screen
        img_tk = ImageTk.PhotoImage(img)
        label.config(image=img_tk)
        label.image = img_tk  # Keep a reference!

    def save_label(answer):
        nonlocal current_index
        image_name = image_files[current_index]
        labels[image_name] = answer
        with open(output_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            number_only = image_name.split(".")[0]
            writer.writerow([number_only, answer])
        current_index += 1
        load_image()

    # === Buttons ===
    btn_yes = Button(root, text="Yes (Y)", width=10, command=lambda: save_label('yes'))
    btn_yes.pack(side='left', padx=20, pady=20)

    btn_no = Button(root, text="No (N)", width=10, command=lambda: save_label('no'))
    btn_no.pack(side='right', padx=20, pady=20)

    # === Keyboard Shortcuts ===
    root.bind('y', lambda event: save_label('yes'))
    root.bind('n', lambda event: save_label('no'))

    # === Start ===
    load_image()
    root.mainloop()



def detect_blobs(tiff, downscale_factor=0.25, sigma=2, min_sigma=20, max_sigma=50, exclude_border=0):
    '''
    Detects blobs in a TIFF image using the Laplacian of Gaussian (LoG) method after downscaling.

    Parameters:
        tiff (numpy.ndarray): Input image as a NumPy array.
        downscale_factor (float, optional): Factor by which to downscale the image before detection 
            (default is 0.25, i.e., image is resized to 25% of original size).

    Returns:
        tuple:
            - blob (numpy.ndarray): Array of shape (1, 3) containing the largest detected blob's (y, x, sigma).
              Empty array if no blobs detected.
            - coordinates_rescaled (numpy.ndarray): Array of shape (1, 2) with (x, y) coordinates of the largest blob,
              rescaled to original image size.
            - radii_rescaled (numpy.ndarray): Array containing the radius of the largest blob, rescaled to original size.

    Notes:
        - The input image is downscaled and smoothed with a Gaussian filter before blob detection.
        - The largest blob by radius (sigma) is selected from all detected blobs.
        - Coordinates are converted from (y, x) to (x, y) and scaled back to the original image dimensions.
        - If no blobs are detected, returns empty arrays.
    '''

    downsampled = rescale(tiff, downscale_factor, anti_aliasing=True)  # 4x smaller

    # Convert to float and normalize
    tiff = img_as_float(downsampled)

    # smooth
    smoothed = gaussian(tiff, sigma)

    # Detect blobs
    blobs = blob_log(image=smoothed, min_sigma=min_sigma, max_sigma=max_sigma, exclude_border=exclude_border)

    if blobs.shape[0] == 0:
        return np.empty((0, 3)), np.empty((0, 2)), np.array([])
    
    # Select the blob with the largest radius (sigma)
    largest_blob_idx = np.argmax(blobs[:, 2])
    blob = blobs[largest_blob_idx].reshape(1, -1)  # shape (1,3)

    # blobs: Nx3 array of (y, x, sigma)
    coordinates = blob[:, :2][:, ::-1]  # convert (y,x) to (x,y)
    radii = blob[:, 2] * np.sqrt(2)

    # Rescale coordinates and radii to match original image
    scale = 1 / downscale_factor
    coordinates_rescaled = coordinates * scale
    radii_rescaled = radii * scale

    return blob, coordinates_rescaled, radii_rescaled


def analyse_by_grid(directory, tiff, grid_mask, marker, condition, repeat, downscale_factor=0.25, mask=True, rescale_switch=True):
    '''
    Analyzes an image by dividing it into grid regions and extracting each region in parallel,
    optionally downscaling before processing and upscaling the results afterward.

    Parameters:
        tiff (numpy.ndarray): The input TIFF image as a NumPy array (2D or 3D).
        grid_mask (numpy.ndarray): A 2D mask with labeled grid regions (same spatial size as `tiff`).
        downscale_factor (float, optional): Factor by which to downscale the image and grid for faster processing (default: 0.25).
        mask (bool, optional): If True, indicates the input is a mask; otherwise treated as an image (default: True).
        rescale_switch (bool, optional): If True, downscales before processing and upscales the cropped boxes afterward (default: True).

    Returns:
        list: A list of cropped (and optionally upscaled) NumPy arrays, one per grid region.

    Side Effects:
        - Saves the resulting list of image/mask patches as a compressed `.npz` file named:
          `"rescale_mask_boxes.npz"`, `"NOrescale_img_boxes.npz"`, etc., based on the options used.

    Notes:
        - Uses multithreading to accelerate both cropping and upscaling steps.
        - Each grid cell is processed individually using the `crop_and_save()` function.
        - If `rescale_switch` is enabled, nearest-neighbor interpolation is used for resizing `grid_mask`.
        - `upscale_box()` must return data compatible with `.npz` saving (e.g., NumPy arrays).
    '''

    # settings for rescale 
    if rescale_switch:
        tiff_small = rescale(tiff, downscale_factor, anti_aliasing=True)
        grid_mask_small = rescale(grid_mask.astype(float), downscale_factor, order=0, anti_aliasing=False).astype(np.uint16)
    else:
        tiff_small = tiff
        grid_mask_small = grid_mask.astype(np.uint16)

    unique_boxes = np.unique(grid_mask_small)

    # make and set all directory paths 
    unique_boxes = np.unique(grid_mask_small)
    npz_directory = f"{directory}/{repeat}/boxes_npz"
    tiff_directory =f"{directory}/{repeat}/boxes_tiff"
    os.makedirs(npz_directory, exist_ok=True)
    os.makedirs(tiff_directory, exist_ok=True)
    heading_npz = f"{npz_directory}/mask_{marker}_{condition}" if mask else f"{npz_directory}/img_{marker}_{condition}"
    heading_tiff = f"{tiff_directory}/mask_{marker}_{condition}" if mask else f"{tiff_directory}/img_{marker}_{condition}"

    # Crop and save tiffs individually (slow but good for manual visual checking)
    args_list = [(box_id, tiff_small, grid_mask_small, heading_tiff) for box_id in unique_boxes]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        cropped_boxes = list(executor.map(crop_and_save, args_list))

    # Parallel upscale only if rescaling was done
    if rescale_switch:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(upscale_box, cropped, downscale_factor) for cropped in cropped_boxes]
            upscaled_boxes = [future.result() for future in concurrent.futures.as_completed(futures)]
    else:
        upscaled_boxes = cropped_boxes

    # save npz file
    save_path = f"{heading_npz}.npz"
    np.savez_compressed(save_path, *upscaled_boxes)
    print(f"Saved {len(upscaled_boxes)} boxes to {save_path}")
    print(f"Boxes done")
    return upscaled_boxes


def crop_and_save(args):
    '''
    Crops a specific region from a downscaled image based on a grid mask and saves it as a PNG.

    Parameters:
        args (tuple): A tuple containing:
            - box_id (int): The unique label of the grid cell to crop.
            - tiff_small (numpy.ndarray): The downscaled image or mask.
            - grid_mask_small (numpy.ndarray): The downscaled grid mask with labeled regions.
            - heading (str): A string indicating the data type ('img' or 'mask'), used for folder naming.

    Returns:
        numpy.ndarray: The cropped image/mask corresponding to the grid cell.

    Side Effects:
        - Saves the cropped region as a PNG image in the path: `boxes/{heading}/{box_id}.png`
        - Prints confirmation when the box is saved.

    Notes:
        - Assumes the output directory `boxes/{heading}/` already exists.
        - The image is saved as 8-bit PNG after scaling to [0, 255].
    '''
    
    box_id, tiff_small, grid_mask_small, heading = args

    # if box_id == 638:
    #     print(box_id) # dupa
    # else:
    #     return None

    image_shape_y, image_shape_x = grid_mask_small.shape
    grid_rows, grid_cols = 26, 26

    box_x = (box_id - 1) % grid_cols
    box_y = (box_id - 1) // grid_cols

    box_width = image_shape_x // grid_cols
    box_height = image_shape_y // grid_rows

    pad = 30  # padding in pixels

    xmin = (box_x * box_width)
    xmax = (box_x + 1) * box_width if box_x < grid_cols - 1 else image_shape_x

    ymin = box_y * box_height
    ymax = (box_y + 1) * box_height if box_y < grid_rows - 1 else image_shape_y

    # add pad if not edges
    if xmin != 0:
        xmin = xmin - pad

    if ymin != 0:
        ymin = ymin - pad

    if xmax != image_shape_x:
        xmax = xmax + pad

    if ymax != image_shape_y:
        ymax = ymax + pad

    xmin = max(xmin, 0)
    ymin = max(ymin, 0)
    xmax = min(xmax, image_shape_x)
    ymax = min(ymax, image_shape_y)


    os.makedirs(f"{heading}", exist_ok=True)

    cropped_small = tiff_small[ymin:ymax, xmin:xmax]

    plt.imsave(f"{heading}/{box_id}.tiff", cropped_small, cmap='gray') 

    # print(f"added Box {box_id}")
    return cropped_small


def upscale_box(cropped, downscale_factor):
    '''
    Upscales a cropped image or mask back to its original resolution using the specified downscale factor.

    Parameters:
        cropped (numpy.ndarray): The downscaled cropped image or mask.
        downscale_factor (float): The factor by which the original image was downscaled (e.g., 0.25 for 4× downscaling).

    Returns:
        numpy.ndarray: The upscaled image or mask, resized to approximate original dimensions.

    Notes:
        - Uses `skimage.transform.resize` with anti-aliasing enabled for smooth interpolation.
        - Output dimensions are calculated as (height / factor, width / factor).
        - If precise alignment with original resolution is critical, consider rounding or padding post-resize.
    '''
    target_shape = (int(cropped.shape[0] / downscale_factor), int(cropped.shape[1] / downscale_factor))
    return resize(cropped, target_shape, anti_aliasing=True)

def load_boxes(path):
    '''
    Loads all arrays from a compressed `.npz` file and returns them as a list.

    Parameters:
        path (str): Path to the `.npz` file containing saved image or mask boxes.

    Returns:
        list: A list of NumPy arrays, one for each saved box.

    Notes:
        - The file must be saved using `np.savez` or `np.savez_compressed`.
        - Uses `allow_pickle=True` to support loading object arrays if present.
        - Preserves the original order of keys as stored in the `.npz` file.
    '''
    data = np.load(path, allow_pickle=True)
    return [data[key] for key in data]


def process_box(args):
    '''
    Detects blobs in a single image box, saves the results, and returns coordinates and radii.

    Parameters:
        args (tuple): A tuple containing:
            - box_id (int): The unique identifier for the box (used in the output filename).
            - box (numpy.ndarray): The image data for the box to process.

    Returns:
        tuple:
            - coordinates (numpy.ndarray): Array of detected (x, y) coordinates. Shape: (N, 2).
            - radii (numpy.ndarray): Array of corresponding blob radii. Shape: (N,).

    Side Effects:
        - Saves the blob detection result to `Blobs/{box_id}.npz` using `save_blobs()`.
        - Prints status messages depending on the number of blobs detected:
            ✓ for exactly one blob,
            X for none or more than one blob.

    Notes:
        - Uses the `detect_blobs()` function for blob detection.
        - Ensures returned arrays have consistent shape using `np.atleast_2d` and `np.ravel`.
    '''
    box_id, box = args
    blob, coordinates, radii = detect_blobs(box)

    # removed saving blobs seperately as I just wanto save all radii and coordinates at the end     

    if len(blob) == 1:
        print(f"Saved blob for box {box_id}")
        model_found = True
    elif len(blob) == 0:
        print(f"X No blobs detected in box {box_id}")
        model_found = False
    else:
        print(f"X Unexpected blob count in box {box_id} (count: {len(blob)})")

    coordinates = np.atleast_2d(coordinates)
    radii = np.ravel(radii)
    return coordinates, radii, model_found

def detect_blob_in_all_boxes(mask_boxes, selection_csv):
    '''
    Detects blobs in a list of image or mask boxes in parallel and collects their coordinates and radii.

    Parameters:
        mask_boxes (list): A list of 2D NumPy arrays, each representing an individual box (typically cropped from a mask or image).

    Returns:
        tuple:
            - all_coordinates (list): A list of NumPy arrays, each containing (x, y) coordinates of blobs detected in the corresponding box.
            - all_radii (list): A list of NumPy arrays, each containing the radii of detected blobs in the corresponding box.

    Notes:
        - Uses `ThreadPoolExecutor` to parallelize processing with `process_box()`.
        - Box IDs are assigned starting from 1 and used to save blob results as `Blobs/{box_id}.npz`.
        - Preserves the original order of boxes in the returned lists.
        - Each box is expected to contain at most one detectable blob. Multiple or no blobs will trigger warning messages from `process_box()`.
    '''
    # load manual selection 
    selected_boxes_ids = load_allowed_ids(selection_csv)

    all_coordinates = []
    all_radii = []
    model_count = 0

    # only select boxes that are confirmed as pass
    print(selected_boxes_ids)
    box_args = [(i + 1, box) for i, box in enumerate(mask_boxes) if (i + 1) in selected_boxes_ids] # (box_id, box)

    with ThreadPoolExecutor() as executor:
        results = executor.map(process_box, box_args)  # preserves order

    for coords, rads, model_found in results:
        all_coordinates.append(coords)
        all_radii.append(rads)
        if model_found:
            model_count = model_count + 1

    return all_coordinates, all_radii, model_count


def grid_split(directory, markers, conditions, repeats, output_list=False):
    '''
    Assumes ALL cropped images are already made and only need to be reloaded 
    List of images (index according to marker where 1:DAPI, 2:SOX2, 3:BRA, 4:GATA3) KEEP THE SAME ALWAYS
        
        img_DAPI, img_SOX2, img_BRA, img_GATA3 = images

        mask_DAPI, mask_SOX2, mask_BRA, mask_GATA3 = masks

    '''
    mask_boxes_list = []
    img_boxes_list = []

    for condition in conditions:
        for repeat in repeats:

            # reload cropped images that are already made 
            images, masks = load_image_and_mask(directory, repeat, condition, markers)

            for i, marker in enumerate(markers):
                
                # draw grid
                grid_mask = draw_grid(images[i])
                print(f"finished making grid for repeat: {repeat}, condition: {condition}, marker: {marker}")

                # make mask boxes per repeat and save 
                print(f"------------------------Starting mask boxes for repeat: {repeat}, condition: {condition}, marker: {marker}")
                mask_boxes = analyse_by_grid(directory, masks[i], grid_mask, marker, condition, repeat, mask=True, rescale_switch = False)
                print(len(mask_boxes), "should be 676")
                print(f"------------------------finished mask boxes for repeat: {repeat}, condition: {condition}, marker: {marker}")

                # make image boxes per repeat and save
                print(f"------------------------Starting image boxes for repeat: {repeat}, condition: {condition}, marker: {marker}")
                img_boxes = analyse_by_grid(directory, images[i], grid_mask, marker, condition, repeat, mask=False, rescale_switch = False)
                print(len(img_boxes), "should be 676")
                print(f"------------------------finished image boxes for repeat: {repeat}, condition: {condition}, marker: {marker}")

                mask_boxes_list.append(mask_boxes)
                img_boxes_list.append(img_boxes)

    if output_list:
        return mask_boxes_list, img_boxes_list


def select_gastruloids(directory, markers, conditions, repeats):

    for repeat in repeats:
        for condition in conditions:
            for marker in markers:
        
                # getting directories
                mask_boxes_path = f"{directory}/{repeat}/boxes_npz/mask_{marker}_{condition}.npz"
                image_boxes_path = f"{directory}/{repeat}/boxes_npz/img_{marker}_{condition}.npz"

                blobs_output_directory = f"{directory}/{repeat}/blobs_npz"
                blobs_output_path = f"{blobs_output_directory}/{marker}_{condition}.npz"
                os.makedirs(blobs_output_directory, exist_ok=True)
                
                # A. RELOAD: IMAGE, MASK BOXES 
                print(f"------------------------Starting Load mask boxes for repeat: {repeat}, condition: {condition}, marker: {marker}")
                mask_boxes = load_boxes(mask_boxes_path) 
                print(f"------------------------Finished Load mask boxes for repeat: {repeat}, condition: {condition}, marker: {marker}")

                print(f"------------------------Starting Load images boxes for repeat: {repeat}, condition: {condition}, marker: {marker}")
                img_boxes = load_boxes(image_boxes_path)
                print(f"------------------------Finished Load images boxes for repeat: {repeat}, condition: {condition}, marker: {marker}")
                
                selection_output_dir = f"{directory}/{repeat}/selection"
                os.makedirs(selection_output_dir, exist_ok=True)
                selection_csv = f"{selection_output_dir}/img_DAPI_{condition}.csv"


                # # B. MANUALLY REVIEW WHICH BOXES TO INCLUDE AND WHICH TO LEAVE OUT
                print(f"------------------------Starting Manual Selection for repeat: {repeat}, condition: {condition}, marker: {marker}")
                images_folder = f"{directory}/{repeat}/boxes_tiff/img_{marker}_{condition}"
                manual_selection(images_folder, selection_csv)
                print(f"------------------------Finished Manual Selection for repeat: {repeat}, condition: {condition}, marker: {marker}")


                # C. DETECT BLOB IN EACH MASK BOX AND SAVE
                print(f"------------------------Starting Blob detection for repeat: {repeat}, condition: {condition}, marker: {marker}")
                all_coordinates, all_radii, model_count = detect_blob_in_all_boxes(mask_boxes, selection_csv)
                print(f"Models detected: {model_count}/676")
                print(f"------------------------Finished Blob detection for repeat: {repeat}, condition: {condition}, marker: {marker}")


                # save all outputs for easy visualization and further analysis
                print(f"------------------------Saving Blob detection outputs for repeat: {repeat}, condition: {condition}, marker: {marker} in {blobs_output_path}")
                
                print(f"------------------------{blobs_output_path} info:")
                print("mask_boxes:", type(mask_boxes), len(mask_boxes), type(mask_boxes[0]))
                print("img_boxes:",type(img_boxes), len(img_boxes), type(img_boxes[0]))
                print("all_coordinates:",type(all_coordinates), len(all_coordinates), type(all_coordinates[0]))
                print("all_radii:",type(all_radii), len(all_radii), type(all_radii[0]))
                print(f"------------------------")
                
                np.savez(blobs_output_path,
                        all_coordinates=np.array(all_coordinates, dtype=object),
                        all_radii=np.array(all_radii, dtype=object))
                
    boxes_tiff_selected(directory, repeats, conditions, markers)

def boxes_tiff_selected(directory, repeats, conditions, markers):
    for repeat in repeats:
        for condition in conditions:
            for marker in markers:
                print(f"------------Repeat: {repeat}, Condition: {condition}, marker: {marker}")

                # output directory for images 
                img_output_dir = f"{directory}/{repeat}/boxes_tiff_selected/img_{marker}_{condition}"
                os.makedirs(img_output_dir, exist_ok=True)

                # get selected boxes
                selected_boxes_ids = load_allowed_ids(f'{directory}/{repeat}/selection/img_DAPI_{condition}.csv')
                selected_boxes_ids.sort()
                selected_set = set(str(num) for num in selected_boxes_ids)

                boxes_tiffs_path = f"{directory}/{repeat}/boxes_tiff/img_{marker}_{condition}"  

                # Get all files in the directory
                all_files = os.listdir(boxes_tiffs_path)

                # Sort files numerically by their base name
                all_files.sort(key=lambda x: int(os.path.splitext(x)[0]))

                counter = 1
                for filename in all_files:
                    name, ext = os.path.splitext(filename)
                    if name in selected_set:
                        counter_str = str(counter)
                        src_path = os.path.join(boxes_tiffs_path, filename)
                        dst_path = os.path.join(img_output_dir, f"{counter_str}.tiff")
                        shutil.copy2(src_path, dst_path)
                        print(f"Copied {filename} into {counter}")
                        counter += 1
