
from pathlib import Path
from tqdm import tqdm
import pyclesperanto_prototype as cle
from tifffile import imread, imwrite
import pandas as pd
import numpy as np
from skimage.measure import regionprops_table
from skimage.color import rgb2gray
from collections import defaultdict
from scipy import stats
import re

def list_images (directory_path, format=None):

    # Transform directory string into a Path object
    directory_path = Path(directory_path)

    # Create an empty list to store all image filepaths within the dataset directory
    images = []

    # Append file_path
    for file_path in directory_path.glob(f"*{format}"):
        if "f00d0" not in file_path.stem: # ignore tiled images from dataset
            images.append(str(file_path))

    return images

def simulate_cytoplasm(nuclei_labels, dilation_radius=2, erosion_radius=0):

    if erosion_radius >= 1:

        # Erode nuclei_labels to maintain a closed cytoplasmic region when labels are touching (if needed)
        eroded_nuclei_labels = cle.erode_labels(nuclei_labels, radius=erosion_radius)
        eroded_nuclei_labels = cle.pull(eroded_nuclei_labels)
        nuclei_labels = eroded_nuclei_labels

    # Dilate nuclei labels to simulate the surrounding cytoplasm
    cyto_nuclei_labels = cle.dilate_labels(nuclei_labels, radius=dilation_radius)
    cytoplasm = cle.pull(cyto_nuclei_labels)

    # Create a binary mask of the nuclei
    nuclei_mask = nuclei_labels > 0

    # Set the corresponding values in the cyto_nuclei_labels array to zero
    cytoplasm[nuclei_mask] = 0

    return cytoplasm

def grayscale_skimage(img):

    # Case 1: TIFF loads as (3, H, W)
    if img.ndim == 3 and img.shape[0] == 3 and img.shape[-1] != 3:
        img = np.moveaxis(img, 0, -1)  # → (H, W, 3)
    
    # Case 2: Already (H, W, 3) → fine
    if img.ndim == 3 and img.shape[-1] == 3:
        gray01 = rgb2gray(img)  # float64 in [0,1]
    
    else:
        raise ValueError(f"Unexpected image shape {img.shape}")

    # Restore original dtype
    if np.issubdtype(img.dtype, np.integer):
        maxval = np.iinfo(img.dtype).max
        gray = (gray01 * maxval).astype(img.dtype)
    else:
        gray = gray01.astype(img.dtype)

    return gray

def generate_multichannel_tif(data_folder):

    # Parse all paths to the single channel image files
    images = list_images(data_folder, format="tif")

    # Create a folder to store the resulting multichannel tif files
    output_dir = Path(data_folder) / "processed_tiffs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Provide feedback to user
    print("\nProcessing input single channel .tif into multichannel .tif files \n")
    print(f"Processed .tif files will be saved under: {output_dir}\n")

    # Regex to extract well, frame, channel
    pattern = re.compile(
        r'.*_(?P<well>[A-Z]\d{2})f(?P<frame>\d{2})d(?P<channel>\d)\.TIF$',
        re.IGNORECASE
    )

    # Group image filenames by (well, frame), and within each group
    # map channel number -> corresponding image file path
    groups = defaultdict(dict)
    for f in images:
        m = pattern.match(f)
        if m:
            groups[(m.group('well'), m.group('frame'))][int(m.group('channel'))] = f

    # Build & save stacks
    for (well, frame), channels in tqdm(groups.items()):
        expected = [1, 2, 4]
        if not all(c in channels for c in expected):
            print(f"Skipping incomplete {well}f{frame}")
            continue

        slices = []
        for c in expected:
            rgb = imread(channels[c], series=None)
            gray = grayscale_skimage(rgb)
            slices.append(gray)

        stack = np.stack(slices, axis=0)
        out_path = output_dir / f"{well}f{frame}.tif"
        imwrite(out_path, stack)

def remap_labels(nuclei_labels, cytoplasm_labels):

    # Label-to-label remapping: each nucleus inherits the cytoplasm label value it lies in
    # Might cause some issues with multinucleated cells (will try to filter them out later)

    out = np.zeros_like(nuclei_labels)

    for nid in np.unique(nuclei_labels):
        if nid == 0:
            continue
        
        mask = nuclei_labels == nid
        cyto_vals = cytoplasm_labels[mask]
        cyto_vals = cyto_vals[cyto_vals != 0]  # ignore background
        
        if len(cyto_vals) == 0:
            continue
        
        cyto_id = stats.mode(cyto_vals, keepdims=False).mode
        out[mask] = cyto_id

    return out

def extract_img_metadata (img_filepath, verbose = False):
    
    # Extract image metadata from filename
    field_of_view = Path(img_filepath).stem.split("f")[1]
    well_id = Path(img_filepath).stem.split("f")[0]

    # Create a dictionary containing all image descriptors
    descriptor_dict = {"well_id": well_id, "FOV": field_of_view}

    if verbose:

        print(f"Visualizing well: {well_id}, FOV: {field_of_view}")

    return descriptor_dict

def rename_cyto_nuclei_cols(props_df, compartment, scikit_props):

    for prop in scikit_props:
        if prop != "label":
            # Rename intensity_mean column to indicate the specific cellular compartment
            props_df.rename(columns={prop: f"{compartment}_{prop}"}, inplace=True)

    return props_df

def extract_features(img, nuclei_remapped, cytoplasm_labels, descriptor_dict, scikit_props=["label", "intensity_mean", "area"]):

    # Create an empty list to hold each props_df (for nuclei and cytoplasm)
    props_list = []

    # Extract nuclei features
    props = regionprops_table(label_image=nuclei_remapped,
                            intensity_image=img[0],
                            properties=scikit_props)

    # Convert to dataframe
    props_df = pd.DataFrame(props)

    # Rename feature column to indicate the specific cellular compartment
    props_df = rename_cyto_nuclei_cols(props_df, compartment="nuclei", scikit_props=scikit_props)

    # Append each props_df to props_list
    props_list.append(props_df)

    # Extract cytoplasm features
    props = regionprops_table(label_image=cytoplasm_labels,
                            intensity_image=img[0],
                            properties=scikit_props)

    # Convert to dataframe
    props_df = pd.DataFrame(props)

    # Rename intensity_mean column to indicate the specific image
    props_df = rename_cyto_nuclei_cols(props_df, compartment="cytoplasm", scikit_props=scikit_props)

    # Append each props_df to props_list
    props_list.append(props_df)

    # Initialize the df with the first df in the list
    props_df = props_list[0]
    # Start looping from the second df in the list
    for df in props_list[1:]:
        props_df = props_df.merge(df, on="label")

    # Add each key-value pair from descriptor_dict to props_df at the specified position
    insertion_position = 0
    for key, value in descriptor_dict.items():
        props_df.insert(insertion_position, key, value)
        insertion_position += 1  # Increment position to maintain the order of keys in descriptor_dict

    # Calculate ratio of cytoplasm to nuclei area to filter out incorrectly segmented entities
    props_df["cyto_to_nuclei_ratio"] = props_df["cytoplasm_area"] / props_df["nuclei_area"]

    return props_df