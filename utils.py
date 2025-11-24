
from pathlib import Path
import pyclesperanto_prototype as cle

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