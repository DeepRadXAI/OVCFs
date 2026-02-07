import os
import json
from typing import Tuple
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import subfiles

def get_identifiers_from_splitted_files(folder: str):
    """
    This function will return the unique identifiers from the files in the folder.
    It assumes the files are named as follows: <identifier>_0000.nii.gz, <identifier>_0001.nii.gz, etc.
    """
    # Assuming the suffix length to remove is 12 characters ('_0000.nii.gz')
    return np.unique([i[:-7] for i in subfiles(folder, suffix='.nii.gz', join=False)])

def generate_dataset_json(output_file: str, imagesTr_dir: str, imagesTs_dir: str, modalities: Tuple,
                          labels: dict, dataset_name: str, license: str = "hands off!", dataset_description: str = "",
                          dataset_reference="", dataset_release='0.0'):
    train_identifiers = get_identifiers_from_splitted_files(imagesTr_dir)

    if imagesTs_dir is not None:
        test_identifiers = get_identifiers_from_splitted_files(imagesTs_dir)
    else:
        test_identifiers = []

    json_dict = {}
    json_dict['name'] = dataset_name
    json_dict['description'] = dataset_description
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = dataset_reference
    json_dict['licence'] = license
    json_dict['release'] = dataset_release
    json_dict['modality'] = {str(i): modalities[i] for i in range(len(modalities))}
    json_dict['labels'] = {str(i): labels[i] for i in labels.keys()}

    json_dict['numTraining'] = len(train_identifiers)
    json_dict['numTest'] = len(test_identifiers)
    json_dict['training'] = [
        {'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in train_identifiers]
    json_dict['test'] = ["./imagesTs/%s.nii.gz" % i for i in test_identifiers]

    if not output_file.endswith("dataset.json"):
        print("WARNING: output file name is not dataset.json! This may be intentional or not. You decide. "
              "Proceeding anyways...")
    save_json(json_dict, os.path.join(output_file))

def save_json(obj, file_path):
    with open(file_path, 'w') as f:
        json.dump(obj, f, indent=4)

# Set the paths and parameters
base_path = "/root/autodl-tmp/U-Mamba/DATASET/nnUNet_raw/Task04_OVCFs" 
output_file = os.path.join(base_path, "dataset.json")
imagesTr_dir = os.path.join(base_path, "imagesTr")
imagesTs_dir = os.path.join(base_path, "imagesTs")
modalities = ("CT",)  # Modify according to your data
labels = {
    0: "background",
    1: "hemorrhage"
}
dataset_name = "OVCFs_Segmentation"
license = "MIT"
dataset_description = "OVCFs Segmentation Dataset"
dataset_reference = ""
dataset_release = "1.0"

# Generate dataset.json
generate_dataset_json(output_file, imagesTr_dir, imagesTs_dir, modalities, labels, dataset_name, license, dataset_description, dataset_reference, dataset_release)

print(f"dataset.json has been created at {output_file}")
