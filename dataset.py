from typing import Any
from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd
import yaml
import numpy as np
import torch
import torchvision
import cv2

from torchvision import transforms

from tqdm import tqdm

from typing import List, Optional, Tuple, Literal
from enum import Enum

import lovely_tensors as lt
lt.monkey_patch()

DEFAULT_PATH = "/fhome/vlia/HelicoDataSet/"

def delete_alpha_channel(image: torch.Tensor) -> torch.Tensor:
    if image.shape[0] == 4:
        image = image[:3, :, :]
    return image

class Patients(Enum):
    LOW = "BAIXA"
    HIGH = "ALTA"
    NEGATIVE = "NEGATIVA"

def cropped_collator(batch: list[dict]) -> dict[str, Any]:
    LABEL_MAP = {"NEGATIVA": 0, "BAIXA": 1, "ALTA": 2}

    images = [item["image"] for item in batch]
    labels_str = [item["label"] for item in batch]

    batched_images = torch.stack(images, dim=0)

    try:
        labels_int = [LABEL_MAP[label] for label in labels_str]
    except KeyError as e:
        print(f"Error: Unknown label encountered in batch: {e}. Available labels: {list(LABEL_MAP.keys())}")
        raise

    batched_labels = torch.tensor(labels_int, dtype=torch.long)

    return {"image": batched_images, "label": batched_labels}

def annotated_collate(batch):
    batch = [item for item in batch if item is not None]
    
    if len(batch) == 0:
        return torch.tensor([]), torch.tensor([])

    return torch.utils.data.dataloader.default_collate(batch)

class HelicoCropped(Dataset):
    def __init__(self, target_id = "NEGATIVA", load_ram: bool=False):
        super().__init__()
        xlsx_path = os.path.join(DEFAULT_PATH, "PatientDiagnosis.csv")
        data = pd.read_csv(xlsx_path)

        patient_ids = data["CODI"].astype(str).to_numpy()
        labels = data["DENSITAT"].astype(str).to_numpy()

        match target_id:
            case "NEGATIVA":
                negative_ids = data[data["DENSITAT"] == "NEGATIVA"]["CODI"].astype(str).to_numpy()
                selected_indices = np.where(labels == "NEGATIVA")[0]
            case "BAIXA":
                low_ids = data[data["DENSITAT"] == "BAIXA"]["CODI"].astype(str).to_numpy()
                selected_indices = np.where(labels == "BAIXA")[0]
            case "ALTA":
                high_ids = data[data["DENSITAT"] == "ALTA"]["CODI"].astype(str).to_numpy()
                selected_indices = np.where(labels == "ALTA")[0]
            case _:
                raise ValueError(f"Invalid target_id: {target_id}. Must be one of 'NEGATIVA', 'BAIXA', 'ALTA'.")

        images_path = os.path.join(DEFAULT_PATH, "CrossValidation", "Cropped")
        images_subfolders = os.listdir(images_path)

        self.samples = []
        for images_subfolder in images_subfolders:
            patient_id = images_subfolder.split("_")[0]

            if patient_id in patient_ids[selected_indices]:
                # store the image paths and labels
                image_filenames = os.listdir(os.path.join(images_path, images_subfolder))
                for image_filename in image_filenames:
                    image_path = os.path.join(images_path, images_subfolder, image_filename)

                    # ensure that is a png image
                    if not image_filename.lower().endswith('.png'):
                        continue

                    label = labels[np.where(patient_ids == patient_id)[0][0]]
                    self.samples.append((image_path, label, patient_id))

        self.load_ram = load_ram

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),      

            # does several stuff:
            # - [0, 255] -> [0, 1]
            # - (H x W x C) -> (C x H x W) 
            transforms.ToTensor(),
        ])            

        if self.load_ram:
            self.ram_data = []
            for image_path, label, pid in tqdm(self.samples, desc="Loading data into RAM"):
                image = Image.open(image_path).convert("RGB")
                image = self.transform(image)
                self.ram_data.append((image, label, pid))

    def __getitem__(self, index) -> Any:
        if self.load_ram:
            image, label, _ = self.ram_data[index]
            
        else:
            image_path, label, _ = self.samples[index]
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image)

        return {"image": image, "label": label}

    def __len__(self) -> int:
        return len(self.samples)


class HelicoPatients(Dataset):
    def __init__(self):
        super().__init__()
        xlsx_path = os.path.join(DEFAULT_PATH, "PatientDiagnosis.csv")
        data = pd.read_csv(xlsx_path)

        patient_ids = data["CODI"].astype(str).to_numpy()
        labels = data["DENSITAT"].astype(str).to_numpy()
        self.id_to_label = dict(zip(patient_ids, labels))

        images_path = os.path.join(DEFAULT_PATH, "CrossValidation", "Cropped")
        images_subfolders = os.listdir(images_path)

        patient_groups = {}

        for subfolder in images_subfolders:
            patient_id = subfolder.split("_")[0]

            if patient_id in self.id_to_label:
                if patient_id not in patient_groups:
                    patient_groups[patient_id] = []

                subfolder_path = os.path.join(images_path, subfolder)
                image_filenames = os.listdir(subfolder_path)

                for img_name in image_filenames:
                    if img_name.lower().endswith('.png'):
                        full_path = os.path.join(subfolder_path, img_name)
                        patient_groups[patient_id].append(full_path)
        self.samples = []
        for pid, paths in patient_groups.items():
            if len(paths) > 0:
                label = self.id_to_label[pid]
                self.samples.append((pid, label, paths))

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),      

            # does several stuff:
            # - [0, 255] -> [0, 1]
            # - (H x W x C) -> (C x H x W) 
            transforms.ToTensor(),
        ])            

    def __getitem__(self, index) -> Any:
        pid, label, paths = self.samples[index]

        img_tensors = []
        for p in paths:
            img = Image.open(p).convert("RGB")
            img = self.transform(img)
            img_tensors.append(img)

        images = torch.stack(img_tensors)

        return {"images": images, "label": label, "p_id": pid}

    def __len__(self) -> int:
        return len(self.samples)


class HelicoAnnotated(Dataset):
    def __init__(self, patient_id: bool=False, only_negative: bool=False, only_positive: bool=False, load_ram: bool=False):
        xlsx_path = os.path.join(DEFAULT_PATH, "HP_WSI-CoordAnnotatedAllPatches.xlsx")
        data = pd.read_excel(xlsx_path)
        patient_ids = data["Pat_ID"].astype(str).to_numpy()
        section_ids = data["Section_ID"].astype(str).to_numpy()
        window_ids = data["Window_ID"].astype(str).to_numpy()
        labels = data["Presence"].astype(int).to_numpy()

        if only_negative:
            indices = np.where(labels == -1)[0]
            patient_ids = patient_ids[indices]
            section_ids = section_ids[indices]
            window_ids = window_ids[indices]
            labels = labels[indices]

            print(f"Number of negative samples: {len(labels)}")
        elif only_positive:
            indices = np.where(labels == 1)[0]
            patient_ids = patient_ids[indices]
            section_ids = section_ids[indices]
            window_ids = window_ids[indices]
            labels = labels[indices]
            print(f"Number of positive samples: {len(labels)}")


        else:
            print(f"Total number of samples: {len(labels)}")

        self.samples = []  # (image_path, label, patient_id)
        for pid, sid, wid, label in zip(patient_ids, section_ids, window_ids, labels):

            split_ = wid.split("_")
            if len(split_) > 1:
                number = int(split_[0])
                string = split_[1]
                wid = f"{number:05d}_{string}"
            else:
                wid = f"{int(wid):05d}"

            image_path = os.path.join(DEFAULT_PATH, "CrossValidation", "Annotated", f"{pid}_{sid}", f"{wid}.png")
            self.samples.append((image_path, label, pid))
            
        self.load_ram = load_ram
        self.transform = transforms.Compose([
          transforms.Resize((256, 256)),      
  
          # does several stuff:
          # - [0, 255] -> [0, 1]
          # - (H x W x C) -> (C x H x W) 
          transforms.ToTensor(),
          ])        
        
        if self.load_ram:
            self.ram_data = []
            for image_path, label, pid in tqdm(self.samples, desc="Loading data into RAM"):
                image = torchvision.io.read_image(image_path)
                image = delete_alpha_channel(image)

                self.ram_data.append((image, label, pid))

    def __getitem__(self, index) -> Any:
        if self.load_ram:
            image, label, _ = self.ram_data[index]
        else:
            try:
                image_path, label, _ = self.samples[index]
                image = Image.open(image_path).convert("RGB")
                image = self.transform(image)
                
            except Exception as e:
                #ME MOLESTABA TANTO EL PRINT AJAJAJJA despues lo agregamos
                #print(f"Error loading image at index {index}: {e}")
                return None

        return image, label

    def __len__(self) -> int:
        return len(self.samples)

class HelicoMixed(Dataset):
    """
    A unified dataset class that combines Benign and Malignant samples
    for Triplet Loss training.
    
    - Benign (Negative) samples are assigned Label 0
    - Malignant (Positive) samples are assigned Label 1
    """
    def __init__(self, load_ram: bool = False):
        super().__init__()
        # Initialize sub-datasets
        self.benign_ds = HelicoAnnotated(only_negative=True, load_ram=load_ram)
        self.malignant_ds = HelicoAnnotated(only_positive=True, load_ram=load_ram)
        
    def __len__(self):
        return len(self.benign_ds) + len(self.malignant_ds)
    
    def __getitem__(self, index):
        # Determine which dataset to pull from based on index
        if index < len(self.benign_ds):
            data = self.benign_ds[index]
            if data is None: return None
            img, _ = data
            return img, 0 # Enforce Label 0 for Benign
        else:
            # Shift index for malignant dataset
            adj_index = index - len(self.benign_ds)
            data = self.malignant_ds[adj_index]
            if data is None: return None
            img, _ = data
            return img, 1 # Enforce Label 1 for Malignant

if __name__ == "__main__":
    dataset = HelicoPatients()
    print(dataset[1])