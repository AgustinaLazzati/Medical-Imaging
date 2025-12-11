import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

# Custom imports (must be importable in your PYTHONPATH)
from dataset import HelicoPatientsHoldout   #
from Models.AEmodels import AutoEncoderCNN, VAECNN
from train_conv_ae import AEConfigs
from train_conv_vae import VAEConfigs
from patient_diagnosis import get_patient_errors  #PATCH-TO-PATIENT ERROR FUNCTION (SCORE FUNCTION)

# ----------------------------
# CONFIGURATION
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "/fhome/vlia01/Medical-Imaging/slurm_output/config_three.pth"  # adjust if needed
BATCH_SIZE = 264
MODEL_NAME = "Autoencoder"  # "Autoencoder" or "Variational Autoencoder"
SAVE_FIG = True
METRIC = 'hsv_red'    # 'hsv_red' or 'mae_red' or 'mse'
RESULTS_DIR = "/fhome/vlia01/Medical-Imaging/HoldOut"
os.makedirs(RESULTS_DIR, exist_ok=True)
NUM_WORKERS = 4
MAX_PATCHES= 300

# ----------------------------
# 1. MODEL LOADER
# ----------------------------
def load_model(config_id, model_path=None, model_name="Autoencoder"):
    print(f"Loading {model_name} from {model_path}...")
    if model_name == "Autoencoder":
        net_paramsEnc, net_paramsDec, inputmodule_paramsEnc, inputmodule_paramsDec = AEConfigs(config_id)
        model = AutoEncoderCNN(inputmodule_paramsEnc, net_paramsEnc, inputmodule_paramsDec, net_paramsDec)
    elif model_name == "Variational Autoencoder":
        net_paramsEnc, net_paramsDec, inputmodule_paramsEnc, inputmodule_paramsDec, net_paramsRep = VAEConfigs(config_id)
        model = VAECNN(inputmodule_paramsEnc, net_paramsEnc, inputmodule_paramsDec, net_paramsDec, net_paramsRep)
    else:
        raise ValueError("Invalid model name")

    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    else:
        print(f"Warning: Weights not found at {model_path}")

    model.to(DEVICE)
    model.eval()
    return model


# -----------------
# HELPER FUNCTION
# ----------------- 
def label_to_binary(label):
    if isinstance(label, str):
        if label.upper() in ['ALTA', 'BAIXA']:
            return 1
        else:  # NEGATIVA or any other negative string
            return 0
    else:  # numeric label
        return 0 if label == -1 else 1

    
# ----------------------------
# 4. MAIN
# ----------------------------
def main():
    # A. Load Data & Model
    model = load_model(config_id="3", model_path=MODEL_PATH, model_name=MODEL_NAME)

    # Load FULL datasets (load_ram=False to save memory)
    patient_dataset = HelicoPatientsHoldout()
    print(f"Loaded patient dataset with {len(patient_dataset)} patients.")
    
    dataloader = DataLoader(
        patient_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS
    )
    
    print(f'Calculating patient patch scores, using maximum of {MAX_PATCHES} patches')
    # 3.3 Compute patient scores
    patient_scores, patient_labels = get_patient_errors(
        model,
        dataloader,
        patch_thr=0.0,
        max_patches=MAX_PATCHES,
        model_name=MODEL_NAME
    )
    
    scores = np.array(list(patient_scores.values()))
    labels = np.array(patient_labels)
    
    # ---------------------------
    #aply all kfolds thresholds, that are in cvs 
    # ---------------------------
    
    
    #----------------------------
    # MAX VOTING to aggregate the results of passing those thresholds
    # ---------------------------
    
    
    
    
    
    
if __name__ == "__main__":
    main()