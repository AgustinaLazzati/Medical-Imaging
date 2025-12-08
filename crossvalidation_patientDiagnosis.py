# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score
import cv2  
from sklearn.model_selection import StratifiedGroupKFold

# Custom imports
from dataset import HelicoPatients  #CROPPED PATIENT-LEVEL dataset
from Models.AEmodels import AutoEncoderCNN, VAECNN
from train_conv_ae import AEConfigs
from train_conv_vae import VAEConfigs
import pandas as pd

# ----------------------------
# CONFIGURATION
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "/fhome/vlia01/Medical-Imaging/slurm_output/config_three.pth"
BATCH_SIZE = 128   #NEED TO ADJUST??
MODEL_NAME = "Autoencoder" # "Autoencoder" or "Variational Autoencoder"
SAVE_FIG = True
K_FOLD = 3
PATCH_THR = 0.000473   # patch-levelthreshold from previous ROC_CURVE.PY
max_patches= 10

# ----------------------------
# 1. LOAD MODEL
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
    
    
def label_to_binary(label):
    if isinstance(label, str):
        if label.upper() in ['ALTA', 'BAIXA']:
            return 1
        else:  # NEGATIVA or any other negative string
            return 0
    else:  # numeric label
        return 0 if label == -1 else 1


# ----------------------------
# 2. PATCH-TO-PATIENT ERROR FUNCTION (adapted for HelicoPatients)
# ----------------------------
def get_patient_errors(model, dataloader, patch_thr=PATCH_THR, model_name=MODEL_NAME, max_patches=32):
    patient_scores = {}
    patient_labels = {}

    with torch.no_grad():
        for batch in dataloader:
            images = batch['images'].to(DEVICE).float()  # shape [1, num_patches, C, H, W]
            images = images.squeeze(0)
            label = batch['label'][0]
            pid = batch['p_id'][0]

            # randomly sample patches if there are too many
            if len(images) > max_patches:
                idxs = np.random.choice(len(images), max_patches, replace=False)
                images = images[idxs]

            if images.max() > 1:
                images /= 255.0

            recon = model(images) if model_name=="Autoencoder" else model(images)[0]

            patch_errors = []
            for orig_img, rec_img in zip(images, recon):
                orig_np = (orig_img.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
                rec_np = (rec_img.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)

                orig_hsv = cv2.cvtColor(orig_np, cv2.COLOR_RGB2HSV)
                rec_hsv = cv2.cvtColor(rec_np, cv2.COLOR_RGB2HSV)

                orig_hue = orig_hsv[:, :, 0]
                orig_sat = orig_hsv[:, :, 1] / 255.0
                rec_hue = rec_hsv[:, :, 0]
                rec_sat = rec_hsv[:, :, 1] / 255.0

                orig_mask = ((orig_hue <= 10) | (orig_hue >= 170)) & (orig_sat > 0.04)
                rec_mask = ((rec_hue <= 10) | (rec_hue >= 170)) & (rec_sat > 0.04)

                error_val = np.sum(orig_mask & (~rec_mask)) / orig_mask.size
                patch_errors.append(error_val)

            patch_errors = np.array(patch_errors)
            pct_positive = np.mean(patch_errors >= patch_thr)
            patient_scores[pid] = pct_positive
            patient_labels[pid] = label_to_binary(label)

    patient_labels_list = [patient_labels[pid] for pid in patient_scores.keys()]
    return patient_scores, patient_labels_list

    
# -----------------------------------------
# 6. PATIENT-LEVEL K-FOLD CROSS VALIDATION
# -----------------------------------------
def kfold_patient_cv(model, dataset, k=K_FOLD, max_patches=32, num_workers=4):
    # Ensure patient_ids and labels are aligned with dataset.samples indices
    patient_ids = np.array([dataset.samples[i][0] for i in range(len(dataset))])
    labels = np.array([label_to_binary(dataset.samples[i][1]) for i in range(len(dataset))])

    skf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=42)

    precision_list, recall_list, f1_list = [], [], []
    all_predictions = []

    fold_num = 1
    plt.figure(figsize=(8,6))
    
    # train_idx and test_idx are the INDICES into the dataset.samples list.
    # Since patient_ids and labels were built in the same order, these are the correct indices.
    for train_idx, test_idx in skf.split(patient_ids, labels, groups=patient_ids): 
        print(f"\nFold {fold_num}/{k}")

        # Use the indices directly to create the subsets.
        # train_patients and test_patients variables are no longer needed
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        test_dataset = torch.utils.data.Subset(dataset, test_idx)
        
        print('Loading patient train and test DataLoaders')
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=num_workers)  # batch_size=1: one patient at a time
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

        # TRAIN fold: compute ROC and threshold
        train_scores, train_labels = get_patient_errors(model, train_loader, max_patches=max_patches)
        train_scores_array = np.array(list(train_scores.values()))
        train_labels_array = np.array(train_labels)

        fpr, tpr, thresholds = roc_curve(train_labels_array, train_scores_array)
        youden_idx = np.argmax(tpr - fpr)
        optimal_thr = thresholds[youden_idx]
        print(f"Optimal patient threshold (train fold): {optimal_thr:.6f}")

        # TEST fold
        test_scores, test_labels = get_patient_errors(model, test_loader, max_patches=max_patches)
        test_scores_array = np.array(list(test_scores.values()))
        test_labels_array = np.array(test_labels)
        test_preds = (test_scores_array >= optimal_thr).astype(int)

        # Metrics
        precision = precision_score(test_labels_array, test_preds, zero_division=0)
        recall = recall_score(test_labels_array, test_preds, zero_division=0)
        f1 = f1_score(test_labels_array, test_preds, zero_division=0)
        print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

        # Store predictions
        for pid, score, pred, label in zip(test_scores.keys(), test_scores_array, test_preds, test_labels_array):
            all_predictions.append({'patient_id': pid, 'score': score, 'pred': pred, 'label': label, 'fold': fold_num})

        # Plot ROC per fold
        fold_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Fold {fold_num} (AUC={fold_auc:.3f})')

        fold_num += 1

    # Plot ROC
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Patient-level ROC Curve per Fold')
    plt.legend(loc='lower right')
    if SAVE_FIG:
        plt.savefig("patient_roc_folds.png", dpi=300)
    plt.show() # Note: This will only work in an interactive environment.

    # Summary
    print("\n=== 10-Fold CV Summary ===")
    print(f"Precision: {np.mean(precision_list):.3f} ± {np.std(precision_list):.3f}")
    print(f"Recall:    {np.mean(recall_list):.3f} ± {np.std(recall_list):.3f}")
    print(f"F1-score:  {np.mean(f1_list):.3f} ± {np.std(f1_list):.3f}")

    df_predictions = pd.DataFrame(all_predictions)
    df_predictions.to_csv("crossvalidation/patient_predictions.csv", index=False)

    return precision_list, recall_list, f1_list, df_predictions
    
# ----------------------------
# 7. MAIN
# ----------------------------
def main():
    # A. Load Data & Model
    model = load_model(config_id="3", model_path=MODEL_PATH, model_name=MODEL_NAME)

    # Load FULL datasets (load_ram=False to save memory)
    patient_dataset = HelicoPatients()
    print(f"Loaded patient dataset with {len(patient_dataset)} patients.")
    precision_list, recall_list, f1_list, df_predictions = kfold_patient_cv(model, patient_dataset, k=K_FOLD, max_patches=32)
    print("All patient predictions saved to crossvalidation/patient_predictions.csv")
    
    
    
if __name__ == "__main__":
    main()