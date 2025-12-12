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
from sklearn.metrics import confusion_matrix
import pandas as pd

# Custom imports
from dataset import HelicoPatients  #CROPPED PATIENT-LEVEL dataset
from Models.AEmodels import AutoEncoderCNN, VAECNN
from train_conv_ae import AEConfigs
from train_conv_vae import VAEConfigs


# ----------------------------
# CONFIGURATION
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_DIR = "/fhome/vlia01/Medical-Imaging"
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
#MODEL_PATH = "/fhome/vlia01/Medical-Imaging/slurm_output/config_three.pth"
MODEL_PATH = "/fhome/vlia01/Medical-Imaging/slurm_output/vae_3.pth"
BATCH_SIZE = 264   
MODEL_NAME = "Variational Autoencoder" # "Autoencoder" or "Variational Autoencoder"
SAVE_FIG = True
PATCH_THR = 0.0005780   #0.000491333008   # patch-levelthreshold from MEAN OF AUTOENCODER CONFIG 3
MAX_PATCHES= 300
NUM_WORKERS = 4


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
    
# -----------------
# HELPER FUNCTION
# -----------------   
def label_to_binary(label):
    #HELPER FUNCTION
    if isinstance(label, str):
        if label.upper() in ['ALTA', 'BAIXA']:
            return 1
        else:  # NEGATIVA or any other negative string
            return 0
    else:  # numeric label
        return 0 if label == -1 else 1
        
    
# ----------------------------------------------------
# 2. PATCH-TO-PATIENT ERROR FUNCTION (SCORE FUNCTION)
# ----------------------------------------------------
def get_patient_errors(model, dataloader, patch_thr=PATCH_THR, model_name=MODEL_NAME, max_patches=100):
    patient_scores = {}
    patient_labels = {}

    with torch.no_grad():
        for batch in dataloader:
            images = batch['images'].to(DEVICE).float()  # shape [1, num_patches, C, H, W]
            images = images.squeeze(0)
            label = batch['label'][0]
            pid = batch['p_id'][0]
            
            # randomly sample patches if there are too many
            num_images = len(images)
            print(f"Processing patient: {pid} having {num_images} patches")
            
            if num_images > max_patches:
                # randomly sample exactly max_patches
                idxs = np.random.choice(num_images, max_patches, replace=False)
                images = images[idxs]
            elif num_images < max_patches:
                # if there are fewer images, just keep all
                images = images  # no sampling needed, just keep them all

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

    
    
# ----------------------------
# 3. MAIN
# ----------------------------
def main():
    # 3.1 Load Data & Model
    model = load_model(config_id="3", model_path=MODEL_PATH, model_name=MODEL_NAME)

    # 3.2 Load FULL datasets (load_ram=False to save memory)
    patient_dataset = HelicoPatients()
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
        patch_thr=PATCH_THR,
        max_patches=MAX_PATCHES,
        model_name=MODEL_NAME
    )
    
    scores = np.array(list(patient_scores.values()))
    labels = np.array(patient_labels)
    
    # 3.4 PATIENT ROC + THRESHOLD
    fpr, tpr, thresholds = roc_curve(labels, scores)
    auc_val = auc(fpr, tpr)
    
    # Best Threshold (Youden's J statistic) OPTIMAL THRESHOLD
    J = tpr - fpr
    best_idx = np.argmax(J)
    patient_thr = thresholds[best_idx]  #BEST THRESHOLD
    best_tpr = tpr[best_idx]
    best_fpr = fpr[best_idx]
    
    print(f"\n--- RESULTS ---")
    print(f"Best Threshold found: {patient_thr:.6f}")
    print(f"At this threshold -> TPR (Recall): {best_tpr:.4f}, FPR: {best_fpr:.4f}")
    print(f"AUC: {auc_val:.4f}")

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC={auc_val:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Random guess line
    plt.scatter(best_fpr, best_tpr, color='red', label=f'Optimal Point (Optimal Thresh={patient_thr:.6f})', zorder=5)
    
    plt.text(best_fpr + 0.02, best_tpr - 0.04,
         f"({best_fpr:.3f}, {best_tpr:.3f})",
         fontsize=9,)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title(f'ROC Curve PATIENT LEVEL - {MODEL_NAME} Conf3')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    if SAVE_FIG:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        save_path = os.path.join(RESULTS_DIR, f"ROC_PATIENT_LEVEL_{MODEL_NAME.replace(' ', '_')}(globalTHR).png")
        plt.savefig(save_path, dpi=300)
        print(f"ROC curve saved to: {save_path}")

    plt.show()
    
    # -----------------------
    # 3.5 APPLY THRESHOLD
    # -----------------------
    preds = (scores >= patient_thr).astype(int)

    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)

    print("\nFINAL PATIENT METRICS...")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    
    
    # 3.6 CONFUSION MATRIX
    cm = confusion_matrix(labels, preds)

    tn, fp, fn, tp = cm.ravel()
    print("\nCONFUSION MATRIX:")
    print(cm)
    print(f"\nTN={tn}, FP={fp}, FN={fn}, TP={tp}")

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix Patient Diagnosis")
    plt.colorbar()

    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["No H. pylori", "H. pylori"])
    plt.yticks(tick_marks, ["No H. pylori", "H. pylori"])

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    # Annotate cells
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            plt.text(
                j, i, format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.tight_layout()

    if SAVE_FIG:
        cm_path = os.path.join(RESULTS_DIR, "patient_diagnosis_VAE(globalTHR).png")
        plt.savefig(cm_path, dpi=300)
        print(f"Confusion matrix saved to: {cm_path}")

    plt.show()
    

    # Saving predictions to CSV 
    df = pd.DataFrame({
        "patient_id": list(patient_scores.keys()),
        "score": scores,
        "prediction": preds,
        "label": labels
    })

    csv_path = os.path.join(RESULTS_DIR, "patient_predictions_diagnosis_VAE(globalTHR).csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved predictions to {csv_path}")
    
    
if __name__ == "__main__":
    main()