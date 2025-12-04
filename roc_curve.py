import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import roc_curve, auc
import cv2  

# Custom imports
from dataset import HelicoAnnotated, annotated_collate
from Models.AEmodels import AutoEncoderCNN, VAECNN
from train_conv_ae import AEConfigs
from train_conv_vae import VAEConfigs

# ----------------------------
# CONFIGURATION
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "/fhome/vlia01/Medical-Imaging/slurm_output/config_three.pth" # Adjust if needed
BATCH_SIZE = 16
MODEL_NAME = "Autoencoder" # "Autoencoder" or "Variational Autoencoder"

# ROC Configuration
NUM_THRESHOLDS = 50  # Increased from 20 to 50 for a smoother curve
SAVE_FIG = True

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

# ----------------------------
# 2. CALCULATE ERRORS
# ----------------------------
def get_reconstruction_errors(dataloader, model, model_name="Autoencoder", metric="hsv_red"):
    """
    Compute HSV-based red-channel reconstruction error per image.
    metric="hsv_red": fraction of bright true-red pixels lost in reconstruction
    """
    all_errors = []

    print(f"Calculating {metric.upper()} errors for dataset...")

    with torch.no_grad():
        for batch in dataloader:
            images, labels = batch
            images = images.to(DEVICE).float()
            if images.max() > 1:
                images /= 255.0

            # Forward pass
            if model_name == "Autoencoder":
                recon = model(images)
            else:
                recon, _, _ = model(images)

            batch_errors = []
            for orig_img, rec_img in zip(images, recon):
                # Convert to numpy [H,W,3] in range 0-255
                orig_np = (orig_img.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
                rec_np = (rec_img.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)

                # Convert RGB -> HSV
                orig_hsv = cv2.cvtColor(orig_np, cv2.COLOR_RGB2HSV)
                rec_hsv = cv2.cvtColor(rec_np, cv2.COLOR_RGB2HSV)

                orig_hue = orig_hsv[:, :, 0]
                orig_sat = orig_hsv[:, :, 1] / 255.0
                rec_hue = rec_hsv[:, :, 0]
                rec_sat = rec_hsv[:, :, 1] / 255.0

                # Red mask: Hue near 0 or 179, saturation > 0.5
                orig_mask = ((orig_hue <= 10) | (orig_hue >= 170))  & (orig_sat > 0.04)
                rec_mask = ((rec_hue <= 10) | (rec_hue >= 170)) & (rec_sat > 0.04)

                # Fraction of bright red pixels lost
                error_val = np.sum(orig_mask & (~rec_mask)) / orig_mask.size
                batch_errors.append(error_val)

            all_errors.extend(batch_errors)

    return np.array(all_errors)


# ----------------------------
# 3. CALCULATE ROC (sklearn)
# ----------------------------
def calculate_roc_metrics(benign_errors, malign_errors):
    """
    Compute FPR, TPR, thresholds and AUC for ROC curve using sklearn
    """
    # Create labels: 0=benign, 1=malignant
    y_true = np.concatenate([np.zeros_like(benign_errors), np.ones_like(malign_errors)])
    y_scores = np.concatenate([benign_errors, malign_errors])  # Higher error = more likely malignant

    # Compute ROC
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # Compute AUC
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, thresholds, roc_auc


# ----------------------------
# 4. MAIN
# ----------------------------
def main():
    # A. Load Data & Model
    model = load_model(config_id="3", model_path=MODEL_PATH, model_name=MODEL_NAME)

    # Load FULL datasets (load_ram=False to save memory)
    dataset_benign = HelicoAnnotated(only_negative=True, load_ram=False)
    dataset_malign = HelicoAnnotated(only_positive=True, load_ram=False)

    loader_benign = DataLoader(dataset_benign, batch_size=BATCH_SIZE, shuffle=False, collate_fn=annotated_collate)
    loader_malign = DataLoader(dataset_malign, batch_size=BATCH_SIZE, shuffle=False, collate_fn=annotated_collate)

    METRIC = 'red'

    # B. Get Error Distributions
    print("Calculating Benign Errors...")
    benign_err = get_reconstruction_errors(loader_benign, model, metric=METRIC, model_name=MODEL_NAME)

    print("Calculating Malignant Errors...")
    malign_err = get_reconstruction_errors(loader_malign, model, metric=METRIC, model_name=MODEL_NAME)

    # C. Calculate ROC automatically
    fpr, tpr, thresholds, roc_auc = calculate_roc_metrics(benign_err, malign_err)

    # D. Best Threshold (Youden's J statistic) OPTIMAL THRESHOLD
    J = tpr - fpr
    best_idx = np.argmax(J)
    best_threshold = thresholds[best_idx]
    best_tpr = tpr[best_idx]
    best_fpr = fpr[best_idx]

    print(f"\n--- RESULTS ---")
    print(f"Best Threshold found: {best_threshold:.6f}")
    print(f"At this threshold -> TPR (Recall): {best_tpr:.4f}, FPR: {best_fpr:.4f}")
    print(f"AUC: {roc_auc:.4f}")

    # E. Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC={roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Random guess line
    plt.scatter(best_fpr, best_tpr, color='red', label='Optimal Point', zorder=5)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title(f'ROC Curve PATCH LEVEL - {MODEL_NAME} Conf3')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    if SAVE_FIG:
        os.makedirs("results", exist_ok=True)
        save_path = f"results/ROC_RED_{MODEL_NAME.replace(' ', '_')}.png"
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")

    plt.show()

if __name__ == "__main__":
    main()