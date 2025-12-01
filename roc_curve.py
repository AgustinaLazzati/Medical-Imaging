import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

# Custom imports
from dataset import HelicoAnnotated, annotated_collate
from Models.AEmodels import AutoEncoderCNN, VAECNN
from train_conv_ae import AEConfigs
from train_conv_vae import VAEConfigs

# ----------------------------
# CONFIGURATION
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "/fhome/vlia01/Medical-Imaging/slurm_output/vivid-armadillo-62.pth" # Adjust if needed
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
def get_all_errors(dataloader, model, model_name="Autoencoder"):
    """
    Runs the model on the dataloader and returns a list of MSE errors for every image.
    """
    errors = []
    metric='mse'
    with torch.no_grad():
        for batch in dataloader:
            images, labels = batch
            images = images.to(DEVICE).float()
            if images.max() > 1: images /= 255.0
            
            if model_name == "Autoencoder":
                recon = model(images)
            else:
                recon, mu, logvar = model(images)
            
            if metric == "mse":
              # Calculate MSE per image: average over [Channels, H, W]
              # Shape: [Batch_Size]
              batch_mse = torch.mean((images - recon) ** 2, dim=[1, 2, 3])
              errors.extend(batch_mse.cpu().numpy())
            elif metric == "red":
              red_original = images[:, 0:1, :, :]
              red_recon = recon[:, 0:1, :, :]
              # Extract the Red Channel (Channel 0 for RGB images)
              # Slicing [:, 0:1, :, :] keeps the channel dimension for correct calculation
              batch_red_mse = torch.mean((red_original - red_recon) ** 2, dim=[1, 2, 3])
              errors.extend(batch_red_mse.cpu().numpy())
            
    return np.array(errors)

# ----------------------------
# 3. ROC CALCULATION
# ----------------------------
def calculate_roc_metrics(benign_errors, malign_errors, thresholds):
    tpr_list = []
    fpr_list = []
    
    print(f"Evaluating {len(thresholds)} thresholds...")

    for threshold in thresholds:
        # --- CLASSIFICATION LOGIC ---
        # If Error > Threshold -> Classify as Malignant (Positive)
        # If Error <= Threshold -> Classify as Benign (Negative)
        
        # True Positives (TP): Malignant samples correctly identified as Malignant
        tp = np.sum(malign_errors > threshold)
        
        # False Negatives (FN): Malignant samples incorrectly identified as Benign
        fn = np.sum(malign_errors <= threshold)
        
        # False Positives (FP): Benign samples incorrectly identified as Malignant
        fp = np.sum(benign_errors > threshold)
        
        # True Negatives (TN): Benign samples correctly identified as Benign
        tn = np.sum(benign_errors <= threshold)
        
        # --- RATES ---
        # TPR (Sensitivity/Recall) = TP / (TP + FN)
        # How many of the actual malignant cases did we catch?
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # FPR (1 - Specificity) = FP / (FP + TN)
        # How many benign cases did we falsely alarm?
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        
    return np.array(fpr_list), np.array(tpr_list)

# ----------------------------
# 4. MAIN PLOTTING LOGIC
# ----------------------------
def main():
    # A. Load Data & Model
    model = load_model(config_id="4", model_path=MODEL_PATH, model_name=MODEL_NAME)
    
    # Load FULL datasets (load_ram=False to save memory)
    dataset_benign = HelicoAnnotated(only_negative=True, load_ram=False)
    dataset_malign = HelicoAnnotated(only_positive=True, load_ram=False)
    
    loader_benign = DataLoader(dataset_benign, batch_size=BATCH_SIZE, shuffle=False, collate_fn=annotated_collate)
    loader_malign = DataLoader(dataset_malign, batch_size=BATCH_SIZE, shuffle=False, collate_fn=annotated_collate)
    
    # B. Get Error Distributions
    print("Calculating Benign Errors...")
    benign_err = get_all_errors(loader_benign, model, MODEL_NAME)
    
    print("Calculating Malignant Errors...")
    malign_err = get_all_errors(loader_malign, model, MODEL_NAME)
    
    # C. Define Thresholds
    # We dynamically find the range based on your data
    min_err = min(benign_err.min(), malign_err.min())
    max_err = max(benign_err.max(), malign_err.max())
    
    # Add a small buffer to the max range to ensure the curve hits (0,0)
    print(f"\nData Range Detected -> Min: {min_err:.6f}, Max: {max_err:.6f}")
    
    # Based on your plot, max is likely ~0.0025. 
    # We create evenly spaced thresholds across this full range.
    thresholds = np.linspace(min_err, max_err, NUM_THRESHOLDS)
    
    # D. Calculate ROC
    fpr, tpr = calculate_roc_metrics(benign_err, malign_err, thresholds)
    
    # Calculate Best Threshold (Youden's J statistic: max(TPR - FPR))
    # This finds the point closest to the top-left corner
    J = tpr - fpr
    best_idx = np.argmax(J)
    best_threshold = thresholds[best_idx]
    best_tpr = tpr[best_idx]
    best_fpr = fpr[best_idx]
    
    print(f"\n--- RESULTS ---")
    print(f"Best Threshold found: {best_threshold:.6f}")
    print(f"At this threshold -> TPR (Recall): {best_tpr:.4f}, FPR: {best_fpr:.4f}")
    
    # E. Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (Best Thresh={best_threshold:.5f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Random guess line
    
    # Mark the best threshold point
    plt.scatter(best_fpr, best_tpr, color='red', label='Optimal Point', zorder=5)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title(f'ROC Curve - {MODEL_NAME}')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    if SAVE_FIG:
        os.makedirs("results", exist_ok=True)
        save_path = f"results/ROC_{MODEL_NAME.replace(' ', '_')}.png"
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
        
    plt.show()

if __name__ == "__main__":
    main()