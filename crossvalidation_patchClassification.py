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
from dataset import HelicoAnnotated, annotated_collate
from Models.AEmodels import AutoEncoderCNN, VAECNN
from train_conv_ae import AEConfigs
from train_conv_vae import VAEConfigs

# ----------------------------
# CONFIGURATION
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "/fhome/vlia01/Medical-Imaging/slurm_output/config_three.pth"  # adjust if needed
#MODEL_PATH = "/fhome/vlia01/Medical-Imaging/slurm_output/vae_2.pth"
BATCH_SIZE = 128
MODEL_NAME = "Autoencoder"  # "Autoencoder" or "Variational Autoencoder"
K_FOLD = 10
SAVE_FIG = True
METRIC = 'hsv_red'    # 'hsv_red' or 'mae_red' or 'mse'
RESULTS_DIR = "/fhome/vlia01/Medical-Imaging/crossvalidation"
os.makedirs(RESULTS_DIR, exist_ok=True)
NUM_WORKERS = 4

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

# -----------------------------------------
# 2. RECONSTRUCTION ERROR CALCULATION (PATCH)
# -----------------------------------------
def get_reconstruction_errors(dataloader, model, model_name="Autoencoder", metric="hsv_red"):
    """
    Returns:
      errors: numpy array of error per sample (higher -> more likely malignant)
      labels: numpy array of labels (0 benign, 1 malignant) aligned with errors
      indices: numpy array of dataset indices corresponding to processed samples
    Expects dataloader built with annotated_collate and yielding (images, labels).
    """
    all_errors = []
    all_labels = []
    all_indices = []

    idx_counter = 0  # Tracks absolute dataset index

    with torch.no_grad():
        for batch in dataloader:
            # annotated_collate returns either (images, labels) or empty tensors
            if isinstance(batch, tuple) or isinstance(batch, list):
                images, labels = batch
            else:
                # unexpected; skip
                continue

            if images is None or len(images) == 0:
                continue

            images = images.to(DEVICE).float()
            if images.max() > 1:
                images = images / 255.0

            # forward
            if model_name == "Autoencoder":
                recon = model(images)
            else:
                recon, _, _ = model(images)

            # ensure recon shape matches images
            # iterate per-sample
            for i in range(images.shape[0]):
                orig_img = images[i].cpu()
                rec_img = recon[i].cpu()

                if metric == 'hsv_red':
                    orig_np = (orig_img.permute(1,2,0).numpy() * 255).astype(np.uint8)
                    rec_np  = (rec_img.permute(1,2,0).numpy() * 255).astype(np.uint8)

                    # Convert RGB->HSV
                    orig_hsv = cv2.cvtColor(orig_np, cv2.COLOR_RGB2HSV)
                    rec_hsv  = cv2.cvtColor(rec_np,  cv2.COLOR_RGB2HSV)

                    orig_hue = orig_hsv[:, :, 0]
                    orig_sat = orig_hsv[:, :, 1] / 255.0
                    rec_hue  = rec_hsv[:, :, 0]
                    rec_sat  = rec_hsv[:, :, 1] / 255.0

                    orig_mask = ((orig_hue <= 10) | (orig_hue >= 170)) & (orig_sat > 0.04)
                    rec_mask  = ((rec_hue <= 10)  | (rec_hue >= 170)) & (rec_sat > 0.04)

                    error_val = np.sum(orig_mask & (~rec_mask)) / orig_mask.size
                    all_errors.append(float(error_val))

                elif metric == 'mae_red':
                    orig_red = orig_img.permute(1,2,0).numpy()[:,:,0].astype(np.float32)
                    rec_red  = rec_img.permute(1,2,0).numpy()[:,:,0].astype(np.float32)
                    all_errors.append(float(np.mean(np.abs(orig_red - rec_red))))

                elif metric == 'mse':
                    orig_arr = orig_img.permute(1,2,0).numpy().astype(np.float32)
                    rec_arr  = rec_img.permute(1,2,0).numpy().astype(np.float32)
                    all_errors.append(float(np.mean((orig_arr - rec_arr)**2)))
                else:
                    raise ValueError(f"Unknown metric: {metric}")

                # Record index of successfully processed sample
                all_indices.append(idx_counter)
                idx_counter += 1

            # flatten labels to numpy
            lab_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else np.array(labels)
            all_labels.extend(np.where(lab_np == -1, 0, lab_np).tolist())

    return np.array(all_errors), np.array(all_labels, dtype=int), np.array(all_indices)


# ----------------------------
# 3. CROSS-VALIDATION (patient-stratified)
# ----------------------------
def crossvalidation_patches(model, dataset, k=10):
    """
    Patient-stratified cross-validation.
    Each fold contains patients, all patches from those patients are used.
    """
    # Get patient info
    all_patient_ids = np.array([s[2] for s in dataset.samples])  # assuming sample[2] = patient ID
    all_labels = np.array([1 if s[1] == 1 else 0 for s in dataset.samples])  # patch-level labels
    unique_patients = np.unique(all_patient_ids)

    # Compute patient-level labels: positive if ANY patch is positive
    patient_labels = np.array([int(np.any(all_labels[all_patient_ids == p])) for p in unique_patients])

    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=42)

    metrics = []
    rocs = []
    predictions = []
    thresholds_per_fold = []

    for fold_idx, (train_pat_idx, test_pat_idx) in enumerate(
        sgkf.split(
            X=np.zeros_like(unique_patients),
            y=patient_labels,
            groups=unique_patients
        )
    ):
        print(f"\n===== FOLD {fold_idx + 1} / {k} =====")

        # Map patient indices to patch indices
        train_patients = unique_patients[train_pat_idx]
        test_patients  = unique_patients[test_pat_idx]

        train_idx = np.where(np.isin(all_patient_ids, train_patients))[0]
        test_idx  = np.where(np.isin(all_patient_ids, test_patients))[0]

        # Create DataLoaders
        train_loader = DataLoader(Subset(dataset, train_idx),
                                  batch_size=BATCH_SIZE,
                                  shuffle=False,
                                  collate_fn=annotated_collate)
        test_loader = DataLoader(Subset(dataset, test_idx),
                                 batch_size=BATCH_SIZE,
                                 shuffle=False,
                                 collate_fn=annotated_collate)

        # Compute reconstruction errors
        err_train, lab_train, _ = get_reconstruction_errors(train_loader, model,
                                                            model_name=MODEL_NAME,
                                                            metric=METRIC)
        err_test, lab_test, idx_test_processed = get_reconstruction_errors(test_loader, model,
                                                                           model_name=MODEL_NAME,
                                                                           metric=METRIC)

        # Skip fold if training labels are single-class
        if len(np.unique(lab_train)) < 2:
            print("Skipping fold: training labels contain only one class.")
            metrics.append((np.nan, np.nan, np.nan, np.nan))
            rocs.append(([], [], np.nan))
            continue

        # ROC train -> determine threshold
        fpr, tpr, thr = roc_curve(lab_train, err_train)
        roc_auc = auc(fpr, tpr)
        best_thr = thr[np.argmax(tpr - fpr)]
        thresholds_per_fold.append(best_thr)

        # Apply threshold to test
        if len(np.unique(lab_test)) < 2:
            y_pred = np.zeros_like(lab_test)
            prec = rec = f1 = np.nan
            print("Warning: test labels contain only one class; metrics set to NaN")
        else:
            y_pred = (err_test >= best_thr).astype(int)
            prec = precision_score(lab_test, y_pred, zero_division=0)
            rec  = recall_score(lab_test, y_pred, zero_division=0)
            f1   = f1_score(lab_test, y_pred, zero_division=0)

        print(f"AUC(TRAIN)={roc_auc:.4f} | Th={best_thr:.12f}")
        print(f"Prec={prec:.4f} Recall={rec:.4f} F1={f1:.4f}")

        metrics.append((prec, rec, f1, roc_auc))

        # ROC curve for test
        if len(np.unique(lab_test)) < 2:
            rocs.append(([], [], np.nan))
        else:
            fpr_t, tpr_t, _ = roc_curve(lab_test, err_test)
            rocs.append((fpr_t, tpr_t, roc_auc))

        # Store predictions
        for loc, ds_idx in enumerate(idx_test_processed):
            predictions.append({
                "fold": fold_idx + 1,
                "sample_idx": int(ds_idx),
                "patient": str(all_patient_ids[ds_idx]),
                "true": int(lab_test[loc]),
                "pred": int(y_pred[loc]) if len(np.unique(lab_test)) >= 2 else -1,
                "error": float(err_test[loc]),
                "threshold": float(best_thr)
            })

    return metrics, rocs, predictions, thresholds_per_fold


# ----------------------------
# MAIN
# ----------------------------
def main():

    # ---------------------------------------------------
    # LOAD MODEL
    # ---------------------------------------------------
    model = load_model(
        config_id="3",
        model_path=MODEL_PATH,
        model_name=MODEL_NAME
    )

    # ---------------------------------------------------
    # LOAD DATASET
    # ---------------------------------------------------
    dataset_all = HelicoAnnotated(load_ram=False)
    print(f"Total samples in dataset: {len(dataset_all)}")

    samples = dataset_all.samples
    benign_indices  = [i for i, s in enumerate(samples) if s[1] == -1]
    malign_indices  = [i for i, s in enumerate(samples) if s[1] ==  1]

    print(f"Benign patches: {len(benign_indices)}")
    print(f"Malignant patches: {len(malign_indices)}")

    benign_loader = DataLoader(
        Subset(dataset_all, benign_indices),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=annotated_collate, 
        num_workers=NUM_WORKERS
    )
    malign_loader = DataLoader(
        Subset(dataset_all, malign_indices),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=annotated_collate, 
        num_workers=NUM_WORKERS
    )

    # ---------------------------------------------------
    # GLOBAL ROC (ALL PATCHES)
    # ---------------------------------------------------
    print("\nComputing GLOBAL ROC...")

    benign_errs, _ , _ = get_reconstruction_errors(
        benign_loader, model, metric=METRIC, model_name=MODEL_NAME
    )
    malign_errs, _ , _ = get_reconstruction_errors(
        malign_loader, model, metric=METRIC, model_name=MODEL_NAME
    )

    if benign_errs.size > 0 and malign_errs.size > 0:
        y_true_global   = np.concatenate([np.zeros_like(benign_errs),
                                          np.ones_like(malign_errs)])
        y_score_global  = np.concatenate([benign_errs, malign_errs])

        fpr_g, tpr_g, thr_g = roc_curve(y_true_global, y_score_global)
        auc_g = auc(fpr_g, tpr_g)

        J = tpr_g - fpr_g
        best_i = np.argmax(J)
        best_thr_global = thr_g[best_i]

        print(f"GLOBAL AUC={auc_g:.4f} | Best threshold={best_thr_global:.12f}")

        # --- Global ROC figure ---
        plt.figure(figsize=(7, 6))
        plt.plot(fpr_g, tpr_g, color='darkorange', lw=2, label=f"AUC={auc_g:.4f}")
        plt.plot([0, 1], [0, 1], "k--", lw=1)
        plt.scatter(fpr_g[best_i], tpr_g[best_i], color="red",
                    label=f"Optimal Th={best_thr_global:.6f}")
                    
        plt.text(fpr_g[best_i] + 0.02, tpr_g[best_i] - 0.04,
         f"({fpr_g[best_i]:.3f}, {tpr_g[best_i]:.3f})",
         fontsize=9,)

        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title(f"Global ROC – {MODEL_NAME} conf3 - {METRIC} ERROR")
        plt.grid(alpha=0.3)
        plt.legend()

        if SAVE_FIG:
            out = os.path.join(RESULTS_DIR, f"ROC_GLOBAL_{MODEL_NAME}_conf3_{METRIC}.png")
            plt.savefig(out, dpi=300)
            print(f"Saved: {out}")
        plt.close()

    else:
        print("WARNING: Global ROC skipped (one class empty).")

    # ---------------------------------------------------
    # PATIENT-STRATIFIED K-FOLD CV
    # ---------------------------------------------------
    print("\nRunning PATIENT-STRATIFIED K-FOLD CV...")

    metrics, rocs, predictions, thresholds_per_fold = crossvalidation_patches(
        model=model,
        dataset=dataset_all,
        k=K_FOLD
    )

    # metrics = [(prec, rec, f1, auc_test)]
    metrics = np.array(metrics)
    if len(metrics) > 0:
        prec_m, rec_m, f1_m, auc_m = metrics.mean(axis=0)
        prec_s, rec_s, f1_s, auc_s = metrics.std(axis=0)
        thresholds_per_fold = np.array(thresholds_per_fold)
        thr_mean = thresholds_per_fold.mean()
        thr_std  = thresholds_per_fold.std()

        print("\n===============================")
        print(" K-FOLD CROSS-VALIDATION SUMMARY")
        print("===============================")
        print(f"Precision : {prec_m:.4f} ± {prec_s:.4f}")
        print(f"Recall    : {rec_m:.4f} ± {rec_s:.4f}")
        print(f"F1-score  : {f1_m:.4f} ± {f1_s:.4f}")
        print(f"AUC (test): {auc_m:.4f} ± {auc_s:.4f}")
        print(f"Optimal Threshold: {thr_mean:.12f} ± {thr_std:.12f}")
    # ---------------------------------------------------
    # SAVE PREDICTIONS CSV
    # ---------------------------------------------------
    if len(predictions) > 0:
        dfp = pd.DataFrame(predictions)
        csv_path = os.path.join(RESULTS_DIR, f"patch_level_predictions_{MODEL_NAME}_conf3_{METRIC}.csv")
        dfp.to_csv(csv_path, index=False)
        print(f"\nSaved predictions to {csv_path}")

    # ---------------------------------------------------
    # PER-FOLD ROC FIGURE
    # ---------------------------------------------------
    plt.figure(figsize=(8, 6))
    print("\nPlotting per-fold ROC curves...")

    for i, (fpr, tpr, auc_val) in enumerate(rocs, 1):
        plt.plot(fpr, tpr, lw=1.5, alpha=0.6,
                 label=f"Fold {i} (AUC={auc_val:.3f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"Per-fold ROC – {MODEL_NAME} conf3 - {METRIC} ERROR")
    plt.grid(alpha=0.3)
    plt.legend(fontsize="small")

    if SAVE_FIG:
        out = os.path.join(RESULTS_DIR, f"ROC_FOLDS_{MODEL_NAME.replace(' ', '_')}_conf3_{METRIC.replace(' ', '_')}.png")
        plt.savefig(out, dpi=300)
        print(f"Saved: {out}")
    plt.close()
    

    print("\nDONE.")


if __name__ == "__main__":
    main()
