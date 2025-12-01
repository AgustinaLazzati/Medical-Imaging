import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

# Custom imports (Keep your existing folder structure)
from dataset import HelicoAnnotated, annotated_collate
from Models.AEmodels import AutoEncoderCNN, VAECNN
from train_conv_ae import AEConfigs
from train_conv_vae import VAEConfigs

# ----------------------------
# CONFIG
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "/fhome/vlia01/Medical-Imaging/slurm_output/vivid-armadillo-62.pth"
#MODEL_PATH = "/fhome/vlia01/Medical-Imaging/slurm_output/vae_2.pth"
BATCH_SIZE = 16
NUM_EXAMPLES = 5  # per class for visualization
SAVE_FIG = True
MODEL_NAME = "Autoencoder"  # "Variational Autoencoder" or "Autoencoder"

# ----------------------------
# LOAD MODEL
def load_model(config_id, model_path=None, model_name="Autoencoder"):
    if model_name == "Autoencoder":
        net_paramsEnc, net_paramsDec, inputmodule_paramsEnc, inputmodule_paramsDec = AEConfigs(config_id)
        model = AutoEncoderCNN(
            inputmodule_paramsEnc,
            net_paramsEnc,
            inputmodule_paramsDec,
            net_paramsDec
        )
    elif model_name == "Variational Autoencoder":
        net_paramsEnc, net_paramsDec, inputmodule_paramsEnc, inputmodule_paramsDec, net_paramsRep = VAEConfigs(config_id)
        model = VAECNN(
            inputmodule_paramsEnc,
            net_paramsEnc,
            inputmodule_paramsDec,
            net_paramsDec,
            net_paramsRep
        )
    else:
        raise ValueError("model_name must be 'Autoencoder' or 'Variational Autoencoder'")
    
    if model_path:
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            print(f"Loaded weights from {model_path}")
        else:
            print(f"Warning: Model path {model_path} not found. Initializing random weights.")
            
    model.to(DEVICE)
    model.eval()
    return model

# ----------------------------
# HELPER FUNCTIONS

def get_examples(dataloader, model, num_examples, model_name="Autoencoder"):
    """
    Get specific examples for visualization.
    """
    examples = []
    count = 0
    skip_count = 0
    
    with torch.no_grad():
        for batch in dataloader:
            images, labels = batch 
            images = images.to(DEVICE).float()
            if images.max() > 1:
                images = images / 255.0
            
            if model_name == "Autoencoder":
                recon = model(images)
            elif model_name == "Variational Autoencoder":
                recon, mu, logvar = model(images)
            
            for img, out in zip(images, recon):
                # Skip the first few images if desired (e.g., if they are black borders)
                if skip_count < 3: 
                    skip_count += 1
                    continue
                
                examples.append((img.cpu(), out.cpu()))
                count += 1
                if count >= num_examples:
                    return examples
                    
    return examples

def get_reconstruction_errors(dataloader, model, model_name="Autoencoder", metric="mse"):
    all_errors = []
    
    print(f"Calculating {metric.upper()} errors for dataset...")
    
    with torch.no_grad():
        for batch in dataloader:
            images, labels = batch
            images = images.to(DEVICE).float()
            if images.max() > 1: images /= 255.0
            
            if model_name == "Autoencoder":
                recon = model(images)
                mu, logvar = None, None
            elif model_name == "Variational Autoencoder":
                recon, mu, logvar = model(images)
            
            # --- METRIC CALCULATIONS ---
            
            # 1. MSE (Standard)
            if metric == "mse":
                batch_errors = torch.mean((images - recon) ** 2, dim=[1, 2, 3])
            
            # 2. L1 / MAE (Focus on sharpness)
            elif metric == "mae":
                batch_errors = torch.mean(torch.abs(images - recon), dim=[1, 2, 3])
            
            elif metric == "red":
                red_original = images[:, 0:1, :, :]
                red_recon = recon[:, 0:1, :, :]
                batch_errors = torch.mean((red_original - red_recon) ** 2, dim=[1, 2, 3])
                
            # 3. KL Divergence (VAE Only - measures "statistical" anomaly)
            elif metric == "kl":
                if mu is None or logvar is None:
                    raise ValueError("KL metric only available for VAE")
                # KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
                # We sum over the latent dimension (dim 1)
                kl_per_image = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
                batch_errors = kl_per_image

            # 4. Combined (ELBO approximation: MSE + KL)
            elif metric == "combined":
                 mse = torch.sum((images - recon) ** 2, dim=[1, 2, 3]) # sum usually better for ELBO balancing
                 if mu is not None:
                     kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
                     batch_errors = mse + kl
                 else:
                     batch_errors = mse

            all_errors.extend(batch_errors.cpu().numpy().tolist())
            
    return np.array(all_errors)

def to_numpy(img_tensor):
    """ Convert tensor [C,H,W] to numpy [H,W,C] """
    img = img_tensor.detach().cpu().permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    return img

def plot_reconstructions_with_error(images, reconstructions, model_name="Autoencoder", subset_name="Subset"):
    """
    Plots Original, Reconstructed, and Difference map.
    """
    n = len(images)
    plt.figure(figsize=(3 * n, 8)) # Increased height for 3 rows

    for i in range(n):
        # 1. Original
        plt.subplot(3, n, i + 1)
        orig_img = to_numpy(images[i])
        plt.imshow(orig_img)
        plt.axis("off")
        if i == n // 2: plt.title("Original")

        # 2. Reconstruction
        plt.subplot(3, n, i + 1 + n)
        recon_img = to_numpy(reconstructions[i])
        plt.imshow(recon_img)
        plt.axis("off")
        if i == n // 2: plt.title("Reconstructed")

        # 3. Error (Difference)
        plt.subplot(3, n, i + 1 + 2 * n)
        # Calculate absolute difference per channel, then mean over channels to get heatmap
        diff_map = np.abs(orig_img - recon_img)
        # Optional: Average over channels for a greyscale heatmap, or keep RGB diff
        # Here we show RGB diff magnitude
        plt.imshow(diff_map) 
        plt.axis("off")
        
        # Calculate individual scalar error for title
        mse = np.mean((orig_img - recon_img)**2)
        plt.title(f"Err: {mse:.4f}", fontsize=9)
        if i == n // 2: 
            plt.text(0.5, 1.15, "Difference Map", transform=plt.gca().transAxes, ha='center', fontsize=12)

    plt.tight_layout()
    
    if SAVE_FIG:
        os.makedirs("reconstruction", exist_ok=True)
        filename = f"reconstruction/{model_name.replace(' ','_')}_{subset_name}_reconstruction.png"
        plt.savefig(filename, dpi=300)
        print(f"Saved visualization to {filename}")
    plt.show()

def plot_error_comparison(benign_errors, malign_errors, model_name="AE"):
    """
    Plots histograms to compare error distributions.
    """
    plt.figure(figsize=(12, 6))
    
    # 1. Histogram
    plt.subplot(1, 2, 1)
    plt.hist(benign_errors, bins=50, alpha=0.6, label='Benign', density=True, color='green')
    plt.hist(malign_errors, bins=50, alpha=0.6, label='Malignant', density=True, color='red')
    plt.xlabel("Reconstruction Error (RED CHANNEL)")
    plt.ylabel("Density")
    plt.title(f"Error Distribution: {model_name}")
    plt.legend()
    plt.grid(alpha=0.3)

    # 2. Box Plot
    plt.subplot(1, 2, 2)
    data = [benign_errors, malign_errors]
    plt.boxplot(data, labels=['Benign', 'Malignant'], patch_artist=True, 
                boxprops=dict(facecolor='lightblue'))
    plt.ylabel("Reconstruction Error (RED CHANNEL)")
    plt.title("Error Separation")
    plt.grid(alpha=0.3)
    
    # Calculate stats
    benign_mean = np.mean(benign_errors)
    malign_mean = np.mean(malign_errors)
    print("\n--- Statistics ---")
    print(f"Benign Mean Error:    {benign_mean:.6f} (std: {np.std(benign_errors):.6f})")
    print(f"Malignant Mean Error: {malign_mean:.6f} (std: {np.std(malign_errors):.6f})")
    
    if SAVE_FIG:
        os.makedirs("reconstruction", exist_ok=True)
        filename = f"reconstruction/{model_name.replace(' ','_')}_Error_Comparison.png"
        plt.savefig(filename, dpi=300)
        print(f"Saved error comparison to {filename}")
    plt.show()

# ----------------------------
# MAIN
def main():
    model = load_model(config_id="4", model_path=MODEL_PATH, model_name=MODEL_NAME)

    dataset_benign = HelicoAnnotated(only_negative=True, load_ram=False)
    dataset_malign = HelicoAnnotated(only_positive=True, load_ram=False) 

    dataloader_benign = DataLoader(dataset_benign, batch_size=BATCH_SIZE, shuffle=False, collate_fn=annotated_collate)
    dataloader_malign = DataLoader(dataset_malign, batch_size=BATCH_SIZE, shuffle=False, collate_fn=annotated_collate)

    print(f"Benign samples: {len(dataset_benign)}")
    print(f"Malignant samples: {len(dataset_malign)}")

    print("\nGenerating Visualizations...")
    benign_examples = get_examples(dataloader_benign, model, NUM_EXAMPLES, model_name=MODEL_NAME)
    malign_examples = get_examples(dataloader_malign, model, NUM_EXAMPLES, model_name=MODEL_NAME)

    plot_reconstructions_with_error(
        [img for img, _ in benign_examples],
        [out for _, out in benign_examples],
        model_name=MODEL_NAME,
        subset_name="Benign"
    )

    plot_reconstructions_with_error(
        [img for img, _ in malign_examples],
        [out for _, out in malign_examples],
        model_name=MODEL_NAME,
        subset_name="Malignant"
    )

    print("\nCalculating Full Dataset Statistics...")
    benign_errors = get_reconstruction_errors(dataloader_benign, model, model_name=MODEL_NAME, metric="red")
    malign_errors = get_reconstruction_errors(dataloader_malign, model, model_name=MODEL_NAME, metric="red")

    plot_error_comparison(benign_errors, malign_errors, model_name=MODEL_NAME)

if __name__ == "__main__":
    main()