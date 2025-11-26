import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch.nn.functional as F

# Custom imports
from dataset import HelicoAnnotated, annotated_collate
from Models.AEmodels import AutoEncoderCNN, VAECNN
from train_conv_ae import AEConfigs
from train_conv_vae import VAEConfigs

# ----------------------------
# CONFIGURATION
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "/fhome/vlia01/Medical-Imaging/slurm_output/config_three.pth"
#MODEL_PATH = "/fhome/vlia01/Medical-Imaging/slurm_output/vae_2.pth"
BATCH_SIZE = 32 
MODEL_NAME = "Autoencoder" # "Autoencoder" or "Variational Autoencoder"
SAVE_FIG = True
MAX_SAMPLES = 2000 # Total samples to visualize (will try to take half from each class)

# ----------------------------
# 1. LOAD MODEL
# ----------------------------
def load_model(config_id="3", model_path=None, model_name="Autoencoder"):
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
        print(f"Warning: Weights not found at {model_path}. Visualization will be random.")
            
    model.to(DEVICE)
    model.eval()
    return model
    
# ----------------------------
# VISTO QUE NO HAY BOTTLENECK, hay que reducir dimensiones (y muchas) + normalizar
# ----------------------------  
class AEEmbeddingNet(nn.Module):
    def __init__(self, ae_model, embedding_dim=128):
        super(AEEmbeddingNet, self).__init__()

        self.encoder = ae_model.encoder

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.LazyLinear(embedding_dim)

    def forward(self, x):
        feat = self.encoder(x)       # [B, C, H, W]

        z = self.pool(feat)          # [B, C, 1, 1]
        z = z.view(z.size(0), -1)    # [B, C]
        z = self.fc(z)               # [B, embedding_dim]

        z = F.normalize(z, p=2, dim=1)

        return z

# ----------------------------
# 2. EXTRACT LATENT VECTORS
# ----------------------------
def get_latent_vectors(dataloader, model, model_name="Autoencoder", max_samples=None, force_label=None, embedding_dim=None):
    """
    Extracts latent vectors from the model.
    Args:
        force_label (int): If provided, overwrites the dataset targets with this value.
                           Useful when the dataset returns masks instead of class ints.
    """
    latents_list = []
    labels_list = []
    total_samples = 0
    
    dataset_name = "Benign" if force_label == 0 else "Malignant"
    print(f"Extracting latent vectors for {dataset_name}...")
    
    with torch.no_grad():
        for batch in dataloader:
            images, targets = batch
            images = images.to(DEVICE).float()
            #if images.max() > 1: images /= 255.0
            if images.dtype == torch.uint8:
                images = images.float() / 255.0
            
            # --- EXTRACT LATENT SPACE ---
            if model_name == "Variational Autoencoder":
                # VAE forward returns: recon, mu, logvar -> use mu
                _, mu, _ = model(images)
                curr_latents = mu
                
            elif model_name == "Autoencoder":
                # Standard AE -> use encoder
                
                if hasattr(model, 'encoder'):
                    curr_latents = model.encoder(images)
        
                elif hasattr(model, 'netEnc'):
                    curr_latents = model.netEnc(images)
                else:
                    raise AttributeError("Could not find 'encoder' or 'netEnc' in model.")
            
            # Flatten: [Batch, C, H, W] -> [Batch, Features]
           
            #print("Latent shape BEFORE flatten:", curr_latents.shape)
            curr_latents = curr_latents.view(curr_latents.size(0), -1)
            #print("Latent shape AFTER flatten:", curr_latents.shape)
            # Store latents
            latents_list.append(curr_latents.cpu().numpy())
            
            """
            AHORA AGREGUE LO DEL POOLING Y LO DE LAS DIMENSIONES
            # --- GLOBAL POOLING ---
            if curr_latents.ndim == 4:  # [B, C, H, W]
                curr_latents = nn.functional.adaptive_avg_pool2d(curr_latents, (1, 1))
            curr_latents = curr_latents.view(curr_latents.size(0), -1)  # [B, C]
            
            latents_list.append(curr_latents.cpu().numpy())
            """
            # Store labels (Force specific label if provided, else use target)
            if force_label is not None:
                # Create array of the forced label (e.g., all 0s or all 1s)
                labels_list.append(np.full(images.size(0), force_label))
            else:
                labels_list.append(targets.numpy())
            
            total_samples += images.size(0)
            if max_samples and total_samples >= max_samples:
                break
                
    if len(latents_list) == 0:
        return np.array([]), np.array([])

    # Concatenate all batches
    latents = np.concatenate(latents_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    
    # Trim if we overshot the max_samples
    if max_samples:
        latents = latents[:max_samples]
        labels = labels[:max_samples]
        
    return latents, labels

# ----------------------------
# 3. TSNE & PLOTTING
# ----------------------------
def plot_tsne(latents, labels, model_name="Model"):
    print(f"Computing t-SNE on {latents.shape[0]} samples with {latents.shape[1]} dimensions...")
    
    # Run TSNE (Updated args)
    latents = PCA(n_components=50).fit_transform(latents) ###agregado 
    tsne = TSNE(n_components=2, random_state=42, perplexity=40, max_iter=1000, learning_rate='auto', init='pca')
    tsne_results = tsne.fit_transform(latents)
    
    print("Plotting...")
    plt.figure(figsize=(10, 8))
    
    # Create masks for classes
    # Ensure labels are integers for comparison
    labels = labels.astype(int)
    benign_mask = (labels == 0)
    malign_mask = (labels == 1)
    
    print(f" - Benign samples plotted: {np.sum(benign_mask)}")
    print(f" - Malignant samples plotted: {np.sum(malign_mask)}")
    
    # Scatter Plot
    plt.scatter(tsne_results[benign_mask, 0], tsne_results[benign_mask, 1], 
                c='green', label='Benign', alpha=0.6, s=15, edgecolors='none')
    
    plt.scatter(tsne_results[malign_mask, 0], tsne_results[malign_mask, 1], 
                c='red', label='Malignant', alpha=0.6, s=15, edgecolors='none')
    
    plt.title(f"t-SNE Visualization of Latent Space\n({model_name})", fontsize=14)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    
    if SAVE_FIG:
        os.makedirs("results", exist_ok=True)
        save_path = f"results/TSNE_{model_name.replace(' ', '_')}.png"
        plt.savefig(save_path, dpi=300)
        print(f"Saved t-SNE plot to {save_path}")
        
    plt.show()

# ----------------------------
# MAIN
# ----------------------------
def main():
    # 1. Load Model
    model = load_model(config_id="3", model_path=MODEL_PATH, model_name=MODEL_NAME)
    
    # Create embedding wrapper (NO freeze)
    #embedding_model = AEEmbeddingNet(model).to(DEVICE)
    #embedding_model.eval()
    
    # 2. Load Data Separately
    # Loading separately allows us to force the labels (0 for Benign, 1 for Malignant)
    # This avoids issues if the dataset returns masks as targets
    dataset_benign = HelicoAnnotated(only_negative=True, load_ram=False)
    dataset_malign = HelicoAnnotated(only_positive=True, load_ram=False)
    
    loader_benign = DataLoader(dataset_benign, batch_size=BATCH_SIZE, shuffle=True, collate_fn=annotated_collate)
    loader_malign = DataLoader(dataset_malign, batch_size=BATCH_SIZE, shuffle=True, collate_fn=annotated_collate)
    
    # 3. Get Vectors
    # We take half of MAX_SAMPLES from each to keep the plot balanced
    samples_per_class = MAX_SAMPLES // 2 if MAX_SAMPLES else None
    
    latents_b, labels_b = get_latent_vectors(loader_benign, model, MODEL_NAME, samples_per_class, force_label=0, embedding_dim=128)
    latents_m, labels_m = get_latent_vectors(loader_malign, model, MODEL_NAME, samples_per_class, force_label=1, embedding_dim=128)
    
    # Concatenate
    if len(latents_b) > 0 and len(latents_m) > 0:
        latents = np.concatenate([latents_b, latents_m], axis=0)
        labels = np.concatenate([labels_b, labels_m], axis=0)
        
        # 4. Visualize
        plot_tsne(latents, labels, model_name=MODEL_NAME)
    else:
        print("Error: Could not extract latents from one or both datasets.")

if __name__ == "__main__":
    main()