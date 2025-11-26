import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import os

# --- Import your local modules ---
# Ensure VAECNN is imported here. Assuming it's in Models.AEmodels or similar.
# If it's in a new file, change this import.
from Models.AEmodels import VAECNN 
from train_conv_ae import AEConfigs
from dataset import HelicoAnnotated, annotated_collate

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
NUM_EXAMPLES = 10  # For the reconstruction check
TSNE_SAMPLES = 1000 # Total points to plot
RESULTS_DIR = "vae_tsne_results" # Folder to save images

# Create results folder for SSH viewing
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_model(config_id=1, model_path=None):
    """
    Loads the VAE model. 
    NOTE: This assumes AEConfigs now returns 5 values to match VAECNN init.
    If AEConfigs only returns 4, you need to manually define net_paramsRep here.
    """
    try:
        # Attempt to unpack 5 values (VAE style)
        configs = AEConfigs(str(config_id))
        if len(configs) == 5:
            inputmodule_paramsEnc, net_paramsEnc, inputmodule_paramsDec, net_paramsDec, net_paramsRep = configs
            
        model = VAECNN(
            inputmodule_paramsEnc,
            net_paramsEnc,
            inputmodule_paramsDec,
            net_paramsDec,
            net_paramsRep
        )
    except Exception as e:
        print(f"Error loading config: {e}")
        raise e

    if model_path:
        if os.path.exists(model_path):
            # Load weights
            state_dict = torch.load(model_path, map_location=DEVICE)
            model.load_state_dict(state_dict)
            print(f"Loaded model from {model_path}")
        else:
            raise ValueError(f"Model weights not found at: {model_path}")
    
    model.to(DEVICE)
    model.eval()
    return model

def get_examples(dataloader, model, num_examples):
    """
    Get input images and their reconstructions.
    Handles VAE output tuple: (recon, mu, logvar)
    """
    examples = []
    count = 0
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch
            
            images = images.to(DEVICE).float()
            if images.max() > 1:
                images = images / 255.0
            
            # VAE Forward returns (recon, mu, logvar)
            outputs = model(images)
            reconstructions = outputs[0] # Take only the first element
            
            for img, recon in zip(images, reconstructions):
                if count < 3:  # Skip first few
                    count += 1
                    continue
                
                examples.append((img.cpu(), recon.cpu()))
                
                if len(examples) >= num_examples:
                    return examples
    return examples

def get_latent_vectors(dataloader, model, max_samples=500):
    """
    Extracts the 'mu' vector from the VAE.
    We use 'mu' for t-SNE because 'z' contains random sampling noise 
    which distorts the visualization.
    """
    vectors = []
    count = 0
    
    print("Extracting VAE Latent Vectors (Mu)...")
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch

            images = images.to(DEVICE).float()
            if images.max() > 1:
                images = images / 255.0
            
            # --- VAE SPECIFIC EXTRACTION ---
            # We replicate the first half of the forward pass to get 'mu'
            # 1. Encoder
            x = model.encoder(images)
            
            # 2. Flatten
            # Matches logic in VAECNN.forward: x = x.view(n, -1)
            x = x.view(x.size(0), -1)
            
            # 3. Get Mu (Mean)
            # We access the linear layer directly
            mu = model.encoder_mu(x)
            
            # Store
            vectors.append(mu.cpu().numpy())
            count += images.size(0)
            
            if count >= max_samples:
                break
                
    return np.concatenate(vectors, axis=0)[:max_samples]

# --- MAIN EXECUTION ---

# 1. Load Model
# Update this path to your VAE .pth file
model_path = "/fhome/vlia01/Medical-Imaging/slurm_output/vae_2.pth" 
model = load_model(config_id=2, model_path=model_path)

# 2. Load Data
dataset_benign = HelicoAnnotated(only_negative=True, load_ram=False)
dataset_malign = HelicoAnnotated(only_positive=True, load_ram=False)

dataloader_benign = DataLoader(dataset_benign, batch_size=BATCH_SIZE, shuffle=False, collate_fn=annotated_collate)
dataloader_malign = DataLoader(dataset_malign, batch_size=BATCH_SIZE, shuffle=False, collate_fn=annotated_collate)

# 3. Check Reconstructions (Sanity Check)
print("Generating reconstruction check...")
try:
    benign_examples = get_examples(dataloader_benign, model, NUM_EXAMPLES)
    fig, axes = plt.subplots(2, NUM_EXAMPLES, figsize=(12, 4))
    for idx in range(NUM_EXAMPLES):
        # Original
        orig = benign_examples[idx][0].permute(1, 2, 0).numpy()
        axes[0, idx].imshow(np.clip(orig, 0, 1))
        axes[0, idx].axis('off')
        
        # Recon
        recon = benign_examples[idx][1].permute(1, 2, 0).numpy()
        axes[1, idx].imshow(np.clip(recon, 0, 1))
        axes[1, idx].axis('off')
        
    axes[0, 0].set_title("Original")
    axes[1, 0].set_title("VAE Recon")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "vae_reconstructions.png"))
    plt.close()
    print(f"Saved reconstructions to {RESULTS_DIR}/vae_reconstructions.png")
except Exception as e:
    print(f"Skipping reconstruction check due to error: {e}")

# 4. Prepare t-SNE Data
samples_per_class = TSNE_SAMPLES // 2

benign_vecs = get_latent_vectors(dataloader_benign, model, max_samples=samples_per_class)
malign_vecs = get_latent_vectors(dataloader_malign, model, max_samples=samples_per_class)

benign_labels = np.zeros(len(benign_vecs))
malign_labels = np.ones(len(malign_vecs))

X = np.concatenate([benign_vecs, malign_vecs], axis=0)
y = np.concatenate([benign_labels, malign_labels], axis=0)

# 5. Run t-SNE
print(f"Running t-SNE on {X.shape[0]} samples with embedding dim {X.shape[1]}...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, metric='euclidean', init='pca', n_iter=1000)
X_embedded = tsne.fit_transform(X)

# 6. Plot Results
plt.figure(figsize=(10, 8))

# Plot Benign
plt.scatter(
    X_embedded[y == 0, 0], 
    X_embedded[y == 0, 1], 
    c='blue', label='Benign', alpha=0.5, s=20, edgecolors='none'
)

# Plot Malign
plt.scatter(
    X_embedded[y == 1, 0], 
    X_embedded[y == 1, 1], 
    c='red', label='Malign', alpha=0.5, s=20, edgecolors='none'
)

plt.title("t-SNE: VAE Latent Space (Mean)")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.legend()
plt.grid(True, alpha=0.3)

save_path = os.path.join(RESULTS_DIR, 'vae_tsne.png')
plt.savefig(save_path)
plt.close()

print(f"Done! t-SNE saved to: {save_path}")