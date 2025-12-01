import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import os

# --- Import your local modules ---
from Models.AEmodels import AutoEncoderCNN
from train_conv_ae import AEConfigs
from dataset import HelicoAnnotated, annotated_collate

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_EXAMPLES = 5  # For the reconstruction check
TSNE_SAMPLES = 1000 # Total points to plot
RESULTS_DIR = "tsne_results" # Folder to save images

# Create results folder if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_model(config_id=1, model_path=None):
    net_paramsEnc, net_paramsDec, inputmodule_paramsEnc, inputmodule_paramsDec = AEConfigs(str(config_id))
    model = AutoEncoderCNN(
        inputmodule_paramsEnc,
        net_paramsEnc,
        inputmodule_paramsDec,
        net_paramsDec
    )
    if model_path:
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            print(f"Loaded model from {model_path}")
        else:
            print(f"Warning: Model path {model_path} not found. Initializing random weights.")
    
    model.to(DEVICE)
    model.eval()
    return model

def get_examples(dataloader, model, num_examples):
    """ Get input images and their reconstructions. """
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
            
            outputs = model(images)
            
            for img, out in zip(images, outputs):
                if count < 2:  # Skip first few
                    count += 1
                    continue
                
                examples.append((img.cpu(), out.cpu()))
                if len(examples) >= num_examples:
                    return examples
    return examples

def get_latent_vectors_and_images(dataloader, model, max_samples=500):
    """
    Extracts embeddings AND keeps the original images for visualization.
    """
    vectors = []
    stored_imgs = []
    count = 0
    
    print(f"Extracting embeddings (limit {max_samples})...")
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch

            # Preprocessing
            images = images.to(DEVICE).float()
            if images.max() > 1:
                images = images / 255.0
            
            # Extract Embeddings
            if hasattr(model, 'encoder'):
                embedding = model.encoder(images)
            else:
                raise AttributeError("Model missing 'encoder' attribute.")

            embedding = embedding.view(embedding.size(0), -1)
            
            # Store Data
            vectors.append(embedding.cpu().numpy())
            
            # Store images (CHW -> HWC for plotting)
            # We detach and move to CPU immediately to save GPU RAM
            imgs_np = images.cpu().permute(0, 2, 3, 1).numpy()
            stored_imgs.append(imgs_np)
            
            count += images.size(0)
            if count >= max_samples:
                break
                
    # Concatenate and slice to max_samples
    return (np.concatenate(vectors, axis=0)[:max_samples], 
            np.concatenate(stored_imgs, axis=0)[:max_samples])

# --- Main Execution ---

# 1. Load Data and Model
model_path = "path/to/your/checkpoint.pth" # UPDATE THIS PATH
model = load_model(config_id=1, model_path=model_path)

dataset_benign = HelicoAnnotated(only_negative=True, load_ram=False)
dataset_malign = HelicoAnnotated(only_positive=True, load_ram=False)

dataloader_benign = DataLoader(dataset_benign, batch_size=BATCH_SIZE, shuffle=False, collate_fn=annotated_collate)
dataloader_malign = DataLoader(dataset_malign, batch_size=BATCH_SIZE, shuffle=False, collate_fn=annotated_collate)

# 2. Quick Reconstruction Check
print("Generating reconstruction check...")
benign_examples = get_examples(dataloader_benign, model, NUM_EXAMPLES)
fig, axes = plt.subplots(2, NUM_EXAMPLES, figsize=(12, 4))
for idx in range(NUM_EXAMPLES):
    orig = benign_examples[idx][0].permute(1, 2, 0).numpy()
    recon = benign_examples[idx][1].permute(1, 2, 0).numpy()
    axes[0, idx].imshow(np.clip(orig, 0, 1)); axes[0, idx].axis('off')
    axes[1, idx].imshow(np.clip(recon, 0, 1)); axes[1, idx].axis('off')
axes[0, 0].set_title("Original"); axes[1, 0].set_title("Reconstructed")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "reconstruction_check.png"))
plt.close()
print(f"Saved reconstruction check to {RESULTS_DIR}/reconstruction_check.png")

# 3. T-SNE Preparation
print("Preparing T-SNE data...")
samples_per_class = TSNE_SAMPLES // 2

# Get vectors AND images
benign_vecs, benign_imgs = get_latent_vectors_and_images(dataloader_benign, model, max_samples=samples_per_class)
malign_vecs, malign_imgs = get_latent_vectors_and_images(dataloader_malign, model, max_samples=samples_per_class)

# Create labels (0: Benign, 1: Malign)
benign_labels = np.zeros(len(benign_vecs))
malign_labels = np.ones(len(malign_vecs))

# Combine
X = np.concatenate([benign_vecs, malign_vecs], axis=0)
y = np.concatenate([benign_labels, malign_labels], axis=0)
all_images = np.concatenate([benign_imgs, malign_imgs], axis=0)

print(f"Running T-SNE on {X.shape[0]} samples...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_embedded = tsne.fit_transform(X)

# 4. Plot T-SNE
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='bwr', alpha=0.6, s=20)
plt.title("t-SNE: Blue=Benign, Red=Malignant")
plt.colorbar(scatter, ticks=[0, 1], label='Class')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(RESULTS_DIR, "tsne_plot.png"))
plt.close()
print(f"Saved t-SNE plot to {RESULTS_DIR}/tsne_plot.png")

# 5. CLUSTER INSPECTION (New Section)
# We use KMeans to mathematically identify the two clusters seen in the plot
print("Analyzing Clusters...")
kmeans = KMeans(n_clusters=2, random_state=42).fit(X_embedded)
cluster_ids = kmeans.labels_

# Show samples from Cluster 0
idxs_c0 = np.where(cluster_ids == 0)[0]
print(f"\nCluster A contains {len(idxs_c0)} images.")
print(f"Composition: {np.mean(y[idxs_c0]==0)*100:.1f}% Benign, {np.mean(y[idxs_c0]==1)*100:.1f}% Malign")

fig, axes = plt.subplots(1, 5, figsize=(15, 3))
fig.suptitle(f"Samples from Cluster A (Dominant content check)", fontsize=14)
for i, idx in enumerate(np.random.choice(idxs_c0, 5, replace=False)):
    axes[i].imshow(np.clip(all_images[idx], 0, 1))
    axes[i].set_title("Benign" if y[idx]==0 else "Malign")
    axes[i].axis('off')
plt.savefig(os.path.join(RESULTS_DIR, "cluster_A_samples.png"))
plt.close()

# Show samples from Cluster 1
idxs_c1 = np.where(cluster_ids == 1)[0]
print(f"\nCluster B contains {len(idxs_c1)} images.")
print(f"Composition: {np.mean(y[idxs_c1]==0)*100:.1f}% Benign, {np.mean(y[idxs_c1]==1)*100:.1f}% Malign")

fig, axes = plt.subplots(1, 5, figsize=(15, 3))
fig.suptitle(f"Samples from Cluster B (Dominant content check)", fontsize=14)
for i, idx in enumerate(np.random.choice(idxs_c1, 5, replace=False)):
    axes[i].imshow(np.clip(all_images[idx], 0, 1))
    axes[i].set_title("Benign" if y[idx]==0 else "Malign")
    axes[i].axis('off')
plt.savefig(os.path.join(RESULTS_DIR, "cluster_B_samples.png"))
plt.close()
print(f"Saved cluster samples to {RESULTS_DIR}")