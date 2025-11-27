import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import os
import random
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
BATCH_SIZE = 64 
MODEL_NAME = "Autoencoder" # "Autoencoder" or "Variational Autoencoder"
SAVE_FIG = True
MAX_SAMPLES = 2000 # Total samples to visualize (will try to take half from each class)
LEARNING_RATE = 1e-4
EMBEDDING_DIM = 128
NUM_EPOCHS = 50

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
# 2. EMBEDDING NETWORK FOR TRIPLET
# ----------------------------
class EmbeddingNet(nn.Module):
    def __init__(self, base_model, embedding_dim=128):
        super().__init__()
        self.encoder = base_model.encoder
        self.pool = nn.AdaptiveMaxPool2d((8, 8))
        self.fc = nn.LazyLinear(embedding_dim)

    def forward(self, x):
        z = self.encoder(x)           # [B, C, H, W]
        z = self.pool(z)              # [B, C, 8, 8]
        z = z.view(z.size(0), -1)     # flatten
        z = self.fc(z)                # [B, embedding_dim]
        z = F.normalize(z, p=2, dim=1)
        return z


# ---------------------------
# 3. Triplet Loss
# ---------------------------

class TripletLoss(nn.Module):
    """
    L = max(0, d(a,p) - d(a,n) + margin)
    where d is squared Euclidean distance.
    """
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def calc_euclidean(self, x1, x2):
        # x1, x2: [B, D] -> returns [B]
        return (x1 - x2).pow(2).sum(dim=1)

    def forward(self, anchor, positive, negative):
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()


# ----------------------------------------
# 4. TripletDataset built from QuironDataset
# ----------------------------------------

class TripletDataset(Dataset):
    def __init__(self, base_dataset):
        super(TripletDataset, self).__init__()
        self.base = base_dataset
        self.samples = base_dataset.samples  # list of (path, label, pat_id)

        # Build dictionary: label - list of indices with that label
        self.label_to_indices = {}
        for idx, (_, label, _) in enumerate(self.samples):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)

        self.labels = np.array([lbl for (_, lbl, _) in self.samples])
        self.unique_labels = list(self.label_to_indices.keys())

        if len(self.unique_labels) < 2:
            raise RuntimeError("Need at least 2 different labels to form triplets.")

    def __len__(self):
        return len(self.samples)
        
    def safe_get(self, index):
        """Get image and label, skip if None"""
        data = self.base[index]
        while data is None:
            index = (index + 1) % len(self.base)
            data = self.base[index]
        return data

    def __getitem__(self, idx):
        # Anchor
        anchor_img, anchor_label = self.safe_get(idx)
    
        # Positive (same label)
        same_label_indices = self.label_to_indices[anchor_label]
        if len(same_label_indices) > 1:
            pos_idx = idx
            while pos_idx == idx:
                pos_idx = random.choice(same_label_indices)
        else:
            pos_idx = idx
        positive_img, _ = self.safe_get(pos_idx)
    
        # Negative (different label)
        neg_label_choices = [l for l in self.unique_labels if l != anchor_label]
        neg_label = random.choice(neg_label_choices)
        neg_idx = random.choice(self.label_to_indices[neg_label])
        negative_img, _ = self.safe_get(neg_idx)
    
        return anchor_img, positive_img, negative_img



# ----------------------------
# 5. EXTRACT LATENT VECTORS FOR TSNE
# ----------------------------
def get_latent_vectors(dataloader, model, model_name="Autoencoder", max_samples=None, force_label=None):
    latents_list = []
    labels_list = []
    total_samples = 0
    
    dataset_name = "Benign" if force_label == 0 else "Malignant"
    print(f"Extracting latent vectors for {dataset_name}...")

    with torch.no_grad():
        for batch in dataloader:
            images, targets = batch
            images = images.to(DEVICE).float()
            if images.dtype == torch.uint8:
                images = images.float() / 255.0

            if model_name == "Variational Autoencoder":
                _, mu, _ = model(images)
                curr_latents = mu
            else:
                curr_latents = model.encoder(images)

            if curr_latents.ndim == 4:
                curr_latents = F.adaptive_max_pool2d(curr_latents, (8, 8))
            curr_latents = curr_latents.view(curr_latents.size(0), -1)
            latents_list.append(curr_latents.cpu().numpy())

            if force_label is not None:
                labels_list.append(np.full(images.size(0), force_label))
            else:
                labels_list.append(targets.numpy())

            total_samples += images.size(0)
            if max_samples and total_samples >= max_samples:
                break

    latents = np.concatenate(latents_list, axis=0) if latents_list else np.array([])
    labels = np.concatenate(labels_list, axis=0) if labels_list else np.array([])

    if max_samples:
        latents = latents[:max_samples]
        labels = labels[:max_samples]

    return latents, labels

# ----------------------------
# 6. TSNE PLOTTING
# ----------------------------
def plot_tsne(latents, labels, model_name="Model"):
    print(f"Computing t-SNE on {latents.shape[0]} samples with {latents.shape[1]} dimensions...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=40, max_iter=1000, learning_rate='auto', init='pca')
    tsne_results = tsne.fit_transform(latents)

    print("Plotting...")
    plt.figure(figsize=(12, 10))
    labels = labels.astype(int)
    benign_mask = (labels == 0)
    malign_mask = (labels == 1)

    plt.scatter(tsne_results[benign_mask, 0], tsne_results[benign_mask, 1], 
                c='blue', label='Benign (-1)', alpha=0.8, s=40, edgecolors='none')
    plt.scatter(tsne_results[malign_mask, 0], tsne_results[malign_mask, 1], 
                c='darkturquoise', label='Malignant (1)', alpha=0.8, s=40, edgecolors='none')

    plt.title(f"t-SNE of Latent Space\n{model_name} conf3 + TRIPLET LOSS", fontsize=16)
    plt.xlabel("t-SNE Dimension 1", fontsize=14)
    plt.ylabel("t-SNE Dimension 2", fontsize=14)
    plt.legend(loc='best', fontsize=12)
    plt.grid(alpha=0.3)

    if SAVE_FIG:
        save_dir = "/fhome/vlia01/Medical-Imaging/results"
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"TripletLoss_{model_name.replace(' ', '_')}.png"), dpi=300)
    plt.show()

# ----------------------------
# 7. MAIN
# ----------------------------
def main():
    # Load AE/VAE model
    model = load_model(config_id="3", model_path=MODEL_PATH, model_name=MODEL_NAME)
    
    ###
    # Triplet Training
    ###
    print(f"Starting embbedings...")
    embedding_model = EmbeddingNet(model, embedding_dim=EMBEDDING_DIM).to(DEVICE)
    embedding_model.train()
    criterion = TripletLoss(margin=1.0)
    optimizer = torch.optim.Adam(embedding_model.parameters(), lr=LEARNING_RATE)

    triplet_dataset = TripletDataset(base_dataset=HelicoAnnotated(load_ram=False))
    triplet_loader = DataLoader(triplet_dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        for anchor, positive, negative in triplet_loader:
            anchor = anchor.to(DEVICE).float()
            positive = positive.to(DEVICE).float()
            negative = negative.to(DEVICE).float()
            if anchor.max() > 1:
                anchor /= 255.0
                positive /= 255.0
                negative /= 255.0

            z_anchor = embedding_model(anchor)
            z_pos = embedding_model(positive)
            z_neg = embedding_model(negative)

            loss = criterion(z_anchor, z_pos, z_neg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {epoch_loss/len(triplet_loader):.4f}")

    ###
    # Extract embeddings for TSNE
    ###
    dataset_benign = HelicoAnnotated(only_negative=True, load_ram=False)
    dataset_malign = HelicoAnnotated(only_positive=True, load_ram=False)

    loader_benign = DataLoader(dataset_benign, batch_size=BATCH_SIZE, shuffle=True, collate_fn=annotated_collate)
    loader_malign = DataLoader(dataset_malign, batch_size=BATCH_SIZE, shuffle=True, collate_fn=annotated_collate)

    samples_per_class = MAX_SAMPLES // 2 if MAX_SAMPLES else None

    latents_b, labels_b = get_latent_vectors(loader_benign, embedding_model, MODEL_NAME, samples_per_class, force_label=0)
    latents_m, labels_m = get_latent_vectors(loader_malign, embedding_model, MODEL_NAME, samples_per_class, force_label=1)

    if len(latents_b) > 0 and len(latents_m) > 0:
        latents = np.concatenate([latents_b, latents_m], axis=0)
        labels = np.concatenate([labels_b, labels_m], axis=0)
        plot_tsne(latents, labels, model_name=MODEL_NAME)
    else:
        print("Error: Could not extract latents from one or both datasets.")

if __name__ == "__main__":
    main()