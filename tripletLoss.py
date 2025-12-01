# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import wandb
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch.nn.functional as F

from dataset import HelicoAnnotated, annotated_collate
from Models.AEmodels import AutoEncoderCNN, VAECNN
from train_conv_ae import AEConfigs
from train_conv_vae import VAEConfigs

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "/fhome/vlia01/Medical-Imaging/slurm_output/config_three.pth"
#MODEL_PATH = "/fhome/vlia01/Medical-Imaging/slurm_output/vae_3.pth"
BATCH_SIZE = 128 #(en vae no me deja 128 poner 96)
MODEL_NAME = "Variational Autoencoder" # "Autoencoder" or "Variational Autoencoder"
SAVE_FIG = True
MAX_SAMPLES = 2000 # Total samples to visualize (will try to take half from each class)
LEARNING_RATE = 1e-3
EMBEDDING_DIM = 256
NUM_EPOCHS = 50
RESULTS_DIR = "/fhome/vlia01/Medical-Imaging/results"
MARGIN = 0.8
NUM_WORKERS = 4

os.makedirs(RESULTS_DIR, exist_ok=True)

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
    

class EmbeddingNet(nn.Module):
    def __init__(self, base_model, embedding_dim=128):
        super().__init__()
        self.encoder = base_model.encoder
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.fc = nn.LazyLinear(embedding_dim)
    def forward(self, x):
        z = self.encoder(x)           # [B, C, H, W]
        z = self.pool(z)              # [B, C, 8, 8]
        z = z.view(z.size(0), -1)     # flatten
        z = self.fc(z)                # [B, embedding_dim]
        z = F.normalize(z, p=2, dim=1) # L2 normalization
        return z


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


class TripletDataset(Dataset):
    """
    TripletDataset robusto:
    - Usa safe_get() para saltar imagenes malas.
    - Convierte labels de HelicoAnnotated (que pueden ser mascaras) a binario 0/1.
    - Elimina cualquier muestra que NO tenga etiqueta valida.
    - Construye triplets limpios y consistentes.
    """
    def __init__(self, base_dataset, max_retries=20):
        super().__init__()
        self.base = base_dataset
        self.max_retries = max_retries

        self.valid_indices = []
        self.labels = []

        print("Building TripletDataset ...")

        # ---- 1. Filtrar muestras validas ----
        for idx in range(len(self.base)):
            sample = self.safe_get(idx)
            if sample is None:
                continue

            img, label = sample[:2]

            # Convertir mascaras ? valor binario
            if isinstance(label, torch.Tensor):
                label = int(label.float().mean().item() > 0)

            # Mapear -1 ? 0, 1 ? 1
            if label == -1:
                mapped = 0
            elif label == 1:
                mapped = 1
            elif label in (0, 1):  # ya binario
                mapped = int(label)
            else:
                continue  # ignorar etiquetas raras

            self.valid_indices.append(idx)
            self.labels.append(mapped)

        self.labels = np.array(self.labels)

        self.label_to_indices = {
            0: np.where(self.labels == 0)[0],
            1: np.where(self.labels == 1)[0],
        }

        print("TripletDataset loaded:")
        print("  Benign (0):     ", len(self.label_to_indices[0]))
        print("  Malignant (1):  ", len(self.label_to_indices[1]))
        print("  Total usable:   ", len(self.valid_indices))

        if len(self.label_to_indices[0]) == 0 or len(self.label_to_indices[1]) == 0:
            raise RuntimeError("ERROR: A class has zero valid samples!")

    def safe_get(self, index):
        #Get image and label, skip if None
        data = self.base[index]
        while data is None:
            index = (index + 1) % len(self.base)
            data = self.base[index]
        return data

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, i):

        # Anchor
        anchor_base_idx = self.valid_indices[i]
        anchor_data = self.safe_get(anchor_base_idx)
        if anchor_data is None:
            # buscar otro indice valido
            j = random.randint(0, len(self.valid_indices)-1)
            anchor_base_idx = self.valid_indices[j]
            anchor_data = self.safe_get(anchor_base_idx)
        
        anchor_img, anchor_label = anchor_data[:2]
        # Normalizar el label a 0/1
        if isinstance(anchor_label, torch.Tensor):
            anchor_label = int(anchor_label.float().mean().item() > 0)
        elif anchor_label == -1:
            anchor_label = 0
        else:
            anchor_label = int(anchor_label)

        # Positive
        same_pool = self.label_to_indices[anchor_label]
        pos_choice = i
        while pos_choice == i:
            pos_choice = int(random.choice(same_pool))
        pos_base_idx = self.valid_indices[pos_choice]
        positive_img = self.safe_get(pos_base_idx)[0]

        # Negative
        neg_label = 1 - anchor_label
        neg_pool = self.label_to_indices[neg_label]
        neg_choice = int(random.choice(neg_pool))
        neg_base_idx = self.valid_indices[neg_choice]
        negative_img = self.safe_get(neg_base_idx)[0]

        return anchor_img, positive_img, negative_img


def get_latent_vectors(dataloader, embedding_model, max_samples=None, force_label=None):
    
    #Extract latent vectors from a trained EmbeddingNet.
    # embedding_model: the trained EmbeddingNet (encoder -> pool -> FC -> normalize)
    #not needed to handle AE or VAE separatelly, since they are handled in EmbeddingNet 
    
    embedding_model.eval()
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

            # Use the full embedding model
            curr_latents = embedding_model(images)

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
    print("Embedding latent shape:", latents.shape)

    return latents, labels


def plot_tsne(latents, labels, model_name="Model", save_dir=RESULTS_DIR):
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
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"TripletLoss_{model_name.replace(' ', '_')}.png")
        plt.savefig(save_path, dpi=300)
        print(f"Saved TSNE to {save_path}")
        # also log to wandb if active
        try:
            wandb.log({"tsne_plot": wandb.Image(save_path)})
        except Exception:
            pass

    plt.show()
    

def init_run():
    config = {
        "BATCH_SIZE": BATCH_SIZE,
        "LEARNING_RATE": LEARNING_RATE,
        "EMBEDDING_DIM": EMBEDDING_DIM,
        "NUM_EPOCHS": NUM_EPOCHS,
        "BATCH_SIZE": BATCH_SIZE,
        "NUM_WORKERS": NUM_WORKERS,
        "MARGIN": MARGIN,
        "PROJECT_NAME": "helicodataset-triplet-embedding"
    }
    wandb.init(project=config["PROJECT_NAME"], config=config)
    cfg = wandb.config
    return cfg

def main():
    cfg = init_run()
    
    model = load_model(config_id="3", model_path=MODEL_PATH, model_name=MODEL_NAME)
    
    print(f"Starting embbedings...")
    embedding_model = EmbeddingNet(model, embedding_dim=EMBEDDING_DIM).to(DEVICE)

    # Prepare training objects
    criterion = TripletLoss(margin=float(MARGIN))
    
    # ensure LazyLinear parameters are initialized (one dummy forward)
    with torch.no_grad():
        try:
            dummy = torch.zeros(1, 3, 256, 256).to(DEVICE)  
            embedding_model(dummy)
        except Exception as e:
            print("Warning: dummy forward failed:", e)
    
    optimizer = torch.optim.Adam(embedding_model.parameters(), lr=float(LEARNING_RATE))

    # Create a TripletDataset and DataLoader
    triplet_dataset = TripletDataset(base_dataset=HelicoAnnotated(load_ram=False))
    triplet_loader = DataLoader(
        triplet_dataset,
        batch_size=int(BATCH_SIZE),
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=False
    )

    try:
        wandb.watch(embedding_model, criterion, log="all", log_freq=100)
    except Exception:
        pass
        
    epoch_losses = []
    print("Starting triplet training...")
    embedding_model.train()
    for epoch in range(int(NUM_EPOCHS)):
        running_loss = 0.0
        n_batches = 0
        for anchor, positive, negative in triplet_loader:
            anchor = anchor.to(DEVICE).float()
            positive = positive.to(DEVICE).float()
            negative = negative.to(DEVICE).float()

            # Normalize image range if needed
            if anchor.max() > 1:
                anchor = anchor / 255.0
                positive = positive / 255.0
                negative = negative / 255.0

            z_a = embedding_model(anchor)
            z_p = embedding_model(positive)
            z_n = embedding_model(negative)

            loss = criterion(z_a, z_p, z_n)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

            # log batch loss
            try:
                wandb.log({"batch_train_loss": loss.item()})
            except Exception:
                pass

        avg_loss = running_loss / max(1, n_batches)
        epoch_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{int(NUM_EPOCHS)}] - Loss: {avg_loss:.6f}")
        wandb.log({"epoch": epoch + 1, "epoch_train_loss": avg_loss})

    print("Training finished. Switching embedding model to eval() for extraction.")
    embedding_model.eval()
    
    dataset_benign = HelicoAnnotated(only_negative=True, load_ram=False)
    dataset_malign = HelicoAnnotated(only_positive=True, load_ram=False)

    loader_benign = DataLoader(dataset_benign, batch_size=int(BATCH_SIZE), shuffle=True, collate_fn=annotated_collate, num_workers=NUM_WORKERS, pin_memory=True)
    loader_malign = DataLoader(dataset_malign, batch_size=int(BATCH_SIZE), shuffle=True, collate_fn=annotated_collate, num_workers=NUM_WORKERS, pin_memory=True)

    samples_per_class = MAX_SAMPLES // 2 if MAX_SAMPLES else None

    latents_b, labels_b = get_latent_vectors(loader_benign, embedding_model, samples_per_class, force_label=0)
    latents_m, labels_m = get_latent_vectors(loader_malign, embedding_model, samples_per_class, force_label=1)

    if len(latents_b) > 0 and len(latents_m) > 0:
        latents = np.concatenate([latents_b, latents_m], axis=0)
        labels = np.concatenate([labels_b, labels_m], axis=0)

        plot_tsne(latents, labels, model_name=MODEL_NAME, save_dir=RESULTS_DIR)
    else:
        print("Error: Could not extract latents from one or both datasets.")

    try:
        wandb.finish()
    except Exception:
        pass

if __name__ == "__main__":
    print("Using device:", DEVICE)
    main()