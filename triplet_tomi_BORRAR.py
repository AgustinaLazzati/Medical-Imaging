# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, default_collate
from sklearn.manifold import TSNE
import wandb

from pytorch_metric_learning import losses, miners, distances, reducers, testers
from dataset import HelicoAnnotated, annotated_collate
from Models.AEmodels import AutoEncoderCNN, VAECNN
from train_conv_ae import AEConfigs
from train_conv_vae import VAEConfigs

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_CONFIG = {
    "project_name": "helicodataset-triplet-embedding",
    "model_path": "/fhome/vlia01/Medical-Imaging/slurm_output/config_three.pth",
    "results_dir": "/fhome/vlia01/Medical-Imaging/results",
    "batch_size": 128,
    "learning_rate": 1e-4,
    "embedding_dim": 128,
    "margin": 0.2,
    "miner_type": "hard",
    "epochs": 40,
    "num_workers": 4,
    "max_samples_vis": 2000
}


def safe_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    
    if len(batch) == 0:
        return None
        
    return default_collate(batch)

def load_backbone(model_name="Autoencoder", config_id="3", model_path=None):
    """Loads the pre-trained AE/VAE and freezes it to act as a feature extractor."""
    print(f"Loading {model_name} backbone from {model_path}...")
    
    if model_name == "Autoencoder":
        enc_p, dec_p, in_enc, in_dec = AEConfigs(config_id)
        model = AutoEncoderCNN(in_enc, enc_p, in_dec, dec_p)
    elif model_name == "Variational Autoencoder":
        enc_p, dec_p, in_enc, in_dec, rep_p = VAEConfigs(config_id)
        model = VAECNN(in_enc, enc_p, in_dec, dec_p, rep_p)
    else:
        raise ValueError("Invalid model name")
    
    if model_path and os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
    else:
        print("Warning: Weights not found. Using random initialization.")

    # We only need the encoder
    encoder = model.encoder
    
    # Freeze the backbone
    for param in encoder.parameters():
        param.requires_grad = False
        
    return encoder.to(DEVICE)


class MetricModel(nn.Module):
    """
    Wraps the backbone with a projection head for metric learning.
    Backbone is frozen; only the head is trained.
    """
    def __init__(self, backbone, embedding_dim=128):
        super().__init__()
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool2d((1, 1)) # Global Average Pooling
        
        # Simple projection head
        # Using LazyLinear to automatically infer input shape from backbone
        self.head = nn.Sequential(
            nn.Flatten(),
            # LOL
            nn.LazyLinear(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim),
            # Note: Normalization is usually handled by the Loss function in PML,
            # but you can add F.normalize here if you prefer.
        )

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)
        
        # If the backbone returns a tuple (VAE), take the first element
        if isinstance(features, tuple):
            features = features[0]
            
        pooled = self.pool(features)
        embeddings = self.head(pooled)
        return embeddings

def visualize_space(model, dataloader, save_dir, epoch):
    model.eval()
    all_embeds = []
    all_labels = []
    
    # Extract features
    with torch.no_grad():
        for batch in dataloader:
            # Handle variable unpacking depending on dataset return
            if len(batch) == 2:
                img, label = batch
            else:
                img, label, *_ = batch # Ignore patient ID etc
                
            img = img.to(DEVICE).float()
            # Normalize if needed
            if img.max() > 1: img /= 255.0
                
            emb = model(img)
            # Normalize for visualization cleanliness
            emb = F.normalize(emb, p=2, dim=1)
            
            all_embeds.append(emb.cpu().numpy())
            all_labels.append(label.numpy())
            
            if len(np.concatenate(all_labels)) > DEFAULT_CONFIG["max_samples_vis"]:
                break

    X = np.concatenate(all_embeds)
    y = np.concatenate(all_labels)

    # Compute t-SNE
    tsne = TSNE(n_components=2, init='pca', learning_rate='auto', random_state=42)
    X_tsne = tsne.fit_transform(X)

    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='coolwarm', alpha=0.6)
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title(f"Latent Space (Epoch {epoch})")
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"tsne_epoch_{epoch}.png")
    plt.savefig(save_path)
    plt.close()
    
    return wandb.Image(save_path)

def train_sweep(config=None):
    # Initialize WandB
    run = wandb.init(config=DEFAULT_CONFIG, project=DEFAULT_CONFIG["project_name"])
    cfg = run.config # Access updated config from Sweep

    print(cfg.keys())

    train_dataset = HelicoAnnotated(load_ram=False) 
    print(train_dataset[0])
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=safe_collate,
    )

    # 2. Model Setup
    backbone = load_backbone(model_path=DEFAULT_CONFIG["model_path"]).to(DEVICE)
    model = MetricModel(backbone, embedding_dim=cfg.embedding_dim).to(DEVICE)

    dummy_input = torch.randn(2, 3, 256, 256).to(DEVICE)
    backbone_output = backbone(dummy_input)
    dummy_output = model(dummy_input)

    print(f"backbone output: {backbone_output}")
    print(f"metric output: {dummy_output}")


    optimizer = torch.optim.Adam(model.head.parameters(), lr=cfg.learning_rate)

    distance = distances.LpDistance(normalize_embeddings=True, p=2, power=1)
    
    # Loss: Triplet Margin Loss
    loss_func = losses.TripletMarginLoss(margin=cfg.margin, distance=distance)
    
    miner = miners.TripletMarginMiner(
        margin=cfg.margin, 
        distance=distance, 
        type_of_triplets=cfg.miner_type 
    )

    model.train()

    for epoch in range(cfg.epochs):
        epoch_loss = 0.0
        batch_count = 0
        
        for batch in train_loader:
            if len(batch) == 2:
                images, labels = batch
            else:
                images, labels, *_ = batch
            
            images, labels = images.to(DEVICE, non_blocking=True).float(), labels.to(DEVICE, non_blocking=True)
            if images.max() > 1: images /= 255.0

            optimizer.zero_grad()
            
            embeddings = model(images)
            
            hard_pairs = miner(embeddings, labels)
            
            loss = loss_func(embeddings, labels, hard_pairs)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
            
            # Log miner stats (how many triplets were active?)
            wandb.log({
                "batch_loss": loss.item(),
                "active_triplets": miner.num_triplets
            })

        avg_loss = epoch_loss / batch_count
        print(f"Epoch {epoch+1}/{cfg.epochs} | Loss: {avg_loss:.4f} | Margin: {cfg.margin}")
        
        wandb.log({"epoch_loss": avg_loss, "epoch": epoch+1})

        # Visualization every 10 epochs
        if (epoch + 1) % 10 == 0:
            tsne_img = visualize_space(model, train_loader, DEFAULT_CONFIG["results_dir"], epoch+1)
            wandb.log({"t-SNE": tsne_img})
            model.train() # Set back to train mode

    run.finish()

# ----------------------------
# 5. SWEEP CONFIGURATION & ENTRY
# ----------------------------

# Define the sweep configuration dict
sweep_configuration = {
    'method': 'bayes', # 'grid', 'random', or 'bayes'
    'metric': {
        'name': 'epoch_loss',
        'goal': 'minimize'   
    },
    'parameters': {
        'margin': {
            'min': 0.05,
            'max': 0.3
        },
    }
}

if __name__ == "__main__":
    print(f"Running on {DEVICE}")
    
    # Option A: Run a single training run (Good for debugging)
    #train_sweep(config=DEFAULT_CONFIG)

    # Option B: Run the Sweep (Uncomment below to activate)
    # 1. Initialize sweep
    sweep_id = wandb.sweep(sweep_configuration, project=DEFAULT_CONFIG["project_name"])
    # 2. Start agent
    wandb.agent(sweep_id, function=train_sweep, count=10) # count = number of runs