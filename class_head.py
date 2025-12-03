# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
import wandb

# Custom imports
from dataset import HelicoAnnotated, annotated_collate
from Models.AEmodels import AutoEncoderCNN, VAECNN
from train_conv_ae import AEConfigs
from train_conv_vae import VAEConfigs

# ----------------------------
# CONFIGURATION
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
AE_MODEL_PATH = "/fhome/vlia01/Medical-Imaging/slurm_output/config_three.pth"
EMBEDDING_MODEL_PATH = "/fhome/vlia01/Medical-Imaging/results/triplet_embedding_model.pth"
RESULTS_DIR = "/fhome/vlia01/Medical-Imaging/results"

# Hyperparams
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
NUM_EPOCHS = 30
EMBEDDING_DIM = 256
CLASSIFIER_HIDDEN_DIM = 64
FREEZE_BACKBONE = True  # True = Train only the new layers; False = Fine-tune everything

# ----------------------------
# 1. MODEL CLASSES
# ----------------------------

class EmbeddingNet(nn.Module):
    """
    Must match the class definition used in the triplet training exactly
    to load the state_dict correctly.
    """
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

class HelicoClassifier(nn.Module):
    """
    The Single Pipeline Model:
    Image -> Encoder -> Embedding -> Classification Head -> Logit
    """
    def __init__(self, embedding_model, input_dim=256, hidden_dim=64):
        super().__init__()
        self.embedding_net = embedding_model
        
        # Binary Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5), # Regularization
            nn.Linear(hidden_dim, 1) # Output 1 raw logit
        )

    def forward(self, x):
        # 1. Get the embedding (normalized)
        # Note: We rely on EmbeddingNet's forward pass
        embedding = self.embedding_net(x)
        
        # 2. Classify
        logit = self.classifier(embedding)
        return logit

# ----------------------------
# 2. LOADING HELPERS
# ----------------------------

def load_base_ae(config_id="3", model_path=None):
    # Load the AE architecture and weights
    print(f"Loading Base AE from {model_path}...")
    net_paramsEnc, net_paramsDec, inputmodule_paramsEnc, inputmodule_paramsDec = AEConfigs(config_id)
    model = AutoEncoderCNN(inputmodule_paramsEnc, net_paramsEnc, inputmodule_paramsDec, net_paramsDec)
    
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    else:
        print("Warning: AE weights not found. Using random init.")
    return model

def build_full_model():
    # 1. Load AE
    ae_model = load_base_ae(config_id="3", model_path=AE_MODEL_PATH)
    
    # 2. Wrap in EmbeddingNet
    embedding_net = EmbeddingNet(ae_model, embedding_dim=EMBEDDING_DIM).to(DEVICE)
    
    # Init LazyLinear with dummy pass
    with torch.no_grad():
        _ = embedding_net(torch.zeros(1, 3, 256, 256).to(DEVICE))

    # 3. Load Embedding Weights
    if os.path.exists(EMBEDDING_MODEL_PATH):
        print(f"Loading Embedding weights from {EMBEDDING_MODEL_PATH}...")
        embedding_net.load_state_dict(torch.load(EMBEDDING_MODEL_PATH, map_location=DEVICE))
    else:
        raise FileNotFoundError("Embedding weights not found! Run embedding training first.")

    # 4. Create Final Classifier
    full_model = HelicoClassifier(embedding_net, input_dim=EMBEDDING_DIM, hidden_dim=CLASSIFIER_HIDDEN_DIM)
    
    return full_model.to(DEVICE)

# ----------------------------
# 3. MAIN TRAINING
# ----------------------------

def main():
    wandb.init(project="helico-binary-classifier")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- Data Prep ---
    # We load both benign and malignant and let DataLoader mix them
    print("Loading Dataset...")
    full_dataset = HelicoAnnotated(load_ram=False) 
    
    # Split into Train/Val (Simple random split for this example)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=4, pin_memory=True, collate_fn=annotated_collate)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=4, pin_memory=True, collate_fn=annotated_collate)

    # --- Model Prep ---
    model = build_full_model()

    if FREEZE_BACKBONE:
        print("Freezing Encoder and Embedding layers...")
        for param in model.embedding_net.parameters():
            param.requires_grad = False
    
    # Optimizer (Only optimize parameters that require grad)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    
    # Loss: BCEWithLogitsLoss is more stable than Sigmoid + BCELoss
    criterion = nn.BCEWithLogitsLoss()

    print(f"Starting training for {NUM_EPOCHS} epochs...")

    for epoch in range(NUM_EPOCHS):
        # --- TRAIN ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch in train_loader:
            images, labels = batch
            
            # Filter invalid labels (if any exist in dataset like -1)
            # We treat 1 as Malignant, 0 (or -1) as Benign
            mask = (labels != -99) # Placeholder if you need to filter
            if not mask.any(): continue
            
            # Map labels: Ensure binary 0.0 vs 1.0
            # Assuming dataset returns -1 or 0 for Benign, 1 for Malignant
            binary_labels = (labels == 1).float().to(DEVICE).unsqueeze(1) # [B, 1]
            images = images.to(DEVICE).float()
            
            if images.max() > 1: images /= 255.0

            # Forward
            logits = model(images)
            loss = criterion(logits, binary_labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Stats
            running_loss += loss.item()
            preds = torch.sigmoid(logits) > 0.5
            correct += (preds == binary_labels.bool()).sum().item()
            total += binary_labels.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        
        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images, labels = batch
                binary_labels = (labels == 1).float().to(DEVICE).unsqueeze(1)
                images = images.to(DEVICE).float()
                if images.max() > 1: images /= 255.0

                logits = model(images)
                loss = criterion(logits, binary_labels)
                
                val_loss += loss.item()
                preds = torch.sigmoid(logits) > 0.5
                val_correct += (preds == binary_labels.bool()).sum().item()
                val_total += binary_labels.size(0)

        val_epoch_loss = val_loss / len(val_loader)
        val_epoch_acc = val_correct / val_total

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
              f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | "
              f"Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")

        wandb.log({
            "epoch": epoch+1,
            "train_loss": epoch_loss, "train_acc": epoch_acc,
            "val_loss": val_epoch_loss, "val_acc": val_epoch_acc
        })

    save_path = os.path.join(RESULTS_DIR, "helico_binary_classifier_full.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Full pipeline model saved to {save_path}")
    wandb.finish()

if __name__ == "__main__":
    main()