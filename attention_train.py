#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 18:20:27 2023
Updated on Tue Dec 02 2025

@author: Guillermo Torres
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import wandb

from tqdm import tqdm

# Assuming these are available in your directory structure
from dataset import HelicoPatients
from Models.AEmodels import AutoEncoderCNN, VAECNN
from Models.Attention import NeuralNetwork, GatedAttention
from train_conv_ae import AEConfigs
from train_conv_vae import VAEConfigs

import lovely_tensors as lt
lt.monkey_patch()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = "/fhome/vlia01/Medical-Imaging/results" # Adjust as needed

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
        print(f"Warning: Weights not found at {model_path}. Visualization will be random.")
            
    model.to(DEVICE)
    model.eval()
    return model


class ClassificationModel(nn.Module):
    def __init__(self, config_number = 4, FREEZE_BACKBONE = True):
        super().__init__()

        if config_number == 4:
            MODEL_PATH = "/fhome/vlia01/Medical-Imaging/slurm_output/config_four.pth"
            ae = load_model(config_id='4', model_path=MODEL_PATH, model_name="Autoencoder")

        self.encoder = ae.encoder
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        
        if FREEZE_BACKBONE:
            for param in self.encoder.parameters():
                param.requires_grad = False

        attention_config = {
            'in_features': 512,
            'decom_space': 256,
            'heads': 1,
        }
        self.attn = GatedAttention(attention_config)

        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(.5),
            nn.Linear(256, 1),
        )

    def forward(self, input_tensor):
        latent = self.encoder(input_tensor) # B, 8, 8, 32
        latent = self.pool(latent)
        latent = torch.flatten(latent, 1) # B, 512

        super_patch, _ = self.attn(latent) # 1, 512
        label = self.head(super_patch) # 1, 1

        return label

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------

def get_label_tensor(raw_label, device):
    """
    Parses the label from the dataset into a binary FloatTensor.
    Assumes: 'NEGATIVA' or 0 -> 0.0 (Benign)
             Anything else    -> 1.0 (Malignant)
    """
    is_malignant = 0.0
    
    if isinstance(raw_label, str):
        if raw_label.upper() == "NEGATIVA":
            is_malignant = 0.0
        else:
            is_malignant = 1.0 # ALTA, BAIXA, etc.
    elif isinstance(raw_label, (int, float)):
         # Assuming 0 is benign, 1 is malignant
        is_malignant = float(raw_label)
    elif isinstance(raw_label, torch.Tensor):
        is_malignant = float(raw_label.item())
        
    return torch.tensor([[is_malignant]], dtype=torch.float, device=device)

def step(model, batch, criterion, device):
    """
    Performs a single forward pass for one patient (one batch).
    """
    # 1. Prepare Data
    # batch['images'] comes out of DataLoader as [1, N, 3, 256, 256] due to batch_size=1
    # We need to squeeze the batch dim to get [N, 3, 256, 256] for the model
    images = batch['images'].to(device)
    if images.dim() == 5:
        images = images.squeeze(0)
        
    # Prepare Label [1, 1]
    raw_label = batch['label']
    # If using DataLoader with batch_size=1, raw_label is a tuple/list of length 1
    if isinstance(raw_label, (list, tuple)):
        raw_label = raw_label[0]
        
    target = get_label_tensor(raw_label, device)

    # 2. Forward
    images = images.to(torch.float16)
    logits = model(images) # [1, 1]

    # 3. Loss
    loss = criterion(logits, target)
    
    return loss, logits, target

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------

def main():
    # --- Configuration ---
    PROJECT_NAME = "helicodataset-binary-classifier"
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 5
    ACCUM_STEPS = 5  # Gradient Accumulation steps
    BATCH_SIZE = 1   # MUST REMAIN ONE per instructions

    # Init WandB
    wandb.init(project=PROJECT_NAME, config={
        "learning_rate": LEARNING_RATE,
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "accum_steps": ACCUM_STEPS,
        "model": "MIL-Attention-CNN"
    })
    
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- Dataset & DataLoader ---
    print("Initializing Dataset...")
    full_dataset = HelicoPatients()
    
    # Random Split (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Loaders (Batch Size MUST be 1)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # --- Model & Optimizer ---
    print("Initializing Model...")
    model = ClassificationModel(FREEZE_BACKBONE=True).to(DEVICE)

    model = model.to(torch.float16)
    
    # Only optimize non-frozen parameters (Attention + Head)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    criterion = torch.nn.BCEWithLogitsLoss()

    # --- Training Loop ---
    print("Starting Training...")
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss_accum = 0.0
        train_correct = 0
        train_total = 0
        
        optimizer.zero_grad() # Zero gradients at start of epoch
        
        # Using tqdm for progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        
        for i, batch in enumerate(pbar):
            # 1. Step (Forward + Loss calculation)
            loss, logits, target = step(model, batch, criterion, DEVICE)
            
            # 2. Normalize Loss for Accumulation
            # We divide by ACCUM_STEPS because gradients sum up
            loss_norm = loss / ACCUM_STEPS
            loss_norm.backward()
            
            # 3. Track Metrics
            train_loss_accum += loss.item()
            pred = torch.sigmoid(logits) > 0.5
            if pred == target.bool():
                train_correct += 1
            train_total += 1
            
            # 4. Optimizer Step (Gradient Accumulation)
            if (i + 1) % ACCUM_STEPS == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
                
            pbar.set_postfix({"Loss": loss.item()})

        avg_train_loss = train_loss_accum / len(train_loader)
        train_acc = train_correct / train_total

        # --- Validation Loop ---
        model.eval()
        val_loss_accum = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]"):
                loss, logits, target = step(model, batch, criterion, DEVICE)
                
                val_loss_accum += loss.item()
                pred = torch.sigmoid(logits) > 0.5
                if pred == target.bool():
                    val_correct += 1
                val_total += 1

        avg_val_loss = val_loss_accum / len(val_loader)
        val_acc = val_correct / val_total

        print(f"Epoch [{epoch+1}] Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.4f}")

        # --- Logging ---
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_acc": train_acc,
            "val_loss": avg_val_loss,
            "val_acc": val_acc
        })
        
        # --- Save Model (Optional: Save best only logic can be added) ---
        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(RESULTS_DIR, f"mil_attention_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

    # Final Save
    final_path = os.path.join(RESULTS_DIR, "mil_attention_final.pth")
    torch.save(model.state_dict(), final_path)
    wandb.finish()

if __name__ == "__main__":
    main()