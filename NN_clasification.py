import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.manifold import TSNE

# Custom imports
from dataset import HelicoAnnotated, annotated_collate
from Models.AEmodels import AutoEncoderCNN, VAECNN
from train_conv_ae import AEConfigs
from train_conv_vae import VAEConfigs

# ----------------------------
# CONFIGURATION
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "/fhome/vlia01/Medical-Imaging/slurm_output/config_two.pth" 
BATCH_SIZE = 32 
MODEL_NAME = "Autoencoder" # "Autoencoder" or "Variational Autoencoder"
SAVE_FIG = True


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
        print(f"Warning: Weights not found at {model_path}. Classification will be random.")
            
    model.to(DEVICE)
    model.eval()
    return model


def extract_latent_features(model, dataloader, device, model_name):
    """
    Extracts latent features and labels for all patches in a DataLoader.
    """
    all_features = []
    all_labels = []
    
    # Get the encoder part of the model
    if model_name == "Autoencoder":
        encoder = model.encoder
    elif model_name == "Variational Autoencoder":
        encoder = model.encoder
    else:
        raise ValueError("model_name must be 'Autoencoder' or 'Variational Autoencoder'")

    encoder.eval() # Ensure encoder is in evaluation mode

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)

            if model_name == "Autoencoder":
                features = encoder(images)
            elif model_name == "Variational Autoencoder":
                # VAE Encoder returns recogn, mu and log_var. We use mu.
                -, mu, _ = encoder(images)
                features = mu
            
            # Reshape features from (B, C, H, W) to (B, C*H*W) for FNN/Linear layer
            features = features.view(features.size(0), -1)
            
            # Map labels: -1 (Negative/Benign) -> 0, 1 (Positive/Malign) -> 1
            mapped_labels = torch.where(labels == -1, 0, 1)

            all_features.append(features.cpu())
            all_labels.append(mapped_labels.cpu())

    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)
    
    
# MAIN MODIFICATION FOR FEATURE EXTRACTION
def main_feature_extraction():
    # 1. Setup
    model = load_model(config_id="2", model_path=MODEL_PATH, model_name=MODEL_NAME)

    # 2. Load Datasets
    # We use HelicoAnnotated for labeled patches
    dataset_all = HelicoAnnotated(only_negative=False, only_positive=False, load_ram=False)
    dataloader_all = DataLoader(dataset_all, batch_size=BATCH_SIZE, shuffle=False, collate_fn=annotated_collate)
    
    # 3. Extract Features
    print("Extracting latent features...")
    features, labels = extract_latent_features(model, dataloader_all, DEVICE, MODEL_NAME)
    print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")

    # You can now proceed to train the classifier with (features, labels)
    return features, labels