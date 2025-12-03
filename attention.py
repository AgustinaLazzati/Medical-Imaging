#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 18:20:27 2023

@author: Guillermo Torres
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

from tqdm import tqdm

from dataset import HelicoPatients
from Models.AEmodels import AutoEncoderCNN, VAECNN
from Models.Attention import NeuralNetwork, GatedAttention
from train_conv_ae import AEConfigs
from train_conv_vae import VAEConfigs

import lovely_tensors as lt
lt.monkey_patch()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self.pool = nn.AdaptiveAvgPool2d((8, 8))

        # nuestra propia configuarion.
        # el output (flatten) es de 8192
        if config_number == 4:
            MODEL_PATH = "/fhome/vlia01/Medical-Imaging/slurm_output/config_four.pth"
            ae = load_model(config_id='4', model_path=MODEL_PATH, model_name="Autoencoder")

        self.encoder = ae.encoder
        
        # la CNN backnbone (AE encoder) no es entrenada
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
        assert input_tensor.ndim == 4
        assert input_tensor.shape[0] != 1

        latent = self.encoder(input_tensor) # B, 8, 8, 32

        latent = self.pool(latent)
        latent = torch.flatten(latent, 1) # B, 512

        super_patch, _ = self.attn(latent) # 1, 512
        label = self.head(super_patch) # 1, 1

        return label


if __name__ == "__main__":
    # 16 patches de 256x256
    example_image = torch.randn((16, 3, 256, 256))

    model = ClassificationModel(FREEZE_BACKBONE=False)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(format(param_count, '_d'))

    out = model(example_image)
    print(out)