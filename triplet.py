import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

# Custom imports
from dataset import HelicoMixed, annotated_collate
from Models.AEmodels import AutoEncoderCNN, VAECNN
from train_conv_ae import AEConfigs
from train_conv_vae import VAEConfigs

# ----------------------------
# CONFIGURATION
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "/fhome/vlia01/Medical-Imaging/slurm_output/config_three.pth"
MODEL_NAME = "Autoencoder" 
BATCH_SIZE = 32
LR = 0.0005  # Reduced LR slightly for stability
EPOCHS = 20  # Increased epochs (fast with pre-computation)
MARGIN = 0.5 # Reduced margin for stability on hypersphere (max dist is 2.0)

# ----------------------------
# 1. MODELS & COMPONENTS
# ----------------------------

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=None):
        super(ProjectionHead, self).__init__()
        # User requested to maintain size, so input_dim -> input_dim
        if hidden_dim is None:
            hidden_dim = input_dim
            
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), # Added for stability
            nn.LeakyReLU(0.1),          # Leaky to prevent dead neurons
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_dim) 
            # Output is same size as input, unnormalized (we normalize in loop)
        )

    def forward(self, x):
        return self.net(x)

def get_latent_dim(model, model_name):
    dummy = torch.randn(1, 3, 256, 256).to(DEVICE)
    with torch.no_grad():
        if model_name == "Autoencoder":
            if hasattr(model, 'encoder'): out = model.encoder(dummy)
            elif hasattr(model, 'netEnc'): out = model.netEnc(dummy)
        elif model_name == "Variational Autoencoder":
            _, mu, _ = model(dummy)
            out = mu
    return out.view(1, -1).shape[1]

def load_pretrained_model(config_id="3", model_path=None, model_name="Autoencoder"):
    print(f"Loading frozen {model_name}...")
    if model_name == "Autoencoder":
        net_paramsEnc, net_paramsDec, inputmodule_paramsEnc, inputmodule_paramsDec = AEConfigs(config_id)
        model = AutoEncoderCNN(inputmodule_paramsEnc, net_paramsEnc, inputmodule_paramsDec, net_paramsDec)
    elif model_name == "Variational Autoencoder":
        net_paramsEnc, net_paramsDec, inputmodule_paramsEnc, inputmodule_paramsDec, net_paramsRep = VAEConfigs(config_id)
        model = VAECNN(inputmodule_paramsEnc, net_paramsEnc, inputmodule_paramsDec, net_paramsDec, net_paramsRep)
    
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    
    model.to(DEVICE)
    model.eval()
    # Freeze weights
    for param in model.parameters():
        param.requires_grad = False
    return model

# ----------------------------
# 2. TRIPLET LOSS
# ----------------------------
def batch_hard_triplet_loss(embeddings, labels, margin=0.5):
    # Pairwise distances
    dot_product = torch.matmul(embeddings, embeddings.t())
    square_norm = torch.diag(dot_product)
    distances_sq = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
    distances_sq = torch.clamp(distances_sq, min=0.0)
    distances = torch.sqrt(distances_sq + 1e-8)

    # Masks
    labels = labels.view(-1)
    mask_pos = labels.unsqueeze(0) == labels.unsqueeze(1)
    mask_neg = labels.unsqueeze(0) != labels.unsqueeze(1)

    # Hardest Positive
    hardest_pos_dist = (distances * mask_pos.float()).max(dim=1)[0]

    # Hardest Negative
    max_dist = distances.max()
    hardest_neg_dist = (distances + max_dist * (~mask_neg).float()).min(dim=1)[0]

    loss = torch.clamp(hardest_pos_dist - hardest_neg_dist + margin, min=0.0)
    return loss.mean()

# ----------------------------
# 3. PRE-COMPUTATION
# ----------------------------
def precompute_latents(dataloader, ae_model, model_name):
    """
    Extracts ALL latent vectors from the dataset using the frozen AE.
    Returns tensors stored in RAM.
    """
    print("Pre-computing latent vectors (this happens once)...")
    ae_model.eval()
    latents_list = []
    labels_list = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting Features"):
            images, targets = batch
            images = images.to(DEVICE).float()
            if images.max() > 1: images /= 255.0
            
            if model_name == "Autoencoder":
                if hasattr(ae_model, 'encoder'): z = ae_model.encoder(images)
                else: z = ae_model.netEnc(images)
            else:
                _, z, _ = ae_model(images) # mu
            
            z = z.view(z.size(0), -1)
            latents_list.append(z.cpu())
            labels_list.append(targets.cpu())
            
    return torch.cat(latents_list), torch.cat(labels_list)

# ----------------------------
# 4. MAIN
# ----------------------------
def main():
    # A. Load Frozen Model
    ae_model = load_pretrained_model(model_path=MODEL_PATH, model_name=MODEL_NAME)
    latent_dim = get_latent_dim(ae_model, MODEL_NAME)
    print(f"Latent Dimension: {latent_dim}")

    # B. Data Loading (Load Images to RAM -> Precompute Latents)
    print("Loading Dataset into RAM...")
    full_ds = HelicoMixed(load_ram=True) # Load images to RAM as requested
    raw_loader = DataLoader(full_ds, batch_size=32, shuffle=False, collate_fn=annotated_collate)
    
    # Extract Latents (The efficiency boost)
    all_latents, all_labels = precompute_latents(raw_loader, ae_model, MODEL_NAME)
    
    # Create TensorDataset (Super fast training)
    train_ds = TensorDataset(all_latents, all_labels)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # C. Projector Setup
    # Output dim = Input dim (Maintain size)
    projector = ProjectionHead(input_dim=latent_dim, hidden_dim=latent_dim).to(DEVICE)
    optimizer = optim.Adam(projector.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # D. Training Loop
    print(f"Starting Triplet Training for {EPOCHS} epochs...")
    
    for epoch in range(EPOCHS):
        projector.train()
        epoch_loss = 0
        count = 0
        
        for latents, labels in train_loader:
            latents, labels = latents.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward MLP
            embeddings = projector(latents)
            
            # Normalize to Unit Hypersphere (Crucial for high dim / preventing collapse)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            loss = batch_hard_triplet_loss(embeddings, labels, margin=MARGIN)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            count += 1
            
        scheduler.step()
        avg_loss = epoch_loss / count
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f}")

    # E. Evaluation (ROC)
    print("\nEvaluating ROC...")
    projector.eval()
    
    # Get all projected embeddings
    with torch.no_grad():
        # Process all precomputed latents in one go (or batches if memory tight)
        # We iterate loader to be safe on GPU memory
        eval_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
        proj_list = []
        lbl_list = []
        for l, t in eval_loader:
            l = l.to(DEVICE)
            emb = projector(l)
            emb = F.normalize(emb, p=2, dim=1) # Remember to normalize here too!
            proj_list.append(emb.cpu())
            lbl_list.append(t)
            
    vecs = torch.cat(proj_list).numpy()
    lbls = torch.cat(lbl_list).numpy()
    
    # Centroid Distance
    benign_vecs = vecs[lbls == 0]
    malign_vecs = vecs[lbls == 1]
    
    if len(benign_vecs) > 0:
        center = np.mean(benign_vecs, axis=0)
        # Normalize center (it should lie on the sphere too approx)
        center = center / np.linalg.norm(center)
        
        benign_dists = np.linalg.norm(benign_vecs - center, axis=1)
        malign_dists = np.linalg.norm(malign_vecs - center, axis=1)
        
        y_true = np.concatenate([np.zeros(len(benign_dists)), np.ones(len(malign_dists))])
        y_scores = np.concatenate([benign_dists, malign_dists])
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        print(f"Latent Space AUC: {roc_auc:.4f}")
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='purple', lw=2, label=f'AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.title('ROC - Triplet Projected Distance')
        plt.legend()
        if True:
            os.makedirs("results", exist_ok=True)
            plt.savefig(f"results/ROC_Triplet_{MODEL_NAME.replace(' ','_')}.png")
        plt.show()

if __name__ == "__main__":
    main()