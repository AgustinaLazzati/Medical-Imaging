import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os


# --- Import your local modules ---
# Assuming these are available in your path as per your snippet
from Models.AEmodels import AutoEncoderCNN
from train_conv_ae import AEConfigs
from dataset import HelicoAnnotated, annotated_collate

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
NUM_EXAMPLES = 100  # For the reconstruction check
TSNE_SAMPLES = 1000 # Total points to plot in TSNE (don't plot whole dataset if it's huge)

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
            raise ValueError(f"model weights not found, given path: {model_path}")
    
    model.to(DEVICE)
    model.eval()
    return model

def load_vgg():


def get_examples(dataloader, model, num_examples):
    """
    Get input images and their reconstructions.
    """
    examples = []
    count = 0
    with torch.no_grad():
        for batch in dataloader:
            # Adjust based on what your dataloader returns (images, labels) or just images
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch
            
            # Normalization check
            images = images.to(DEVICE).float()
            if images.max() > 1:
                images = images / 255.0
            
            outputs = model(images)
            
            for img, out in zip(images, outputs):
                if count < 3:  # Skip first few if needed
                    count += 1
                    continue
                
                examples.append((img.cpu(), out.cpu()))
                
                if len(examples) >= num_examples:
                    return examples
    return examples

def get_latent_vectors(dataloader, model, max_samples=500):
    vectors = []
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
            
            embedding = model.encoder(images)

            embedding = embedding.view(embedding.size(0), -1)
            
            vectors.append(embedding.cpu().numpy())
            count += images.size(0)
            
            if count >= max_samples:
                break
                
    return np.concatenate(vectors, axis=0)[:max_samples]


model = load_model(config_id=1, model_path="/fhome/vlia01/Medical-Imaging/slurm_output/config_one.pth")

dataset_benign = HelicoAnnotated(only_negative=True, load_ram=False)
dataset_malign = HelicoAnnotated(only_positive=True, load_ram=False)

dataloader_benign = DataLoader(dataset_benign, batch_size=BATCH_SIZE, shuffle=False, collate_fn=annotated_collate)
dataloader_malign = DataLoader(dataset_malign, batch_size=BATCH_SIZE, shuffle=False, collate_fn=annotated_collate)

samples_per_class = TSNE_SAMPLES // 2

benign_vecs = get_latent_vectors(dataloader_benign, model, max_samples=samples_per_class)
malign_vecs = get_latent_vectors(dataloader_malign, model, max_samples=samples_per_class)

benign_labels = np.zeros(len(benign_vecs)) # ZEROS
malign_labels = np.ones(len(malign_vecs)) # ONES

X = np.concatenate([benign_vecs, malign_vecs], axis=0)
y = np.concatenate([benign_labels, malign_labels], axis=0)

tsne = TSNE(n_components=2, random_state=42, perplexity=30, metric='euclidean', init='pca')
X_embedded = tsne.fit_transform(X)
print(X_embedded)

plt.figure(figsize=(10, 8))

plt.scatter(
    X_embedded[y == 0, 0],  # x dim
    X_embedded[y == 0, 1],  # y dim
    c='blue', label='Benign', alpha=0.6, s=15
)

plt.scatter(
    X_embedded[y == 1, 0], 
    X_embedded[y == 1, 1], 
    c='red', label='Malign', alpha=0.6, s=15
)

plt.title("t-SNE Visualization of AutoEncoder Latent Space")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('tsne.png')