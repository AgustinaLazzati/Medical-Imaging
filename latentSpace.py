import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
import numpy as np
from tqdm import tqdm

from dataset import HelicoDatasetPatientDiagnosis
from train import Autoencoder, BEST_MODEL_PATH

# CONFIG
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
LATENT_VISUALIZER = "tsne"  # "pca" or "tsne"
SAVE_LATENT = True
SAVE_FIG = True
OUTPUT_FILE = "AE_latentSpace.png"

# ----------------------------
# LOAD MODEL
model = Autoencoder().to(DEVICE).eval()
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
print(f"Loaded best model from {BEST_MODEL_PATH}")

# We only need the encoder part
encoder = model.encoder
encoder.eval()

# ----------------------------
# LOAD DATA
print("Loading annotated dataset (HelicoDatasetPatientDiagnosis)...")
test_dataset = HelicoDatasetPatientDiagnosis(split="test")
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

# ----------------------------
# EXTRACT LATENT FEATURES
all_latents = []
all_labels = []
reconstruction_errors = []

criterion = nn.MSELoss(reduction="mean")

with torch.no_grad():
    for images, _, labels in tqdm(test_loader, desc="Extracting latents"):
        images = images.to(DEVICE, non_blocking=True)

        # Encode latent representation
        z = encoder(images).flatten(start_dim=1)
        all_latents.append(z.cpu())

        # Compute reconstruction error
        reconstructions = model(images)
        loss = torch.mean((images - reconstructions) ** 2, dim=[1,2,3])  # per-image MSE
        reconstruction_errors.extend(loss.cpu().numpy())

        all_labels.extend(labels.cpu().numpy())

# Concatenate
latents = torch.cat(all_latents, dim=0).numpy()
labels = np.array(all_labels)
reconstruction_errors = np.array(reconstruction_errors)

print(f"Latent space shape: {latents.shape}")

# ----------------------------
# DIMENSIONALITY REDUCTION
if LATENT_VISUALIZER == "pca":
    reducer = PCA(n_components=2)
    latents_2d = reducer.fit_transform(latents)
elif LATENT_VISUALIZER == "tsne":
    reducer = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000)
    latents_2d = reducer.fit_transform(latents)
else:
    raise ValueError("Invalid LATENT_VISUALIZER option")

# ----------------------------
# VISUALIZE LATENT SPACE
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x=latents_2d[:, 0],
    y=latents_2d[:, 1],
    hue=labels,
    palette=["#3776ab", "#E74C3C"],  # blue=NEGATIVA, red=POSITIVA
    alpha=0.6
)
plt.title(f"Latent Space Visualization ({LATENT_VISUALIZER.upper()})")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.legend(title="Diagnosis", labels=["NEGATIVE", "POSITIVE"])
plt.grid(True)

if SAVE_FIG:
    plt.savefig(OUTPUT_FILE, dpi=300)
    print(f"Saved latent space plot as... {OUTPUT_FILE}")

plt.show()

if SAVE_LATENT:
    os.makedirs("results", exist_ok=True)
    np.savez("results/latent_features.npz",
             latents=latents,
             labels=labels,
             recon_error=reconstruction_errors)
    print("Stored in results/latent_features.npz")