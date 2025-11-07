from dataset import HelicoDatasetAnomalyDetection

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb  # Import wandb
import os

config = {
    "LEARNING_RATE": 0.002,
    "BATCH_SIZE": 256,
    "EPOCHS": 50,
    "PROJECT_NAME": "helicodataset-autoencoder-anomaly",
    "scheduler_patience": 3,
}

BEST_MODEL_PATH = "best_autoencoder_model.pth"

# --- 1. Initialize W&B ---
# Start a new wandb run to track this experiment
wandb.init(
    project=config["PROJECT_NAME"],
    config=config
)
# wandb.config now holds all your hyperparameters
config = wandb.config


# --- Dataset and DataLoaders ---
dataset = HelicoDatasetAnomalyDetection()

# Split dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
training_data, validation_data = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create data loaders
training_data_loader = DataLoader(training_data, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=64, pin_memory=True, persistent_workers=True)
validation_data_loader = DataLoader(validation_data, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=64, pin_memory=True, persistent_workers=True)


# --- Model Definition ---
class Autoencoder(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            nn.Conv2d(3, 16, kernel_size = 3, stride=2, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(16, 32, kernel_size = 3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size = 3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size = 3, stride=2, padding=1),
            nn.ReLU(),
        )
        
        self.bottleneck = torch.nn.Sequential(
            nn.Conv2d(128, 128, kernel_size = 3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size = 3, stride=1, padding=1),
            nn.ReLU(),
            
        ) 

        self.decoder = torch.nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size = 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size = 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 16, kernel_size = 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(16, 3, kernel_size = 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid() # Sigmoid activation to output values between 0 and 1
        )
    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Streamlined forward pass
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x


model = Autoencoder()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).to(torch.bfloat16)


optimizer = torch.optim.RMSprop(model.parameters(), lr=config.LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

criterion = nn.MSELoss()

# --- 2. Track model with wandb ---
# This will log gradients and model parameters (optional)
wandb.watch(model, criterion, log="all", log_freq=100)

print("Starting training...")

best_train_loss = float('inf') # Initialize best loss for model saving

for epoch in range(1, config.EPOCHS + 1):
    
    model.train()
    train_loss = 0.0
    for data in tqdm(training_data_loader, desc=f"Epoch {epoch}/{config.EPOCHS} [Train]"):
        inputs = data.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(inputs)
            loss = criterion(outputs, inputs)

        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * inputs.size(0)
    
    train_loss /= len(training_data_loader.dataset)
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad(): # Disable gradient calculation
        for data in tqdm(validation_data_loader, desc=f"Epoch {epoch}/{config.EPOCHS} [Val]"):
            inputs = data
            inputs = inputs.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
        
            val_loss += loss.item() * inputs.size(0)
            
    val_loss /= len(validation_data_loader.dataset)

    scheduler.step(train_loss)
    
    print(f'Epoch {epoch}, Training Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}')
    
    wandb.log({
        "epoch": epoch,
        "learning_rate": optimizer.param_groups[0]['lr'],
        "train_loss": train_loss,
        "val_loss": val_loss
    })
    
    if train_loss < best_train_loss:
        best_train_loss = train_loss
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"  -> New best model saved with train_loss: {best_train_loss:.6f}")

print("Training finished.")

print(f"Logging best model ('{BEST_MODEL_PATH}') to wandb as an artifact...")
artifact = wandb.Artifact(
    'best-autoencoder', 
    type='model',
    description='Best autoencoder model based on minimum validation loss.'
)
artifact.add_file(BEST_MODEL_PATH)
wandb.run.log_artifact(artifact)
print("Artifact logged.")

wandb.finish()
