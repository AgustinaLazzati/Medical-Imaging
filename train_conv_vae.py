from dataset import HelicoCropped, cropped_collator
from Models.AEmodels import VAECNN
import math
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import os
import argparse

import lovely_tensors as lt
lt.monkey_patch()

def VAEConfigs(Config):
    inputmodule_paramsEnc = {'dim_input': 256, 'num_input_channels': 3}
    inputmodule_paramsDec = {'dim_input': 256}
    dim_in = inputmodule_paramsEnc['dim_input']
    
    net_paramsEnc = {}
    net_paramsDec = {}
    net_paramsRep = {}

    if Config == '1':
        net_paramsEnc['block_configs'] = [[32, 32], [64, 64], [64, 64]]
        net_paramsEnc['stride'] = [[1, 2], [1, 2], [1, 2]]
        net_paramsDec['block_configs'] = [[64, 64], [64, 32], [32, inputmodule_paramsEnc['num_input_channels']]]
        net_paramsDec['stride'] = net_paramsEnc['stride']
        inputmodule_paramsDec['num_input_channels'] = net_paramsDec['block_configs'][0][0]

        net_paramsRep['h_dim'] = 262144
        net_paramsRep['z_dim'] = 256

    elif Config == '2':
        net_paramsEnc['block_configs'] = [[32], [64], [128], [256]]
        net_paramsEnc['stride'] = [[2], [2], [2], [2]]
        net_paramsDec['block_configs'] = [[256], [128], [64], [inputmodule_paramsEnc['num_input_channels']]]
        net_paramsDec['stride'] = net_paramsEnc['stride']
        inputmodule_paramsDec['num_input_channels'] = net_paramsDec['block_configs'][0][0]
        
        net_paramsRep['h_dim'] = 65536
        net_paramsRep['z_dim'] = 512

    elif Config == '3':
        net_paramsEnc['block_configs'] = [[32], [64], [64]]
        net_paramsEnc['stride'] = [[1], [2], [2]]
        net_paramsDec['block_configs'] = [[64], [32], [inputmodule_paramsEnc['num_input_channels']]]
        net_paramsDec['stride'] = net_paramsEnc['stride']
        inputmodule_paramsDec['num_input_channels'] = net_paramsDec['block_configs'][0][0]
        
        net_paramsRep['h_dim'] = 262144
        net_paramsRep['z_dim'] = 256

    return net_paramsEnc, net_paramsDec, inputmodule_paramsEnc, inputmodule_paramsDec, net_paramsRep

def init_run(config_number: int):
    assert config_number in [1, 2, 3], "Config number must be 1, 2, or 3."
    net_paramsEnc, net_paramsDec, inputmodule_paramsEnc, inputmodule_paramsDec, net_paramsRep = VAEConfigs(str(config_number))
    config = {
        "LEARNING_RATE": 1e-4, #0.002,
        "BATCH_SIZE": 128, #512,
        "EPOCHS": 25,
        "PROJECT_NAME": "helicodataset-autoencoder-anomaly",
        "scheduler_patience": 3,
        "net_paramsEnc": net_paramsEnc,
        "net_paramsDec": net_paramsDec,
        "inputmodule_paramsEnc": inputmodule_paramsEnc,
        "inputmodule_paramsDec": inputmodule_paramsDec,
        "net_paramsRep": net_paramsRep,
    }

    wandb.init(
        project=config["PROJECT_NAME"],
        config=config
    )
    config = wandb.config

    # we save the model with the run name
    run_name = wandb.run.name
    MODEL_PATH = f"{run_name}.pth"
    

    model = VAECNN(
        config.inputmodule_paramsEnc,
        config.net_paramsEnc,
        config.inputmodule_paramsDec,
        config.net_paramsDec,
        config.net_paramsRep,
    )

    print(model)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model initialized with {param_count} trainable parameters.")

    wandb.config.update({"param_count": param_count})

    return config, model, MODEL_PATH

def get_dataloaders(config):
    # WE ONLY ONE THE NEGATIVE CLASS IMAGES
    # we grab the images that belong to entire patients with negative diagnosis
    dataset = HelicoCropped(target_id="NEGATIVA", load_ram=False)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    training_data, validation_data = torch.utils.data.random_split(dataset, [train_size, val_size])

    training_data_loader = DataLoader(
        training_data,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=cropped_collator,
    )

    validation_data_loader = DataLoader(
        validation_data,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=cropped_collator,
    )

    return training_data_loader, validation_data_loader

def train(config, dataloader, optimizer, model, criterion):
    train_loss = 0.0
    for batch in dataloader:
        optimizer.zero_grad()
        
        images = batch['image'].to(device, non_blocking=True).float()
        
        outputs, mu, logvar = model(images)
        loss = criterion(outputs, images)
        
        loss.backward()
        optimizer.step()

        wandb.log({"train_batch_loss": loss.item()})

        train_loss += loss.item() * images.size(0)

    train_loss = train_loss / len(dataloader.dataset)

    return train_loss

def eval(config, dataloader, model, criterion):
    val_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
          images = batch['image'].to(device, non_blocking=True).float()
          
          outputs, mu, logvar = model(images)
          loss = criterion(outputs, images)
          
          val_loss += loss.item() * images.size(0)
            
    val_loss /= len(dataloader.dataset)

    return val_loss

def main(config_number: int):
    config, model, MODEL_PATH = init_run(config_number)
    
    training_data_loader, validation_data_loader = get_dataloaders(config)

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=config.scheduler_patience, verbose=True)

    wandb.watch(model, criterion, log="all", log_freq=100)
    global_step = 0  # initialize before training

    for epoch in range(1, config.EPOCHS + 1):
        train_loss = train(config, training_data_loader, optimizer, model, criterion)
        val_loss = eval(config, validation_data_loader, model, criterion)

        # we want to overfit the model
        # so we use the training loss for the scheduler
        scheduler.step(train_loss)

        print(f'Epoch {epoch}, Training Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}')

        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch": epoch,
        })

    # save the final model
    torch.save(model.state_dict(), MODEL_PATH)

    print(f"Model saved to {MODEL_PATH}")

    artifact = wandb.Artifact(
        'best-autoencoder', 
        type='model',
    )
    artifact.add_file(MODEL_PATH)
    wandb.run.log_artifact(artifact)

    print("Training complete.")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Autoencoder on Helico Dataset")
    parser.add_argument(
        "--config_number",
        type=int,
        choices=[1, 2, 3],
        required=True,
        help="Configuration number for the autoencoder architecture (1, 2, or 3)."
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Using device:', device)
    main(args.config_number)