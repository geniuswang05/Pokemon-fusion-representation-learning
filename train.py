# Imports & Seed
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import ConvVAE, vae_loss
import copy

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import os
from torchvision import transforms
from PIL import Image


# ✅ Dataset Loading
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        data = np.load(path)
        self.images = data['images'] / 255.0
        self.labels = data['labels']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return torch.tensor(self.images[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

full_dataset = CustomDataset("train.npz")
train_len = int(0.9 * len(full_dataset))
val_len = len(full_dataset) - train_len
train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# sample data batch
images, labels = next(iter(train_loader))
print(f"images shape: {images.shape}")
print(f"labels shape: {labels.shape}")
import matplotlib.pyplot as plt


with open("label2type.txt", 'r') as f:
    label2type = eval(f.read())

# plot samples
fig, axes = plt.subplots(1, 4, figsize=(12, 4))
for i, ax in enumerate(axes):
    label = labels[i].item()
    # (C, H, W) to (H, W, C) for plotting
    ax.imshow(images[i].numpy().transpose(1, 2, 0))
    ax.axis('off')
    ax.set_title(f"{label}: {label2type[label]}")
plt.tight_layout()
plt.show()


# ====== Training loop using new loss ======
def train_vae(model, train_loader, val_loader, optimizer, device,
              num_epochs=1000, early_stop_patience=400,
              freeze_encoder_epochs=160):

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')
    no_improve_epochs = 0

    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        model.train()

        if epoch < freeze_encoder_epochs:
            for name, param in model.named_parameters():
                if 'encoder' in name:
                    param.requires_grad = False
        else:
            for param in model.parameters():
                param.requires_grad = True

        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for x, labels in loop:
            x, labels = x.to(device), labels.to(device)
            optimizer.zero_grad()
            x_recon, z, mean, logvar, logits = model(x)
            loss, recon_loss, class_loss = vae_loss(x, x_recon, z, mean, logvar, labels, model.classifier)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item(), recon=recon_loss.item(), cls=class_loss.item())

        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                x_recon, z, mean, logvar, _ = model(x_val)
                loss, _, _= vae_loss(x_val, x_recon, z, mean, logvar, y_val, model.classifier)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f"Validation loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), "best_checkpoint_112.pt")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_model_wts)
    return model

model = ConvVAE(latent_channels=32).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
trained_model = train_vae(model, train_loader, val_loader, optimizer, device,
                          num_epochs=1000,
                          freeze_encoder_epochs=160)

# Visualization of Reconstructions
def plot_reconstructions(model, dataloader, device, num_images=8):
    model.eval()
    with torch.no_grad():
        try:
            x, _ = next(iter(dataloader))
        except StopIteration:
            print("Warning: Dataloader is empty!")
            return
        x = x.to(device)
        x_recon, z, _, _, _ = model(x)
        x = x.cpu().numpy()
        x_recon = x_recon.cpu().numpy()
        print(f"Latent bottleneck dim: {z.view(z.shape[0], -1).shape[1]}")
        plt.figure(figsize=(16, 4))
        for i in range(min(num_images, len(x))):
            plt.subplot(2, num_images, i+1)
            plt.imshow(x[i].transpose(1, 2, 0))
            plt.axis('off')
            plt.subplot(2, num_images, i+1+num_images)
            plt.imshow(x_recon[i].transpose(1, 2, 0))
            plt.axis('off')
        plt.tight_layout()
        plt.show()


plot_reconstructions(model, train_loader, device, num_images=8)
torch.save(model.state_dict(), "checkpoint.pt")