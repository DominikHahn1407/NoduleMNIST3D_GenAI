import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from torch.utils.data import Dataset
from torchvision import transforms

def set_seet(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ToTensor3D:
    def __call__(self, img):
        img = torch.from_numpy(img).float()
        if img.max() > 1.0:
            img = img / 255.0
        img = transforms.Normalize(mean=[0.5], std=[0.5])(img)
        return img
    
class SyntheticVAEDataset(Dataset):
    def __init__(self, vae, device, length, label_probs=None, num_classes=2):
        self.vae = vae
        self.length = length
        self.num_classes = num_classes
        self.device = device
        if label_probs is None:
            self.label_probs = np.ones(num_classes) / num_classes
        else:
            self.label_probs = label_probs

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        self.vae.eval()
        with torch.no_grad():
            z = torch.randn(1, self.vae.latent_dim, device=self.device)
            x_syn = self.vae.decode(z).cpu()
        label = np.random.choice(self.num_classes, p=self.label_probs)
        label = torch.tensor(label).long()
        return x_syn.squeeze(0), label
    
def train_classifier(model, device, train_loader, val_loader, epochs=10, lr=1e-3):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_acc = 0.0
    best_state = deepcopy(model.state_dict())

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)

            y = y.squeeze()
            if y.ndim > 1:
                y = y.argmax(dim=1)
            y = y.long()

            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds==y).sum().item()
            total += x.size(0)
        train_loss = running_loss / total
        train_acc = correct / total

        val_acc, val_loss = evaluate_classifier(model, device, val_loader, criterion)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = deepcopy(model.state_dict())

        print(f"[Epoch {epoch:02d}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
    model.load_state_dict(best_state)
    return model, best_val_acc

def evaluate_classifier(model, device, val_loader, criterion=None):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds==y).sum().item()
            total += x.size(0)
            if criterion is not None:
                y = y.squeeze()
                if y.ndim > 1:
                    y = y.argmax(dim=1)
                y = y.long()
                loss = criterion(logits, y)
                running_loss += loss.item() * x.size(0)
    acc = correct / total
    loss = running_loss / total if criterion is not None else None
    return acc, loss

def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    recon = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)
    # KL divergence
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon + beta * kl, recon, kl

def train_vae(vae, device, train_loader, epochs=10, lr=1e-3):
    vae = vae.to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)

    for epoch in range(1, epochs+1):
        vae.train()
        epoch_loss = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            x_hat, mu, logvar = vae(x)
            loss, recon, kl = vae_loss(x_hat, x, mu, logvar, beta=1.0)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)
        print(f"[VAE Epoch {epoch:02d}] Loss: {epoch_loss/len(train_loader.dataset):.4f}")
    return vae