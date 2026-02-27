"""
NephroLens - Step 7: ResNet Model Training
Author: Sanket Kelzarkar
Description: Train ResNet50 for kidney stone classification
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

# Project Configuration
PROJECT_DIR = r"C:\Sem VI\ML Projects\KSPML"
PREPROCESSED_DIR = os.path.join(PROJECT_DIR, "preprocessed_data")
RESNET_DIR = os.path.join(PROJECT_DIR, "resnet_training")
RESULTS_DIR = os.path.join(RESNET_DIR, "results")

os.makedirs(RESNET_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

class KidneyStoneDataset(Dataset):
    """Custom dataset for kidney stone CT scans"""
    
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.samples = []
        self.labels = []
        
        # Load data
        for category_idx, category in enumerate(['stone', 'normal']):
            category_path = os.path.join(data_dir, split, category)
            
            if os.path.exists(category_path):
                images = [f for f in os.listdir(category_path) if f.endswith('.png')]
                
                for img_file in images:
                    img_path = os.path.join(category_path, img_file)
                    self.samples.append(img_path)
                    self.labels.append(category_idx)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class ResNetKidneyStone:
    """ResNet50 training pipeline"""
    
    def __init__(self, num_classes=2, pretrained=True):
        self.num_classes = num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = models.resnet50(pretrained=pretrained)
        
        # Modify final layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        self.model = self.model.to(self.device)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def get_data_transforms(self):
        """Define data augmentation transforms"""
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        return train_transform, val_transform
    
    def prepare_dataloaders(self, batch_size=32):
        """Create data loaders"""
        print("=" * 60)
        print("NEPHROLENS - RESNET50 TRAINING")
        print("Powered by Sanket Kelzarkar")
        print("=" * 60)
        
        print("\n[1/5] Preparing data loaders...")
        
        train_transform, val_transform = self.get_data_transforms()
        
        # Create datasets
        train_dataset = KidneyStoneDataset(PREPROCESSED_DIR, 'train', train_transform)
        val_dataset = KidneyStoneDataset(PREPROCESSED_DIR, 'val', val_transform)
        test_dataset = KidneyStoneDataset(PREPROCESSED_DIR, 'test', val_transform)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                 shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                               shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                                shuffle=False, num_workers=4, pin_memory=True)
        
        print(f"✓ Train samples: {len(train_dataset)}")
        print(f"✓ Validation samples: {len(val_dataset)}")
        print(f"✓ Test samples: {len(test_dataset)}")
        print(f"✓ Device: {self.device}")
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, train_loader, criterion, optimizer):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': loss.item(), 'acc': 100 * correct / total})
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader, criterion):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, train_loader, val_loader, epochs=50, lr=0.001):
        """Complete training pipeline"""
        print(f"\n[2/5] Training ResNet50...")
        print(f"  Epochs: {epochs}")
        print(f"  Learning rate: {lr}")
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                         factor=0.5, patience=5)
        
        best_val_acc = 0.0
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 
                          os.path.join(RESNET_DIR, 'resnet50_best.pth'))
                print(f"✓ Best model saved! Val Acc: {val_acc:.2f}%")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        print(f"\n✓ Training completed! Best Val Acc: {best_val_acc:.2f}%")
    
    def evaluate(self, test_loader):
        """Evaluate on test set"""
        print("\n[3/5] Evaluating on test set...")
        
        # Load best model
        self.model.load_state_dict(torch.load(os.path.join(RESNET_DIR, 'resnet50_best.pth')))
        
        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Testing"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        test_acc = 100 * correct / total
        print(f"✓ Test Accuracy: {test_acc:.2f}%")
        
        return test_acc, all_preds, all_labels
    
    def plot_training_history(self):
        """Plot training curves"""
        print("\n[4/5] Plotting training history...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.history['train_loss'], label='Train Loss', linewidth=2)
        ax1.plot(self.history['val_loss'], label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(self.history['train_acc'], label='Train Acc', linewidth=2)
        ax2.plot(self.history['val_acc'], label='Val Acc', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('ResNet50 Training History - NephroLens\nPowered by Sanket Kelzarkar',
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'resnet_training_history.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Training history saved")
    
    def save_results(self, test_acc):
        """Save training results"""
        print("\n[5/5] Saving results...")
        
        results = {
            'model': 'ResNet50',
            'test_accuracy': test_acc,
            'best_val_accuracy': max(self.history['val_acc']),
            'total_epochs': len(self.history['train_loss']),
            'device': str(self.device),
            'history': self.history
        }
        
        with open(os.path.join(RESULTS_DIR, 'resnet_results.json'), 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"✓ Results saved")

def main():
    """Main training pipeline"""
    # Initialize model
    resnet_trainer = ResNetKidneyStone(num_classes=2, pretrained=True)
    
    # Prepare data
    train_loader, val_loader, test_loader = resnet_trainer.prepare_dataloaders(batch_size=32)
    
    # Train
    resnet_trainer.train(train_loader, val_loader, epochs=15, lr=0.001)
    
    # Evaluate
    test_acc, _, _ = resnet_trainer.evaluate(test_loader)
    
    # Plot history
    resnet_trainer.plot_training_history()
    
    # Save results
    resnet_trainer.save_results(test_acc)
    
    print("\n" + "=" * 60)
    print("RESNET50 TRAINING COMPLETED SUCCESSFULLY!")
    print("Powered by Sanket Kelzarkar")
    print("=" * 60)

if __name__ == "__main__":
    main()
    print("\n✓ Ready for next step: VM-UNet Mamba Model Training")