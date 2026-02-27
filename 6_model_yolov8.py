"""
NephroLens - Step 6: YOLOv8 Model Training
Author: Sanket Kelzarkar
Description: Train YOLOv8 for kidney stone detection and localization
"""

import os
import yaml
import torch
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Project Configuration
PROJECT_DIR = r"C:\Sem VI\ML Projects\KSPML"
PREPROCESSED_DIR = os.path.join(PROJECT_DIR, "preprocessed_data")
YOLO_DIR = os.path.join(PROJECT_DIR, "yolov8_training")
RESULTS_DIR = os.path.join(YOLO_DIR, "results")

os.makedirs(YOLO_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

class YOLOv8KidneyStone:
    """YOLOv8 training pipeline for kidney stone detection"""
    
    def __init__(self, model_size='n'):
        """
        Initialize YOLOv8 model
        model_size: 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (xlarge)
        """
        self.model_size = model_size
        self.model = None
        self.dataset_yaml = os.path.join(YOLO_DIR, 'dataset.yaml')
        
    def prepare_yolo_dataset(self):
        """Convert dataset to YOLO format"""
        print("=" * 60)
        print("NEPHROLENS - YOLOV8 TRAINING")
        print("Powered by Sanket Kelzarkar")
        print("=" * 60)
        
        print("\n[1/5] Preparing YOLO dataset format...")
        
        # Create YOLO directory structure
        yolo_data_dir = os.path.join(YOLO_DIR, 'data')
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(yolo_data_dir, split, 'images'), exist_ok=True)
            os.makedirs(os.path.join(yolo_data_dir, split, 'labels'), exist_ok=True)
        
        # Copy images and create pseudo-labels for classification
        for split in ['train', 'val', 'test']:
            print(f"  Processing {split} split...")
            
            for category in ['stone', 'normal']:
                src_dir = os.path.join(PREPROCESSED_DIR, split, category)
                
                if not os.path.exists(src_dir):
                    continue
                
                images = [f for f in os.listdir(src_dir) if f.endswith('.png')]
                
                for img_file in images:
                    # Copy image
                    src_path = os.path.join(src_dir, img_file)
                    dst_path = os.path.join(yolo_data_dir, split, 'images', img_file)
                    
                    img = cv2.imread(src_path)
                    cv2.imwrite(dst_path, img)
                    
                    # Create label file (for classification, use full image as bounding box)
                    label_file = os.path.splitext(img_file)[0] + '.txt'
                    label_path = os.path.join(yolo_data_dir, split, 'labels', label_file)
                    
                    class_id = 0 if category == 'stone' else 1
                    
                    # YOLO format: class_id x_center y_center width height (normalized)
                    # For classification, use full image
                    with open(label_path, 'w') as f:
                        f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
        
        # Create dataset.yaml
        dataset_config = {
            'path': yolo_data_dir,
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': 2,  # number of classes
            'names': ['stone', 'normal']
        }
        
        with open(self.dataset_yaml, 'w') as f:
            yaml.dump(dataset_config, f)
        
        print(f"✓ YOLO dataset prepared at: {yolo_data_dir}")
        return yolo_data_dir
    
    def train_model(self, epochs=100, batch_size=16, img_size=640):
        """Train YOLOv8 model"""
        print(f"\n[2/5] Training YOLOv8{self.model_size} model...")
        
        # Initialize model
        self.model = YOLO(f'yolov8{self.model_size}.pt')
        
        # Training parameters
        train_params = {
            'data': self.dataset_yaml,
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': img_size,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'project': YOLO_DIR,
            'name': f'yolov8{self.model_size}_kidney_stone',
            'exist_ok': True,
            'patience': 20,
            'save': True,
            'plots': True,
            'verbose': True,
            'optimizer': 'Adam',
            'lr0': 0.001,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'augment': True,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 10.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.5,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.0,
        }
        
        print(f"  Device: {train_params['device']}")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Image size: {img_size}")
        
        # Train
        results = self.model.train(**train_params)
        
        print("✓ Training completed!")
        return results
    
    def validate_model(self):
        """Validate trained model"""
        print("\n[3/5] Validating model...")
        
        metrics = self.model.val(data=self.dataset_yaml)
        
        print(f"✓ Validation metrics:")
        print(f"  mAP50: {metrics.box.map50:.4f}")
        print(f"  mAP50-95: {metrics.box.map:.4f}")
        print(f"  Precision: {metrics.box.mp:.4f}")
        print(f"  Recall: {metrics.box.mr:.4f}")
        
        return metrics
    
    def test_model(self):
        """Test model on test set"""
        print("\n[4/5] Testing model on test set...")
        
        test_dir = os.path.join(YOLO_DIR, 'data', 'test', 'images')
        results = self.model.predict(
            source=test_dir,
            save=True,
            project=RESULTS_DIR,
            name='test_predictions',
            conf=0.25
        )
        
        print(f"✓ Test predictions saved to: {RESULTS_DIR}/test_predictions")
        return results
    
    def export_model(self):
        """Export model to various formats"""
        print("\n[5/5] Exporting model...")
        
        export_dir = os.path.join(YOLO_DIR, 'exported_models')
        os.makedirs(export_dir, exist_ok=True)
        
        # Export to ONNX
        try:
            onnx_path = self.model.export(format='onnx', dynamic=True)
            print(f"✓ ONNX model exported: {onnx_path}")
        except Exception as e:
            print(f"  ONNX export failed: {e}")
        
        # Save best weights
        best_weights = os.path.join(YOLO_DIR, f'yolov8{self.model_size}_kidney_stone', 'weights', 'best.pt')
        if os.path.exists(best_weights):
            print(f"✓ Best model weights: {best_weights}")
        
        return best_weights
    
    def visualize_predictions(self, num_samples=10):
        """Visualize sample predictions"""
        print("\n[VISUALIZATION] Creating prediction samples...")
        
        test_dir = os.path.join(YOLO_DIR, 'data', 'test', 'images')
        images = [f for f in os.listdir(test_dir) if f.endswith('.png')][:num_samples]
        
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        fig.suptitle('YOLOv8 Predictions - NephroLens\nPowered by Sanket Kelzarkar', 
                     fontsize=16, fontweight='bold')
        
        for idx, img_file in enumerate(images):
            if idx >= 10:
                break
                
            img_path = os.path.join(test_dir, img_file)
            
            # Predict
            results = self.model.predict(img_path, verbose=False)[0]
            
            # Plot
            row = idx // 5
            col = idx % 5
            
            # Load and display image with predictions
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Draw predictions
            if len(results.boxes) > 0:
                for box in results.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    label = f"{results.names[cls]} {conf:.2f}"
                    
                    cv2.rectangle(img_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(img_rgb, label, (int(x1), int(y1)-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            axes[row, col].imshow(img_rgb)
            axes[row, col].axis('off')
            axes[row, col].set_title(img_file[:15], fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'yolov8_predictions_sample.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Visualization saved to: {RESULTS_DIR}/yolov8_predictions_sample.png")

def main():
    """Main training pipeline"""
    # Initialize YOLOv8 (using nano model for faster training)
    yolo_trainer = YOLOv8KidneyStone(model_size='n')
    
    # Prepare dataset
    yolo_trainer.prepare_yolo_dataset()
    
    # Train model
    yolo_trainer.train_model(epochs=100, batch_size=16, img_size=640)
    
    # Validate
    yolo_trainer.validate_model()
    
    # Test
    yolo_trainer.test_model()
    
    # Export
    yolo_trainer.export_model()
    
    # Visualize
    yolo_trainer.visualize_predictions()
    
    print("\n" + "=" * 60)
    print("YOLOV8 TRAINING COMPLETED SUCCESSFULLY!")
    print("Powered by Sanket Kelzarkar")
    print("=" * 60)

if __name__ == "__main__":
    main()
    print("\n✓ Ready for next step: ResNet Model Training")