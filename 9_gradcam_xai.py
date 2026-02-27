"""
NephroLens - Step 9: Grad-CAM Explainable AI
Author: Sanket Kelzarkar
Description: Generate Grad-CAM visualizations for model interpretability with advanced features
"""

import os
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from typing import Tuple, List, Dict, Optional
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
import logging

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
PROJECT_DIR = r"C:\Sem VI\ML Projects\KSPML"
PREPROCESSED_DIR = os.path.join(PROJECT_DIR, "preprocessed_data")
RESNET_DIR = os.path.join(PROJECT_DIR, "resnet_training")
GRADCAM_DIR = os.path.join(PROJECT_DIR, "gradcam_results")

# Grad-CAM parameters
GRADCAM_ALPHA = 0.4
ATTENTION_THRESHOLD = 0.5
VISUALIZATION_DPI = 300
IMAGE_SIZE = 224
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

os.makedirs(GRADCAM_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for model interpretability.
    Generates heat maps that highlight regions contributing to predictions.
    """
    
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        """
        Args:
            model: PyTorch model for which to generate CAM
            target_layer: Target layer for activation/gradient capture
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
        logger.info(f"Grad-CAM initialized for layer: {target_layer}")
    
    def save_activation(self, module, input, output) -> None:
        """Save forward pass activations."""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output) -> None:
        """Save backward pass gradients."""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_image: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate Class Activation Map.
        
        Args:
            input_image: Input tensor of shape (1, C, H, W)
            target_class: Target class for CAM generation. If None, uses predicted class.
            
        Returns:
            CAM as numpy array normalized to [0, 1]
        """
        # Forward pass
        model_output = self.model(input_image)
        
        if target_class is None:
            target_class = model_output.argmax(dim=1)
        
        # Reset gradients
        self.model.zero_grad()
        
        # Create one-hot encoded target
        one_hot = torch.zeros_like(model_output)
        one_hot[0, target_class] = 1
        
        # Backward pass
        model_output.backward(gradient=one_hot, retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Compute weights: global average pooling of gradients
        weights = gradients.mean(dim=(1, 2))  # [C]
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU to get positive activations
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        
        return cam.cpu().numpy()

class GradCAMVisualizer:
    """Generate and visualize Grad-CAM heatmaps with comprehensive analysis."""
    
    def __init__(self, model_path: str, device: str = 'cuda') -> None:
        """
        Initialize Grad-CAM visualizer.
        
        Args:
            model_path: Path to trained model weights
            device: Device to use ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.model = self.load_model(model_path)
        self.transform = self.get_transform()
        
        # Initialize Grad-CAM on last conv layer
        target_layer = self.model.layer4[-1]
        self.gradcam = GradCAM(self.model, target_layer)
        self.classes = ['Stone', 'Normal']
    
    def load_model(self, model_path: str) -> torch.nn.Module:
        """
        Load pre-trained ResNet model.
        
        Args:
            model_path: Path to model weights
            
        Returns:
            Loaded model in eval mode
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
            
        model = models.resnet50(weights=None)
        num_features = model.fc.in_features
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(num_features, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, 2)
        )
        
        try:
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model = model.to(self.device)
            model.eval()
            logger.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        return model
    
    def get_transform(self) -> transforms.Compose:
        """Get image transform with standard ImageNet normalization."""
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
        ])
    
    def overlay_heatmap(self, img: np.ndarray, heatmap: np.ndarray, 
                       alpha: float = GRADCAM_ALPHA) -> Tuple[np.ndarray, np.ndarray]:
        """
        Overlay Grad-CAM heatmap on original image.
        
        Args:
            img: Original image as numpy array
            heatmap: Grad-CAM heatmap normalized to [0, 1]
            alpha: Blending factor [0, 1]
            
        Returns:
            Tuple of (overlayed image, colored heatmap)
        """
        # Ensure heatmap is in [0, 1]
        heatmap = np.clip(heatmap, 0, 1)
        
        # Resize heatmap to match image size
        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        
        # Convert heatmap to RGB with JET colormap
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized), 
            cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Blend images
        overlayed = cv2.addWeighted(img.astype(np.float32), 1-alpha, 
                                   heatmap_colored.astype(np.float32), alpha, 0)
        
        return overlayed.astype(np.uint8), heatmap_colored
    
    def visualize_gradcam(self, image_path: str, save_path: Optional[str] = None) -> Tuple[np.ndarray, int, float]:
        """
        Generate and visualize Grad-CAM for single image.
        
        Args:
            image_path: Path to input image
            save_path: Optional path to save visualization
            
        Returns:
            Tuple of (CAM, predicted_class, confidence)
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load and preprocess image
        img_pil = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
        
        # Generate Grad-CAM
        cam = self.gradcam.generate_cam(img_tensor)
        
        # Get prediction
        with torch.no_grad():
            output = self.model(img_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        # Load original image for visualization
        img_np = np.array(img_pil)
        
        # Create overlay
        overlayed, heatmap_colored = self.overlay_heatmap(img_np, cam)
        
        # Create attention regions mask
        threshold = ATTENTION_THRESHOLD
        attention_mask = cam > threshold
        attention_mask_resized = cv2.resize(
            attention_mask.astype(np.uint8), 
            (img_np.shape[1], img_np.shape[0])
        ).astype(bool)
        attention_img = img_np.copy()
        attention_img[~attention_mask_resized] = (attention_img[~attention_mask_resized] * 0.4).astype(np.uint8)
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        axes[0, 0].imshow(img_np)
        axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(heatmap_colored)
        axes[0, 1].set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(overlayed)
        axes[1, 0].set_title('Overlay (α=0.4)', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(attention_img)
        axes[1, 1].set_title(f'Attention Regions (threshold={threshold})', fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')
        
        # Add prediction info
        pred_label = self.classes[predicted.item()]
        conf_percent = confidence.item() * 100
        prediction_text = f"Prediction: {pred_label}\nConfidence: {conf_percent:.2f}%"
        
        fig.suptitle(f'Grad-CAM Analysis - NephroLens\n{prediction_text}\nPowered by Sanket Kelzarkar', 
                    fontsize=14, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=VISUALIZATION_DPI, bbox_inches='tight')
            logger.info(f"Saved: {save_path}")
        
        plt.close()
        
        return cam, predicted.item(), confidence.item()
    
    def batch_visualize(self, data_dir: str, num_samples: int = 20) -> List[Dict]:
        """
        Generate Grad-CAM visualizations for multiple images.
        
        Args:
            data_dir: Root directory containing test data
            num_samples: Total number of samples to process
            
        Returns:
            List of result dictionaries with analysis data
        """
        logger.info("=" * 70)
        logger.info("NEPHROLENS - GRAD-CAM EXPLAINABLE AI ANALYSIS")
        logger.info("Powered by Sanket Kelzarkar")
        logger.info("=" * 70)
        logger.info(f"Generating Grad-CAM visualizations for {num_samples} samples")
        
        results = []
        sample_count = 0
        
        for category in ['stone', 'normal']:
            category_dir = os.path.join(data_dir, 'test', category)
            
            if not os.path.exists(category_dir):
                logger.warning(f"Category directory not found: {category_dir}")
                continue
            
            images = [f for f in os.listdir(category_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            if not images:
                logger.warning(f"No images found in {category_dir}")
                continue
                
            samples = np.random.choice(images, min(num_samples//2, len(images)), replace=False)
            logger.info(f"Processing {len(samples)} {category} samples...")
            
            for img_file in tqdm(samples, desc=f"{category.capitalize()} samples"):
                img_path = os.path.join(category_dir, img_file)
                save_dir = os.path.join(GRADCAM_DIR, category)
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"gradcam_{sample_count:04d}.png")
                
                try:
                    cam, prediction, confidence = self.visualize_gradcam(img_path, save_path)
                    
                    results.append({
                        'image': img_file,
                        'true_label': category,
                        'prediction': prediction,
                        'predicted_label': self.classes[prediction],
                        'confidence': float(confidence),
                        'confidence_percent': float(confidence * 100),
                        'cam_path': save_path,
                        'correct': category == self.classes[prediction].lower()
                    })
                    
                    sample_count += 1
                except Exception as e:
                    logger.error(f"Failed to process {img_file}: {e}")
                    continue
        
        logger.info(f"✓ Processed {len(results)} images successfully")
        return results
    
    def create_comparison_grid(self, results: List[Dict], num_display: int = 8) -> None:
        """
        Create comparison grid of Grad-CAM results.
        
        Args:
            results: List of result dictionaries from batch_visualize
            num_display: Number of samples to display in grid
        """
        if not results:
            logger.warning("No results to display")
            return
            
        logger.info("Creating comparison grid...")
        
        num_display = min(num_display, len(results))
        selected = np.random.choice(results, num_display, replace=False)
        
        rows = (num_display + 3) // 4
        fig, axes = plt.subplots(rows, 4, figsize=(20, 5*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('Grad-CAM Analysis Results - NephroLens\nPowered by Sanket Kelzarkar',
                    fontsize=16, fontweight='bold')
        
        for idx, result in enumerate(selected):
            row = idx // 4
            col = idx % 4
            
            try:
                img = plt.imread(result['cam_path'])
                axes[row, col].imshow(img)
                
                true_label = result['true_label'].capitalize()
                pred_label = result['predicted_label']
                conf = result['confidence_percent']
                correct = result['correct']
                
                title = f"True: {true_label}\nPred: {pred_label}\nConf: {conf:.1f}%"
                color = 'green' if correct else 'red'
                
                axes[row, col].set_title(title, fontsize=10, color=color, fontweight='bold')
                axes[row, col].axis('off')
            except Exception as e:
                logger.error(f"Failed to display result: {e}")
                axes[row, col].axis('off')
        
        # Hide unused subplots
        for idx in range(num_display, rows * 4):
            row = idx // 4
            col = idx % 4
            axes[row, col].axis('off')
        
        plt.tight_layout()
        grid_path = os.path.join(GRADCAM_DIR, 'gradcam_comparison_grid.png')
        plt.savefig(grid_path, dpi=VISUALIZATION_DPI, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Comparison grid saved to {grid_path}")
    
    def save_statistics(self, results: List[Dict]) -> None:
        """
        Save analysis statistics to JSON file.
        
        Args:
            results: List of result dictionaries
        """
        if not results:
            return
            
        # Calculate metrics
        correct = sum(1 for r in results if r['correct'])
        accuracy = (correct / len(results)) * 100
        
        # Per-class metrics
        stone_results = [r for r in results if r['true_label'] == 'stone']
        normal_results = [r for r in results if r['true_label'] == 'normal']
        
        stone_acc = (sum(1 for r in stone_results if r['correct']) / len(stone_results) * 100) if stone_results else 0
        normal_acc = (sum(1 for r in normal_results if r['correct']) / len(normal_results) * 100) if normal_results else 0
        
        stone_conf = np.mean([r['confidence'] for r in stone_results]) * 100 if stone_results else 0
        normal_conf = np.mean([r['confidence'] for r in normal_results]) * 100 if normal_results else 0
        
        statistics = {
            'total_samples': len(results),
            'correct_predictions': correct,
            'overall_accuracy': round(accuracy, 2),
            'per_class_metrics': {
                'stone': {
                    'samples': len(stone_results),
                    'accuracy': round(stone_acc, 2),
                    'avg_confidence': round(stone_conf, 2)
                },
                'normal': {
                    'samples': len(normal_results),
                    'accuracy': round(normal_acc, 2),
                    'avg_confidence': round(normal_conf, 2)
                }
            },
            'detailed_results': results
        }
        
        stats_path = os.path.join(GRADCAM_DIR, 'gradcam_statistics.json')
        with open(stats_path, 'w') as f:
            json.dump(statistics, f, indent=2)
        
        logger.info(f"✓ Statistics saved to {stats_path}")

def main():
    """Main Grad-CAM analysis pipeline."""
    try:
        # Verify model exists
        model_path = os.path.join(RESNET_DIR, 'resnet50_best.pth')
        if not os.path.exists(model_path):
            logger.error(f"Model not found: {model_path}")
            logger.error("Please train ResNet model first (Step 7)")
            return
        
        # Verify data exists
        test_data_path = os.path.join(PREPROCESSED_DIR, 'test')
        if not os.path.exists(test_data_path):
            logger.error(f"Test data not found: {test_data_path}")
            logger.error("Please preprocess data first")
            return
        
        # Initialize visualizer
        logger.info("Initializing Grad-CAM visualizer...")
        visualizer = GradCAMVisualizer(model_path)
        
        # Generate Grad-CAM visualizations
        logger.info("Starting batch visualization...")
        results = visualizer.batch_visualize(PREPROCESSED_DIR, num_samples=20)
        
        if not results:
            logger.error("No results generated. Check input data and logs.")
            return
        
        # Create comparison grid
        visualizer.create_comparison_grid(results)
        
        # Save detailed statistics
        visualizer.save_statistics(results)
        
        # Print summary
        logger.info("=" * 70)
        logger.info("GRAD-CAM ANALYSIS SUMMARY")
        logger.info("=" * 70)
        
        correct = sum(1 for r in results if r['correct'])
        accuracy = (correct / len(results)) * 100
        
        logger.info(f"Total samples analyzed: {len(results)}")
        logger.info(f"Correct predictions: {correct}")
        logger.info(f"Overall accuracy: {accuracy:.2f}%")
        
        # Per-class stats
        for category in ['stone', 'normal']:
            category_results = [r for r in results if r['true_label'] == category]
            if category_results:
                cat_correct = sum(1 for r in category_results if r['correct'])
                cat_acc = (cat_correct / len(category_results)) * 100
                avg_conf = np.mean([r['confidence_percent'] for r in category_results])
                logger.info(f"\n{category.upper()}:")
                logger.info(f"  Samples: {len(category_results)}")
                logger.info(f"  Accuracy: {cat_acc:.2f}%")
                logger.info(f"  Avg Confidence: {avg_conf:.2f}%")
        
        logger.info(f"\nResults saved to: {GRADCAM_DIR}")
        logger.info("=" * 70)
        logger.info("GRAD-CAM ANALYSIS COMPLETED!")
        logger.info("Powered by Sanket Kelzarkar")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
    logger.info("✓ Ready for next step: Model Evaluation (Step 10)")