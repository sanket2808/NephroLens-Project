"""
NephroLens - Step 2: Data Preprocessing (Fixed)
Author: Sanket Kelzarkar
Description: Clean, normalize, and augment CT scan images
"""

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# Project Configuration
PROJECT_DIR = r"C:\Sem VI\ML Projects\KSPML"
ORGANIZED_DIR = os.path.join(PROJECT_DIR, "organized_data")
PREPROCESSED_DIR = os.path.join(PROJECT_DIR, "preprocessed_data")
IMG_SIZE = (512, 512)

class CTImagePreprocessor:
    """Preprocess CT scan images for kidney stone detection"""
    
    def __init__(self, img_size=(512, 512)):
        self.img_size = img_size
        self.preprocessing_log = []
        
    def load_image(self, image_path):
        """Load image from various formats"""
        try:
            # Handle DICOM files
            if image_path.lower().endswith('.dcm'):
                import pydicom
                dcm = pydicom.dcmread(image_path)
                img = dcm.pixel_array
                img = self.normalize_dicom(img)
            else:
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            return img
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return None
    
    def normalize_dicom(self, img):
        """Normalize DICOM pixel values to 0-255 range"""
        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min()) * 255
        return img.astype(np.uint8)
    
    def remove_noise(self, img):
        """Apply denoising filters"""
        # Gaussian blur to reduce noise
        denoised = cv2.GaussianBlur(img, (5, 5), 0)
        return denoised
    
    def enhance_contrast(self, img):
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img)
        return enhanced
    
    def resize_image(self, img):
        """Resize image to target size"""
        resized = cv2.resize(img, self.img_size, interpolation=cv2.INTER_AREA)
        return resized
    
    def normalize_pixels(self, img):
        """Normalize pixel values to 0-1 range"""
        normalized = img.astype(np.float32) / 255.0
        return normalized
    
    def check_image_quality(self, img):
        """Check if image meets quality standards"""
        # Check if image is too dark or too bright
        mean_intensity = np.mean(img)
        if mean_intensity < 20 or mean_intensity > 235:
            return False, "Poor brightness"
        
        # Check if image has sufficient variance
        if np.std(img) < 10:
            return False, "Low variance"
        
        return True, "Good quality"
    
    def preprocess_image(self, image_path, output_path):
        """Complete preprocessing pipeline"""
        # Load image
        img = self.load_image(image_path)
        if img is None:
            return False, "Failed to load"
        
        # Check quality
        is_good, quality_msg = self.check_image_quality(img)
        
        # Preprocess
        img = self.remove_noise(img)
        img = self.enhance_contrast(img)
        img = self.resize_image(img)
        
        # Save preprocessed image
        cv2.imwrite(output_path, img)
        
        return True, quality_msg

def create_basic_augmentations(img):
    """Create basic augmentations without albumentations library"""
    augmented_images = []
    
    # 1. Horizontal flip
    h_flip = cv2.flip(img, 1)
    augmented_images.append(('h_flip', h_flip))
    
    # 2. Vertical flip
    v_flip = cv2.flip(img, 0)
    augmented_images.append(('v_flip', v_flip))
    
    # 3. Rotation (15 degrees)
    h, w = img.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, 15, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    augmented_images.append(('rotate', rotated))
    
    # 4. Brightness adjustment
    bright = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
    augmented_images.append(('bright', bright))
    
    # 5. Add Gaussian noise
    noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
    noisy = cv2.add(img, noise)
    augmented_images.append(('noise', noisy))
    
    return augmented_images

def preprocess_dataset():
    """Preprocess entire dataset"""
    print("=" * 60)
    print("NEPHROLENS - DATA PREPROCESSING")
    print("Powered by Sanket Kelzarkar")
    print("=" * 60)
    
    preprocessor = CTImagePreprocessor(img_size=IMG_SIZE)
    
    # Create preprocessed directories
    for split in ['train', 'val', 'test']:
        for category in ['stone', 'normal']:
            os.makedirs(os.path.join(PREPROCESSED_DIR, split, category), exist_ok=True)
    
    stats = {'total': 0, 'success': 0, 'failed': 0, 'poor_quality': 0}
    
    # Process each split
    for split in ['train', 'val', 'test']:
        print(f"\n[Processing {split.upper()} set]")
        
        for category in ['stone', 'normal']:
            input_dir = os.path.join(ORGANIZED_DIR, split, category)
            output_dir = os.path.join(PREPROCESSED_DIR, split, category)
            
            if not os.path.exists(input_dir):
                continue
            
            image_files = [f for f in os.listdir(input_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.dcm', '.nii'))]
            
            print(f"  Processing {category}: {len(image_files)} images")
            
            for img_file in tqdm(image_files, desc=f"  {category}"):
                input_path = os.path.join(input_dir, img_file)
                output_path = os.path.join(output_dir, 
                                          os.path.splitext(img_file)[0] + '.png')
                
                stats['total'] += 1
                success, quality = preprocessor.preprocess_image(input_path, output_path)
                
                if success:
                    stats['success'] += 1
                    if quality != "Good quality":
                        stats['poor_quality'] += 1
                else:
                    stats['failed'] += 1
    
    # Generate preprocessing report
    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total images processed: {stats['total']}")
    print(f"Successfully preprocessed: {stats['success']}")
    print(f"Failed: {stats['failed']}")
    print(f"Poor quality (but processed): {stats['poor_quality']}")
    print(f"Success rate: {stats['success']/stats['total']*100:.2f}%")
    
    # Save report
    report_path = os.path.join(PROJECT_DIR, "preprocessing_report.txt")
    with open(report_path, 'w') as f:
        f.write("NephroLens - Preprocessing Report\n")
        f.write("Powered by Sanket Kelzarkar\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total images processed: {stats['total']}\n")
        f.write(f"Successfully preprocessed: {stats['success']}\n")
        f.write(f"Failed: {stats['failed']}\n")
        f.write(f"Poor quality: {stats['poor_quality']}\n")
    
    print(f"\n✓ Report saved to: {report_path}")
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETED!")
    print("Powered by Sanket Kelzarkar")
    print("=" * 60)

def augment_training_data():
    """Apply data augmentation to training set using basic OpenCV operations"""
    print("\n[AUGMENTING TRAINING DATA]")
    print("Using basic OpenCV augmentations (no albumentations required)")
    
    train_dir = os.path.join(PREPROCESSED_DIR, 'train')
    augmented_dir = os.path.join(PROJECT_DIR, 'augmented_data', 'train')
    
    for category in ['stone', 'normal']:
        input_dir = os.path.join(train_dir, category)
        output_dir = os.path.join(augmented_dir, category)
        os.makedirs(output_dir, exist_ok=True)
        
        image_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
        
        print(f"  Augmenting {category}: {len(image_files)} images")
        
        for img_file in tqdm(image_files, desc=f"  {category}"):
            img_path = os.path.join(input_dir, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            # Save original
            cv2.imwrite(os.path.join(output_dir, img_file), img)
            
            # Create augmented versions
            augmentations = create_basic_augmentations(img)
            
            # Save first 3 augmentations
            for i, (aug_type, aug_img) in enumerate(augmentations[:3]):
                aug_filename = f"{os.path.splitext(img_file)[0]}_aug{i+1}.png"
                cv2.imwrite(os.path.join(output_dir, aug_filename), aug_img)
    
    print("✓ Data augmentation completed!")

if __name__ == "__main__":
    preprocess_dataset()
    
    # Optional: Augment training data
    augment_choice = input("\nDo you want to augment training data? (y/n): ")
    if augment_choice.lower() == 'y':
        augment_training_data()
    
    print("\n✓ Ready for next step: Exploratory Data Analysis")