"""
NephroLens - Step 1: Data Extraction
Author: Sanket Kelzarkar
Description: Extract and organize CT scan dataset for kidney stone detection
"""

import os
import zipfile
import shutil
from pathlib import Path
import requests

# Project Configuration
ZIP_PATH = r"C:\Sem VI\ML Projects\KSPML\Axial CT Imaging Dataset for AI-Powered Kidney Stone Detection A Resource for Deep Learning Research.zip"
PROJECT_DIR = r"C:\Sem VI\ML Projects\KSPML"
EXTRACTED_DIR = os.path.join(PROJECT_DIR, "extracted_data")
ORGANIZED_DIR = os.path.join(PROJECT_DIR, "organized_data")

def extract_dataset():
    """Extract the ZIP file containing CT scan images"""
    print("=" * 60)
    print("NEPHROLENS - DATA EXTRACTION")
    print("Powered by Sanket Kelzarkar")
    print("=" * 60)
    
    # Create directories
    os.makedirs(EXTRACTED_DIR, exist_ok=True)
    os.makedirs(ORGANIZED_DIR, exist_ok=True)
    
    # Extract ZIP file
    print(f"\n[1/3] Extracting dataset from: {ZIP_PATH}")
    try:
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(EXTRACTED_DIR)
        print("✓ Dataset extracted successfully!")
    except Exception as e:
        print(f"✗ Error extracting dataset: {e}")
        return False
    
    return True

def organize_data():
    """Organize data into structured folders"""
    print("\n[2/3] Organizing data into structured folders...")
    
    # Create organized structure
    train_dir = os.path.join(ORGANIZED_DIR, "train")
    val_dir = os.path.join(ORGANIZED_DIR, "val")
    test_dir = os.path.join(ORGANIZED_DIR, "test")
    
    for directory in [train_dir, val_dir, test_dir]:
        os.makedirs(os.path.join(directory, "stone"), exist_ok=True)
        os.makedirs(os.path.join(directory, "normal"), exist_ok=True)
    
    # Walk through extracted directory
    image_count = 0
    for root, dirs, files in os.walk(EXTRACTED_DIR):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.dcm', '.nii')):
                image_count += 1
                src_path = os.path.join(root, file)
                
                # Categorize based on filename or folder structure
                # Adjust this logic based on your dataset structure
                if 'stone' in root.lower() or 'stone' in file.lower():
                    category = 'stone'
                else:
                    category = 'normal'
                
                # For now, just copy to train folder (we'll split properly later)
                dest_path = os.path.join(train_dir, category, file)
                shutil.copy2(src_path, dest_path)
    
    print(f"✓ Organized {image_count} images successfully!")
    return True

def create_directory_summary():
    """Create a summary of the organized data"""
    print("\n[3/3] Creating data summary...")
    
    summary = {}
    for split in ['train', 'val', 'test']:
        summary[split] = {}
        split_dir = os.path.join(ORGANIZED_DIR, split)
        for category in ['stone', 'normal']:
            category_dir = os.path.join(split_dir, category)
            if os.path.exists(category_dir):
                count = len([f for f in os.listdir(category_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.dcm', '.nii'))])
                summary[split][category] = count
    
    # Print summary
    print("\n" + "=" * 60)
    print("DATA ORGANIZATION SUMMARY")
    print("=" * 60)
    for split, categories in summary.items():
        print(f"\n{split.upper()}:")
        for category, count in categories.items():
            print(f"  {category}: {count} images")
    
    # Save summary to file
    summary_file = os.path.join(PROJECT_DIR, "data_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("NephroLens - Data Summary\n")
        f.write("Powered by Sanket Kelzarkar\n")
        f.write("=" * 60 + "\n\n")
        for split, categories in summary.items():
            f.write(f"{split.upper()}:\n")
            for category, count in categories.items():
                f.write(f"  {category}: {count} images\n")
            f.write("\n")
    
    print(f"\n✓ Summary saved to: {summary_file}")
    print("\n" + "=" * 60)
    print("DATA EXTRACTION COMPLETED SUCCESSFULLY!")
    print("Powered by Sanket Kelzarkar")
    print("=" * 60)

if __name__ == "__main__":
    if extract_dataset():
        if organize_data():
            create_directory_summary()
            print("\n✓ Ready for next step: Data Preprocessing")
        else:
            print("\n✗ Error organizing data")
    else:
        print("\n✗ Error extracting dataset")