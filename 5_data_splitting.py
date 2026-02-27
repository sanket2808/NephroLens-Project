"""
NephroLens - Step 5: Data Splitting
Author: Sanket Kelzarkar
Description: Split dataset into Train, Validation, and Test sets
"""

import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json

# Project Configuration
PROJECT_DIR = r"C:\Sem VI\ML Projects\KSPML"
PREPROCESSED_DIR = os.path.join(PROJECT_DIR, "preprocessed_data")
SPLIT_DIR = os.path.join(PROJECT_DIR, "split_data")

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

def validate_ratios():
    """Validate split ratios sum to 1.0"""
    total = TRAIN_RATIO + VAL_RATIO + TEST_RATIO
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")

def get_all_images(base_dir):
    """Get all images from preprocessed directory"""
    images_data = {'stone': [], 'normal': []}
    
    for category in ['stone', 'normal']:
        category_dir = os.path.join(base_dir, 'train', category)
        
        if os.path.exists(category_dir):
            images = [f for f in os.listdir(category_dir) if f.endswith('.png')]
            
            for img_file in images:
                img_path = os.path.join(category_dir, img_file)
                images_data[category].append((img_file, img_path))
    
    return images_data

def stratified_split(images_data):
    """Perform stratified splitting"""
    print("=" * 60)
    print("NEPHROLENS - DATA SPLITTING")
    print("Powered by Sanket Kelzarkar")
    print("=" * 60)
    
    print(f"\n[SPLIT CONFIGURATION]")
    print(f"Train: {TRAIN_RATIO*100:.0f}%")
    print(f"Validation: {VAL_RATIO*100:.0f}%")
    print(f"Test: {TEST_RATIO*100:.0f}%")
    
    splits = {'train': {}, 'val': {}, 'test': {}}
    split_stats = {}
    
    for category, images in images_data.items():
        print(f"\n[Splitting {category.upper()} class: {len(images)} images]")
        
        # Extract file paths
        files = [img[1] for img in images]
        
        # First split: train + (val + test)
        train_files, temp_files = train_test_split(
            files, 
            test_size=(VAL_RATIO + TEST_RATIO),
            random_state=42,
            shuffle=True
        )
        
        # Second split: val + test
        val_size_adjusted = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
        val_files, test_files = train_test_split(
            temp_files,
            test_size=(1 - val_size_adjusted),
            random_state=42,
            shuffle=True
        )
        
        splits['train'][category] = train_files
        splits['val'][category] = val_files
        splits['test'][category] = test_files
        
        split_stats[category] = {
            'total': len(files),
            'train': len(train_files),
            'val': len(val_files),
            'test': len(test_files)
        }
        
        print(f"  Train: {len(train_files)}")
        print(f"  Validation: {len(val_files)}")
        print(f"  Test: {len(test_files)}")
    
    return splits, split_stats

def copy_files_to_splits(splits):
    """Copy files to respective split directories"""
    print("\n[COPYING FILES TO SPLIT DIRECTORIES]")
    
    # Create split directories
    for split in ['train', 'val', 'test']:
        for category in ['stone', 'normal']:
            os.makedirs(os.path.join(SPLIT_DIR, split, category), exist_ok=True)
    
    total_copied = 0
    
    for split_name, categories in splits.items():
        print(f"\n  Processing {split_name} split...")
        
        for category, file_paths in categories.items():
            for src_path in tqdm(file_paths, desc=f"    {category}"):
                filename = os.path.basename(src_path)
                dst_path = os.path.join(SPLIT_DIR, split_name, category, filename)
                
                shutil.copy2(src_path, dst_path)
                total_copied += 1
    
    print(f"\n✓ Total files copied: {total_copied}")

def verify_splits(split_stats):
    """Verify split integrity"""
    print("\n[VERIFYING SPLITS]")
    
    for category, stats in split_stats.items():
        total_split = stats['train'] + stats['val'] + stats['test']
        
        if total_split != stats['total']:
            print(f"✗ WARNING: {category} - Total mismatch!")
            print(f"  Expected: {stats['total']}, Got: {total_split}")
        else:
            print(f"✓ {category} - Verification passed")
    
    # Verify actual files
    print("\n[VERIFYING FILE COUNTS]")
    
    for split in ['train', 'val', 'test']:
        for category in ['stone', 'normal']:
            split_dir = os.path.join(SPLIT_DIR, split, category)
            
            if os.path.exists(split_dir):
                count = len([f for f in os.listdir(split_dir) if f.endswith('.png')])
                expected = split_stats[category][split]
                
                if count == expected:
                    print(f"✓ {split}/{category}: {count} files")
                else:
                    print(f"✗ {split}/{category}: Expected {expected}, Found {count}")

def generate_split_report(split_stats):
    """Generate detailed split report"""
    print("\n" + "=" * 60)
    print("DATA SPLITTING SUMMARY")
    print("=" * 60)
    
    # Calculate percentages
    total_images = sum(stats['total'] for stats in split_stats.values())
    
    print(f"\nTotal Images: {total_images}")
    print(f"\nClass Distribution:")
    for category, stats in split_stats.items():
        percentage = (stats['total'] / total_images) * 100
        print(f"  {category}: {stats['total']} ({percentage:.1f}%)")
    
    print(f"\nSplit Distribution:")
    for split in ['train', 'val', 'test']:
        split_total = sum(stats[split] for stats in split_stats.values())
        split_percentage = (split_total / total_images) * 100
        print(f"\n{split.upper()} ({split_percentage:.1f}%):")
        
        for category, stats in split_stats.items():
            category_split_pct = (stats[split] / stats['total']) * 100
            print(f"  {category}: {stats[split]} ({category_split_pct:.1f}% of {category} class)")
    
    # Save report to file
    report_path = os.path.join(PROJECT_DIR, "data_splitting_report.json")
    
    report_data = {
        'total_images': total_images,
        'split_ratios': {
            'train': TRAIN_RATIO,
            'val': VAL_RATIO,
            'test': TEST_RATIO
        },
        'class_stats': split_stats,
        'metadata': {
            'author': 'Sanket Kelzarkar',
            'project': 'NephroLens',
            'stratified': True
        }
    }
    
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=4)
    
    print(f"\n✓ Detailed report saved to: {report_path}")
    
    print("\n" + "=" * 60)
    print("DATA SPLITTING COMPLETED SUCCESSFULLY!")
    print("Powered by Sanket Kelzarkar")
    print("=" * 60)

def create_data_manifest():
    """Create manifest file for dataset"""
    print("\n[CREATING DATA MANIFEST]")
    
    manifest = {
        'dataset_name': 'NephroLens Kidney Stone Detection',
        'version': '1.0',
        'author': 'Sanket Kelzarkar',
        'splits': {}
    }
    
    for split in ['train', 'val', 'test']:
        manifest['splits'][split] = {}
        
        for category in ['stone', 'normal']:
            split_dir = os.path.join(SPLIT_DIR, split, category)
            
            if os.path.exists(split_dir):
                files = [f for f in os.listdir(split_dir) if f.endswith('.png')]
                manifest['splits'][split][category] = {
                    'count': len(files),
                    'files': files
                }
    
    manifest_path = os.path.join(SPLIT_DIR, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"✓ Manifest created: {manifest_path}")

def main():
    """Main data splitting pipeline"""
    # Validate ratios
    validate_ratios()
    
    # Get all images
    images_data = get_all_images(PREPROCESSED_DIR)
    
    # Perform stratified split
    splits, split_stats = stratified_split(images_data)
    
    # Copy files to split directories
    copy_files_to_splits(splits)
    
    # Verify splits
    verify_splits(split_stats)
    
    # Generate report
    generate_split_report(split_stats)
    
    # Create manifest
    create_data_manifest()
    
    print("\n✓ Ready for next step: Model Training (YOLOv8)")

if __name__ == "__main__":
    main()