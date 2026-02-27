"""
Organize all extracted data (Original + Augmented) into train/val/test splits
"""
import os
import shutil
import random
from tqdm import tqdm

PROJECT_DIR = r"C:\Sem VI\ML Projects\KSPML"
EXTRACTED_DIR = os.path.join(PROJECT_DIR, "extracted_data")
ORGANIZED_DIR = os.path.join(PROJECT_DIR, "organized_data")

# Train/Val/Test split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

def organize_all_data():
    """Organize all extracted data into train/val/test splits"""
    print("=" * 70)
    print("ORGANIZING ALL EXTRACTED DATA")
    print("=" * 70)
    
    # Create organized directory structure
    for split in ['train', 'val', 'test']:
        for category in ['stone', 'normal']:
            os.makedirs(os.path.join(ORGANIZED_DIR, split, category), exist_ok=True)
    
    # Collect all images from extracted_data
    all_images = {
        'stone': [],
        'normal': []
    }
    
    print("\n[1/3] Collecting all images from extracted_data...")
    
    # Walk through Original and Augmented datasets
    extracted_base = EXTRACTED_DIR
    
    for root, dirs, files in os.walk(extracted_base):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.dcm', '.nii')):
                full_path = os.path.join(root, file)
                
                # Determine category based on folder name
                if 'non-stone' in root.lower() or 'normal' in root.lower():
                    all_images['normal'].append(full_path)
                elif 'stone' in root.lower():
                    all_images['stone'].append(full_path)
    
    total_stone = len(all_images['stone'])
    total_normal = len(all_images['normal'])
    total = total_stone + total_normal
    
    print(f"✓ Found {total_stone} stone images")
    print(f"✓ Found {total_normal} normal images")
    print(f"✓ Total: {total} images")
    
    # Shuffle images
    for category in all_images:
        random.shuffle(all_images[category])
    
    # Split and organize
    print("\n[2/3] Splitting data into train/val/test...")
    
    for category in ['stone', 'normal']:
        images = all_images[category]
        total = len(images)
        
        train_end = int(total * TRAIN_RATIO)
        val_end = train_end + int(total * VAL_RATIO)
        
        splits = {
            'train': images[:train_end],
            'val': images[train_end:val_end],
            'test': images[val_end:]
        }
        
        print(f"\n  {category.upper()}:")
        for split_name, split_images in splits.items():
            output_dir = os.path.join(ORGANIZED_DIR, split_name, category)
            print(f"    {split_name}: {len(split_images)} images", end=" -> ")
            
            for idx, src_path in enumerate(tqdm(split_images, desc=f"      {split_name}", disable=True)):
                try:
                    filename = os.path.basename(src_path)
                    dst_path = os.path.join(output_dir, filename)
                    
                    # Avoid overwriting existing files
                    if not os.path.exists(dst_path):
                        shutil.copy2(src_path, dst_path)
                except Exception as e:
                    print(f"Error copying {src_path}: {e}")
            
            print(f"copied")
    
    # Verify organization
    print("\n[3/3] Verifying organization...")
    
    total_organized = 0
    for split in ['train', 'val', 'test']:
        for category in ['stone', 'normal']:
            dir_path = os.path.join(ORGANIZED_DIR, split, category)
            count = len([f for f in os.listdir(dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.dcm', '.nii'))])
            total_organized += count
            print(f"  {split}/{category}: {count} images")
    
    print("\n" + "=" * 70)
    print(f"ORGANIZATION COMPLETED!")
    print(f"Total images organized: {total_organized}")
    print("=" * 70)

if __name__ == "__main__":
    organize_all_data()
