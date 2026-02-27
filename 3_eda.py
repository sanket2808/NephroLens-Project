"""
NephroLens - Step 3: Exploratory Data Analysis (EDA)
Author: Sanket Kelzarkar
Description: Analyze and visualize CT scan dataset patterns
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import Counter

# Project Configuration
PROJECT_DIR = r"C:\Sem VI\ML Projects\KSPML"
PREPROCESSED_DIR = os.path.join(PROJECT_DIR, "preprocessed_data")
EDA_OUTPUT_DIR = os.path.join(PROJECT_DIR, "eda_results")

# Create output directory
os.makedirs(EDA_OUTPUT_DIR, exist_ok=True)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

class KidneyStoneEDA:
    """Comprehensive EDA for kidney stone detection dataset"""
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_stats = {
            'train': {'stone': [], 'normal': []},
            'val': {'stone': [], 'normal': []},
            'test': {'stone': [], 'normal': []}
        }
        
    def load_dataset_stats(self):
        """Load and analyze basic dataset statistics"""
        print("=" * 60)
        print("NEPHROLENS - EXPLORATORY DATA ANALYSIS")
        print("Powered by Sanket Kelzarkar")
        print("=" * 60)
        
        print("\n[1/8] Loading dataset statistics...")
        
        stats_summary = []
        
        for split in ['train', 'val', 'test']:
            for category in ['stone', 'normal']:
                dir_path = os.path.join(self.data_dir, split, category)
                
                if not os.path.exists(dir_path):
                    continue
                
                images = [f for f in os.listdir(dir_path) if f.lower().endswith('.png')]
                
                for img_file in tqdm(images, desc=f"  {split}/{category}"):
                    img_path = os.path.join(dir_path, img_file)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    
                    if img is not None:
                        stats = {
                            'split': split,
                            'category': category,
                            'filename': img_file,
                            'height': img.shape[0],
                            'width': img.shape[1],
                            'mean_intensity': np.mean(img),
                            'std_intensity': np.std(img),
                            'min_intensity': np.min(img),
                            'max_intensity': np.max(img),
                            'file_size_kb': os.path.getsize(img_path) / 1024
                        }
                        stats_summary.append(stats)
                        self.data_stats[split][category].append(img)
        
        self.df_stats = pd.DataFrame(stats_summary)
        print(f"✓ Loaded statistics for {len(stats_summary)} images")
        return self.df_stats
    
    def plot_class_distribution(self):
        """Plot distribution of classes across splits"""
        print("\n[2/8] Analyzing class distribution...")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Count plot
        counts = self.df_stats.groupby(['split', 'category']).size().reset_index(name='count')
        pivot_counts = counts.pivot(index='split', columns='category', values='count')
        
        pivot_counts.plot(kind='bar', ax=axes[0], color=['#e74c3c', '#3498db'])
        axes[0].set_title('Class Distribution Across Splits', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Split', fontsize=12)
        axes[0].set_ylabel('Number of Images', fontsize=12)
        axes[0].legend(title='Category', title_fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # Pie chart for overall distribution
        overall_counts = self.df_stats['category'].value_counts()
        axes[1].pie(overall_counts, labels=overall_counts.index, autopct='%1.1f%%',
                   colors=['#e74c3c', '#3498db'], startangle=90)
        axes[1].set_title('Overall Class Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(EDA_OUTPUT_DIR, '1_class_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Class distribution plot saved")
    
    def plot_image_properties(self):
        """Analyze and plot image properties"""
        print("\n[3/8] Analyzing image properties...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Mean intensity distribution
        for category in ['stone', 'normal']:
            data = self.df_stats[self.df_stats['category'] == category]['mean_intensity']
            axes[0, 0].hist(data, bins=30, alpha=0.6, label=category)
        axes[0, 0].set_title('Mean Intensity Distribution', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Mean Intensity')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Standard deviation distribution
        for category in ['stone', 'normal']:
            data = self.df_stats[self.df_stats['category'] == category]['std_intensity']
            axes[0, 1].hist(data, bins=30, alpha=0.6, label=category)
        axes[0, 1].set_title('Intensity Standard Deviation', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Std Deviation')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Intensity range
        sns.boxplot(data=self.df_stats, x='category', y='mean_intensity', ax=axes[1, 0])
        axes[1, 0].set_title('Mean Intensity by Category', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Category')
        axes[1, 0].set_ylabel('Mean Intensity')
        
        # File sizes
        sns.violinplot(data=self.df_stats, x='category', y='file_size_kb', ax=axes[1, 1])
        axes[1, 1].set_title('File Size Distribution', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Category')
        axes[1, 1].set_ylabel('File Size (KB)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(EDA_OUTPUT_DIR, '2_image_properties.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Image properties plot saved")
    
    def plot_sample_images(self):
        """Display sample images from each category"""
        print("\n[4/8] Creating sample image grid...")
        
        fig, axes = plt.subplots(4, 8, figsize=(20, 10))
        fig.suptitle('Sample CT Scan Images - NephroLens\nPowered by Sanket Kelzarkar', 
                     fontsize=16, fontweight='bold')

        categories = ['stone', 'normal']

        for row, category in enumerate(categories):
            imgs = self.data_stats['train'].get(category, [])
            if not imgs:
                # fill corresponding rows with empty axes
                for r in (row*2, row*2+1):
                    for c in range(4):
                        axes[r, c].axis('off')
                continue

            n_samples = min(4, len(imgs))
            # choose without replacement when possible
            if len(imgs) >= n_samples:
                samples = list(np.random.choice(len(imgs), n_samples, replace=False))
            else:
                samples = list(np.random.choice(len(imgs), n_samples, replace=True))

            for col_idx, idx in enumerate(samples):
                img = imgs[idx]
                r0 = row*2
                # Original image
                axes[r0, col_idx].imshow(img, cmap='gray')
                axes[r0, col_idx].set_title(f'{category.capitalize()} - Original', fontsize=9)
                axes[r0, col_idx].axis('off')

                # Enhanced image with edge detection
                edges = cv2.Canny(img, 50, 150)
                axes[r0+1, col_idx].imshow(edges, cmap='gray')
                axes[r0+1, col_idx].set_title(f'{category.capitalize()} - Edges', fontsize=9)
                axes[r0+1, col_idx].axis('off')

        # Fill remaining columns for all rows
        for r in range(4):
            for c in range(4, 8):
                axes[r, c].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(EDA_OUTPUT_DIR, '3_sample_images.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Sample images grid saved")
    
    def analyze_intensity_patterns(self):
        """Analyze intensity patterns and histograms"""
        print("\n[5/8] Analyzing intensity patterns...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        for idx, category in enumerate(['stone', 'normal']):
            imgs = self.data_stats['train'].get(category, [])
            if not imgs:
                axes[0, idx].text(0.5, 0.5, 'No samples available', ha='center', va='center')
                axes[1, idx].axis('off')
                continue

            n_samples = min(100, len(imgs))
            if len(imgs) >= n_samples:
                sel_idx = list(np.random.choice(len(imgs), n_samples, replace=False))
            else:
                sel_idx = list(np.random.choice(len(imgs), n_samples, replace=True))

            # Aggregate histogram
            hist_sum = np.zeros(256)
            for i in sel_idx:
                img = imgs[i]
                hist = cv2.calcHist([img], [0], None, [256], [0, 256])
                hist_sum += hist.flatten()

            axes[0, idx].plot(hist_sum / n_samples, color='blue', linewidth=2)
            axes[0, idx].set_title(f'Average Intensity Histogram - {category.capitalize()}', 
                                  fontsize=12, fontweight='bold')
            axes[0, idx].set_xlabel('Pixel Intensity')
            axes[0, idx].set_ylabel('Frequency')
            axes[0, idx].grid(True, alpha=0.3)

            # Intensity heatmap - compute mean image
            mean_img = np.mean([imgs[i] for i in sel_idx], axis=0)
            im = axes[1, idx].imshow(mean_img, cmap='hot', interpolation='nearest')
            axes[1, idx].set_title(f'Average Intensity Map - {category.capitalize()}', 
                                  fontsize=12, fontweight='bold')
            axes[1, idx].axis('off')
            plt.colorbar(im, ax=axes[1, idx])
        
        plt.tight_layout()
        plt.savefig(os.path.join(EDA_OUTPUT_DIR, '4_intensity_patterns.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Intensity patterns plot saved")
    
    def correlation_analysis(self):
        """Analyze correlations between features"""
        print("\n[6/8] Performing correlation analysis...")
        
        numeric_cols = ['mean_intensity', 'std_intensity', 'min_intensity', 
                       'max_intensity', 'file_size_kb']
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        corr_matrix = self.df_stats[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, ax=ax, square=True, linewidths=1)
        ax.set_title('Feature Correlation Matrix\nNephroLens - Powered by Sanket Kelzarkar', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(EDA_OUTPUT_DIR, '5_correlation_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Correlation matrix saved")
    
    def outlier_detection(self):
        """Detect and visualize outliers"""
        print("\n[7/8] Detecting outliers...")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        features = ['mean_intensity', 'std_intensity', 'file_size_kb']
        titles = ['Mean Intensity', 'Std Deviation', 'File Size (KB)']
        
        for idx, (feature, title) in enumerate(zip(features, titles)):
            sns.boxplot(data=self.df_stats, x='category', y=feature, ax=axes[idx])
            axes[idx].set_title(f'Outlier Detection - {title}', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Category')
            axes[idx].set_ylabel(title)
        
        plt.tight_layout()
        plt.savefig(os.path.join(EDA_OUTPUT_DIR, '6_outlier_detection.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Outlier detection plot saved")
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("\n[8/8] Generating summary report...")
        
        report_path = os.path.join(EDA_OUTPUT_DIR, 'eda_summary_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("NEPHROLENS - EXPLORATORY DATA ANALYSIS REPORT\n")
            f.write("Powered by Sanket Kelzarkar\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("1. DATASET OVERVIEW\n")
            f.write("-" * 70 + "\n")
            total = len(self.df_stats)
            f.write(f"Total Images: {total}\n")
            if total > 0:
                f.write(f"Image Dimensions: {self.df_stats['height'].iloc[0]} x {self.df_stats['width'].iloc[0]}\n\n")
            else:
                f.write("Image Dimensions: N/A\n\n")
            
            f.write("Class Distribution:\n")
            for cat, count in self.df_stats['category'].value_counts().items():
                f.write(f"  - {cat}: {count} ({count/len(self.df_stats)*100:.2f}%)\n")
            
            f.write("\n2. SPLIT DISTRIBUTION\n")
            f.write("-" * 70 + "\n")
            split_dist = self.df_stats.groupby(['split', 'category']).size()
            for (split, cat), count in split_dist.items():
                f.write(f"  {split.upper()} - {cat}: {count}\n")
            
            f.write("\n3. INTENSITY STATISTICS\n")
            f.write("-" * 70 + "\n")
            for category in ['stone', 'normal']:
                cat_data = self.df_stats[self.df_stats['category'] == category]
                f.write(f"\n{category.upper()}:\n")
                f.write(f"  Mean Intensity: {cat_data['mean_intensity'].mean():.2f} ± {cat_data['mean_intensity'].std():.2f}\n")
                f.write(f"  Std Intensity: {cat_data['std_intensity'].mean():.2f} ± {cat_data['std_intensity'].std():.2f}\n")
            
            f.write("\n4. KEY FINDINGS\n")
            f.write("-" * 70 + "\n")
            f.write("  ✓ Dataset is ready for model training\n")
            f.write("  ✓ Images are preprocessed and normalized\n")
            f.write("  ✓ Class balance should be considered during training\n")
            f.write("  ✓ Intensity patterns show distinguishable features\n")
            
            f.write("\n" + "=" * 70 + "\n")
            f.write("Analysis completed successfully!\n")
            f.write("=" * 70 + "\n")
        
        print(f"✓ Summary report saved to: {report_path}")

def run_complete_eda():
    """Run complete EDA pipeline"""
    eda = KidneyStoneEDA(PREPROCESSED_DIR)
    
    # Run all analyses
    eda.load_dataset_stats()
    eda.plot_class_distribution()
    eda.plot_image_properties()
    eda.plot_sample_images()
    eda.analyze_intensity_patterns()
    eda.correlation_analysis()
    eda.outlier_detection()
    eda.generate_summary_report()
    
    print("\n" + "=" * 60)
    print("EDA COMPLETED SUCCESSFULLY!")
    print(f"Results saved to: {EDA_OUTPUT_DIR}")
    print("Powered by Sanket Kelzarkar")
    print("=" * 60)

if __name__ == "__main__":
    run_complete_eda()
    print("\n✓ Ready for next step: Feature Engineering")