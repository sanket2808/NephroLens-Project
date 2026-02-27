"""
NephroLens - Step 4: Feature Engineering & Selection
Author: Sanket Kelzarkar
Description: Extract and engineer features from CT scan images
"""

import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.measure import shannon_entropy
from scipy.stats import skew, kurtosis
from tqdm import tqdm
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Project Configuration
PROJECT_DIR = r"C:\Sem VI\ML Projects\KSPML"
PREPROCESSED_DIR = os.path.join(PROJECT_DIR, "preprocessed_data")
FEATURES_DIR = os.path.join(PROJECT_DIR, "features")

os.makedirs(FEATURES_DIR, exist_ok=True)

class CTFeatureExtractor:
    """Extract comprehensive features from CT scan images"""
    
    def __init__(self):
        self.feature_names = []
        
    def extract_intensity_features(self, img):
        """Extract statistical intensity features"""
        features = {}
        
        # Basic statistics
        features['mean_intensity'] = np.mean(img)
        features['std_intensity'] = np.std(img)
        features['median_intensity'] = np.median(img)
        features['min_intensity'] = np.min(img)
        features['max_intensity'] = np.max(img)
        features['range_intensity'] = features['max_intensity'] - features['min_intensity']
        
        # Higher order statistics
        flat_img = img.flatten()
        features['skewness'] = skew(flat_img)
        features['kurtosis'] = kurtosis(flat_img)
        features['entropy'] = shannon_entropy(img)
        
        # Quartiles
        features['q1_intensity'] = np.percentile(img, 25)
        features['q3_intensity'] = np.percentile(img, 75)
        features['iqr_intensity'] = features['q3_intensity'] - features['q1_intensity']
        
        return features
    
    def extract_texture_features_glcm(self, img):
        """Extract GLCM (Gray-Level Co-occurrence Matrix) texture features"""
        features = {}
        
        # Normalize image to 0-255 range
        img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Calculate GLCM for multiple angles
        distances = [1, 3, 5]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        glcm = graycomatrix(img_norm, distances=distances, angles=angles, 
                           levels=256, symmetric=True, normed=True)
        
        # Extract properties
        properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
        
        for prop in properties:
            values = graycoprops(glcm, prop)
            features[f'glcm_{prop}_mean'] = np.mean(values)
            features[f'glcm_{prop}_std'] = np.std(values)
            features[f'glcm_{prop}_range'] = np.ptp(values)
        
        return features
    
    def extract_lbp_features(self, img):
        """Extract Local Binary Pattern (LBP) features"""
        features = {}
        
        # Calculate LBP
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(img, n_points, radius, method='uniform')
        
        # LBP histogram
        hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-6)  # Normalize
        
        # Statistical features from LBP
        features['lbp_mean'] = np.mean(lbp)
        features['lbp_std'] = np.std(lbp)
        features['lbp_entropy'] = shannon_entropy(lbp)
        
        # Histogram features
        for i in range(min(10, len(hist))):
            features[f'lbp_hist_bin_{i}'] = hist[i]
        
        return features
    
    def extract_edge_features(self, img):
        """Extract edge detection features"""
        features = {}
        
        # Canny edge detection
        edges_canny = cv2.Canny(img.astype(np.uint8), 50, 150)
        features['edge_density_canny'] = np.sum(edges_canny > 0) / edges_canny.size
        
        # Sobel edge detection
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        features['sobel_mean'] = np.mean(sobel_magnitude)
        features['sobel_std'] = np.std(sobel_magnitude)
        features['sobel_max'] = np.max(sobel_magnitude)
        
        # Laplacian
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        features['laplacian_var'] = np.var(laplacian)
        features['laplacian_mean'] = np.mean(np.abs(laplacian))
        
        return features
    
    def extract_shape_features(self, img):
        """Extract shape and morphological features"""
        features = {}
        
        # Threshold image
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            # Largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Contour features
            features['contour_area'] = cv2.contourArea(largest_contour)
            features['contour_perimeter'] = cv2.arcLength(largest_contour, True)
            
            # Circularity
            if features['contour_perimeter'] > 0:
                features['circularity'] = 4 * np.pi * features['contour_area'] / (features['contour_perimeter'] ** 2)
            else:
                features['circularity'] = 0
            
            # Moments
            moments = cv2.moments(largest_contour)
            if moments['m00'] != 0:
                features['hu_moment_1'] = moments['mu20'] / moments['m00']
                features['hu_moment_2'] = moments['mu02'] / moments['m00']
            else:
                features['hu_moment_1'] = 0
                features['hu_moment_2'] = 0
        else:
            features['contour_area'] = 0
            features['contour_perimeter'] = 0
            features['circularity'] = 0
            features['hu_moment_1'] = 0
            features['hu_moment_2'] = 0
        
        return features
    
    def extract_frequency_features(self, img):
        """Extract frequency domain features using FFT"""
        features = {}
        
        # Apply FFT
        f_transform = np.fft.fft2(img)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        # Features from magnitude spectrum
        features['fft_mean'] = np.mean(magnitude_spectrum)
        features['fft_std'] = np.std(magnitude_spectrum)
        features['fft_energy'] = np.sum(magnitude_spectrum ** 2)
        
        # High frequency content
        rows, cols = img.shape
        crow, ccol = rows // 2, cols // 2
        mask_size = 30
        high_freq_region = magnitude_spectrum.copy()
        high_freq_region[crow-mask_size:crow+mask_size, ccol-mask_size:ccol+mask_size] = 0
        features['high_freq_energy'] = np.sum(high_freq_region ** 2)
        
        return features
    
    def extract_all_features(self, img):
        """Extract all features from image"""
        all_features = {}
        
        # Extract features from different categories
        all_features.update(self.extract_intensity_features(img))
        all_features.update(self.extract_texture_features_glcm(img))
        all_features.update(self.extract_lbp_features(img))
        all_features.update(self.extract_edge_features(img))
        all_features.update(self.extract_shape_features(img))
        all_features.update(self.extract_frequency_features(img))
        
        return all_features

def extract_features_from_dataset():
    """Extract features from entire dataset"""
    print("=" * 60)
    print("NEPHROLENS - FEATURE ENGINEERING")
    print("Powered by Sanket Kelzarkar")
    print("=" * 60)
    
    extractor = CTFeatureExtractor()
    all_features = []
    
    for split in ['train', 'val', 'test']:
        print(f"\n[Processing {split.upper()} set]")
        
        for category in ['stone', 'normal']:
            data_dir = os.path.join(PREPROCESSED_DIR, split, category)
            
            if not os.path.exists(data_dir):
                continue
            
            images = [f for f in os.listdir(data_dir) if f.endswith('.png')]
            
            for img_file in tqdm(images, desc=f"  {category}"):
                img_path = os.path.join(data_dir, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is not None:
                    # Extract features
                    features = extractor.extract_all_features(img)
                    features['filename'] = img_file
                    features['split'] = split
                    features['category'] = category
                    features['label'] = 1 if category == 'stone' else 0
                    
                    all_features.append(features)
    
    # Create DataFrame
    df_features = pd.DataFrame(all_features)
    
    # Save features
    for split in ['train', 'val', 'test']:
        split_features = df_features[df_features['split'] == split]
        output_path = os.path.join(FEATURES_DIR, f'{split}_features.csv')
        split_features.to_csv(output_path, index=False)
        print(f"✓ Saved {split} features: {len(split_features)} samples")
    
    return df_features

def feature_selection(df_features):
    """Perform feature selection"""
    print("\n[FEATURE SELECTION]")
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
    
    # Prepare data
    train_features = df_features[df_features['split'] == 'train']
    feature_cols = [col for col in train_features.columns 
                   if col not in ['filename', 'split', 'category', 'label']]
    
    X = train_features[feature_cols].fillna(0)
    y = train_features['label']
    
    # Random Forest feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Save feature importance
    importance_path = os.path.join(FEATURES_DIR, 'feature_importance.csv')
    feature_importance.to_csv(importance_path, index=False)
    
    # Plot top features
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(20)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance Score')
    plt.title('Top 20 Most Important Features\nNephroLens - Powered by Sanket Kelzarkar', 
              fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FEATURES_DIR, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Feature importance saved")
    print(f"\nTop 10 Features:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Select top K features
    top_k = 50
    selected_features = feature_importance.head(top_k)['feature'].tolist()
    
    # Save selected features
    joblib.dump(selected_features, os.path.join(FEATURES_DIR, 'selected_features.pkl'))
    print(f"\n✓ Selected top {top_k} features for model training")
    
    return selected_features

if __name__ == "__main__":
    # Extract features
    df_features = extract_features_from_dataset()
    
    # Feature selection
    selected_features = feature_selection(df_features)
    
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING COMPLETED!")
    print(f"Total features extracted: {len(df_features.columns) - 4}")
    print(f"Selected features: {len(selected_features)}")
    print("Powered by Sanket Kelzarkar")
    print("=" * 60)
    print("\n✓ Ready for next step: Data Splitting")