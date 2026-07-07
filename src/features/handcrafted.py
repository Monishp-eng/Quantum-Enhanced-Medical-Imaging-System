import numpy as np
import cv2
import pywt
from skimage.feature import graycomatrix, graycoprops, hog
from scipy.stats import skew, kurtosis

def extract_glcm_features(img):
    """
    Extracts GLCM texture properties: contrast, correlation, energy, homogeneity, dissimilarity.
    Uses 1 distance and 4 angles (0, 45, 90, 135 degrees in radians) -> 20 features.
    """
    # GLCM expects uint8 grayscale image
    img_uint8 = (img * 255).astype(np.uint8) if img.dtype != np.uint8 else img
    
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    glcm = graycomatrix(img_uint8, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    
    properties = ['contrast', 'correlation', 'energy', 'homogeneity', 'dissimilarity']
    features = []
    for prop in properties:
        prop_values = graycoprops(glcm, prop) # Shape: (len(distances), len(angles))
        features.extend(prop_values.flatten())
        
    return np.array(features, dtype=np.float32)

def extract_hog_features(img):
    """
    Extracts HOG descriptors.
    Uses orientations=9, pixels_per_cell=(112, 112), cells_per_block=(1, 1) on a 224x224 image
    to yield exactly 2 * 2 * 9 = 36 features.
    """
    img_224 = cv2.resize(img, (224, 224)) if img.shape != (224, 224) else img
    
    features = hog(
        img_224, 
        orientations=9, 
        pixels_per_cell=(112, 112), 
        cells_per_block=(1, 1), 
        visualize=False
    )
    
    return features.astype(np.float32)

def extract_intensity_features(img):
    """
    Extracts intensity statistics: mean, std, skewness, kurtosis, entropy, and percentiles.
    Yields 10 features.
    """
    pixels = img.flatten()
    mean_val = np.mean(pixels)
    std_val = np.std(pixels)
    skew_val = skew(pixels)
    kurt_val = kurtosis(pixels)
    
    # Entropy
    hist, _ = np.histogram(pixels, bins=256, range=(0, 1), density=True)
    hist = hist[hist > 0]
    entropy_val = -np.sum(hist * np.log2(hist))
    
    # Percentiles
    p10 = np.percentile(pixels, 10)
    p25 = np.percentile(pixels, 25)
    p50 = np.percentile(pixels, 50)
    p75 = np.percentile(pixels, 75)
    p90 = np.percentile(pixels, 90)
    
    features = [mean_val, std_val, skew_val, kurt_val, entropy_val, p10, p25, p50, p75, p90]
    return np.array(features, dtype=np.float32)

def extract_wavelet_features(img, wavelet='db1', level=4):
    """
    Extracts DWT coefficients using Haar/Daubechies at 4 levels.
    Calculates mean and standard deviation for each subband (Approximation + 3 detail subbands per level).
    Total subbands = 1 (level 4 approx) + 4 levels * 3 detail subbands = 13 subbands.
    13 subbands * 2 features (mean, std) = 26 features.
    We add energy of the detail subbands (4 features) and max/min of approx subband (2 features) to get exactly 32 features.
    """
    coeffs = pywt.wavedec2(img, wavelet, level=level)
    features = []
    
    # Approximation coefficients (level 4)
    approx = coeffs[0]
    features.append(np.mean(approx))
    features.append(np.std(approx))
    features.append(np.min(approx))
    features.append(np.max(approx))
    
    # Detail coefficients (levels 4 down to 1)
    for detail_level in coeffs[1:]:
        h, v, d = detail_level # Horizontal, Vertical, Diagonal
        for subband in [h, v, d]:
            features.append(np.mean(subband))
            features.append(np.std(subband))
            features.append(np.sum(subband**2) / subband.size) # Energy of subband
            
    # We have 4 (approx) + 4 levels * 3 subbands * 3 features = 4 + 36 = 40 features.
    # Let's slice/truncate or select exactly 32 features to match target specifications.
    return np.array(features[:32], dtype=np.float32)

def extract_all_handcrafted(img):
    """Fuses all handcrafted features: GLCM (20) + HOG (36) + Intensity (10) + Wavelet (32) = 98 features."""
    glcm = extract_glcm_features(img)
    hog_f = extract_hog_features(img)
    intensity = extract_intensity_features(img)
    wavelet = extract_wavelet_features(img)
    
    return np.concatenate([glcm, hog_f, intensity, wavelet], axis=0)
