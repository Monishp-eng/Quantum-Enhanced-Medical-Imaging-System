import cv2
import numpy as np

def load_image(path, grayscale=True):
    """Loads an image either in grayscale or RGB."""
    if grayscale:
        return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return None

def skull_strip(img):
    """
    Simple contour-based skull stripping.
    Finds the largest contour (assuming it's the brain) and masks everything else.
    Works best on grayscale images.
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
        
    # Thresholding to find the main brain structure
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return img
        
    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Create mask and apply it
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    
    if len(img.shape) == 3:
        mask = cv2.merge([mask, mask, mask])
        
    stripped = cv2.bitwise_and(img, mask)
    return stripped

def resize_image(img, size=(224, 224)):
    """Resizes the image to the target size."""
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Applies CLAHE contrast enhancement."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    if len(img.shape) == 3:
        # Convert to LAB space to apply CLAHE to L channel
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    else:
        return clahe.apply(img)

def denoise_image(img, kernel_size=(3, 3)):
    """Applies Gaussian Blur for denoising."""
    return cv2.GaussianBlur(img, kernel_size, 0)

def normalize_image(img):
    """Normalizes pixel values to [0, 1]."""
    return img.astype(np.float32) / 255.0

def preprocess_image(path, size=(224, 224), grayscale=True, apply_skull_strip=True):
    """
    Executes the full preprocessing pipeline:
    Load -> Skull Strip -> Resize -> CLAHE -> Denoise -> Normalize
    """
    img = load_image(path, grayscale=grayscale)
    if img is None:
        raise ValueError(f"Could not load image at {path}")
        
    if apply_skull_strip:
        img = skull_strip(img)
        
    img = resize_image(img, size)
    img = apply_clahe(img)
    img = denoise_image(img)
    img = normalize_image(img)
    
    return img
