import cv2
import math
import json
import os
import numpy as np

REGISTRY_PATH = "reid_registry.json"

def extract_appearance_vector(crop_img):
    """
    Extracts a robust appearance vector using OpenCV:
    1. HSV color histogram (64 dims)
    2. Sobel edge texture histogram (8 dims)
    3. Spatial vertical crop HSV descriptors (3 regions x 16 dims = 48 dims)
    Conjoined into a normalized 120-dimensional feature vector.
    """
    if crop_img is None or crop_img.size == 0:
        return None
    try:
        # Resize to standard size to normalize scale effects
        resized = cv2.resize(crop_img, (64, 128))
        
        # 1. Global HSV Color Histogram (8x8 bins = 64 dims)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        color_hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
        cv2.normalize(color_hist, color_hist)
        color_feat = color_hist.flatten().tolist()
        
        # 2. Texture descriptor (Sobel magnitude histogram, 8 bins = 8 dims)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(sobel_x, sobel_y)
        magnitude_uint8 = np.uint8(np.clip(magnitude, 0, 255))
        tex_hist = cv2.calcHist([magnitude_uint8], [0], None, [8], [0, 256])
        cv2.normalize(tex_hist, tex_hist)
        tex_feat = tex_hist.flatten().tolist()
        
        # 3. Spatial Crop Descriptor (3 vertical zones, 4x4 HSV bins = 48 dims)
        h, w = resized.shape[:2]
        zone_h = h // 3
        spatial_feats = []
        for i in range(3):
            y_start = i * zone_h
            y_end = (i + 1) * zone_h if i < 2 else h
            zone = resized[y_start:y_end, :]
            zone_hsv = cv2.cvtColor(zone, cv2.COLOR_BGR2HSV)
            zone_hist = cv2.calcHist([zone_hsv], [0, 1], None, [4, 4], [0, 180, 0, 256])
            cv2.normalize(zone_hist, zone_hist)
            spatial_feats.extend(zone_hist.flatten().tolist())
            
        # Combine all features
        combined_feat = color_feat + tex_feat + spatial_feats
        
        # L2 Normalize the combined feature vector
        feat_arr = np.array(combined_feat, dtype=np.float32)
        norm = np.linalg.norm(feat_arr)
        if norm > 0:
            feat_arr = feat_arr / norm
            
        return feat_arr.tolist()
    except Exception as e:
        print(f"Feature extraction failed: {e}")
        return None

def cosine_similarity(v1, v2):
    """
    Calculates Cosine Similarity between two visual appearance vectors.
    """
    if not v1 or not v2:
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    norm_a = math.sqrt(sum(a * a for a in v1))
    norm_b = math.sqrt(sum(b * b for b in v2))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

def load_registry():
    if os.path.exists(REGISTRY_PATH):
        try:
            with open(REGISTRY_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_registry(registry):
    try:
        with open(REGISTRY_PATH, "w") as f:
            json.dump(registry, f, indent=4)
    except Exception as e:
        print(f"Error saving Re-ID registry: {e}")
