import cv2
import numpy as np
import os
import threading
import queue

# Try importing torch to check environment availability
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

def load_midas_with_timeout(timeout=5.0):
    """
    Attempts to load the MiDaS small model from PyTorch Hub with a strict timeout limit.
    This prevents the pipeline from hanging on slow network requests or interactive prompts.
    """
    q = queue.Queue()
    
    def target():
        try:
            import torch.hub
            # Bypass interactive git archive verification prompt
            trusted = getattr(torch.hub, "_TRUSTED_REPO_OWNERS", ())
            if "rwightman" not in trusted:
                torch.hub._TRUSTED_REPO_OWNERS = trusted + ("rwightman",)
            trusted = getattr(torch.hub, "_TRUSTED_REPO_OWNERS", ())
            if "intel-isl" not in trusted:
                torch.hub._TRUSTED_REPO_OWNERS = trusted + ("intel-isl",)
                
            model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            q.put((model, midas_transforms.small_transform))
        except Exception as e:
            q.put(e)
            
    t = threading.Thread(target=target)
    t.daemon = True
    t.start()
    
    try:
        res = q.get(timeout=timeout)
        if isinstance(res, Exception):
            raise res
        return res[0], res[1]
    except queue.Empty:
        print("[Depth Engine] PyTorch Hub load timed out. Falling back to heuristic depth estimator.")
        return None, None
    except Exception as e:
        print(f"[Depth Engine] Failed to load from hub: {e}. Falling back to heuristic depth estimator.")
        return None, None

class MonocularDepthEstimator:
    def __init__(self):
        self.device = "cpu"
        self.model = None
        self.transform = None
        self.model_loaded = False
        
        if TORCH_AVAILABLE:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
                
            print(f"[Depth Engine] Initializing depth mapping on device: {self.device}")
            
            # Load with a 5.0 seconds strict timeout fallback
            model, transform = load_midas_with_timeout(timeout=5.0)
            if model is not None:
                self.model = model
                self.transform = transform
                self.model.to(self.device)
                self.model.eval()
                self.model_loaded = True
                print("[Depth Engine] PyTorch MiDaS model ready.")
        else:
            print("[Depth Engine] PyTorch not detected. Using heuristic mapper.")

    def estimate_depth(self, frame):
        """
        Computes a normalized depth map (Z-channel).
        Returns:
            depth_map: np.ndarray (H, W), values range from 0.0 (near) to 1.0 (far).
        """
        h, w = frame.shape[:2]
        
        if self.model_loaded:
            try:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                input_batch = self.transform(img).to(self.device)
                
                with torch.no_grad():
                    prediction = self.model(input_batch)
                    prediction = torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size=img.shape[:2],
                        mode="bicubic",
                        align_corners=False,
                    ).squeeze()
                    
                depth_map = prediction.cpu().numpy()
                d_min = depth_map.min()
                d_max = depth_map.max()
                
                if d_max - d_min > 0:
                    norm_depth = (depth_map - d_min) / (d_max - d_min)
                    # Convert to Z-depth (0.0=near, 1.0=far)
                    return 1.0 - norm_depth
                else:
                    return np.zeros_like(depth_map, dtype=np.float32)
            except Exception as e:
                print(f"[Depth Engine] PyTorch inference error ({e}). Using heuristic fallback.")
                
        # Heuristic fallback: lower region of screen is closer, upper region is farther
        y_indices, x_indices = np.indices((h, w), dtype=np.float32)
        gradient_depth = 1.0 - (y_indices / h)
        
        # Add visual texture features
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Sobel(gray, cv2.CV_32F, 1, 1, ksize=3)
        edges = np.abs(edges)
        cv2.GaussianBlur(edges, (21, 21), 0, dst=edges)
        
        e_min, e_max = edges.min(), edges.max()
        if e_max - e_min > 0:
            edges = (edges - e_min) / (e_max - e_min)
        else:
            edges = np.zeros_like(edges)
            
        combined = gradient_depth - 0.2 * edges
        return np.clip(combined, 0.0, 1.0)

def get_box_depth(depth_map, box):
    """
    Retrieves the mean depth value inside the object's bounding box area.
    """
    h, w = depth_map.shape[:2]
    x1, y1, x2, y2 = map(int, [max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])])
    if x2 <= x1 or y2 <= y1:
        return 0.5
    crop = depth_map[y1:y2, x1:x2]
    return float(np.mean(crop))
