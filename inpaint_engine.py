import cv2
import numpy as np
import os

class VFXPatchGenerator:
    def __init__(self):
        pass
        
    def generate_mask(self, image_shape, box, dilation_kernel_size=7):
        """
        Generates a binary ROI mask for the bounding box, dilated to smooth edges.
        image_shape: tuple representing (height, width, channels)
        box: bounding box coordinates [x1, y1, x2, y2]
        """
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Unpack box coordinates
        x1, y1, x2, y2 = map(int, [max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])])
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255
            
        # Apply binary dilation to smooth boundary edges
        if dilation_kernel_size > 0:
            kernel = np.ones((dilation_kernel_size, dilation_kernel_size), dtype=np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
            
        return mask

    def inpaint_opencv(self, image, mask, method="telea"):
        """
        Applies OpenCV local inpainting algorithms.
        method: "telea" (INPAINT_TELEA) or "ns" (INPAINT_NS)
        """
        flag = cv2.INPAINT_TELEA if method.lower() == "telea" else cv2.INPAINT_NS
        # Inpaint with an inpainting neighborhood radius of 3px
        return cv2.inpaint(image, mask, 3, flag)

    def inpaint_diffusion(self, image, mask, prompt="clean backplate, empty room, background"):
        """
        Extensible hook for external generative inpainting model endpoints (e.g. Stable Diffusion, Flux).
        Falls back to local OpenCV inpainting as default baseline.
        """
        print("[VFX Fixes] External diffusion hook invoked. Falling back to OpenCV Telea baseline.")
        return self.inpaint_opencv(image, mask, method="telea")

    def generate_patch(self, image, box, method="telea"):
        """
        Main runner creating a clean patched frame.
        """
        mask = self.generate_mask(image.shape, box)
        return self.inpaint_opencv(image, mask, method=method)
