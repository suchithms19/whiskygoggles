import cv2
import numpy as np
from typing import Tuple, List

class ImageProcessor:
    def __init__(self):
        """Initialize SIFT detector and matcher."""
        self.sift = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher()
        # Maximum dimensions for effective SIFT and OCR processing
        self.MAX_IMAGE_DIMENSION = 4096
        self.MIN_IMAGE_DIMENSION = 50

    def validate_image_size(self, img: np.ndarray) -> np.ndarray:
        """
        Validate and resize image if necessary for SIFT and OCR processing.
        Raises ValueError if image is too small or can't be processed.
        """
            
        height, width = img.shape[:2]
        
        # Check minimum size
        if width < self.MIN_IMAGE_DIMENSION or height < self.MIN_IMAGE_DIMENSION:
            raise ValueError(f"Image is too small. Minimum dimension is {self.MIN_IMAGE_DIMENSION}px")
            
        # If image is too large, resize while maintaining aspect ratio
        if width > self.MAX_IMAGE_DIMENSION or height > self.MAX_IMAGE_DIMENSION:
            scale = self.MAX_IMAGE_DIMENSION / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
        return img

    def preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """Preprocess image for feature detection."""
        if img is None:
            raise ValueError("Invalid image provided")
            
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
        # Denoise
        denoised = cv2.fastNlMeansDenoising(thresh)
        return denoised

    def extract_features(self, image: np.ndarray) -> Tuple[List, np.ndarray]:
        """Extract SIFT features from image."""
        keypoints, descriptors = self.sift.detectAndCompute(image, None)
        if descriptors is None:
            return [], None
        return keypoints, descriptors

    def match_features(self, query_keypoints, query_descriptors, ref_img: np.ndarray) -> float:
        """Match SIFT features between query and reference image."""
        if ref_img is None:
            print("Invalid reference image")
            return 0.0
        
        ref_processed = self.preprocess_image(ref_img)
        ref_keypoints, ref_descriptors = self.extract_features(ref_processed)
        
        if query_descriptors is None or ref_descriptors is None:
            print("No descriptors found for query or reference image")
            return 0.0
        
        try:
            matches = self.matcher.knnMatch(query_descriptors, ref_descriptors, k=2)
            good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
            return len(good_matches) / len(query_keypoints) if query_keypoints else 0
        except Exception:
            return 0.0

    def check_blur(self, img: np.ndarray) -> None:
        """
        Check if image is blurry using Laplacian variance.
        Raises ValueError if image is too blurry.
        """
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        variance = laplacian.var()
        
        if variance < 10:
            raise ValueError("Image is too blurry. Please upload again.") 