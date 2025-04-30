import cv2
import numpy as np
import pandas as pd
from google.cloud import vision
from typing import List, Dict, Tuple
import os
from pathlib import Path
import json
from collections import defaultdict 
from fuzzywuzzy import fuzz

class WhiskyGogglesV2:
    def __init__(self, dataset_path: str):
        """Initialize with path to bottle dataset CSV."""
        # Load the full DataFrame to preserve all columns
        self.df = pd.read_csv(dataset_path)
        # Convert to dict format for processing
        self.dataset = self.df.to_dict('records')
        self.sift = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher()
        
        # Initialize Google Vision client with credentials
        try:
            # First try environment variable
            self.google_vision_client = vision.ImageAnnotatorClient()
        except Exception as e:
            # If environment variable fails, try local credentials
            try:
                local_credentials_path = os.path.join(os.path.dirname(__file__), 'google_credentials.json')
                if os.path.exists(local_credentials_path):
                    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = local_credentials_path
                    self.google_vision_client = vision.ImageAnnotatorClient()
                else:
                    raise Exception("No Google Cloud credentials found. Please set GOOGLE_APPLICATION_CREDENTIALS or place google_credentials.json in the project directory.")
            except Exception as inner_e:
                raise Exception(f"Failed to initialize Google Vision client. Error: {str(inner_e)}")
        
        self.text_index = self._build_text_index()
        
    def _load_dataset(self, dataset_path: str) -> List[Dict]:
        """Load dataset from CSV and convert to dictionary format."""
        df = pd.read_csv(dataset_path)
        return df.to_dict('records')
    
    def _build_text_index(self) -> Dict:
        """Build an inverted index for text matching."""
        index = defaultdict(list)
        for idx, entry in enumerate(self.dataset):
            # Index relevant text fields
            text_fields = [
                str(entry.get('name', '')),
                str(entry.get('brand', '')),
                str(entry.get('description', ''))
            ]
            for field in text_fields:
                words = set(field.lower().split())
                for word in words:
                    if len(word) > 2:  # Skip very short words
                        index[word].append(idx)
        return dict(index)

    def google_ocr_scan(self, image_path: str) -> str:
        """Perform OCR using Google Vision API."""
        with open(image_path, 'rb') as image_file:
            content = image_file.read()
        image = vision.Image(content=content)
        response = self.google_vision_client.text_detection(image=image)
        texts = response.text_annotations
        return texts[0].description if texts else ""

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

    def get_initial_matches(self, text: str, max_matches: int = 40) -> List[Dict]:
        """Get initial matches based on text similarity."""
        # Extract words from query text
        query_words = set(text.lower().split())
        
        # Get candidate indices from inverted index
        candidate_indices = set()
        for word in query_words:
            if len(word) > 2:
                candidate_indices.update(self.text_index.get(word, []))
        
        # Score candidates
        scored_candidates = []
        for idx in candidate_indices:
            entry = self.dataset[idx]
            reference_text = f"{entry.get('name', '')} {entry.get('brand', '')} {entry.get('description', '')}"
            score = fuzz.token_set_ratio(text, reference_text)
            scored_candidates.append((score, idx))
        
        # Sort and get top matches
        scored_candidates.sort(reverse=True)
        top_matches = []
        for score, idx in scored_candidates[:max_matches]:
            entry = self.dataset[idx].copy()
            entry['text_score'] = score / 100.0
            top_matches.append(entry)
        
        return top_matches

    def match_features(self, query_keypoints, query_descriptors, image_path: str) -> float:
        """Match SIFT features between query and reference image."""
        if not os.path.exists(image_path):
            return 0.0
        
        ref_img = cv2.imread(image_path)
        if ref_img is None:
            return 0.0
        
        ref_processed = self.preprocess_image(ref_img)
        ref_keypoints, ref_descriptors = self.extract_features(ref_processed)
        
        if query_descriptors is None or ref_descriptors is None:
            return 0.0
        
        try:
            matches = self.matcher.knnMatch(query_descriptors, ref_descriptors, k=2)
            good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
            return len(good_matches) / len(query_keypoints) if query_keypoints else 0
        except:
            return 0.0

    def identify_bottle(self, image_path: str) -> List[Dict]:
        """Main bottle identification function."""
        # Step 1: Initial OCR with Google Vision
        google_text = self.google_ocr_scan(image_path)
        
        # Step 2: Get initial matches based on text
        initial_matches = self.get_initial_matches(google_text)
        
        # Step 3: Load and preprocess query image
        query_img = cv2.imread(image_path)
        if query_img is None:
            raise ValueError("Failed to load query image")
        
        query_processed = self.preprocess_image(query_img)
        query_keypoints, query_descriptors = self.extract_features(query_processed)
        
        # Step 4: Final matching combining SIFT and Google OCR
        final_matches = []
        for match in initial_matches:
            # Get SIFT similarity
            sift_score = self.match_features(query_keypoints, query_descriptors, match.get('image_path', ''))
            
            # Combine scores - adjusted weights since EasyOCR is removed
            final_score = 0.5 * match['text_score'] + 0.5 * sift_score
            
            if final_score > 0.1:  # Minimum threshold
                # Get all data from the original DataFrame row
                result = match.copy()  # This contains all columns from the CSV
                result['confidence'] = final_score
                final_matches.append(result)
        
        # Return top 3 matches
        return sorted(final_matches, key=lambda x: x['confidence'], reverse=True)[:3]

def main():
    """Test the improved bottle identification system."""
    goggles = WhiskyGogglesV2('data.csv')
    image_path = 'image.jpg'
    results = goggles.identify_bottle(image_path)
    
    print("Top matches:")
    for result in results:
        # Print all columns from the CSV plus confidence
        for key, value in result.items():
            print(f"{key}: {value}")
        print("---")

if __name__ == "__main__":
    main() 