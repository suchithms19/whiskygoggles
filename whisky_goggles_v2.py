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
import requests
import hashlib
from urllib.parse import urlparse

class WhiskyGogglesV2:
    def __init__(self, dataset_path: str):
        """Initialize with path to bottle dataset CSV."""
        # Load the full DataFrame to preserve all columns
        self.df = pd.read_csv(dataset_path)
        # Convert to dict format for processing
        self.dataset = self.df.to_dict('records')
        self.sift = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher()
        
        # Create cache directory for downloaded images
        self.cache_dir = Path('image_cache')
        self.cache_dir.mkdir(exist_ok=True)
        
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

    def get_initial_matches(self, text: str, max_matches: int = 10) -> List[Dict]:
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

    def _download_and_cache_image(self, image_url: str) -> str:
        """Download and cache an image from URL, return path to cached file."""
        # Handle nan, None, or empty URLs
        if pd.isna(image_url) or not image_url or image_url == 'nan':
            print(f"Invalid image URL: {image_url}")
            return ""
            
        # Create filename from URL hash
        url_hash = hashlib.md5(image_url.encode()).hexdigest()
        file_ext = os.path.splitext(urlparse(image_url).path)[1] or '.jpg'
        cached_path = self.cache_dir / f"{url_hash}{file_ext}"
        
        # If already cached, return cached path
        if cached_path.exists():
            return str(cached_path)
            
        try:
            # Download image
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            
            # Save to cache
            with open(cached_path, 'wb') as f:
                f.write(response.content)
            return str(cached_path)
        except Exception as e:
            print(f"Failed to download image from {image_url}: {str(e)}")
            return ""

    def match_features(self, query_keypoints, query_descriptors, image_url: str) -> float:
        """Match SIFT features between query and reference image."""
        print(f"Starting SIFT matching for image: {image_url}")
        
        if not image_url:
            print("No image URL provided")
            return 0.0
        
        # Download and cache the reference image
        image_path = self._download_and_cache_image(image_url)
        if not image_path:
            print(f"Failed to download/cache image from {image_url}")
            return 0.0
        
        ref_img = cv2.imread(image_path)
        if ref_img is None:
            print(f"Failed to load reference image from {image_path}")
            return 0.0
        
        print(f"Processing reference image: {image_path}")
        ref_processed = self.preprocess_image(ref_img)
        ref_keypoints, ref_descriptors = self.extract_features(ref_processed)
        
        if query_descriptors is None or ref_descriptors is None:
            print("No descriptors found for query or reference image")
            return 0.0
        
        try:
            print(f"Found {len(query_keypoints)} query keypoints and {len(ref_keypoints)} reference keypoints")
            matches = self.matcher.knnMatch(query_descriptors, ref_descriptors, k=2)
            good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
            match_score = len(good_matches) / len(query_keypoints) if query_keypoints else 0
            print(f"Found {len(good_matches)} good matches out of {len(matches)} total matches. Score: {match_score:.3f}")
            return match_score
        except Exception as e:
            print(f"Error during feature matching: {str(e)}")
            return 0.0

    def identify_bottle(self, image_path: str) -> List[Dict]:
        """Main bottle identification function."""
        print("\nStarting bottle identification process...")
        
        # Step 1: Initial OCR with Google Vision
        print("Performing Google Vision OCR...")
        google_text = self.google_ocr_scan(image_path)
        print(f"Extracted text: {google_text[:100]}...")
        
        # Load and preprocess query image first as it's needed in both paths
        print("Loading and preprocessing query image...")
        query_img = cv2.imread(image_path)
        if query_img is None:
            raise ValueError("Failed to load query image")
        
        query_processed = self.preprocess_image(query_img)
        query_keypoints, query_descriptors = self.extract_features(query_processed)
        print(f"Extracted {len(query_keypoints)} keypoints from query image")

        # If no text found or text is too short, fall back to pure visual matching
        if not google_text or len(google_text.strip()) < 3:
            print("\nNo significant text found in image. Falling back to pure visual matching...")
            return self._pure_visual_matching(query_keypoints, query_descriptors)
        
        # Step 2: Get initial matches based on text
        print("Getting initial matches based on text...")
        initial_matches = self.get_initial_matches(google_text)
        print(f"Found {len(initial_matches)} initial matches")
        
        # If no text matches found, fall back to pure visual matching
        if not initial_matches:
            print("\nNo text matches found. Falling back to pure visual matching...")
            return self._pure_visual_matching(query_keypoints, query_descriptors)
        
        # Step 4: Final matching combining SIFT and Google OCR
        print("\nPerforming final matching...")
        final_matches = []
        for match in initial_matches:
            print(f"\nProcessing match: {match.get('name', 'Unknown')}")
            # Get SIFT similarity
            sift_score = self.match_features(query_keypoints, query_descriptors, match.get('image_url', ''))
            
            # Combine scores - adjusted weights since EasyOCR is removed
            final_score = 0.5 * match['text_score'] + 0.5 * sift_score
            print(f"Final score: {final_score:.3f} (Text: {match['text_score']:.3f}, SIFT: {sift_score:.3f})")
            
            if final_score > 0.1:  # Minimum threshold
                # Get all data from the original DataFrame row
                result = match.copy()  # This contains all columns from the CSV
                result['confidence'] = final_score
                final_matches.append(result)
        
        print(f"\nFound {len(final_matches)} matches above threshold")
        # Return top 5 matches
        return sorted(final_matches, key=lambda x: x['confidence'], reverse=True)[:5]

    def _pure_visual_matching(self, query_keypoints, query_descriptors) -> List[Dict]:
        """Perform pure visual matching against all images in dataset."""
        print("\nStarting pure visual matching against all images...")
        visual_matches = []
        
        # Filter out entries with invalid image URLs
        valid_entries = [entry for entry in self.dataset if not pd.isna(entry.get('image_url')) and entry.get('image_url')]
        total_images = len(valid_entries)
        
        print(f"Found {total_images} valid images out of {len(self.dataset)} total entries")
        
        for idx, entry in enumerate(valid_entries):
            try:
                print(f"\rProcessing image {idx + 1}/{total_images}", end="", flush=True)
                
                # Get SIFT similarity
                sift_score = self.match_features(query_keypoints, query_descriptors, entry.get('image_url', ''))
                
                if sift_score > 0.1:  # Minimum threshold for visual matching
                    result = entry.copy()
                    result['confidence'] = sift_score
                    visual_matches.append(result)
                    
            except Exception as e:
                print(f"\nError processing entry {idx}: {str(e)}")
                continue
        
        print(f"\nFound {len(visual_matches)} visual matches above threshold")
        # Return top 5 matches
        return sorted(visual_matches, key=lambda x: x['confidence'], reverse=True)[:5]

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