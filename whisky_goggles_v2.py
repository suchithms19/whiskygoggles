import cv2
import pandas as pd
from typing import List, Dict
import os

from image_processor import ImageProcessor
from image_downloader import ImageDownloader
from text_processor import TextProcessor

class WhiskyGogglesV2:
    def __init__(self, dataset_path: str):
        """Initialize with path to bottle dataset CSV."""
        # Load the full DataFrame to preserve all columns
        self.df = pd.read_csv(dataset_path)
        # Convert to dict format for processing
        self.dataset = self.df.to_dict('records')
        
        # Initialize components
        self.image_processor = ImageProcessor()
        self.image_downloader = ImageDownloader()
        self.text_processor = TextProcessor()
        
        # Build text index
        self.text_index = self.text_processor.build_text_index(self.dataset)
        
    def identify_bottle(self, image_path: str) -> List[Dict]:
        """Main bottle identification function."""
        print("\n[1/4] Starting bottle identification...")
        
        # Load query image first to validate it
        query_img = cv2.imread(image_path)
        if query_img is None:
            raise ValueError("Failed to load query image")
            
        # Check blur and validate size before any processing
        self.image_processor.check_blur(query_img)
        query_img = self.image_processor.validate_image_size(query_img)
        
        # Enchance lighting if needed
        query_img = self.image_processor.enhance_lighting(query_img)


        # Step 1: Initial OCR with Google Vision
        # Save the processed image temporarily for OCR
        temp_path = "temp_processed.jpg"
        cv2.imwrite(temp_path, query_img)
        google_text = self.text_processor.google_ocr_scan(temp_path)
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        
        # Now preprocess for feature extraction
        query_processed = self.image_processor.preprocess_image(query_img)
        query_keypoints, query_descriptors = self.image_processor.extract_features(query_processed)

        # If no text found or text is too short, fall back to pure visual matching
        if not google_text or len(google_text.strip()) < 3:
            print("[2/4] No text found, using visual matching...")
            matches = self._pure_visual_matching(query_keypoints, query_descriptors)
            self._print_results(matches)
            return matches
        
        # Step 2: Get initial matches based on text
        print("[2/4] Processing text matches...")
        initial_matches = self.text_processor.get_initial_matches(google_text, self.dataset, self.text_index)
        
        # If no text matches found, fall back to pure visual matching
        if not initial_matches:
            print("[3/4] No text matches found, using visual matching...")
            matches = self._pure_visual_matching(query_keypoints, query_descriptors)
            self._print_results(matches)
            return matches
        
        # Step 3: Final matching combining SIFT and Google OCR
        print("[3/4] Processing visual matches...")
        final_matches = []
        for match in initial_matches:
            # Download and load reference image
            image_path = self.image_downloader.download_and_cache_image(match.get('image_url', ''))
            if not image_path:
                continue
                
            ref_img = cv2.imread(image_path)
            if ref_img is None:
                continue
            
            # Get SIFT similarity
            sift_score = self.image_processor.match_features(query_keypoints, query_descriptors, ref_img)
            
            # Combine scores
            final_score = 0.5 * match['text_score'] + 0.5 * sift_score
            
            if final_score > 0.1:  # Minimum threshold
                result = match.copy()
                result['confidence'] = final_score
                final_matches.append(result)
        
        print("[4/4] Finalizing results...")
        matches = sorted(final_matches, key=lambda x: x['confidence'], reverse=True)[:5]
        self._print_results(matches)
        return matches

    def _pure_visual_matching(self, query_keypoints, query_descriptors) -> List[Dict]:
        """Perform pure visual matching against all images in dataset."""
        visual_matches = []
        
        # Filter out entries with invalid image URLs
        valid_entries = [entry for entry in self.dataset if not pd.isna(entry.get('image_url')) and entry.get('image_url')]
        total_images = len(valid_entries)
        
        print(f"[3/4] Processing {total_images} images...")
        
        for idx, entry in enumerate(valid_entries):
            try:
                if idx % 10 == 0:  # Show progress every 10 images
                    print(f"\rProgress: {idx}/{total_images} images processed", end="", flush=True)
                
                # Download and load reference image
                image_path = self.image_downloader.download_and_cache_image(entry.get('image_url', ''))
                if not image_path:
                    continue
                    
                ref_img = cv2.imread(image_path)
                if ref_img is None:
                    continue
                
                # Get SIFT similarity
                sift_score = self.image_processor.match_features(query_keypoints, query_descriptors, ref_img)
                
                if sift_score > 0.1:  # Minimum threshold for visual matching
                    result = entry.copy()
                    result['confidence'] = sift_score
                    visual_matches.append(result)
                    
            except Exception:
                continue
        
        print("\n[4/4] Finalizing results...")
        matches = sorted(visual_matches, key=lambda x: x['confidence'], reverse=True)[:5]
        self._print_results(matches)
        return matches

    def _print_results(self, results: List[Dict]) -> None:
        """Print formatted results."""
        print("\n=== MATCHES ===")
        if not results:
            print("No matches found.")
            return

        for idx, result in enumerate(results, 1):
            name = result.get('name', 'Unknown')
            confidence = result.get('confidence', 0)
            print(f"{idx}. {name:<40} (confidence: {confidence:.3f})")

def main():
    """Test the improved bottle identification system."""
    try:
        goggles = WhiskyGogglesV2('data.csv')
        image_path = 'image.jpg'
        goggles.identify_bottle(image_path)
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Please make sure both data.csv and image.jpg exist in the current directory.")

if __name__ == "__main__":
    main() 