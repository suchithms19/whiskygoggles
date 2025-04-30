from typing import List, Dict
from collections import defaultdict
from fuzzywuzzy import fuzz
from google.cloud import vision
import os

class TextProcessor:
    def __init__(self):
        """Initialize Google Vision client."""
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

    def build_text_index(self, dataset: List[Dict]) -> Dict:
        """Build an inverted index for text matching."""
        index = defaultdict(list)
        for idx, entry in enumerate(dataset):
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

    def get_initial_matches(self, text: str, dataset: List[Dict], text_index: Dict, max_matches: int = 10) -> List[Dict]:
        """Get initial matches based on text similarity."""
        # Extract words from query text
        query_words = set(text.lower().split())
        
        # Get candidate indices from inverted index
        candidate_indices = set()
        for word in query_words:
            if len(word) > 2:
                candidate_indices.update(text_index.get(word, []))
        
        # Score candidates
        scored_candidates = []
        for idx in candidate_indices:
            entry = dataset[idx]
            reference_text = f"{entry.get('name', '')} {entry.get('brand', '')} {entry.get('description', '')}"
            score = fuzz.token_set_ratio(text, reference_text)
            scored_candidates.append((score, idx))
        
        # Sort and get top matches
        scored_candidates.sort(reverse=True)
        top_matches = []
        for score, idx in scored_candidates[:max_matches]:
            entry = dataset[idx].copy()
            entry['text_score'] = score / 100.0
            top_matches.append(entry)
        
        return top_matches 