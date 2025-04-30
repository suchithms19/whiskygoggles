import os
import requests
import hashlib
from pathlib import Path
from urllib.parse import urlparse
import pandas as pd

class ImageDownloader:
    def __init__(self, cache_dir: str = 'image_cache'):
        """Initialize with cache directory path."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def download_and_cache_image(self, image_url: str) -> str:
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
        except Exception:
            return "" 