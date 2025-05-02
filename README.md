# Whiskey Goggles - Spirit Recognition System

A robust spirit recognition system that combines OCR and computer vision to identify bottles and extract pricing information.

## 🎯 Features

- **Dual Recognition System**: Combines Google Vision OCR and SIFT feature matching
- **Price Detection**: Automatically extracts prices from images
- **Web Interface**: Easy-to-use drag-and-drop or camera capture interface
- **Robust Error Handling**: Handles various edge cases and image quality issues

## 🔄 Recognition Flow

1. **Primary Path (OCR + SIFT)**
   - Extracts text using Google Vision OCR
   - Matches text against database
   - Confirms matches using SIFT visual features
   - Combines scores (50% text, 50% visual)

2. **Fallback Path (Pure SIFT)**
   - Activates when:
     - No text detected in image
     - Text too short (< 3 characters)
     - No matches found in database
   - Performs full visual comparison against database

## 🛡️ Edge Case Handling

### Image Quality
- **Blur Detection**: Laplacian variance check prevents blurry images
- **Size Validation**: Enforces minimum (50px) and maximum (4096px) dimensions
- **Lighting Enhancement**: Automatic adjustment for poor lighting conditions

### Data Issues
- **Missing URLs**: Skips invalid image URLs in database
- **Corrupted Images**: Gracefully handles PNG/JPG read errors

### Recognition Robustness
- **OCR Robustness**: Google Vision API handles partial, rotated, and distorted text
- **Confidence Scoring**: Minimum threshold (0.1) for match validity

## 🚀 Quick Start

See [setup.md](setup.md) for detailed installation and configuration instructions.

Quick run after setup:
```bash
python app.py
```
Then visit `http://localhost:5000` in your browser.

## 📊 Output

- Top 5 matches with confidence scores
- Extracted prices and product details
- Match confidence and text match percentages
- Results logged in `bottle_price_log.csv`

## 🔍 Technical Details

- **OCR**: Google Cloud Vision API
- **Visual Matching**: SIFT (Scale-Invariant Feature Transform)
- **Frontend**: Vanilla JS with modern CSS
- **Backend**: Flask with OpenCV

## 📝 Logging

- All matches logged to `bottle_price_log.csv`
- Reference images cached in `image_cache/`
- Temporary uploads in `uploads/`

## ⚠️ Requirements

- Python 3.7+
- Google Cloud Account
- OpenCV
- Flask
- Internet connection for OCR 