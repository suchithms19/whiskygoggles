# Whisky Goggles Setup Guide

## Prerequisites
- Python 3.7 or higher
- Google Cloud account

## Google Cloud Vision API Setup

1. Create a Google Cloud Project:
   - Go to [Google Cloud Console](https://console.cloud.google.com)
   - Click "New Project" and create one
   - Note down your Project ID

2. Enable the Vision API:
   - Search for "Cloud Vision API"
   - Click "Enable"

3. Create Service Account & Download Credentials:
   - Go to "APIs & Services" > "Credentials"
   - Click "Create Credentials" > "Service Account"
   - Fill in service account details and click "Create"
   - Under "Keys", click "Add Key" > "Create New Key"
   - Choose JSON format
   - Save the downloaded file as `google_credentials.json`

4. Place Credentials:
   - Move `google_credentials.json` to the project root directory
   - Don't Delete Data.csv

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Web Interface

1. Start the Flask server:
```bash
python app.py
```

2. Open browser and visit:
```
http://localhost:5000
```

### Option 2: Direct Image Processing
1. Place your bottle image as `image.jpg` in the project root directory
2. Run the command:
```bash
python whisky_goggles_v2.py
```

## Output
All results (from both web interface and direct processing) are logged in:
- `bottle_price_log.csv` (Contains date, bottle name, detected price, and shelf price)
- Images are cached in `image_cache` folder
- Temporary uploads go to `uploads` folder

## Note
The application requires Google Cloud Vision API credentials to function. Make sure you've completed the Google Cloud Vision API Setup section above. 