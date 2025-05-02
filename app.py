from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
from whisky_goggles_v2 import WhiskyGogglesV2
from flask_cors import CORS
import math

app = Flask(__name__, static_folder='frontend')
CORS(app)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Initialize WhiskyGoggles
whisky_goggles = WhiskyGogglesV2('data.csv')

@app.route('/')
def serve_frontend():
    return send_from_directory('frontend', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('frontend', path)

def clean_nan_values(obj):
    """Convert NaN values to None for JSON serialization"""
    if isinstance(obj, float) and math.isnan(obj):
        return None
    elif isinstance(obj, dict):
        return {k: clean_nan_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan_values(x) for x in obj]
    return obj

@app.route('/api/recognize', methods=['POST'])
def recognize_spirit():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        try:
            # Save the uploaded file
            file.save(filepath)
            
            # Process the image for bottle recognition
            results = whisky_goggles.identify_bottle(filepath)
            
            # Clean up the uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
            
            # Clean NaN values from results
            cleaned_results = clean_nan_values(results)
            
            # Format the response
            formatted_results = []
            for result in cleaned_results:
                formatted_result = {
                    'id': result.get('id', ''),
                    'name': result.get('name', ''),
                    'size': result.get('size', ''),
                    'proof': result.get('proof', None),
                    'abv': result.get('abv', None),
                    'spirit_type': result.get('spirit_type', ''),
                    'image_url': result.get('image_url', ''),
                    'avg_msrp': result.get('avg_msrp', None),
                    'fair_price': result.get('fair_price', None),
                    'shelf_price': result.get('shelf_price', None),
                    'total_score': result.get('total_score', None),
                    'wishlist_count': result.get('wishlist_count', 0),
                    'vote_count': result.get('vote_count', 0),
                    'bar_count': result.get('bar_count', 0),
                    'ranking': result.get('ranking', None),
                    'text_score': result.get('text_score', 0),
                    'confidence': round(result.get('confidence', 0) * 100, 2),
                    'extracted_price': result.get('extracted_price', None)
                }
                formatted_results.append(formatted_result)
            
            return jsonify({'results': formatted_results})
            
        except Exception as e:
            # Clean up the uploaded file in case of error
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 