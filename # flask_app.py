# flask_app.py
from flask import Flask, request, jsonify, render_template
from ml_core import ml_model
import json

app = Flask(__name__)

@app.route('/')
def index():
    """Render HTML interface"""
    return render_template('index.html', 
                         accuracy=f"{ml_model.accuracy*100:.2f}%")

@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction API endpoint"""
    try:
        data = request.get_json()
        
        # Validate input
        required_fields = ['total_accidents', 'non_fatal', 'injured', 'vehicles']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        # Make prediction
        result = ml_model.predict(
            data['total_accidents'],
            data['non_fatal'],
            data['injured'],
            data['vehicles']
        )
        
        return jsonify({
            'success': True,
            'prediction': result,
            'model_accuracy': ml_model.accuracy,
            'model_accuracy_percentage': f"{ml_model.accuracy*100:.2f}%"
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': ml_model.model is not None,
        'model_accuracy': ml_model.accuracy
    })

@app.route('/model-info')
def model_info():
    """Model information endpoint"""
    return jsonify({
        'model': 'RandomForestClassifier',
        'n_estimators': 100,
        'features': ['Total_Accidents', 'Non_Fatal_Accidents', 'Injured', 'Vehicles_Involved'],
        'target': 'Risk_Level (1=High, 0=Low)',
        'accuracy': ml_model.accuracy
    })

@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint"""
    try:
        data = request.get_json()
        if not isinstance(data, list):
            return jsonify({'error': 'Expected a list of inputs'}), 400
        
        results = []
        for item in data:
            result = ml_model.predict(
                item['total_accidents'],
                item['non_fatal'],
                item['injured'],
                item['vehicles']
            )
            results.append({
                'input': item,
                'prediction': result
            })
        
        return jsonify({
            'success': True,
            'predictions': results,
            'count': len(results)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)