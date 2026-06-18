from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import joblib
import os

# Initialize the Flask application
app = Flask(__name__)

# ==========================================
# 1. LOAD AI ASSETS ON STARTUP
# ==========================================
# We load these outside of the routes so they only load ONCE when the server boots.
# If we put this inside the predict route, the server would lag trying to reload 
# the massive TensorFlow model every time a user clicked the button!

try:
    print("Loading StandardScaler...")
    scaler = joblib.load('mobile_price_scaler.pkl')
    
    print("Loading Neural Network...")
    ann = tf.keras.models.load_model('mobile_price_model.keras')
    
    print("All AI Assets Loaded Successfully!")
except Exception as e:
    print(f"CRITICAL ERROR LOADING ASSETS: {e}")

# ==========================================
# 2. FLASK ROUTES (The Web Endpoints)
# ==========================================

@app.route('/')
def home():
    """
    When the user visits the main URL (e.g., your-app.onrender.com),
    this function looks inside the 'templates' folder and serves the UI.
    """
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    This is the hidden API endpoint. The JavaScript in your index.html
    silently sends the 20 numbers here when the user clicks 'INITIALIZE NETWORK'.
    """
    try:
        # 1. Catch the data sent from the frontend
        data = request.get_json()
        
        # 2. Extract the 'features' array (the 20 numbers)
        # We wrap it in an extra set of brackets [] so NumPy knows it's 1 row of 20 columns,
        # rather than just a flat list of 20 numbers.
        features = np.array([data['features']])
        
        # 3. Scale the data (Crucial! The model will fail if we skip this)
        features_scaled = scaler.transform(features)
        
        # 4. Ask the Neural Network for a prediction
        # This outputs 4 probabilities (e.g., [0.01, 0.05, 0.14, 0.80])
        probabilities = ann.predict(features_scaled)
        
        # 5. Find the winner (argmax finds the index of the highest probability)
        predicted_class = int(np.argmax(probabilities))
        
        # 6. Send the winning number (0, 1, 2, or 3) back to the frontend
        return jsonify({
            'success': True,
            'predicted_price_range': predicted_class
        })

    except Exception as e:
        # If anything goes wrong (like a missing number), send an error safely
        print(f"Prediction Error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

# ==========================================
# 3. SERVER RUNNER
# ==========================================
if __name__ == '__main__':
    # Render uses Gunicorn to run the app, so it will actually ignore this block.
    # But this allows you to test the app locally on your own computer first!
    # Set port dynamically for cloud environments
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)