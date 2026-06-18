from flask import Flask, request, jsonify, render_template
import numpy as np
import tflite_runtime.interpreter as tflite
import joblib
import os

app = Flask(__name__)

# ==========================================
# 1. LOAD AI ASSETS (TFLite Version)
# ==========================================
try:
    print("Loading StandardScaler...")
    scaler = joblib.load('mobile_price_scaler.pkl')
    
    print("Loading TFLite Model...")
    # Load the compressed model
    interpreter = tflite.Interpreter(model_path="mobile_price_model.tflite")
    interpreter.allocate_tensors()
    
    # Get the specific memory addresses for the input/output channels
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("All AI Assets Loaded Successfully!")
except Exception as e:
    print(f"CRITICAL ERROR LOADING ASSETS: {e}")

# ==========================================
# 2. FLASK ROUTES
# ==========================================

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # TFLite is strict about data types, so we explicitly force it to float32
        features = np.array([data['features']], dtype=np.float32)
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Feed the scaled numbers into the model's input channel
        interpreter.set_tensor(input_details[0]['index'], features_scaled.astype(np.float32))
        
        # Run the neural network
        interpreter.invoke()
        
        # Pull the 4 probabilities out of the model's output channel
        probabilities = interpreter.get_tensor(output_details[0]['index'])
        
        # Find the winning class (0, 1, 2, or 3)
        predicted_class = int(np.argmax(probabilities))
        
        return jsonify({
            'success': True,
            'predicted_price_range': predicted_class
        })

    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
