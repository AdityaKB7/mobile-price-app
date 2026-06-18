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
        
        # 1. Format input array as float32
        features = np.array([data['features']], dtype=np.float32)
        
        # 2. Scale features
        features_scaled = scaler.transform(features)
        
        # --- THE BATCH HACK STARTS HERE ---
        # 3. Create a fake batch of 32 phones (filled with zeros)
        batch_features = np.zeros((32, 20), dtype=np.float32)
        
        # 4. Put our REAL phone data into the very first row
        batch_features[0] = features_scaled[0]
        
        # 5. Push the batch of 32 into the model
        interpreter.set_tensor(input_details[0]['index'], batch_features)
        
        # 6. Run the inference
        interpreter.invoke()
        
        # 7. Pull all 32 prediction results
        all_probabilities = interpreter.get_tensor(output_details[0]['index'])
        
        # 8. Grab just the probabilities for our real phone (the 1st one)
        probabilities = all_probabilities[0]
        # --- THE BATCH HACK ENDS HERE ---
        
        # Find the winning class for our real phone
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
