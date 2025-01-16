import joblib
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

try:
 
    model = joblib.load(r"D:\Placement Preparation\Self Online Internship\CustomerChurn\logreg_customer_churn_model.pkl")
    print("Model loaded successfully!")

    label_encoder = joblib.load(r"D:\Placement Preparation\Self Online Internship\CustomerChurn\label_encoder.pkl")
    print("Label encoder loaded successfully!")

    scaler = joblib.load(r"D:\Placement Preparation\Self Online Internship\CustomerChurn\scaler.pkl")
    print("Scaler loaded successfully!")
except Exception as e:
    print(f"Error loading files: {e}")

@app.route('/')
def home():
    return "Customer Churn Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        print("Received a request!")

        data = request.get_json()
        print(f"Input data: {data}")

        input_data = np.array([
            data['gender'],  
            data['SeniorCitizen'],  
            data['Partner'],  
            data['Dependents'], 
            data['tenure'],  
            data['PhoneService'], 
            data['MultipleLines'], 
            data['InternetService'],  
            data['OnlineSecurity'],  
            data['OnlineBackup'],  
            data['DeviceProtection'], 
            data['TechSupport'],  
            data['StreamingTV'], 
            data['StreamingMovies'], 
            data['Contract'],  
            data['PaperlessBilling'],  
            data['PaymentMethod'],  
            data['TotalCharges']  
        ]).reshape(1, -1)  
        
        print(f"Prepared input: {input_data}")

        input_data_scaled = scaler.transform(input_data)
        print(f"Scaled input: {input_data_scaled}")

      
        prediction = model.predict(input_data_scaled)
        print(f"Prediction: {prediction}")

    
        decoded_prediction = label_encoder.inverse_transform([prediction[0]])[0]
        print(f"Decoded prediction: {decoded_prediction}")

   
        return jsonify({'prediction': decoded_prediction})

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    print("Starting Flask app...")
    app.run(debug=True)
