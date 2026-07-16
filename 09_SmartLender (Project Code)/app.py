from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model  = joblib.load('model/best_model.pkl')
scaler = joblib.load('model/scaler.pkl')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form inputs
        gender         = int(request.form['gender'])        
        married        = int(request.form['married'])       
        dependents     = int(request.form['dependents'])   
        education      = int(request.form['education'])     
        self_employed  = int(request.form['self_employed']) 
        applicant_inc  = float(request.form['applicant_income'])
        coapplicant_inc= float(request.form['coapplicant_income'])
        loan_amount    = float(request.form['loan_amount'])
        loan_term      = float(request.form['loan_term'])
        credit_history = float(request.form['credit_history']) 
        property_area  = int(request.form['property_area'])  

        features = np.array([[gender, married, dependents, education,
                              self_employed, applicant_inc, coapplicant_inc,
                              loan_amount, loan_term, credit_history, property_area]])

        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]

        result = 'Approved ✅' if prediction == 1 else 'Rejected ❌'
        confidence = round(max(probability) * 100, 2)

        return render_template('result.html',
                               prediction=result,
                               confidence=confidence)
    except Exception as e:
        return render_template('result.html',
                               prediction=f'Error: {str(e)}',
                               confidence=0)

if __name__ == '__main__':
    app.run(debug=True)
    