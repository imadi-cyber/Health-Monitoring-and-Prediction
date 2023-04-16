import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

app = Flask(__name__)
model = pickle.load(open('LR.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('stroke.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    non_binary = ['age', 'avg_glucose_level', 'bmi']
    data = pd.DataFrame({"age":[int_features[0]], "avg_glucose_level":[int_features[1]], "bmi":[int_features[2]], "Sex":[int_features[3]], "Married":[int_features[4]], "Employement":[int_features[5]], "Residency":[int_features[6]], "Smoker":[int_features[7]]})
    data[non_binary] = scaler.fit_transform(data[non_binary])
    prediction = model.predict(data) 

    output = int(prediction[0])
    if output == 0:
        return render_template('stroke.html', prediction_text='No need to go to the doctor')
    else:
        return render_template('stroke.html', prediction_text='You should visit the doctor')

    
@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True,port=8080,use_reloader=False)
