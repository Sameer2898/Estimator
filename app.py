'''
Created By:- Sameer Goel
Created On:- 12-04-2021
Language Used:- Python
IDE Used:- VS Code
Purpose:- To predict the price of the used car on the basis of some features
'''

from flask import Flask, render_template, request
import pandas as pd 
import numpy as np 
import pickle

app = Flask(__name__)
df = pd.read_csv('cleaned_data.csv')
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    companies = sorted(df['company'].unique())
    car_models = sorted(df['name'].unique())
    years = sorted(df['year'].unique(), reverse=True)
    fuel_types = sorted(df['fuel_type'].unique())

    companies.insert(0, 'Select Company')
    return render_template('index.html', companies=companies, car_models=car_models, years=years, fuel_types=fuel_types)

# Prediction Logic
@app.route('/predict', methods=['POST'])
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_models')
    fuel_type = request.form.get('fuel_type')
    year = request.form.get('year')
    kms_driven = request.form.get('kms_driven')
    prediction = model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'], data=np.array([car_model, company, year, kms_driven, fuel_type]).reshape(1, 5)))
    return str(np.round(prediction[0],2))

if __name__ == '__main__':
    app.run(debug=True)