from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

# Load model pipeline
pipe = pickle.load(open('model_pipeline.pkl', 'rb'))

# Load CSV for dropdown values
car = pd.read_csv('Car_details_separated.csv')
car.dropna(subset=['name', 'company', 'fuel_type', 'year'], inplace=True)
car['name'] = car['name'].astype(str).str.strip()
car['company'] = car['company'].astype(str).str.strip()
car['fuel_type'] = car['fuel_type'].astype(str).str.strip()

@app.route('/')
def index():
    models_by_company = {}

    for company in car['company'].unique():
        models = car[car['company'] == company]['name'].unique()
        models_by_company[company] = sorted(models)

    return render_template('index.html',
                           companies=sorted(car['company'].unique()),
                           models_by_company=models_by_company,
                           years=sorted(car['year'].unique(), reverse=True),
                           fuel_types=sorted(car['fuel_type'].unique()))
@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    try:
        company = request.form.get('company')
        car_model = request.form.get('car_models')
        year = int(request.form.get('year'))
        fuel_type = request.form.get('fuel_type')
        driven = int(request.form.get('kilo_driven'))

        input_df = pd.DataFrame([[car_model, company, year, driven, fuel_type]],
                                columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])

        prediction = pipe.predict(input_df)[0]
        return str(np.round(prediction, 2))

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
