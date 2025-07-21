# Car Price Predictor

This project is a machine learning web application that predicts the price of a car based on its features. It uses a trained regression model and provides a simple web interface for users to input car details and get price predictions.

## Features
- Predicts car prices using a trained machine learning model
- User-friendly web interface built with Flask
- Handles various car attributes for accurate predictions

## Project Structure
- `application.py` - Main Flask application
- `model_pipeline.pkl`, `LinearRegressionModel.pkl`, `encoders.pkl` - Trained model and encoders
- `templates/` - HTML templates
- `static/` - CSS and static files
- `*.csv` - Data files used for training and analysis

## Getting Started
1. Clone the repository
2. Install dependencies from `requirements.txt`
3. Run the Flask app:
   ```
   python application.py
   ```
4. Open your browser and go to `http://localhost:5000`

## Requirements
- Python 3.10+
- See `requirements.txt` for Python packages

## License
This project is for educational purposes.
