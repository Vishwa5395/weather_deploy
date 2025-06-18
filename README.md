# Weather Forecasting

A Flask-powered web application that predicts the **any future day's average temperature** for a selected city using a trained machine learning model. It supports forecasts for **future dates**, leveraging historical weather data and engineered rolling features.

---

##  Features

-  Dropdown to select cities
-  Choose any future date
-  Predict average temperature (in °C)
-  Trained ML model using `scikit-learn` and joblib
-  Rolling average features for smarter prediction
-  Built with Flask + HTML + Bootstrap

---

##  Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/yourusername/weather-forecasting-app.git
cd weather-forecasting-app
2. Install Dependencies
bash

pip install -r requirements.txt


3. Add Weather Datasets
Place your city-wise weather CSV files inside the DATASETS/ directory. Each file should be named like cityname.csv and must include columns like DATE, TAVG, TMIN, TMAX.

4. Run the App

python app.py
Visit http://127.0.0.1:5000 in your browser.

How it Works
Weather data is loaded and cleaned (prepare_weather_data()).

Rolling means and percentage changes are computed for features.

A trained scikit-learn pipeline (Ridge + ColumnTransformer) is loaded from weather_model.pkl.

For each selected future date, the model simulates the next day's temperature based on the last known values and updates the rolling statistics.

Known Warnings
Ensure your scikit-learn version matches the one used to train weather_model.pkl. You may see warnings like:

InconsistentVersionWarning: Trying to unpickle estimator from version 1.7.0 using version 1.5.2
➤ Fix by upgrading: pip install scikit-learn==1.7.0

Acknowledgements
NOAA/NASA – Weather data

Flask

scikit-learn

Bootstrap – For responsive UI



Contributions
Pull requests are welcome! If you'd like to contribute:

Fork the repo

Make changes

