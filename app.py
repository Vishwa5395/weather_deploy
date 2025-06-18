from flask import Flask, render_template, request
import pandas as pd
import glob
import os
import joblib
from datetime import timedelta,date

app = Flask(__name__)
model = joblib.load('weather_model.pkl')
# print("Model loaded successfully:", model)

# Rebuild weather dataset (you might cache this in production)
def prepare_weather_data():
    all_data = []
    for file in glob.glob("DATASETS/*.csv"):
        city_name = os.path.basename(file).replace('.csv', '').strip().lower()
        df = pd.read_csv(file, index_col='DATE')
        df.columns = df.columns.str.lower()
        df['city'] = city_name
        all_data.append(df)

    weather = pd.concat(all_data)
    weather.index = pd.to_datetime(weather.index, format='mixed', errors='coerce')
    weather = weather.reset_index().rename(columns={'DATE': 'date'})
    weather.columns = weather.columns.str.lower()

    weather = weather.groupby('city', group_keys=False).apply(lambda group: group.ffill())
    null_pct = weather.isnull().mean()
    weather = weather.loc[:, null_pct < 0.3]
    weather['target'] = weather.groupby('city')['tavg'].shift(-1)
    weather = weather.ffill()

    def pct_diff(old, new):
        return (new - old) / old

    def compute_rolling(df, horizon, col):
        label = f"rolling_{horizon}_{col}"
        df[label] = df[col].rolling(horizon).mean()
        df[f"{label}_pct"] = pct_diff(df[col], df[label])
        return df

    for horizon in [3, 14]:
        for col in ['tavg', 'tmin', 'tmax']:
            if col in weather.columns:
                weather = weather.groupby('city', group_keys=False).apply(lambda df: compute_rolling(df, horizon, col))

    def expand_mean(df):
        return df.expanding(1).mean()

    for col in ['tavg', 'tmin', 'tmax']:
        if col in weather.columns:
            weather[f'month_avg_{col}'] = weather.groupby([weather['date'].dt.month, weather['city']])[col].transform(expand_mean)
            weather[f'day_avg_{col}'] = weather.groupby([weather['date'].dt.dayofyear, weather['city']])[col].transform(expand_mean)

    weather = weather.iloc[14:].copy().fillna(0)
    return weather

def fahrenheit_to_celsius(fahrenheit):
    return (fahrenheit - 32) * 5 / 9

weather_data = prepare_weather_data()
predictors = weather_data.columns.difference(['target', 'name', 'station', 'date'])

@app.route('/', methods=['GET', 'POST'])
def index():
    cities = sorted(set(weather_data['city'].str.lower()))
    prediction = None

    if request.method == 'POST':
        city = request.form['city'].strip().lower()
        date_str = request.form['date']
        try:
            forecast_date = pd.to_datetime(date_str)
        except ValueError:
            return render_template('index.html', cities=cities, prediction=prediction, today=date.today().isoformat())

        if city not in cities:
            return render_template('index.html', cities=cities, error="City not found.")

        city_data = weather_data[weather_data['city'] == city]
        latest_row = city_data.iloc[-1:].copy()
        current_date = latest_row['date'].values[0]

        while pd.to_datetime(current_date) < forecast_date:
            next_day = pd.to_datetime(current_date) + timedelta(days=1)
            latest_row['date'] = next_day

            for col in ['tavg', 'tmin', 'tmax']:
                if col in latest_row.columns:
                    for horizon in [3, 14]:
                        r_mean = latest_row[col].values[0]
                        latest_row[f'rolling_{horizon}_{col}'] = r_mean
                        latest_row[f'rolling_{horizon}_{col}_pct'] = 0
                    latest_row[f'month_avg_{col}'] = r_mean
                    latest_row[f'day_avg_{col}'] = r_mean

            try:
                predicted_temp = model.predict(latest_row[predictors])[0]
            except Exception as e:
                print("Prediction error:", e)
                return render_template('index.html', cities=cities, error="Model failed to make prediction.")


            for col in ['tavg', 'tmin', 'tmax']:
                if col in latest_row.columns:
                    latest_row[col] = predicted_temp

            current_date = next_day

        prediction = {
            'city': city.title(),
            'date': forecast_date.strftime('%B %d, %Y'),
            'temp_c': round(fahrenheit_to_celsius(predicted_temp), 2)
        }

    return render_template('index.html', cities=cities, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=False)
