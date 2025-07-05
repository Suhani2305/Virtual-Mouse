import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load calibration data
csv_file = 'calib_data.csv'
data = pd.read_csv(csv_file, header=None)

# Last two columns are screen_x, screen_y; rest are eye landmarks
X = data.iloc[:, :-2].values  # eye landmarks
Y = data.iloc[:, -2:].values  # screen_x, screen_y

# Train LinearRegression
reg_x_lin = LinearRegression().fit(X, Y[:, 0])
reg_y_lin = LinearRegression().fit(X, Y[:, 1])

# Train RandomForestRegressor
reg_x_rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, Y[:, 0])
reg_y_rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, Y[:, 1])

# Save both models
joblib.dump({
    'reg_x_lin': reg_x_lin,
    'reg_y_lin': reg_y_lin,
    'reg_x_rf': reg_x_rf,
    'reg_y_rf': reg_y_rf
}, 'eye_mouse_model.pkl')
print('Models (LinearRegression and RandomForest) trained and saved as eye_mouse_model.pkl') 