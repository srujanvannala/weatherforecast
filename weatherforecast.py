import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_excel("weather_dataset_20rows.xlsx")

# Features (independent variables)
X = df[["Temperature_C", "Humidity_%", "Wind_Speed_kmph", "Rainfall_mm"]]

# Target (dependent variable) - Next day's temperature
y = df["Next_Day_Temperature_C"]

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate performance
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Predict for a new example (today’s weather)
new_data = pd.DataFrame({
    "Temperature_C": [28],
    "Humidity_%": [70],
    "Wind_Speed_kmph": [12],
    "Rainfall_mm": [4]
})
forecast = model.predict(new_data)
print("Predicted Next Day Temperature:", round(forecast[0], 2), "°C")
