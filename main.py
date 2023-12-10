import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# Load the dataset
data = pd.read_csv('Autralian_Vehicle_Prices.csv')

# Filter data for Toyota brand
toyota_data = data[data['Brand'] == 'Toyota']

# Select relevant features
features = ['Year', 'Kilometres', 'Transmission', 'Engine', 'DriveType', 'FuelType', 'FuelConsumption', 'CylindersinEngine', 'BodyType', 'Doors', 'Seats']
target = 'Price'

# Handle categorical variables using one-hot encoding
toyota_data = pd.get_dummies(toyota_data, columns=['Transmission', 'DriveType', 'FuelType', 'BodyType'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(toyota_data[features], toyota_data[target], test_size=0.2, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a model (Linear Regression in this example)
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Alternatively, you can use a more complex model like Random Forest
# model = RandomForestRegressor(n_estimators=100, random_state=42)
# model.fit(X_train_scaled, y_train)
# y_pred = model.predict(X_test_scaled)
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
