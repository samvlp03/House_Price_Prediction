import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

dataset = pd.read_csv('data.csv')

# Assuming 'price' is the target variable
y = dataset['price'].values  # Target variable

# Identify columns with non-numeric data
non_numeric_columns = dataset.select_dtypes(exclude=[np.number]).columns

# Exclude non-numeric columns from features
X = dataset.drop(['price', 'date'] + list(non_numeric_columns), axis=1)

# Convert 'date' to numerical representation (assuming it's a datetime column)
X['date'] = pd.to_datetime(dataset['date']).astype('int64') // 10**9  # Convert datetime to Unix timestamp

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train a simple neural network with TensorFlow/Keras
model_tf = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(1)
])

model_tf.compile(optimizer='adam', loss='mean_squared_error')

history = model_tf.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_data=(X_test_scaled, y_test), verbose=0)

# Make predictions on the test set
predictions_tf = model_tf.predict(X_test_scaled).flatten()

# Evaluate the model
mse_tf = mean_squared_error(y_test, predictions_tf)
r2_tf = r2_score(y_test, predictions_tf)

print('\nTensorFlow Model:')
print(f'Mean Squared Error (MSE): {mse_tf:.2f}')
print(f'R-squared (R2) Score: {r2_tf:.2f}')

# Visualize the predictions
plt.scatter(y_test, predictions_tf)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("House Price Prediction - TensorFlow Model")
plt.show()
