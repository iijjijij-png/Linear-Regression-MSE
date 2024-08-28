import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Sample dataset (replace this with your actual dataset)
data = {
    'YearsExperience': [1.1, 2.0, 3.2, 4.0, 5.0, 6.0, 7.1, 8.2, 9.0, 10.0],
    'Salary': [39343, 46205, 60150, 64445, 66029, 83088, 91738, 101302, 105582, 121872]
}
df = pd.DataFrame(data)

# Features and target
X = df[['YearsExperience']]
y = df['Salary']

# Split the dataset into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model using Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

# Print the MSE
print(f"Mean Squared Error on the test set: {mse:.2f}")
