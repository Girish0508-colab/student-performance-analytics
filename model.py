import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pickle

# Load dataset
data = pd.read_csv("dataset.csv")

# Features and target
X = data[["study_hours", "attendance", "internal_marks"]]
y = data["final_score"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
score = r2_score(y_test, predictions)

print("Model R2 Score:", score)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("Model trained and saved successfully!")
