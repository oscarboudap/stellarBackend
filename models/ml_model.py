import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Simulated dataset for training
def generate_star_dataset():
    np.random.seed(42)
    num_samples = 1000

    mass = np.random.uniform(0.1, 50, num_samples)  # Solar masses
    temperature = np.random.uniform(2000, 40000, num_samples)  # Kelvin
    luminosity = np.random.uniform(-5, 6, num_samples)  # Absolute magnitude

    # Regions based on simplified classification rules
    labels = []
    for m, t, l in zip(mass, temperature, luminosity):
        if l > 0 and t < 5000:
            labels.append("giants")
        elif l < 5 and t < 10000:
            labels.append("main sequence")
        elif l < 0 and t > 20000:
            labels.append("white dwarfs")
        else:
            labels.append("unknown")

    return pd.DataFrame({"mass": mass, "temperature": temperature, "luminosity": luminosity, "label": labels})

# Train the model
def train_model():
    data = generate_star_dataset()
    X = data[["mass", "temperature", "luminosity"]]
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print(f"Model Accuracy: {model.score(X_test, y_test)}")

    # Save the model
    joblib.dump(model, "star_classifier.pkl")

# Train and save
train_model()
