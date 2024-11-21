import numpy as np
import pandas as pd

# Physical constants
SIGMA = 5.67e-8  # Stefan-Boltzmann constant in W/m^2/K^4
MAX_TIME = 100  # Maximum duration of the light curve in days

# Function to calculate luminosity
def calculate_luminosity(radius, temperature):
    return 4 * np.pi * (radius**2) * SIGMA * (temperature**4)

# Generate simulated star data
def generate_star_data(num_samples=1000):
    data = []

    for _ in range(num_samples):
        # Random parameters for each star
        mass = np.random.uniform(0.8, 2.0)  # Mass in solar masses
        temperature = np.random.uniform(3000, 10000)  # Temperature in Kelvin
        luminosity = np.random.uniform(0.5, 10)  # Luminosity in solar luminosities

        # Determine state based on the Chandrasekhar limit
        state = "exploding" if mass > 1.4 else "collapsing"

        # Evolution of radius and temperature
        time = np.linspace(0, MAX_TIME, 100)  # Time in days
        radius = mass * (1 - time / 200)  # Radius evolves linearly
        temperature_curve = temperature * (1 - time / 300)

        # Light curve
        luminosity_curve = calculate_luminosity(radius, temperature_curve)

        # Save data
        data.append({
            "mass": mass,
            "temperature": temperature,
            "luminosity": luminosity,
            "state": state,
            "curve": luminosity_curve.tolist()  # Save the curve as a list
        })

    return pd.DataFrame(data)

# Generate the dataset
dataset = generate_star_data(num_samples=1000)

# Save as CSV
dataset.to_csv("star_data.csv", index=False)

print("Dataset saved as star_data.csv")
