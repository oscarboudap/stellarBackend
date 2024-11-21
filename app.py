from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import joblib

app = Flask(__name__)
CORS(app)

# Constants
SIGMA = 5.67e-8  # Stefan-Boltzmann constant (W/m^2/K^4)
HUBBLE_CONSTANT = 70  # Hubble's constant (km/s/Mpc)
C = 3e5  # Speed of light (km/s)

# Load the trained AI model
model = joblib.load("./data/star_classifier.pkl")


# Function to calculate luminosity
def calculate_luminosity(radius, temperature):
    return 4 * np.pi * (radius**2) * SIGMA * (temperature**4)


# Generate a light curve dynamically based on user inputs
def generate_light_curve(initial_brightness, peak_time, decay_rate):
    """
    Generate a light curve for a supernova Type Ia based on user parameters.

    Parameters:
    - initial_brightness: The initial magnitude (higher = dimmer).
    - peak_time: The time (in years) at which the brightness peaks.
    - decay_rate: The rate at which the magnitude decreases after the peak.

    Returns:
    - time: A list of time values (in years).
    - magnitude: A list of magnitude values at each time step.
    """
    time = np.linspace(0, 10, 500)  # Time in years

    # Phase 1: Rising phase (from start to peak_time)
    rise = initial_brightness + (peak_time - time[time <= peak_time]) / 4

    # Phase 2: Peak phase (constant brightness at the peak)
    peak_duration = 0.5  # Fixed duration of the peak in years
    peak = np.full(
        len(time[(time > peak_time) & (time <= peak_time + peak_duration)]),
        initial_brightness,
    )

    # Phase 3: Decay phase (after the peak phase)
    decay_start = peak_time + peak_duration
    decay = initial_brightness + decay_rate * (time[time > decay_start] - decay_start)

    # Combine all phases
    magnitude = np.concatenate([rise, peak, decay])

    return time.tolist(), magnitude.tolist()


# Predict the position of the star in the H-R diagram using AI
@app.route('/predict_hr_position', methods=['POST'])
def predict_hr_position():
    data = request.json
    mass = data.get('mass', 1.0)
    temperature = data.get('temperature', 5000)
    luminosity = data.get('luminosity', 1.0)

    prediction = model.predict([[mass, temperature, luminosity]])[0]
    return jsonify({"region": prediction})


# Simulate a star and return its light curve and classification
@app.route('/simulate', methods=['POST'])
def simulate():
    data = request.json
    mass = data.get('mass', 1.0)
    temperature = data.get('temperature', 5000)
    luminosity = data.get('luminosity', 1.0)
    radius = data.get('radius', 1.0)
    initial_brightness = data.get('initial_brightness', -19.3)
    peak_time = data.get('peak_time', 2)  # Years
    decay_rate = data.get('decay_rate', 0.1)  # Magnitude per year

    # Generate light curve based on parameters
    time, magnitude = generate_light_curve(initial_brightness, peak_time, decay_rate)

    # Classify lifecycle stage, luminosity class, and spectral type
    lifecycle_stage = classify_lifecycle_stage(mass)
    luminosity_class = classify_luminosity_temperature(luminosity, temperature)
    spectral_type = classify_spectral_type(temperature)

    return jsonify({
        "state": "exploding" if mass > 1.4 else "collapsing",
        "light_curve": {"time": time, "luminosity": magnitude},
        "classification": {
            "lifecycle_stage": lifecycle_stage,
            "luminosity_class": luminosity_class,
            "spectral_type": spectral_type
        }
    })


# Calculate universe expansion based on redshift
@app.route('/expansion', methods=['POST'])
def expansion():
    data = request.json
    redshift = data.get('redshift', 0.1)  # z
    distance = (C * redshift) / HUBBLE_CONSTANT  # Distance in Mpc

    return jsonify({"redshift": redshift, "distance": distance})


# Classify the lifecycle stage of the star
def classify_lifecycle_stage(mass):
    if mass < 0.5:
        return "Proto-Star"
    if mass < 1.4:
        return "Main Sequence Star"
    if 1.4 <= mass < 8:
        return "Red Giant or Supergiant"
    if mass >= 8:
        return "Final Stage (Neutron Star or Black Hole)"
    return "Unknown Stage"


# Classify the star based on luminosity and temperature
def classify_luminosity_temperature(luminosity, temperature):
    if luminosity < 0.1:
        return "White Dwarf"
    if luminosity < 1:
        return "Sub-Dwarf"
    if luminosity < 10:
        return "Main Sequence Star"
    if luminosity < 100:
        return "Giant Star"
    if luminosity < 1000:
        return "Supergiant Star"
    return "Hypergiant Star"


# Classify the star's spectral type based on its temperature
def classify_spectral_type(temperature):
    if temperature > 30000:
        return "O-type (Violet)"
    if temperature > 10000:
        return "B-type (Blue)"
    if temperature > 7500:
        return "A-type (White-Blue)"
    if temperature > 6000:
        return "F-type (Yellow-White)"
    if temperature > 5200:
        return "G-type (Yellow - Sun)"
    if temperature > 3700:
        return "K-type (Orange-Yellow)"
    return "M-type (Red)"


# Handle favicon.ico requests
@app.route('/favicon.ico')
def favicon():
    return '', 204  # Empty response with status code 204 (No Content)


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
