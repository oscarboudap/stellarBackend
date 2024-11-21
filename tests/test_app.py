import json
import pytest
from app import app


@pytest.fixture
def client():
    """Set up the test client."""
    with app.test_client() as client:
        yield client


def test_luminosity_calculation():
    """Test the luminosity calculation function."""
    radius = 1.0
    temperature = 6000  # Temperature in Kelvin
    expected_luminosity = 4 * 3.1416 * (radius**2) * 5.67e-8 * (temperature**4)

    actual_luminosity = app.calculate_luminosity(radius, temperature)
    assert actual_luminosity == pytest.approx(expected_luminosity, rel=1e-5)


def test_generate_light_curve():
    """Test the light curve generation function."""
    initial_brightness = -19.3
    peak_time = 2  # Time in years
    decay_rate = 0.1

    time, magnitude = app.generate_light_curve(initial_brightness, peak_time, decay_rate)

    assert len(time) == len(magnitude)
    assert time[0] == 0
    assert time[-1] == 10
    assert magnitude[0] == pytest.approx(initial_brightness + (peak_time - 0) / 4, rel=1e-1)


def test_predict_hr_position(client):
    """Test the '/predict_hr_position' endpoint."""
    # Input data
    data = {
        "mass": 1.0,
        "temperature": 6000,
        "luminosity": 1.0
    }

    # Make a POST request
    response = client.post('/predict_hr_position', json=data)

    # Check if the response is correct
    assert response.status_code == 200
    json_data = response.get_json()
    assert 'region' in json_data
    assert json_data['region'] in ['G-type (Yellow)', 'O-type (Violet)', 'A-type (White-Blue)', 'B-type (Blue)']


def test_simulate(client):
    """Test the '/simulate' endpoint."""
    data = {
        "mass": 1.0,
        "temperature": 6000,
        "luminosity": 1.0,
        "radius": 1.0,
        "initial_brightness": -19.3,
        "peak_time": 2,
        "decay_rate": 0.1
    }

    # Make a POST request
    response = client.post('/simulate', json=data)

    # Check if the response is correct
    assert response.status_code == 200
    json_data = response.get_json()
    assert 'state' in json_data
    assert 'light_curve' in json_data
    assert 'classification' in json_data


def test_expansion(client):
    """Test the '/expansion' endpoint."""
    data = {
        "redshift": 0.1
    }

    # Make a POST request
    response = client.post('/expansion', json=data)

    # Check if the response is correct
    assert response.status_code == 200
    json_data = response.get_json()
    assert 'redshift' in json_data
    assert json_data['redshift'] == 0.1
    assert 'distance' in json_data


if __name__ == '__main__':
    pytest.main()
