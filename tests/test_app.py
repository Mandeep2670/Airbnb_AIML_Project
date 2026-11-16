from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

sample_listing = {
    "neighbourhood_group": "Manhattan",
    "room_type": "Entire home/apt",
    "latitude": 40.758,
    "longitude": -73.9855,
    "minimum_nights": 2,
    "number_of_reviews": 25,
    "reviews_per_month": 1.5,
    "calculated_host_listings_count": 1,
    "availability_365": 180,
}


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_homepage():
    response = client.get("/")

    assert response.status_code == 200
    assert "Airbnb" in response.text


def test_valid_prediction():
    response = client.post("/predict", json=sample_listing)

    assert response.status_code == 200

    result = response.json()

    assert "predicted_price_usd" in result
    assert result["predicted_price_usd"] >= 0
    assert result["currency"] == "USD"


def test_invalid_availability():
    invalid_listing = sample_listing.copy()
    invalid_listing["availability_365"] = 400

    response = client.post("/predict", json=invalid_listing)

    assert response.status_code == 422


def test_invalid_room_type():
    invalid_listing = sample_listing.copy()
    invalid_listing["room_type"] = "Luxury Palace"

    response = client.post("/predict", json=invalid_listing)

    assert response.status_code == 422


def test_missing_required_feature():
    invalid_listing = sample_listing.copy()
    del invalid_listing["latitude"]

    response = client.post("/predict", json=invalid_listing)

    assert response.status_code == 422


def test_domain_logic():
    manhattan_listing = sample_listing.copy()

    brooklyn_listing = sample_listing.copy()
    brooklyn_listing["neighbourhood_group"] = "Brooklyn"

    response1 = client.post("/predict", json=manhattan_listing)

    response2 = client.post("/predict", json=brooklyn_listing)

    assert response1.status_code == 200
    assert response2.status_code == 200

    price1 = response1.json()["predicted_price_usd"]
    price2 = response2.json()["predicted_price_usd"]

    assert price1 >= 0
    assert price2 >= 0
    assert price1 != price2
