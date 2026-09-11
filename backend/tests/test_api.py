from fastapi.testclient import TestClient
from sign_avatar_api.app import app

client = TestClient(app)


def test_catalogue_contract():
    result = client.get("/api/signs").json()
    assert result["total"] == 49
    assert len({s["id"] for s in result["items"]}) == 49
    assert client.get("/api/signs?category=letter").json()["total"] == 26
    assert client.get("/api/signs?category=digit").json()["total"] == 10
    assert client.get("/api/signs?category=word").json()["total"] == 13


def test_label_identity_and_unknowns():
    assert client.get("/api/signs/alphabet_i").json()["category"] == "letter"
    assert client.get("/api/signs?category=unknown").status_code == 422
    assert client.get("/api/signs/missing").status_code == 404
    assert client.post("/api/recognise").status_code == 404


def test_health_does_not_claim_unimplemented_inference():
    data = client.get("/api/health").json()
    assert data["service"] == "sign-avatar"
    assert data["capabilities"]["recognition"] is False
