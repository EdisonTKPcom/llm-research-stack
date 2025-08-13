from fastapi.testclient import TestClient
from binance_llm.api.main import app

def test_health():
    c = TestClient(app)
    r = c.get('/health')
    assert r.status_code == 200
    assert r.json().get('status') == 'ok'
