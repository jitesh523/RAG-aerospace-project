from fastapi.testclient import TestClient
from tests.conftest import reload_app_with_env


def test_metrics_available_when_public_true():
    appmod = reload_app_with_env(METRICS_PUBLIC="true")
    client = TestClient(appmod.app)
    resp = client.get("/metrics")
    assert resp.status_code == 200
    # basic scrape format contains HELP/TYPE lines
    assert b"HELP" in resp.content and b"TYPE" in resp.content
