from starlette.testclient import TestClient

from pi_agent_runtime.server import build_app


def test_agent_card_advertises_a2a_v1_streaming() -> None:
    response = TestClient(build_app()).get("/.well-known/agent-card.json")

    assert response.status_code == 200
    card = response.json()
    assert card["capabilities"]["streaming"] is True
    assert card["supportedInterfaces"] == [
        {
            "url": "http://0.0.0.0:8000/",
            "protocolBinding": "JSONRPC",
            "protocolVersion": "1.0",
        }
    ]
