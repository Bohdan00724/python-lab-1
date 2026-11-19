import pytest
from fastapi.testclient import TestClient

from findex.index import Index
from findex.web.app import create_app, get_optional_index


def _build_in_memory_index() -> Index:
    idx = Index(store_positions=True)
    idx.add_document(0, ["python", "asyncio", "event", "loop", "fastapi"], path="async_guide.txt")
    idx.add_document(1, ["python", "pydantic", "validation", "schemas"], path="pydantic_doc.txt")
    idx.add_document(2, ["docker", "multi", "stage", "python", "deploy"], path="docker_doc.txt")
    return idx


def test_health_returns_503_when_index_not_loaded() -> None:
    app = create_app()
    app.dependency_overrides[get_optional_index] = lambda: None

    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "unavailable"
        assert data["index_loaded"] is False


def test_search_pagination_and_ranking_via_dependency_override() -> None:
    app = create_app()
    mem_index = _build_in_memory_index()
    app.dependency_overrides[get_optional_index] = lambda: mem_index

    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["num_docs"] == 3

        page1 = client.get("/search", params={"q": "python", "k": 2, "scorer": "bm25", "page": 1})
        assert page1.status_code == 200
        assert "X-Request-ID" in page1.headers
        body1 = page1.json()
        assert body1["total"] == 3
        assert body1["total_pages"] == 2
        assert body1["has_next"] is True
        assert body1["has_prev"] is False
        assert len(body1["results"]) == 2

        page2 = client.get("/search", params={"q": "python", "k": 2, "scorer": "tfidf", "page": 2})
        assert page2.status_code == 200
        body2 = page2.json()
        assert body2["page"] == 2
        assert body2["has_prev"] is True
        assert body2["has_next"] is False
        assert len(body2["results"]) == 1


@pytest.mark.parametrize(
    "params",
    [
        {"q": ""},
        {"q": "python", "k": 0},
        {"q": "python", "k": 100},
        {"q": "python", "page": 0},
        {"q": "python", "scorer": "invalid_scorer"},
        {"q": "python AND (unclosed"},
    ],
)
def test_search_validation_returns_422(params: dict[str, object]) -> None:
    app = create_app()
    mem_index = _build_in_memory_index()
    app.dependency_overrides[get_optional_index] = lambda: mem_index

    with TestClient(app) as client:
        resp = client.get("/search", params=params)
        assert resp.status_code == 422


def test_document_endpoint_200_and_404() -> None:
    app = create_app()
    mem_index = _build_in_memory_index()
    app.dependency_overrides[get_optional_index] = lambda: mem_index

    with TestClient(app) as client:
        ok_resp = client.get("/docs/1")
        assert ok_resp.status_code == 200
        doc_json = ok_resp.json()
        assert doc_json["doc_id"] == 1
        assert doc_json["path"] == "pydantic_doc.txt"
        assert "pydantic" in doc_json["content"]

        not_found = client.get("/docs/999")
        assert not_found.status_code == 404


def test_stats_and_html_ui_page() -> None:
    app = create_app()
    mem_index = _build_in_memory_index()
    app.dependency_overrides[get_optional_index] = lambda: mem_index

    with TestClient(app) as client:
        stats_resp = client.get("/stats")
        assert stats_resp.status_code == 200
        assert stats_resp.json()["num_docs"] == 3

        html_resp = client.get("/", params={"q": "python", "scorer": "bm25"})
        assert html_resp.status_code == 200
        assert "text/html" in html_resp.headers["content-type"]
        assert "async_guide.txt" in html_resp.text


def test_unexpected_exception_returns_500_with_request_id_without_traceback() -> None:
    app = create_app()

    @app.get("/boom")
    def boom_route() -> None:
        raise RuntimeError("Secret internal failure details")

    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.get("/boom", headers={"X-Request-ID": "req-test-500"})
        assert resp.status_code == 500
        payload = resp.json()
        assert payload["detail"] == "Internal Server Error"
        assert payload["request_id"] == "req-test-500"
        assert "RuntimeError" not in resp.text
        assert "Secret internal failure" not in resp.text
