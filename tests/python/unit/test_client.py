from omniff.client import AsyncClient, SyncClient


def test_async_client_init():
    client = AsyncClient("http://localhost:8000")
    assert client.base_url == "http://localhost:8000"


def test_async_client_trailing_slash():
    client = AsyncClient("http://localhost:8000/")
    assert client.base_url == "http://localhost:8000"


def test_sync_client_init():
    client = SyncClient()
    assert client.base_url == "http://localhost:8000"
