from pathlib import Path
from handyllm import OpenAIClient


TEST_ROOT = Path(__file__).parent
ASSETS_ROOT = TEST_ROOT / "assets"


def test_client():
    client_key = 'client_key'
    client_base = 'https://api.example.com'
    client_org = 'client_org'
    client_api_version = '2024-01-01'
    client_api_type = 'azure'
    client_model_engine_map = {'model1': 'engine1', 'model2': 'engine2'}
    test_key_in_file1 = "test-key-in-file-1"
    test_key_in_file2 = "test-key-in-file-2"
    client = OpenAIClient(load_path=ASSETS_ROOT / "fake_credentials.yml")
    assert client.api_key == client_key
    assert client.api_base == client_base
    assert client.organization == client_org
    assert client.api_version == client_api_version
    assert client.api_type == client_api_type
    assert client.model_engine_map == client_model_engine_map
    assert client.endpoint_manager is not None
    assert client.endpoint_manager[0].api_key == test_key_in_file1
    assert client.endpoint_manager[1].api_key == test_key_in_file2
