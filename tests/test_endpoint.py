from pathlib import Path
from handyllm import OpenAIClient, EndpointManager, Endpoint
import pytest


TEST_ROOT = Path(__file__).parent
ASSETS_ROOT = TEST_ROOT / "assets"

def test_endpoint_manager():
    test_key1 = "test-key-1"
    test_key2 = "test-key-2"
    test_key3 = "test-key-3"
    test_key4 = "test-key-4"
    test_key_in_file1 = "test-key-in-file-1"
    test_key_in_file2 = "test-key-in-file-2"
    tmp_key = "tmp-key"
    
    endpoint_manager = EndpointManager(
        endpoints=[
            { "api_key": test_key1 },
            Endpoint(api_key=test_key2)
        ], 
        load_path=ASSETS_ROOT / "fake_credentials.json"
    )
    endpoint_manager.add_endpoint_by_info(api_key=test_key3)
    endpoint_manager.append(Endpoint(api_key=test_key4))
    endpoint_manager.load_from(ASSETS_ROOT / "fake_credentials.yml")
    assert len(endpoint_manager) == 8
    assert endpoint_manager[0].api_key == test_key1
    assert endpoint_manager[1].api_key == test_key2
    assert endpoint_manager[2].api_key == test_key_in_file1
    assert endpoint_manager[3].api_key == test_key_in_file2
    assert endpoint_manager[4].api_key == test_key3
    assert endpoint_manager[5].api_key == test_key4
    assert endpoint_manager[6].api_key == test_key_in_file1
    assert endpoint_manager[7].api_key == test_key_in_file2
    
    # override the current endpoints with the new ones from the file
    endpoint_manager.load_from(ASSETS_ROOT / "fake_credentials.yml", override=True)
    
    with OpenAIClient(endpoint_manager=endpoint_manager) as client:
        assert client.chat(messages=[]).api_key == test_key_in_file1
        # use the temporary key; 
        # current endpoint in endpoint_manager will also be rotated, but not used
        assert client.chat(messages=[], api_key=tmp_key).api_key == tmp_key
        assert client.chat(messages=[], endpoint_manager=endpoint_manager).api_key == test_key_in_file1 # rotate back to the first one
        assert client.chat(messages=[]).api_key == test_key_in_file2 # rotate to the second one

def test_endpoint_param():
    test_key1 = "test-key-1"
    test_key2 = "test-key-2"
    test_key3 = "test-key-3"
    test_key4 = "test-key-4"
    endpoints = [
        { "api_key": test_key1 },
        { "api_key": test_key2 },
    ]
    endpoint_manager = EndpointManager(endpoints=[
        { "api_key": test_key3 },
        { "api_key": test_key4 },
    ])
    with OpenAIClient(endpoint_manager=endpoint_manager) as client:
        # internal endpoint_manager will be used
        assert client.chat(messages=[]).api_key == test_key3
        assert client.chat(messages=[]).api_key == test_key4
        # current endpoint in endpoint_manager will also be rotated, but not used
        assert client.chat(messages=[], endpoint=Endpoint(api_key=test_key2)).api_key == test_key2
        assert client.chat(messages=[], endpoint={ "api_key": test_key1 }).api_key == test_key1
        # always use the first in the endpoints, because each time a new temporary EndpointManager is created
        assert client.chat(messages=[], endpoints=endpoints).api_key == test_key1
        assert client.chat(messages=[], endpoints=endpoints).api_key == test_key1
    with OpenAIClient(endpoints=endpoints) as client:
        # internal endpoint_manager will be used
        assert client.chat(messages=[]).api_key == test_key1
        assert client.chat(messages=[]).api_key == test_key2
        # current endpoint in endpoint_manager will also be rotated, but not used
        assert client.chat(messages=[], endpoint=Endpoint(api_key=test_key3)).api_key == test_key3
        assert client.chat(messages=[], endpoint={ "api_key": test_key4 }).api_key == test_key4
        # always use the first in the endpoints, because each time a new temporary EndpointManager is created
        assert client.chat(messages=[], endpoints=endpoints).api_key == test_key1
        assert client.chat(messages=[], endpoints=endpoints).api_key == test_key1

def test_endpoint_manager_exception():
    with pytest.raises(ValueError) as excinfo:
        # empty endpoint_manager
        with OpenAIClient(endpoint_manager=EndpointManager()) as client:
            client.chat(messages=[])
    assert "No endpoint available" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        EndpointManager(endpoints=[1,2,3])
    assert "Unsupported type" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        EndpointManager(endpoints="asdf")
    assert "non-str iterable" in str(excinfo.value)

