import json
from pathlib import Path
from handyllm import CacheManager
from pytest import CaptureFixture
import pytest


def func():
    print("In func")
    return 1

async def async_func():
    print("In async_func")
    return "hello"

def func_multiple():
    print("In func_multiple")
    return 2, "world", {'a': 1}

def test_cache_manager(tmp_path: Path, capsys: CaptureFixture[str]):
    cm = CacheManager(base_dir=tmp_path, enabled=True)
    wrapped_func = cm.cache(func=func, out="test.txt", load_method=int)
    out = wrapped_func()
    assert (tmp_path / "test.txt").read_text() == "1"
    assert out == 1
    # check print: func() should be called
    captured = capsys.readouterr()
    assert captured.out == "In func\n"

    out2 = wrapped_func()
    assert out2 == 1
    # check print: func() should not be called again
    captured = capsys.readouterr()
    assert captured.out == ""

@pytest.mark.asyncio
async def test_cache_manager_async(tmp_path: Path, capsys: CaptureFixture[str]):
    cm = CacheManager(base_dir=tmp_path, enabled=True)
    wrapped_func = cm.cache(func=async_func, out="test_async.txt")
    out = await wrapped_func()
    assert (tmp_path / "test_async.txt").read_text() == "hello"
    assert out == "hello"
    # check print: func() should be called
    captured = capsys.readouterr()
    assert captured.out == "In async_func\n"

    out2 = await wrapped_func()
    assert out2 == "hello"
    # check print: func() should not be called again
    captured = capsys.readouterr()
    assert captured.out == ""

def test_multiple_output(tmp_path: Path):
    cm = CacheManager(base_dir=tmp_path)
    wrapped_func = cm.cache(
        func=func_multiple, 
        out=["test1.txt", "test2.txt", "test3.json"], 
        load_method=(int, None, None)
    )
    out = wrapped_func()
    assert (tmp_path / "test1.txt").read_text() == "2"
    assert (tmp_path / "test2.txt").read_text() == "world"
    assert (tmp_path / "test3.json").read_text() == json.dumps({'a': 1}, indent=2)
    assert out == (2, "world", {'a': 1})

    out2 = wrapped_func()
    assert out2 == (2, "world", {'a': 1})

def test_custom_load_dump(tmp_path: Path):
    def load_method(s: str) -> int:
        s = s.split(": ", 1)[1]
        return int(s)
    
    def dump_method(i: int):
        return "dumped: " + str(i)
    
    cm = CacheManager(base_dir=tmp_path)
    wrapped_func = cm.cache(
        func=func, 
        out="test.txt", 
        load_method=load_method,
        dump_method=dump_method
    )
    out = wrapped_func()
    assert (tmp_path / "test.txt").read_text() == "dumped: 1"
    assert out == 1

    out2 = wrapped_func()
    assert out2 == 1
