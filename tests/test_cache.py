from pathlib import Path
from handyllm import CacheManager
from pytest import CaptureFixture
import pytest


def func():
    print("In func")
    return "hello"

async def async_func():
    print("In async_func")
    return "hello"

def test_cache_manager(tmp_path: Path, capsys: CaptureFixture[str]):
    cm = CacheManager(base_dir=tmp_path, enabled=True, save_only=False)
    out = cm.cache(func=func, out="test.txt")()
    assert (tmp_path / "test.txt").read_text() == "hello"
    assert out == "hello"
    # check print: func() should be called
    captured = capsys.readouterr()
    assert captured.out == "In func\n"

    out2 = cm.cache(func=func, out="test.txt")()
    assert out2 == "hello"
    # check print: func() should not be called again
    captured = capsys.readouterr()
    assert captured.out == ""

@pytest.mark.asyncio
async def test_cache_manager_async(tmp_path: Path, capsys: CaptureFixture[str]):
    cm = CacheManager(base_dir=tmp_path, enabled=True, save_only=False)
    out = await cm.cache(func=async_func, out="test_async.txt")()
    assert (tmp_path / "test_async.txt").read_text() == "hello"
    assert out == "hello"
    # check print: func() should be called
    captured = capsys.readouterr()
    assert captured.out == "In async_func\n"

    out2 = await cm.cache(func=async_func, out="test_async.txt")()
    assert out2 == "hello"
    # check print: func() should not be called again
    captured = capsys.readouterr()
    assert captured.out == ""


