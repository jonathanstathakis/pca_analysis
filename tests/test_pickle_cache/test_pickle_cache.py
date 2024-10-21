import pytest
from ..pickle_cache import PickleCache
from pathlib import Path


@pytest.fixture(scope="module")
def cache_name():
    return "test_cache"


@pytest.fixture(scope="module")
def cache_parent():
    return str(Path(__file__).parent)


@pytest.fixture(scope="module")
def cache_root(cache_parent, cache_name):
    return str(Path(cache_parent) / cache_name)


@pytest.fixture(scope="module")
def pickle_cache(cache_parent, cache_name):
    return PickleCache(cache_parent=cache_parent, cache_name=cache_name)


def test_pickle_cache_cache_root(pickle_cache, cache_root):
    assert pickle_cache._cache_root == cache_root


@pytest.fixture
def existing_cache():
    path_to_existing_cache = Path(__file__).parent / "test_cache_loading"
    return PickleCache.load_cache(path_to_existing_cache)


def test_pickle_cache_load_cache(existing_cache):
    """test if PickleCache can load an already existing cache"""

    objs = {"obj1": "a string", "obj2": 1}

    for key, obj in objs.items():
        cached_obj = existing_cache.fetch_from_cache(key)

        assert obj == cached_obj


@pytest.fixture
def test_obj_dict():
    return {"obj1": "a string", "obj2": 1}


def test_pickle_cache_get_all_pickle_paths(
    pickle_cache, cache_root: str, test_obj_dict
):
    expected_paths = sorted(
        [str(Path(cache_root) / Path(x).with_suffix(".pk")) for x in test_obj_dict]
    )

    assert expected_paths == pickle_cache._get_all_pickle_paths()


def test_write_to_cache(test_obj_dict, tmp_path):
    """test if can write objs to a path"""
    cache_name = "test_write_cache"
    pc = PickleCache(tmp_path, cache_name)
    pc.add_many_to_cache(test_obj_dict)

    obj_paths = list((Path(tmp_path) / cache_name).glob("*.pk"))

    assert obj_paths


def test_del_cache(test_obj_dict, tmpdir):
    """test deleting a cache"""
    cache_name = "test_del_cache"
    pc = PickleCache(tmpdir, cache_name)
    pc.add_many_to_cache(test_obj_dict)
    pc.clear_cache()

    assert not Path(pc._cache_root).exists()
