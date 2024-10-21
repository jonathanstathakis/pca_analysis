from pathlib import Path
import pickle
from typing import Any
import logging

logger = logging.getLogger(__name__)


class PickleCache:
    def __init__(self, cache_parent: str, cache_name: str = "cache"):
        """
        handle pickle caching for a dir. Stores cached pickles in conjunction with pytestconfig within "<cache_root> / cache".

        # see https://snyk.io/blog/guide-to-python-pickle/ for a good guide to pickling

        Parameters
        ----------
        cache_parent : str, optional.
            the parent path for the cache. Typically the test dir. By default "", in which case PickleCache will use the current file directory.
        cache_name : str, optional.
            the name of the cache. By default "cache". Used to name the directory within which the pickles are stored.

        """

        self._cache_parent = cache_parent

        self._cache_name = cache_name
        self._cache_root = str(Path(self._cache_parent) / self._cache_name)

        # check if pickles present, reading from the file in the cache dir

        if Path(self._cache_root).exists():
            self.cached = [str(x.stem) for x in Path(self._cache_root).glob("*.pk")]
        else:
            self.cached = []

    @classmethod
    def load_cache(cls, path):
        """load an already existing cache at `path`"""

        cache_name = str(Path(path).name)
        cache_parent = str(Path(path).parent)

        return PickleCache(cache_parent=cache_parent, cache_name=cache_name)

    def add_many_to_cache(self, objs: dict[str, Any]) -> None:
        """
        add pickle key, path pairs to `pickles`.

        Parameters
        ----------
        pickles : list[str]
            A list of pickle labels to be added to the cache.
        """
        # ensure we're not overwriting

        dup_keys = [key for key in objs if key in self.cached]
        if dup_keys:
            raise ValueError(
                f"tried adding already existing keys to pickles: {dup_keys}"
            )

        for key, val in objs.items():
            self.add_to_cache(key, val)
            self.cached = sorted(self.cached)

    def add_to_cache(self, key: str, obj):
        """add an object to the cache.

        1. add key to internal list
        2. dump pickle.
        """
        self.cached.append(key)
        self.write_to_cache(key, obj)

    def _get_all_pickle_paths(self):
        """return all stored pickle full paths"""

        return sorted([self._full_pickle_path(x) for x in self.cached])

    def _get_pickle_path(self, pickle_name: str) -> str:
        """return the full path of `pickle_name`"""
        return self._full_pickle_path(pickle_name=pickle_name)

    def _full_pickle_path(self, pickle_name: str) -> str:
        """form the full pickle path"""
        return str((Path(self._cache_root) / pickle_name).with_suffix(".pk"))

    def write_to_cache(self, pickle_name: str, obj):
        """pickle `obj` in cache"""

        path = Path(self._get_pickle_path(pickle_name))

        logger.debug(f"writing obj to {path}..")

        if not path.parent.exists():
            logger.debug(f"creating parent dir {path.parent}..")
            path.parent.mkdir()
        if not path.exists():
            logger.debug(f"creating new file {path}..")
            path.touch()
        with open(path, "wb") as f:
            logger.debug("writing pickle..")
            pickle.dump(obj, f)

        logger.debug(f"pickle written at {path}.")

    def fetch_from_cache(self, key: str) -> Any | None:
        """load `pickle_name`"""

        if key in self.cached:
            with open(self._get_pickle_path(key), "rb") as f:
                obj = pickle.load(f)
        else:
            obj = None

        return obj

    def remove_cached_obj(self, key):
        """remove the cached object at `key`"""

        logger.debug(f"deleting {key}..")
        Path(self._get_pickle_path(key)).unlink()

    def clear_cache(self):
        """clear the cache"""

        # clear the individual files
        logger.debug("clearing cache..")
        logger.debug(f"current cache contents: {self.cached}..")

        for x in self.cached:
            self.remove_cached_obj(x)

        # remove the cache dir
        Path(self._cache_root).rmdir()

        logger.debug("cache cleared.")

    def __repr__(self):
        repr_str = f"""
        {self._cache_name}
        {"-"*len(self._cache_name)}

        dirpath: {self._cache_root}

        items in cache: {len(self.cached)}
        
        keys:
        
        {"- " + self.cached[0] + "\n" + "- ".join(self.cached[1:])}
        """

        return repr_str
