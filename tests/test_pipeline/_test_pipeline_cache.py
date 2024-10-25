from tests.pickle_cache import PickleCache
from tests.test_pipeline import CACHE_PATH
import argparse


def start():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d",
        "--delete",
        help="deletes all cache entries",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "-l", "--list", help="lists cache keys", default=False, action="store_true"
    )
    parser.add_argument(
        "-s", "--show", help="show cache root", default=False, action="store_true"
    )
    parser.add_argument(
        "-c", "--clear", help="clear cache entry at the given key", action="store"
    )

    args = vars(parser.parse_args())
    pipeline_cache(**args)


def pipeline_cache(delete, list, show, clear):
    """
    see <https://stackoverflow.com/questions/59286983/how-to-run-a-script-using-pyproject-toml-settings-and-poetry> for PATH integration
    """

    cache = PickleCache.load_cache(CACHE_PATH)

    if not any([delete, list, show]):
        print(cache)
    if delete:
        cache.clear_cache()
    if list:
        print("items in cache:", cache._cached)
    if show:
        print(f"cache_path: {cache._cache_db_path}")
    if clear:
        cache.remove_cached_obj(key=clear)


if __name__ == "__main__":
    start()
