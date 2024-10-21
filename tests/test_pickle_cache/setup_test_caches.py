from pathlib import Path
import pickle
import logging
import sys
import argparse

logger = logging.getLogger()
logging.basicConfig(level=logging.DEBUG)


def setup_test_cache_loading(overwrite: bool = False):
    """setup a cache dir with pickle objects already present"""

    obj1 = {"obj1": "a string"}
    obj2 = {"obj2": 1}

    cache_path = Path(__file__).parent / "test_cache_loading"

    cache_path.mkdir(exist_ok=True)

    for key, obj in {**obj1, **obj2}.items():
        path = (Path(cache_path) / key).with_suffix(".pk")
        exists = Path(path).exists()
        if exists and overwrite:
            logger.debug(f"overwriting {key}..")
            Path(path).unlink()
        elif exists and not overwrite:
            raise ValueError(
                f"{path} already exists. set overwite to True to overwrite.."
            )

        with open(path, "wb") as f:
            logger.debug(f"writing '{obj}' to {path}")
            pickle.dump(obj, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--overwrite", action="store_true")

    args = parser.parse_args()
    print(args)
    setup_test_cache_loading(overwrite=args.overwrite)
