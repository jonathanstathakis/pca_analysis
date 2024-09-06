from pathlib import Path
from nbconvert import MarkdownExporter
import pandas as pd
import polars as pl
from frontmatter import Frontmatter


class Notebook:
    def __init__(self, path):
        self.path = path
        self.attrs = self.get_attrs(path=self.path)
        self.attrs["filename"] = Path(self.path).stem
        self.attrs["link"] = f"[link]({Path(self.path)})"

    def get_attrs(self, path):
        fm = Frontmatter()
        md_exporter = MarkdownExporter()

        markdown = md_exporter.from_filename(path)

        frontmatter = markdown[0]

        attrs = fm.read(frontmatter)["attributes"]

        if not attrs:
            attrs = {}

        return attrs


def build_toc(path: Path | str, recursive: bool = True) -> pl.DataFrame:
    """
    Create a table of contents as a polars dataframe from querying over the ipynb in `path`.

    `path` is searched recursively.
    """
    if not isinstance(recursive, bool):
        raise ValueError("expect `recursive` to be type `bool`")

    if recursive:
        glob_str = "**/*.ipynb"
    elif not recursive:
        glob_str = "*.ipynb"

    notebooks = list(Path(path).glob(glob_str))

    if not notebooks:
        raise ValueError(f"No notebooks found at {path}")

    attrs = [Notebook(str(path)).attrs for path in notebooks]

    toc = pd.DataFrame.from_records(attrs)

    try:
        toc["cdt"] = pd.to_datetime(toc["cdt"])
        pl.from_pandas(toc).sort("cdt", descending=True)

    except KeyError:
        from warnings import warn

        warn("tried to sort by creation date but no key 'cdt' present")

    toc = pl.DataFrame(toc)

    return toc
