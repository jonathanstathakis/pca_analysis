import duckdb as db


import pandas as pd
import xmltodict
from pathlib import Path


def xml_table_to_df(table) -> pd.DataFrame:
    # sometimes its not a list
    if not isinstance(table["Row"], list):
        table["Row"] = [table["Row"]]

    row_0 = table["Row"][0]

    if not isinstance(row_0, dict):
        raise TypeError

    columns = [parameter["Name"] for parameter in row_0["Parameter"]]

    column_dict = {col: [] for col in columns}
    for row in table["Row"]:
        for param in row["Parameter"]:
            name = param["Name"]
            column_dict[name].append(param["Value"])
    df = pd.DataFrame.from_dict(column_dict)

    return df


def get_bin_pump(macaml):
    sections = macaml["ACAML"]["Doc"]["Content"]["MethodConfiguration"][
        "MethodDescription"
    ]["Section"]["Section"]
    bin_pump_idx = find_bin_pump(sections)
    bin_pump = sections[bin_pump_idx]
    return bin_pump


def get_solvent_comp(bin_pump):
    solvent_comp = bin_pump["Table"][0]
    return solvent_comp


def get_timetable(bin_pump):
    try:
        timetable = bin_pump["Table"][1]
    except KeyError as e:
        e.add_note(f"candidate keys: {bin_pump.keys()}")
        raise e
    return timetable


def format_timetable(timetable: pd.DataFrame, id: str):
    """
    add the id column and clean the existing column names
    """
    timetable["id"] = id
    timetable.columns = [col.lower() for col in timetable.columns]
    timetable = timetable.reset_index(names="idx")
    try:
        timetable = timetable[["id", "idx", "time", "a", "b", "flow", "pressure"]]
    except KeyError as e:
        e.add_note(str(timetable.columns))
        raise e
    return timetable


def format_solvent_comp(solvent_comp: pd.DataFrame, id: str):
    """
    add the id column and clean the existing column names
    """
    solvent_comp["id"] = id
    solvent_comp.columns = [col.lower() for col in solvent_comp.columns]
    solvent_comp = solvent_comp.reset_index(names="idx")
    solvent_comp = solvent_comp.rename(
        {
            "ch. 1 solv.": "ch1_solv",
            "ch2 solv.": "ch2_solv",
            "name 1": "name_1",
            "name 2": "name_2",
        },
        axis=1,
    )
    try:
        solvent_comp = solvent_comp[
            [
                "id",
                "idx",
                "channel",
                "ch1_solv",
                "name_1",
                "ch2_solv",
                "name_2",
                "selected",
                "used",
                "percent",
            ]
        ]
    except KeyError as e:
        e.add_note(str(solvent_comp.columns))
        raise e
    return solvent_comp


def open_xml(path: str):
    with open(path, "rb") as f:
        xmldict = xmltodict.parse(f)
    return xmldict


def get_tables(bin_pump) -> tuple[dict, dict]:
    if not isinstance(bin_pump, dict):
        raise TypeError

    if "Name" not in bin_pump:
        raise ValueError

    solvent_comp = get_solvent_comp(bin_pump)
    timetable = get_timetable(bin_pump)

    return solvent_comp, timetable


sequence_path = "/Users/jonathan/uni/0_jono_data/raw_uv/45.D/sequence.acam_"
macaml_path = "/Users/jonathan/uni/0_jono_data/raw_uv/45.D/acq.macaml"


def get_seq_acam_path(dpath: str):
    return str(Path(dpath) / "sequence.acam_")


def get_sample_acaml_path(dpath: str):
    return str(Path(dpath) / "sample.acaml")


def get_macaml_path(dpath: str):
    return str(Path(dpath) / "acq.macaml")


def validate_table(table):
    if "Row" not in table:
        return None


def find_bin_pump(sections):
    for idx, section in enumerate(sections):
        if section["Name"] == "Binary Pump":
            return idx


def get_id(dpath: str) -> str:
    """
    get the id string from the 'sequence.acam_' or 'sample.acaml' file.

    Single runs have 'sample.acaml', sequence runs have 'sequence.acam_', however the
    XML structures appear very similar, and the ID is in the same location.
    """
    if Path(get_seq_acam_path(dpath)).exists():
        path = get_seq_acam_path(dpath)
        acam_ = open_xml(str(path))
        id = get_id_(acam_)

    elif Path(get_sample_acaml_path(dpath)).exists():
        path = get_sample_acaml_path(dpath)
        acaml = open_xml(str(path))
        id = get_id_(acaml)

    else:
        raise ValueError(f"cant find sequence.acam_ or sample.acaml in {dpath}")

    return id


def get_id_(sequence: dict) -> str:
    """
    Extract the 'id' string at the location given below
    """
    id = sequence["ACAML"]["Doc"]["Content"]["SampleContexts"]["Setup"]["@id"]
    return id


def get_bin_pump_tables(dpath: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """

    part 1 of the process.

    For  a .D dir at `dpath` find either a 'sequence.acam_' or 'sample.acaml' to extract
    the 'id', and
    """

    if not isinstance(dpath, (str, Path)):
        raise TypeError("dpath must be str or Path")

    # get the id string for the sample
    id = get_id(dpath)

    # get the acq.macaml XML as a dict
    macaml_path = get_macaml_path(dpath)
    macaml = open_xml(macaml_path)

    # extract the branch containing the binary pump information
    bin_pump = get_bin_pump(macaml)

    # stract the branches containing the tables in dict form
    solvent_comp, timetable = get_tables(bin_pump)

    # parse the tables, returning them as pandas DataFrames

    # check that the tables have rows
    try:
        if "Row" in solvent_comp:
            solvent_comp_df = xml_table_to_df(solvent_comp).pipe(
                format_solvent_comp, id
            )
        else:
            solvent_comp_df = pd.DataFrame()
        if "Row" in timetable:
            timetable_df = xml_table_to_df(timetable).pipe(format_timetable, id)
        else:
            timetable_df = pd.DataFrame()
    except KeyError as e:
        e.add_note(Path(dpath).stem)
        raise e

    # confirm the tables have been created as expected.
    if timetable_df.empty or solvent_comp_df.empty:
        raise ValueError

    return solvent_comp_df, timetable_df


def sample_tables_to_db(
    sample: tuple[pd.DataFrame, pd.DataFrame],
    con: db.DuckDBPyConnection,
) -> int:
    """
    adds sample tables to database tables 'solvcomps', 'timetables' in schema 'bin_pump'.

    returns the table number.
    """

    with open(Path(__file__).parent / "to_db.sql") as f:
        query = f.read()

    con.sql(query)

    solvcomp = sample[0]
    timetable = sample[1]
    incoming_id = solvcomp["id"][0]

    try:
        con.execute(
            """--sql
        INSERT
            INTO bin_pump.id as tbl BY NAME (SELECT ? as id);
        """,
            parameters=[incoming_id],
        )
    except db.ConstraintException as e:
        e.add_note(
            "you just tried adding a sample to bin_pump tables that was already present."
        )
        raise e

    # now that the id is in, get the table nums and populate the incoming tables while dropping their id columns

    new_tbl_num = (
        con.execute(
            """--sql
    SELECT
        tbl_num
    FROM
        bin_pump.id
    WHERE
        id = ?
    """,
            parameters=[incoming_id],
        )
        .pl()["tbl_num"]
        .item()
    )

    solvcomp_columns = [
        "tbl_num",
        "idx",
        "channel",
        "ch1_solv",
        "name_1",
        "ch2_solv",
        "name_2",
        "selected",
        "used",
        "percent",
    ]

    solvcomp = solvcomp.drop("id", axis=1)
    solvcomp["tbl_num"] = new_tbl_num
    solvcomp = solvcomp[solvcomp_columns]  # correct order

    timetable_columns = ["tbl_num", "idx", "time", "a", "b", "flow", "pressure"]
    timetable = timetable.drop("id", axis=1)
    timetable["tbl_num"] = new_tbl_num
    timetable = timetable[timetable_columns]

    # insert the new data into the tables
    con.sql(
        """--sql
    INSERT INTO
        bin_pump.solvcomps
    SELECT
        *
    FROM
        solvcomp
    """
    )

    con.sql(
        """--sql
    INSERT INTO
        bin_pump.timetables
    SELECT
        *
    FROM
        timetable
    """
    )

    return new_tbl_num


def extract_bin_pump_tables_to_dbase(paths: list, con):
    """ """

    tables = {Path(dpath).stem: get_bin_pump_tables(str(dpath)) for dpath in paths}

    tables_to_db(tables, con)
