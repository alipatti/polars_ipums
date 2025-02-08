from pathlib import Path
from typing import Any, Collection, Optional
from polars.io.csv import BatchedCsvReader

import polars as pl
import polars.selectors as cs
import us
import xmltodict
import rich

IpumsMetadata = dict[str, dict[str, Any]]


def parse_variable_metadata(xml: list[dict]) -> IpumsMetadata:
    return {
        d["@ID"].lower(): {
            "name": d["@ID"].lower(),
            "longname": d["labl"],
            "start": int(d["location"]["@StartPos"]) - 1,
            "end": int(d["location"]["@EndPos"]) - 1,
            "length": int(d["location"]["@width"]),
            "type": d["varFormat"]["@type"],
            "implied_decimals": int(d["@dcml"]),
            "description": d["txt"],
            "labels": (
                {int(i["catValu"]): i["labl"] for i in d["catgry"]}
                if "catgry" in d
                else None
            ),
        }
        for d in xml
    }


def add_state_abbreviation(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Adds column containing state abbreviation (e.g. WY, MN).
    """

    abbr_to_fips = pl.LazyFrame(
        [(s.abbr, s.fips) for s in us.states.STATES_AND_TERRITORIES + [us.states.DC]],
        schema=["state", "statefip"],
        orient="row",
    )

    return df.join(abbr_to_fips, on="statefip").cast(
        {"state": pl.Categorical(ordering="lexical")}
    )


def separate_columns(
    df: pl.LazyFrame,
    *,
    variable_metadata: dict,
    raw_column=pl.col("raw"),
) -> pl.LazyFrame:
    """
    Extracts individual columns from the fixed-width text encoding using info from the IPUMS XML
    metadata file.
    """
    return df.with_columns(
        raw_column.str.slice(offset=v["start"], length=v["length"]).alias(name)
        for name, v in variable_metadata.items()
    )


def convert_to_numeric(
    df: pl.LazyFrame,
    variable_metadata: IpumsMetadata,
    *,
    exclude: Optional[list[str]] = None,
    strict=True,
) -> pl.LazyFrame:
    """
    Converts numeric columns to the appropriate integer or float type with the correct number
    of decimal places.
    """
    exclude = exclude or []

    integer_columns = cs.by_dtype(pl.String)
    if exclude:
        integer_columns &= cs.exclude(exclude)

    decimal_columns = {
        name: v["implied_decimals"]
        for name, v in variable_metadata.items()
        if v["implied_decimals"] > 0 and name not in exclude
    }

    return df.with_columns(
        # cast columns to integers
        integer_columns.str.to_integer(strict=strict),
    ).with_columns(
        # cast cols to floats and scale appropriately
        pl.col(name).cast(pl.Float64).mul(0.1**decimal_places)
        for name, decimal_places in decimal_columns.items()
    )


def label_column(
    column: pl.Expr,
    labels: dict[int, str | None],
    overrides: dict[str, str | None] = {},
    null_values: Collection[str] = set(),
    sort_by_key=True,
) -> pl.Expr:
    """
    Labels a column with labels provided as a dictionary.
    """
    overrides |= {
        l: None
        for l in labels.values()
        if l is not None
        and (l.startswith("Missing") or l.startswith("N/A") or l in null_values)
    }

    labels = {k: (overrides[v] if v in overrides else v) for k, v in labels.items()}

    # sort by integer key
    if sort_by_key:
        labels = dict(sorted(labels.items()))

    indices = pl.Series(labels.keys(), dtype=pl.Int64)
    categories = pl.Series(
        labels.values(),
        dtype=pl.Enum(list({v for v in labels.values() if v is not None})),
    )

    return column.replace_strict(old=indices, new=categories)


def label_columns(
    df: pl.LazyFrame,
    *,
    cols_to_label: Collection[str],
    variable_metadata: IpumsMetadata,
    label_overrides: dict[str, dict],
) -> pl.LazyFrame:
    """
    Labels the indicated columns with labels provided in the IPUMS XML metadata.
    """
    return df.with_columns(
        pl.col(name).pipe(
            label_column,
            labels=metadata["labels"],
            overrides=label_overrides.get(name) or {},
        )
        for name, metadata in variable_metadata.items()
        if name in cols_to_label
    )


def fix_geographic_column(
    df: pl.LazyFrame,
    col: str,
    *,
    state_fips=pl.col("statefip"),
    dtype=pl.Categorical(ordering="lexical"),
) -> pl.LazyFrame:
    """
    Takes df with three-digit county fips code and two-digit state fips codes (as integers).
    Returns df with categorical county column with five-digit fips codes.
    """

    null_values = [
        " " * 3,
        "0" * 3,
        " " * 5,
        "0" * 5,
    ]

    return df.with_columns(
        (state_fips + pl.col(col).replace({v: None for v in null_values}))
        .cast(dtype)
        .alias(col)
    )


def get_ipums_metadata(path: Path):
    """
    Loads the IPUMS metadata from the provided XML file into a Python dictionary.
    """
    xml = xmltodict.parse(path.glob("*.xml").__next__().open("rb"))
    variable_metadata = parse_variable_metadata(xml["codeBook"]["dataDscr"]["var"])

    return variable_metadata


def get_ipums_batched_reader(folder: Path) -> BatchedCsvReader:
    """
    Loads the IPUMS data (expected as an unzipped .dat file in the provided folder) into a BatchedCsvReader.
    """

    return pl.read_csv_batched(
        folder.glob("*.dat").__next__(),
        has_header=False,
        separator="\n",
        new_columns=["raw"],
    )


def clean_ipums(
    raw: pl.LazyFrame,
    variable_metadata: IpumsMetadata,
    labels: dict,
) -> pl.LazyFrame:
    return (
        raw
        # separate out individual columns
        .pipe(separate_columns, variable_metadata=variable_metadata)
        # prepend county and puma with state fips codes so they are unique
        .pipe(
            fix_geographic_column if "countyfip" in variable_metadata else _identity,
            "countyfip",
        ).pipe(
            fix_geographic_column if "puma" in variable_metadata else _identity,
            "puma",
        )
        # add state abbreviations (in addition to numeric fips codes)
        .pipe(add_state_abbreviation if "statefip" in variable_metadata else _identity)
        # convert remaining str cols to integers/floats
        .pipe(convert_to_numeric, variable_metadata, strict=False)
        # label requested columns
        .pipe(
            label_columns,
            cols_to_label=labels.keys(),
            label_overrides=labels,
            variable_metadata=variable_metadata,
        )
        # subset to requested columns
        .drop("raw")
    )


def _identity(df, *args, **kwargs):
    return df


def create_parquet_dataset(
    ipums_directory: Path,
    dataset_directory: Path,
    partition_by: list[str] = ["year"],
    labels: Optional[dict[str, dict]] = None,
    renames: Optional[dict[str, str]] = None,
    keep_only: Optional[list[str]] = None,
    additional_columns: list[pl.Expr] = [],
    override_output=False,
    verbose=False,
) -> None:
    """
    Converts IPUMS .dat files to a hive-partitioned Parquet
    dataset, optionally labeling and renaming columns.

    To load the resulting dataset with Polars:
    ```python
    pl.scan_parquet(dataset_directory, hive_partitioning=True)
    ```

    Parameters:
    ----------
    ipums_directory
        The path to the directory containing the IPUMS export.

    dataset_directory
        The directory where the partitioned Parquet dataset will be created.

    partition_by
        A list of column names to partition the dataset by.

    labels
        A dictionary indicating what columns to label and how to label them.
        Keys indicate the columns to label, and their values are a dictionary
        mapping IPUMS labels to custom user-provided labels. For example:

        ```python
        labels = {
            "sex": {}, # use the default IPUMS labels
            "rachsing": {
                "White": "White",
                "Black/African American": "Black",
                "American Indian/Alaska Native": "AIAN",
                "Asian/Pacific Islander": "Asian",
                "Hispanic/Latino": "Hispanic",
            },
        }
        ```

    renames
        A dictionary mapping IPUMS names to user-provided names, e.g.
        ```python
        {"perwt": "person_weight"}
        ```

    keep_only
        A list of columns to retain in the output.

    additional_columns
        Additional expressions to include as columns in the dataset.

    override_output
        If True, the function will overwrite the content in the
        `dataset_directory` if it already exists.

    verbose
        Print additional information during execution.

    Returns:
    -------
    None
    """

    rich.print(
        "[blue][bold]Cleaning raw IPUMS data",
        f" - From: {ipums_directory}",
        f" - To: {dataset_directory}",
        sep="\n",
    )

    # setup output path
    dataset_directory.expanduser().resolve().mkdir(exist_ok=override_output)

    # load metadata and data
    reader = get_ipums_batched_reader(ipums_directory)
    variable_metadata = get_ipums_metadata(ipums_directory)

    # loop through batches of csv
    n_simultaneous_batches = 50
    i, batches = 0, reader.next_batches(n_simultaneous_batches)
    while batches:
        df = (
            pl.concat(b.lazy() for b in batches)
            .pipe(clean_ipums, variable_metadata, labels=labels or {})
            .with_columns(
                additional_columns,
                batch=pl.lit(i),
            )
            .rename(renames or {})
            .select(keep_only or pl.all())
            .collect()
        )

        if verbose:
            print(df)
        else:
            print(".", end="")

        df.write_parquet(
            dataset_directory,
            partition_by=partition_by + ["batch"],
        )

        i, batches = i + 1, reader.next_batches(n_simultaneous_batches)

    print()


def test_parquet_conversion(
    input_path=Path(
        "/Users/ali/.devpod/agent/contexts/default/"
        "workspaces/health-inequality/content/data/raw/ipums"
    ),
    output_path=Path("~/Downloads/ipums"),
):

    labels = {
        "sex": {},
        "rachsing": {
            "White": "White",
            "Black/African American": "Black",
            "American Indian/Alaska Native": "AIAN",
            "Asian/Pacific Islander": "Asian",
            "Hispanic/Latino": "Hispanic",
        },
    }
    renames = {
        "rachsing": "race",
        "countyfip": "county",
        "ftotinc": "family_income",
        "hhincome": "household_income",
    }

    # custom column
    educd = pl.col("educd")
    my_education = (
        pl.when(educd.le(61))
        .then(0)
        .when(educd.is_between(62, 64))
        .then(1)
        .when(educd.is_between(65, 100))
        .then(2)
        .when(educd.is_between(101, 116))
        .then(3)
        .alias("my_education")
    )

    create_parquet_dataset(
        input_path,
        output_path,
        labels=labels,
        renames=renames,
        additional_columns=[my_education],
        override_output=True,
    )

    # load back into memory
    df = (
        pl.scan_parquet(output_path / "**/*.parquet", hive_partitioning=True)
        .head()
        .collect()
    )

    # check renames worked
    assert all(col in df.columns for col in renames.values())
    assert all(col not in df.columns for col in renames.keys())

    # TODO: wirte more tests

    print(df)


if __name__ == "__main__":
    test_parquet_conversion()
