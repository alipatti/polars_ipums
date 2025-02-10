# polars_ipums

A package to work with IPUMS microata in Python. Used in-house at
[Opportunity Insights](https://opportunityinsights.org).

## Example

```python
# convert IPUMS microdata export to a hive-partioned Parquet dataset

import polars as pl
from polars_ipums import create_parquet_dataset

input_path = "~/Downloads/ipums_export"
output_path = "~/Desktop/parquet_ipums"

labels = {
    # use the default IPUMS labels for the sex column
    "sex": {},
    # use custom labels for the race/hispanic origin column
    "rachsing": {
        "White": "White",
        "Black/African American": "Black",
        "American Indian/Alaska Native": "AIAN",
        "Asian/Pacific Islander": "Asian",
        "Hispanic/Latino": "Hispanic",
    },
}

# give a few columns more human-readable names
renames = {
    "rachsing": "race",
    "countyfip": "county",
    "ftotinc": "family_income",
    "hhincome": "household_income",
}

# custom education column!
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
    partition_by=["year"],
    renames=renames,
    additional_columns=[my_education],
    override_output=True,
    verbose=True,
)

# load a few rows back into memory
ipums_microdata = (
    pl.scan_parquet(output_path / "**/*.parquet", hive_partitioning=True)
    .head()
    .collect()
)

```
