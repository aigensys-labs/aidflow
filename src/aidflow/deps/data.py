"""
The Dependency Abstraction Layer (Dependency Firewall).

This module serves as the sole authorized gateway for all external dependencies 
and Python standard library imports across the entire AIDflow ecosystem. 
Following a strict architectural mandate, no module within the AIDflow package 
is permitted to import third-party libraries or native Python modules directly; 
instead, they must consume these dependencies via the `.deps` abstraction layer.
"""

import pyarrow as pa
import numpy   as np
import pandas  as pd

from pyarrow import (
    compute as pa_pc,
    parquet as pa_pq,
    feather as pa_ft,
    dataset as pa_ds,
    ipc     as pa_ipc,
    csv     as pa_csv,
)

from pandas import (
    DataFrame as pDF, 
    Series    as pSe
)

from pandas.io.formats.style import (
    Styler as pd_Styler
)

def pg_dbapi():
    """DBAPI 2.0-compatible facade for the ADBC libpq driver."""
    import adbc_driver_postgresql.dbapi as pg_dbapi
    return pg_dbapi

def spark_session():
    """
    The entry point to programming Spark with the Dataset and DataFrame API.

    A SparkSession can be used to create :class:`DataFrame`, register :class:`DataFrame` as
    tables, execute SQL over tables, cache tables, and read parquet files.
    """    
    from pyspark.sql import SparkSession
    return SparkSession