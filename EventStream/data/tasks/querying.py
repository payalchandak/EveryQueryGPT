"""Task querying utilities.

Will be re-organized prior to final merge.
"""

from datetime import timedelta
from typing import Any

import polars as pl


def query_temporal_window(
    predicates_df: pl.LazyFrame | pl.DataFrame,
    node_col: str | pl.Expr,
    st_inclusive: bool,
    window_size: timedelta,
    end_inclusive: bool,
    window_constraints: Any,
) -> pl.LazyFrame | pl.DataFrame:
    """Queries predicates_df to determine if the passed temporal window can be satisfied from node_col.

    Args:
        predicates_df: A dataframe with columns ``subject_id``, containing subject identifiers, ``timestamp``
            containing timestamps of the events in each row, a collection of arbitrarily named string columns
            containing boolean indicators of whether or not events occurring at that timestamp satisfy given
            predicates.
        node_col: The column of the starting node of the window.
        st_inclusive: Is the starting point inclusive?
        window_size: The (possibly negative) duration of the window.
        end_inclusive: Is the end point inclusive?
        window_constraints: ???

    Returns:
        ???

    Examples:
        >>> from datetime import datetime, timedelta
        >>> import polars as pl
        >>> pl.Config.set_tbl_width_chars(100)
        <class 'polars.config.Config'>
        >>> predicates_df = pl.DataFrame({
        ...     'subject_id': [1, 1, 1, 1, 1, 2, 2, 2, 2],
        ...     'timestamp': [
        ...         datetime(2020, 1, 1),
        ...         datetime(2020, 1, 2),
        ...         datetime(2020, 1, 5),
        ...         datetime(2020, 1, 16),
        ...         datetime(2020, 3, 1),
        ...         datetime(2020, 1, 3),
        ...         datetime(2020, 1, 4),
        ...         datetime(2020, 1, 9),
        ...         datetime(2020, 1, 11),
        ...     ],
        ...     'event_A': [True,  False, False,  True, False, False,  True,  True, False],
        ...     'event_B': [False,  True,  True, False,  True, False, False, False, True],
        ...     'event_C': [True,   True,  True, False, False, False, False, False, True],
        ... })
        >>> cols = ['subject_id', 'timestamp', 'event_A', 'event_B']
        >>> out_df = query_temporal_window(
        ...     predicates_df, 'event_A', False, timedelta(days=7), True, {"event_B": (1, None)}
        ... )
        >>> out_df.filter(pl.col('valid_window').is_not_null())
        shape: (3, 6)
        ┌────────────┬─────────────────────┬─────────┬─────────┬─────────┬─────────────────────────────────┐
        │ subject_id ┆ timestamp           ┆ event_A ┆ event_B ┆ event_C ┆ valid_window                    │
        │ ---        ┆ ---                 ┆ ---     ┆ ---     ┆ ---     ┆ ---                             │
        │ i64        ┆ datetime[μs]        ┆ bool    ┆ bool    ┆ bool    ┆ struct[2]                       │
        ╞════════════╪═════════════════════╪═════════╪═════════╪═════════╪═════════════════════════════════╡
        │ 1          ┆ 2020-01-01 00:00:00 ┆ true    ┆ false   ┆ true    ┆ {2020-01-01 00:00:00,2020-01-08 │
        │            ┆                     ┆         ┆         ┆         ┆ …                               │
        │ 2          ┆ 2020-01-04 00:00:00 ┆ true    ┆ false   ┆ false   ┆ {2020-01-04 00:00:00,2020-01-11 │
        │            ┆                     ┆         ┆         ┆         ┆ …                               │
        │ 2          ┆ 2020-01-09 00:00:00 ┆ true    ┆ false   ┆ false   ┆ {2020-01-09 00:00:00,2020-01-16 │
        │            ┆                     ┆         ┆         ┆         ┆ …                               │
        └────────────┴─────────────────────┴─────────┴─────────┴─────────┴─────────────────────────────────┘
        >>> out_df
        >>> assert (out_df.drop('valid_window') == predicates_df)
        >>> query_temporal_window(
        ...     predicates_df, 'event_A', False, timedelta(days=7), True, {"event_B": (2, None)}
        ... )
        subject_id | timestamp | event_A | event_B | output
        1          | 2020-1-1  | True    | False   | {"st": (2020-1-1, False), "end": (2020-1-8, True)}
        1          | 2020-1-2  | False   | True    | None
        1          | 2020-1-5  | False   | True    | None
        1          | 2020-1-16 | True    | False   | None
        1          | 2020-3-1  | False   | False   | None
        2          | 2020-1-3  | False   | False   | None
        2          | 2020-1-4  | True    | False   | None
        2          | 2020-1-9  | True    | False   | None
        2          | 2020-1-11 | True    | True    | None
    """

    if st_inclusive and end_inclusive:
        closed = "both"
    elif st_inclusive:
        closed = "left"
    elif end_inclusive:
        closed = "right"
    else:
        closed = "none"

    if window_size < timedelta(days=0):
        period = -window_size
        offset = -period
        end_time_expr = pl.col("timestamp") - window_size
    else:
        period = window_size
        offset = timedelta(days=0)
        end_time_expr = pl.col("timestamp") + window_size

    aggs = {}
    valid_exprs = []
    for col, (cnt_ge, cnt_le) in window_constraints.items():
        if cnt_ge is None and cnt_le is None:
            raise ValueError(f"Empty constraint for {col}!")

        if col == "*":
            aggs["__ALL_EVENTS"] = pl.count()
            col = "__ALL_EVENTS"
        else:
            aggs[col] = pl.col(col).sum()

        if cnt_ge is not None:
            valid_exprs.append(pl.col(col) >= cnt_ge)
        if cnt_le is not None:
            valid_exprs.append(pl.col(col) <= cnt_le)

    return (
        predicates_df.groupby_rolling(
            index_column="timestamp",
            by="subject_id",
            closed=closed,
            period=period,
            offset=offset,
        )
        .agg(**aggs)
        .select("subject_id", "timestamp", pl.all(valid_exprs).alias("is_valid"))
        .join(predicates_df, on=["subject_id", "timestamp"], how="outer")
        .with_columns(
            pl.when(pl.col(node_col) & pl.col("is_valid"))
            .then(pl.struct(pl.col("timestamp").alias("start"), end_time_expr.alias("end")))
            .otherwise(pl.lit(None))
            .alias("valid_window")
        )
        .drop("is_valid")
        .sort(by=["subject_id", "timestamp"])
    )


def query_event_bound_window(
    predicates_df: pl.LazyFrame | pl.DataFrame,
    node_col: str | pl.Expr,
    st_inclusive: bool,
    window_size_negative: bool,
    end_event_col: str,
    end_inclusive: bool,
    window_constraints: Any,
) -> pl.Expr:
    """Queries predicates_df to determine if the passed temporal window can be satisfied from node_col.

    Args:
        predicates_df: A dataframe with columns ``subject_id``, containing subject identifiers, ``timestamp``
            containing timestamps of the events in each row, a collection of arbitrarily named string columns
            containing boolean indicators of whether or not events occurring at that timestamp satisfy given
            predicates.
        node_col: The column of the starting node of the window.
        st_inclusive: Is the starting point inclusive?
        window_size_negative: Does the window span until the last event prior to the st event, or until the
            first event after?
        end_event_bound: The column name of the event predicate that defines the end of the window.
        end_inclusive: Is the end point inclusive?
        window_constraints: ???

    Returns:
        ???

    Examples:
        >>> from datetime import datetime, timedelta
        >>> predicates_df = pl.DataFrame({
        ...     'subject_id': [1, 1, 1, 1, 1, 2, 2, 2, 2],
        ...     'timestamp': [
        ...         datetime(2020, 1, 1),
        ...         datetime(2020, 1, 2),
        ...         datetime(2020, 1, 5),
        ...         datetime(2020, 1, 16),
        ...         datetime(2020, 3, 1),
        ...         datetime(2020, 1, 3),
        ...         datetime(2020, 1, 4),
        ...         datetime(2020, 1, 9),
        ...         datetime(2020, 1, 11),
        ...     ],
        ...     'event_A': [True,  False, False,  True, False, False,  True,  True, False],
        ...     'event_B': [False,  True,  True, False,  True, False, False, False, True],
        ...     'event_C': [True,   True,  True, False, False, False, False, False, True],
        ... })
        >>> cols = ['subject_id', 'timestamp', 'event_A', 'event_B']
        >>> query_event_bound_window(
        ...     predicates_df, 'event_A', False, False, "event_C", False, {"event_B": (None, 0)}
        ... )
        subject_id | timestamp | event_A | event_B | event_C | output
        1          | 2020-1-1  | True    | False   | True    | {"st": 2020-1-1, "end": 2020-1-2}
        1          | 2020-1-2  | False   | True    | True    | False
        1          | 2020-1-5  | False   | True    | True    | False
        1          | 2020-1-16 | True    | False   | False   | False
        1          | 2020-3-1  | False   | False   | False   | False
        2          | 2020-1-3  | False   | False   | False   | False
        2          | 2020-1-4  | True    | False   | False   | True
        2          | 2020-1-9  | True    | False   | False   | True
        2          | 2020-1-11 | True    | True    | True    | False
        >>> query_event_bound_window(
        ...     predicates_df, 'event_A', False, False, "event_C", False,
        ...     {"event_B": (None, 0), "event_A": (None, 0)}
        ... )
        subject_id | timestamp | event_A | event_B | event_C | output
        1          | 2020-1-1  | True    | False   | True    | True
        1          | 2020-1-2  | False   | True    | True    | False
        1          | 2020-1-5  | False   | True    | True    | False
        1          | 2020-1-16 | True    | False   | False   | False
        1          | 2020-3-1  | False   | False   | False   | False
        2          | 2020-1-3  | False   | False   | False   | False
        2          | 2020-1-4  | True    | False   | False   | False
        2          | 2020-1-9  | True    | False   | False   | True
        2          | 2020-1-11 | True    | True    | True    | False
        >>> query_event_bound_window(
        ...     predicates_df, 'event_A', True, False, "event_C", True,
        ...     {"*": (3, None)}
        ... )
        subject_id | timestamp | event_A | event_B | event_C | output
        1          | 2020-1-1  | True    | False   | True    | False
        1          | 2020-1-2  | False   | True    | True    | False
        1          | 2020-1-5  | False   | True    | True    | False
        1          | 2020-1-16 | True    | False   | False   | False
        1          | 2020-3-1  | False   | False   | False   | False
        2          | 2020-1-3  | False   | False   | False   | False
        2          | 2020-1-4  | True    | False   | False   | True
        2          | 2020-1-9  | True    | False   | False   | False
        2          | 2020-1-11 | True    | True    | True    | False
        >>> query_event_bound_window(
        ...     predicates_df, 'event_C', True, True, "event_A", True,
        ...     {"*": (3, None)}
        ... )
        subject_id | timestamp | event_A | event_B | event_C | output
        1          | 2020-1-1  | True    | False   | True    | False
        1          | 2020-1-2  | False   | True    | True    | False
        1          | 2020-1-5  | False   | True    | True    | True
        1          | 2020-1-16 | True    | False   | False   | False
        1          | 2020-3-1  | False   | False   | False   | False
        2          | 2020-1-3  | False   | False   | False   | False
        2          | 2020-1-4  | True    | False   | False   | False
        2          | 2020-1-9  | True    | False   | False   | False
        2          | 2020-1-11 | True    | True    | True    | False
    """
    raise NotImplementedError


def query_window(
    predicates_df: pl.LazyFrame | pl.DataFrame,
    node_col: str | pl.Expr,
    endpoint_expr: Any,
    window_constraints: Any,
) -> pl.Expr:
    """Queries predicates_df to determine if the passed window can be satisfied from node_col.

    Args:
        predicates_df: A dataframe with columns ``subject_id``, containing subject identifiers, ``timestamp``
            containing timestamps of the events in each row, a collection of arbitrarily named string columns
            containing boolean indicators of whether or not events occurring at that timestamp satisfy given
            predicates.
        node_col: The column of the starting node of the window.
        endpoint_expr: ???
        window_constraints: ???

    Returns:
        ???
    """
    st_inclusive, end_expr = endpoint_expr
    end_delta, end_inclusive = end_expr

    match end_delta:
        case timedelta():
            return query_temporal_window(
                predicates_df, node_col, st_inclusive, end_delta, end_inclusive, window_constraints
            )
        case str():
            window_size_negative = end_delta.startswith("-")
            if window_size_negative:
                end_delta = end_delta[1:]
            return query_event_bound_window(
                predicates_df,
                node_col,
                st_inclusive,
                window_size_negative,
                end_delta,
                end_inclusive,
                window_constraints,
            )
        case _:
            raise ValueError(f"Invalid expression {end_delta}.")
