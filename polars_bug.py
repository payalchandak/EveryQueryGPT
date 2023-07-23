from datetime import datetime, timedelta

import polars as pl

df = pl.DataFrame(
    {
        "subject_id": [1, 1, 1, 1, 1, 2, 2, 2, 2],
        "timestamp": [
            datetime(2020, 1, 1),
            datetime(2020, 1, 2),
            datetime(2020, 1, 5),
            datetime(2020, 1, 16),
            datetime(2020, 3, 1),
            datetime(2020, 1, 3),
            datetime(2020, 1, 4),
            datetime(2020, 1, 9),
            datetime(2020, 1, 11),
        ],
        "event_A": [True, False, False, True, False, False, True, True, False],
        "event_B": [False, True, True, False, True, False, False, False, True],
        "event_C": [True, True, True, False, False, False, False, False, True],
    }
)

print(
    """
    df
    .groupby_rolling('timestamp', period='7d', offset=timedelta(days=0), by='subject_id', closed='right')
    .agg(pl.col('event_A').sum())
"""
)
print(
    df.groupby_rolling(
        "timestamp", period="7d", offset=timedelta(days=0), by="subject_id", closed="right"
    ).agg(pl.col("event_A").sum())
)
