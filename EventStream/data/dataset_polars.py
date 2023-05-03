import dataclasses, multiprocessing, numpy as np, pandas as pd, polars as pl

from mixins import TimeableMixin
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Sequence, Union

from .config import MeasurementConfig
from .dataset_base import DatasetBase
from .types import DataModality, InputDataType, TemporalityType, NumericDataModalitySubtype
from .vocabulary import Vocabulary

from ..utils import lt_count_or_proportion
from .preprocessing import (
    Preprocessor, StddevCutoffOutlierDetector, StandardScaler
)

pl.toggle_string_cache(True)

@dataclasses.dataclass
class Query:
    connection_uri: str
    query: Union[str, Path, List[Union[str, Path]]]
    partition_on: Optional[str] = None
    partition_num: Optional[int] = None
    protocol: str = 'binary'


DF_T = Union[pl.LazyFrame, pl.DataFrame, pl.Expr, pl.Series]
INPUT_DF_T = Union[Path, pd.DataFrame, pl.DataFrame, Query]

class Dataset(DatasetBase[DF_T, INPUT_DF_T]):
    # Dictates what models can be fit on numerical metadata columns, for both outlier detection and
    # normalization.
    PREPROCESSORS: Dict[str, Preprocessor] = {
        # Outlier Detectors
        'stddev_cutoff': StddevCutoffOutlierDetector,

        # Normalizers
        'standard_scaler': StandardScaler,
    }

    METADATA_SCHEMA = {
        'drop_upper_bound': pl.Float64,
        'drop_upper_bound_inclusive': pl.Boolean,
        'drop_lower_bound': pl.Float64,
        'drop_lower_bound_inclusive': pl.Boolean,
        'censor_upper_bound': pl.Float64,
        'censor_lower_bound': pl.Float64,
        'outlier_model': lambda outlier_params_schema: pl.Struct(outlier_params_schema),
        'normalizer': lambda normalizer_params_schema: pl.Struct(normalizer_params_schema),
        'value_type': pl.Categorical,
    }

    @staticmethod
    def get_smallest_valid_int_type(num: Union[int, float, pl.Expr]) -> pl.DataType:
        if num >= (2**64)-1: raise ValueError("Value is too large to be expressed as an int!")
        if num >= (2**32)-1: return pl.UInt64
        elif num >= (2**16)-1: return pl.UInt32
        elif num >= (2**8)-1: return pl.UInt16
        else: return pl.UInt8

    @classmethod
    def _load_input_df(
        cls, df: INPUT_DF_T, columns: List[Tuple[str, Union[InputDataType, Tuple[InputDataType, str]]]],
        subject_id_col: Optional[str] = None, subject_ids_map: Optional[Dict[Any, int]] = None,
        subject_id_dtype: Optional[Any] = None,
        filter_on: Optional[Dict[str, Union[bool, List[Any]]]] = None,
    ) -> DF_T:
        """
        Loads an input dataframe into the format expected by the processing library.
        """
        if subject_id_col is None:
            if subject_ids_map is not None:
                raise ValueError("Must not set subject_ids_map if subject_id_col is not set")
            if subject_id_dtype is not None:
                raise ValueError("Must not set subject_id_dtype if subject_id_col is not set")
        else:
            if subject_ids_map is None:
                raise ValueError("Must set subject_ids_map if subject_id_col is set")
            if subject_id_dtype is None:
                raise ValueError("Must set subject_id_dtype if subject_id_col is set")

        match df:
            case Path() as fp:
                if fp.suffix == '.csv': df = pl.scan_csv(df, null_values='')
                elif fp.suffix == '.parquet': df = pl.scan_parquet(df)
                else: raise ValueError(f"Can't read dataframe from file of suffix {fp.suffix}")
            case pd.DataFrame(): df = pl.from_pandas(df, include_index=True).lazy()
            case pl.DataFrame(): df = df.lazy()
            case pl.LazyFrame(): pass
            case Query() as q:

                query = q.query
                if type(query) is not list: query = [query]
                out_query = []
                for qq in query:
                    if type(qq) is Path:
                        with open(qq, mode='r') as f: qq = f.read()
                    elif type(qq) is not str:
                        raise ValueError(f"{type(qq)} is an invalid query.")
                    out_query.append(qq)

                partition_on = subject_id_col if q.partition_on is None else q.partition_on
                partition_num = multiprocessing.cpu_count() if q.partition_num is None else q.partition_num

                df = pl.read_database(
                    query=q.query, connection_uri=q.connection_uri, partition_on=partition_on,
                    partition_num=partition_num, protocol=q.protocol,
                ).lazy()
            case _: raise TypeError(f"Input dataframe `df` is of invalid type {type(df)}!")

        col_exprs = []

        if filter_on: df = cls._filter_col_inclusion(df, filter_on)

        if subject_id_col is None:
            df = df.with_row_count('subject_id')
            col_exprs.append('subject_id')
        else:
            df = df.with_columns(pl.col(subject_id_col).cast(pl.Utf8).cast(pl.Categorical))
            df = cls._filter_col_inclusion(df, {subject_id_col: list(subject_ids_map.keys())})
            col_exprs.append(
                pl.col(subject_id_col).map_dict(subject_ids_map).cast(subject_id_dtype).alias('subject_id')
            )

        for in_col, out_dt in columns:
            match out_dt:
                case InputDataType.FLOAT: col_exprs.append(pl.col(in_col).cast(pl.Float32, strict=False))
                case InputDataType.CATEGORICAL:
                    col_exprs.append(pl.col(in_col).cast(pl.Utf8).cast(pl.Categorical))
                case InputDataType.BOOLEAN: col_exprs.append(pl.col(in_col).cast(pl.Boolean, strict=False))
                case InputDataType.TIMESTAMP: col_exprs.append(pl.col(in_col).cast(pl.Datetime, strict=False))
                case (InputDataType.TIMESTAMP, str() as ts_format):
                    col_exprs.append(pl.col(in_col).str.strptime(pl.Datetime, ts_format, strict=False))
                case _: raise ValueError(f"Invalid out data type {out_dt}!")

        return df.select(col_exprs)

    @classmethod
    def resolve_ts_col(cls, df: DF_T, ts_col: Union[str, List[str]], out_name: str = 'timestamp') -> DF_T:
        match ts_col:
            case list():
                ts_expr = pl.min(ts_col)
                ts_to_drop = [c for c in ts_col if c != out_name]
            case str():
                ts_expr = pl.col(ts_col)
                ts_to_drop = [ts_col] if ts_col != out_name else []

        return df.with_columns(ts_expr.alias(out_name)).drop(ts_to_drop)

    @classmethod
    def process_events_and_measurements_df(
        cls, df: DF_T, event_type: str, columns_schema: Dict[str, Tuple[str, InputDataType]],
    ) -> Tuple[DF_T, Optional[DF_T]]:
        """
        Performs the following pre-processing steps on an input events and measurements dataframe:
          1. Adds a categorical event type column with value `event_type`.
          2. Extracts and renames the columns present in `columns_schema`.
          3. Adds an integer `event_id` column.
          4. Splits the dataframe into an events dataframe, storing `event_id`, `subject_id`, `event_type`,
             and `timestamp`, and a `measurements` dataframe, storing `event_id` and all other data columns.
        """

        cols_select_exprs = [
            'timestamp', 'subject_id', 'event_id',
            pl.lit(event_type).cast(pl.Categorical).alias('event_type')
        ]
        for in_col, (out_col, _) in columns_schema.items():
            cols_select_exprs.append(pl.col(in_col).alias(out_col))

        df = df.filter(
            pl.col('timestamp').is_not_null() & pl.col('subject_id').is_not_null()
        ).with_row_count('event_id').select(
            cols_select_exprs
        )

        events_df = df.select('event_id', 'subject_id', 'timestamp', 'event_type')

        if len(df.columns) > 4: dynamic_measurements_df = df.drop('subject_id', 'timestamp', 'event_type')
        else: dynamic_measurements_df = None

        return events_df, dynamic_measurements_df

    @classmethod
    def split_range_events_df(cls, df: DF_T) -> Tuple[DF_T, DF_T, DF_T]:
        """
        Performs the following steps:
          1. Produces unified start and end timestamp columns representing the minimum of the passed start and end
             timestamps, respectively.
          2. Filters out records where the end timestamp is earlier than the start timestamp.
          3. Splits the dataframe into 3 events dataframes, all with only a single timestamp column, named
             `'timestamp'`:
             (a) An "EQ" dataframe, where start_ts_col == end_ts_col,
             (b) A "start" dataframe, with start events, and
             (c) An "end" dataframe, with end events.
        """

        df = df.filter(pl.col('start_time') <= pl.col('end_time'))

        eq_df = df.filter(pl.col('start_time') == pl.col('end_time'))
        ne_df = df.filter(pl.col('start_time') != pl.col('end_time'))

        st_col, end_col = pl.col('start_time').alias('timestamp'), pl.col('end_time').alias('timestamp')
        drop_cols = ['start_time', 'end_time']
        return (
            eq_df.with_columns(st_col).drop(drop_cols),
            ne_df.with_columns(st_col).drop(drop_cols),
            ne_df.with_columns(end_col).drop(drop_cols)
        )

    @classmethod
    def _inc_df_col(cls, df: DF_T, col: str, inc_by: int) -> DF_T:
        """
        Increments the values in a column by a given amount and returns a dataframe with the incremented
        column.
        """
        return df.with_columns(pl.col(col) + inc_by).collect()

    @classmethod
    def _concat_dfs(cls, dfs: List[DF_T]) -> DF_T:
        """ Concatenates a list of dataframes into a single dataframe. """
        return pl.concat(dfs, how='diagonal')

    @classmethod
    def _read_df(cls, fp: Path, **kwargs) -> DF_T: return pl.read_parquet(fp)

    @classmethod
    def _write_df(cls, df: DF_T, fp: Path, **kwargs): df.write_parquet(fp)

    def get_metadata_schema(self, config: MeasurementConfig) -> Dict[str, pl.DataType]:
        schema = {
            'value_type': self.METADATA_SCHEMA['value_type'],
        }

        if self.config.outlier_detector_config is not None:
            M = self._get_metadata_model(self.config.outlier_detector_config, for_fit=False)
            schema['outlier_model'] = self.METADATA_SCHEMA['outlier_model'](M.params_schema())
        if self.config.normalizer_config is not None:
            M = self._get_metadata_model(self.config.normalizer_config, for_fit=False)
            schema['normalizer'] = self.METADATA_SCHEMA['normalizer'](M.params_schema())

        metadata = config.measurement_metadata
        if metadata is None: return schema

        for col in (
            'drop_upper_bound', 'drop_lower_bound', 'censor_upper_bound', 'censor_lower_bound',
            'drop_upper_bound_inclusive', 'drop_lower_bound_inclusive'
        ):
            if col in metadata: schema[col] = self.METADATA_SCHEMA[col]

        return schema

    @staticmethod
    def drop_or_censor(
        col: pl.Expr,
        drop_lower_bound: Optional[pl.Expr] = None,
        drop_lower_bound_inclusive: Optional[pl.Expr] = None,
        drop_upper_bound: Optional[pl.Expr] = None,
        drop_upper_bound_inclusive: Optional[pl.Expr] = None,
        censor_lower_bound: Optional[pl.Expr] = None,
        censor_upper_bound: Optional[pl.Expr] = None,
        **ignored_kwargs
    ) -> pl.Expr:
        """
        Appropriately either drops (returns np.NaN) or censors (returns the censor value) the value `val`
        based on the bounds in `row`.

        TODO(mmd): could move this code to an outlier model in Preprocessing and have it be one that is
        pre-set in metadata.

        Args:
            `val` (`pl.Expr`): The value to drop, censor, or return unchanged.
            `drop_lower_bound` (`pl.Expr`):
              A lower bound such that if `val` is either below or at or below this level, `np.NaN`
              will be returned.
              If `None` or `np.NaN`, no bound will be applied.
            `drop_lower_bound_inclusive`:
              If `True`, returns `np.NaN` if `val <= row['drop_lower_bound']`. Else, returns
              `np.NaN` if `val < row['drop_lower_bound']`.
            `drop_upper_bound`:
              An upper bound such that if `val` is either above or at or above this level, `np.NaN`
              will be returned.
              If `None` or `np.NaN`, no bound will be applied.
            `drop_upper_bound_inclusive`:
              If `True`, returns `np.NaN` if `val >= row['drop_upper_bound']`. Else, returns
              `np.NaN` if `val > row['drop_upper_bound']`.
            `censor_lower_bound`:
              A lower bound such that if `val` is below this level but above `drop_lower_bound`,
              `censor_lower_bound` will be returned.
              If `None` or `np.NaN`, no bound will be applied.
            `censor_upper_bound`:
              An upper bound such that if `val` is above this level but below `drop_upper_bound`,
              `censor_upper_bound` will be returned.
              If `None` or `np.NaN`, no bound will be applied.

        """

        conditions = []

        if drop_lower_bound is not None:
            conditions.append(
                ((col < drop_lower_bound) | ((col == drop_lower_bound) & drop_lower_bound_inclusive), np.NaN)
            )

        if drop_upper_bound is not None:
            conditions.append(
                ((col > drop_upper_bound) | ((col == drop_upper_bound) & drop_upper_bound_inclusive), np.NaN)
            )

        if censor_lower_bound is not None:
            conditions.append((col < censor_lower_bound, censor_lower_bound))
        if censor_upper_bound is not None:
            conditions.append((col > censor_upper_bound, censor_upper_bound))

        if not conditions: return col

        expr = pl.when(conditions[0][0]).then(conditions[0][1])
        for cond, val in conditions[1:]: expr = expr.when(cond).then(val)
        return expr.otherwise(col)

    @staticmethod
    def _validate_id_col(id_col: pl.Series) -> Tuple[pl.Series, pl.datatypes.DataTypeClass]:
        """
        Validate the given ID column.

        Args:
            id_col (pl.Expr): The ID column to validate.

        Returns:
            pl.Expr: The validated ID column.

        Raises:
            AssertionError: If the ID column is not unique.
        """

        if not id_col.is_unique().all(): raise ValueError(f"ID column {id_col.name} is not unique!")
        match id_col.dtype:
            case pl.Float32 | pl.Float64:
                if not (id_col == id_col.round(0)).all() and (id_col >= 0).all():
                    raise ValueError(f"ID column {id_col.name} is not a non-negative integer type!")
            case pl.Int8 | pl.Int16 | pl.Int32 | pl.Int64:
                if not (id_col >= 0).all():
                    raise ValueError(f"ID column {id_col.name} is not a non-negative integer type!")
            case pl.UInt8 | pl.UInt16 | pl.UInt32 | pl.UInt64:
                pass
            case _:
                raise ValueError(f"ID column {id_col.name} is not a non-negative integer type!")

        max_val = id_col.max()
        dt = Dataset.get_smallest_valid_int_type(max_val)

        id_col = id_col.cast(dt)

        return id_col, dt

    def _validate_initial_df(
        self, source_df: Optional[DF_T], id_col_name: str,
        valid_temporality_type: TemporalityType,
        linked_id_cols: Optional[Dict[str, pl.datatypes.DataTypeClass]] = None
    ) -> Tuple[Optional[DF_T], pl.datatypes.DataTypeClass]:
        if source_df is None: return None, None

        if linked_id_cols:
            for id_col, id_col_dt in linked_id_cols.items():
                if id_col not in source_df: raise ValueError(f"Missing mandatory linkage col {id_col}")
                source_df = source_df.with_columns(pl.col(id_col).cast(id_col_dt))

        if id_col_name not in source_df: source_df = source_df.with_row_count(name=id_col_name)

        id_col, id_col_dt = self._validate_id_col(source_df.get_column(id_col_name))
        source_df = source_df.with_columns(id_col)

        for col, cfg in self.config.measurement_configs.items():
            if cfg.is_dropped: continue
            match cfg.modality:
                case DataModality.DROPPED: continue
                case DataModality.UNIVARIATE_REGRESSION: cat_col, val_col = None, col
                case DataModality.MULTIVARIATE_REGRESSION: cat_col, val_col = col, cfg.values_column
                case _: cat_col, val_col = col, None

            if cat_col is not None and cat_col in source_df:
                if cfg.temporality != valid_temporality_type:
                    raise ValueError(f"Column {cat_col} found in dataframe of wrong temporality")

                source_df = source_df.with_columns(pl.col(cat_col).cast(pl.Utf8).cast(pl.Categorical))

            if val_col is not None and val_col in source_df:
                if cfg.temporality != valid_temporality_type:
                    raise ValueError(f"Column {val_col} found in dataframe of wrong temporality")

                source_df = source_df.with_columns(pl.col(val_col).cast(pl.Float64))

        return source_df, id_col_dt

    def _validate_initial_dfs(
        self, subjects_df: Optional[DF_T], events_df: Optional[DF_T], dynamic_measurements_df: Optional[DF_T]
    ) -> Tuple[Optional[DF_T], Optional[DF_T], Optional[DF_T]]:
        """
        Validate and preprocess the given subjects, events, and dynamic_measurements dataframes.

        For each dataframe, this method checks for the presence of specific columns and unique IDs.
        It also casts certain columns to appropriate data types and performs necessary joins.

        Args:
            subjects_df (Optional[DF_T]):
                A dataframe containing subjects information, with an optional 'subject_id' column.
            events_df (Optional[DF_T]):
                A dataframe containing events information, with optional 'event_id', 'event_type', and
                'subject_id' columns.
            dynamic_measurements_df (Optional[DF_T]):
                A dataframe containing dynamic measurements information, with an optional
                'dynamic_measurement_id' column and other measurement-specific columns.

        Returns:
            Tuple[Optional[DF_T], Optional[DF_T], Optional[DF_T]]:
                A tuple containing the preprocessed subjects, events, and dynamic_measurements dataframes.

        Raises:
            ValuesError: If any of the required columns are missing or invalid.
        """
        subjects_df, subjects_id_type = self._validate_initial_df(
            subjects_df, 'subject_id', TemporalityType.STATIC
        )
        events_df, event_id_type = self._validate_initial_df(
            events_df, 'event_id', TemporalityType.FUNCTIONAL_TIME_DEPENDENT,
            {'subject_id': subjects_id_type} if subjects_df is not None else None,
        )
        if events_df is not None:
            if 'event_type' not in events_df: raise ValueError("Missing event_type column!")
            events_df = events_df.with_columns(pl.col('event_type').cast(pl.Categorical))

            if 'timestamp' not in events_df or events_df['timestamp'].dtype != pl.Datetime:
                raise ValueError("Malformed timestamp column!")

        if dynamic_measurements_df is not None:
            linked_ids = {}
            if events_df is not None:
                linked_ids['event_id'] = event_id_type

            dynamic_measurements_df, dynamic_measurement_id_types = self._validate_initial_df(
                dynamic_measurements_df, 'measurement_id', TemporalityType.DYNAMIC, linked_ids
            )

        return subjects_df, events_df, dynamic_measurements_df

    @TimeableMixin.TimeAs
    def sort_events(self):
        """Sorts events by subject ID and timestamp in ascending order."""
        self.events_df = self.events_df.sort('subject_id', 'timestamp', descending=False)

    @TimeableMixin.TimeAs
    def agg_by_time(self):
        """
        Aggregates the events_df by subject_id, timestamp, combining event_types into grouped categories,
        tracking all associated metadata. Note that no numerical aggregation (e.g., mean, etc.) happens here;
        all data is retained, and only dynamic measurement event IDs are updated.
        """

        event_id_dt = self.events_df['event_id'].dtype

        if self.config.agg_by_time_scale is None:
            grouped = self.events_df.groupby(['subject_id', 'timestamp'], maintain_order=True)
        else:
            grouped = self.events_df.sort(
                ['subject_id', 'timestamp'], descending=False
            ).groupby_dynamic(
                'timestamp',
                every=self.config.agg_by_time_scale,
                truncate=True,
                closed='left',
                start_by='datapoint',
                by='subject_id',
            )

        grouped = grouped.agg(
            pl.col('event_type').unique().sort(),
            pl.col('event_id').unique().alias('old_event_id')
        ).sort(
            'subject_id', 'timestamp', descending=False
        ).with_row_count(
            'event_id'
        ).with_columns(
            pl.col('event_id').cast(event_id_dt),
            pl.col(
                'event_type'
            ).arr.eval(
                pl.col('').cast(pl.Utf8)
            ).arr.join('&').cast(pl.Categorical).alias('event_type')
        )

        new_to_old_set = grouped[['event_id', 'old_event_id']].explode('old_event_id')

        self.events_df = grouped.drop('old_event_id')

        self.dynamic_measurements_df = self.dynamic_measurements_df.rename(
            {'event_id': 'old_event_id'}
        ).join(
            new_to_old_set, on='old_event_id', how='left'
        ).drop('old_event_id')

    @TimeableMixin.TimeAs
    def agg_by_time_type(self):
        """
        Aggregates the events_df by subject_id, timestamp, and event_type, tracking all associated metadata.
        Note that no numerical aggregation (e.g., mean, etc.) happens here; duplicate entries will both be
        captured in the output metadata object.
        """

        grouped = self.events_df.groupby(
            ['subject_id', 'timestamp', 'event_type'], maintain_order=True
        ).all().with_columns(pl.col('event_id').arr.unique())

        grouped = grouped.rename({'event_id': 'old_event_id'}).sort(
            'subject_id', 'timestamp', descending=False
        )
        grouped = grouped.with_row_count('event_id')
        grouped = grouped.with_columns(self._validate_id_col(grouped['event_id'])[0])

        new_to_old_set = grouped[['event_id', 'old_event_id']].explode('old_event_id')

        self.events_df = grouped.drop('old_event_id')

        self.dynamic_measurements_df = self.dynamic_measurements_df.rename(
            {'event_id': 'old_event_id'}
        ).join(
            new_to_old_set, on='old_event_id', how='left'
        ).drop('old_event_id')

    def _update_subject_event_properties(self):
        if self.events_df is not None:
            self.event_types = self.events_df.get_column(
                'event_type'
            ).value_counts(sort=True).get_column('event_type').to_list()

            n_events_pd = self.events_df.get_column('subject_id').value_counts(sort=False).to_pandas()
            self.n_events_per_subject = n_events_pd.set_index('subject_id')['counts'].to_dict()
            self.subject_ids = set(self.n_events_per_subject.keys())

        if self.subjects_df is not None:
            subjects_with_no_events = (
                set(self.subjects_df.get_column('subject_id').to_list()) - self.subject_ids
            )
            for sid in subjects_with_no_events: self.n_events_per_subject[sid] = 0
            self.subject_ids.update(subjects_with_no_events)

    @classmethod
    def _filter_col_inclusion(
        cls, df: DF_T, col_inclusion_targets: Dict[str, Union[bool, Sequence[Any]]]
    ) -> DF_T:
        filter_exprs = []
        for col, incl_targets in col_inclusion_targets.items():
            match incl_targets:
                case True: filter_exprs.append(pl.col(col).is_not_null())
                case False: filter_exprs.append(pl.col(col).is_null())
                case _: filter_exprs.append(pl.col(col).is_in(list(incl_targets)))

        return df.filter(pl.all(filter_exprs))

    @TimeableMixin.TimeAs
    def _get_valid_event_types(self) -> Dict[str, List[str]]:
        measures = []
        for measure, config in self.config.measurement_configs.items():
            if (
                (config.is_dropped) or
                (config.temporality != TemporalityType.DYNAMIC) or
                (config.present_in_event_types is not None) or
                (measure not in self.dynamic_measurements_df.columns)
            ): continue
            measures.append(measure)

        if not measures: return {}

        event_type_cnts = self._filter_measurements_df(
            split='train'
        ).join(
            self.train_events_df.select('event_id', 'event_type'), on='event_id'
        ).groupby('event_type').agg(
            *[pl.col(c).drop_nulls().count() for c in measures]
        )

        out = {}
        for measure in measures:
            out[measure] = event_type_cnts.filter(pl.col(measure) > 0)['event_type'].to_list()
        return out

    @TimeableMixin.TimeAs
    def add_time_dependent_measurements(self):
        exprs = []
        join_cols = set()
        for col, cfg in self.config.measurement_configs.items():
            if cfg.temporality != TemporalityType.FUNCTIONAL_TIME_DEPENDENT: continue
            fn = cfg.functor
            join_cols.update(fn.link_static_cols)
            exprs.append(fn.pl_expr().alias(col))

        join_cols = list(join_cols)

        if join_cols:
            self.events_df = self.events_df.join(
                self.subjects_df.select('subject_id', *join_cols), on='subject_id'
            ).with_columns(
                exprs
            ).drop(join_cols)
        else:
            self.events_df = self.events_df.with_columns(exprs)

    @TimeableMixin.TimeAs
    def _prep_numerical_source(
        self, measure: str, config: MeasurementConfig, source_df: DF_T
    ) -> Tuple[DF_T, str, str, str, pl.DataFrame]:
        metadata = config.measurement_metadata

        metadata_schema = self.get_metadata_schema(config)

        match config.modality:
            case DataModality.UNIVARIATE_REGRESSION:
                key_col = 'const_key'
                val_col = measure
                metadata_as_polars = pl.DataFrame(
                    {key_col: [measure], **{c: [v] for c, v in metadata.items()}}
                )
                source_df = source_df.with_columns(pl.lit(measure).cast(pl.Categorical).alias(key_col))
            case DataModality.MULTIVARIATE_REGRESSION:
                key_col = measure
                val_col = config.values_column
                metadata_as_polars = pl.from_pandas(metadata, include_index=True)
            case _:
                raise ValueError(f"Called _pre_numerical_source on {config.modality} measure {measure}!")

        if 'outlier_model' in metadata_as_polars and len(metadata_as_polars.drop_nulls('outlier_model')) == 0:
            metadata_as_polars = metadata_as_polars.with_columns(pl.lit(None).alias('outlier_model'))
        if 'normalizer' in metadata_as_polars and len(metadata_as_polars.drop_nulls('normalizer')) == 0:
            metadata_as_polars = metadata_as_polars.with_columns(pl.lit(None).alias('normalizer'))

        metadata_as_polars = metadata_as_polars.with_columns(
            pl.col(key_col).cast(pl.Categorical),
            **{k: pl.col(k).cast(v) for k, v in metadata_schema.items()}
        )

        source_df = source_df.join(metadata_as_polars, on=key_col, how='left')
        return source_df, key_col, val_col, f"{measure}_is_inlier", metadata_as_polars

    def _total_possible_and_observed(
        self, measure: str, config: MeasurementConfig, source_df: DF_T
    ) -> Tuple[int, int]:
        agg_by_col = pl.col('event_id') if config.temporality == TemporalityType.DYNAMIC else None

        if agg_by_col is None:
            num_possible = len(source_df)
            num_non_null = len(source_df.drop_nulls(measure))
        else:
            num_possible = source_df.select(pl.col('event_id').n_unique()).item()
            num_non_null = source_df.select(
                pl.col('event_id').filter(pl.col(measure).is_not_null()).n_unique()
            ).item()
        return num_possible, num_non_null

    @TimeableMixin.TimeAs
    def _add_inferred_val_types(
        self, measurement_metadata: DF_T, source_df: DF_T, vocab_keys_col: str, vals_col: str,
    ) -> DF_T:
        """
        Infers the appropriate type of the passed metadata column values. Performs the following steps:
            1. Determines if the column should be dropped for having too few measurements.
            2. Determines if the column actually contains integral, not floating point values.
            3. Determines if the column should be partially or fully re-categorized as a categorical column.

        Args:
            `vals` (`pd.Series`): The values to be pre-processed.
                The total number of column observations that were observed for this metadata column (_not_
                just this key!)

        Returns: The appropriate `NumericDataModalitySubtype` for the values.
        """

        vals_col = pl.col(vals_col)

        if 'value_type' in measurement_metadata:
            missing_val_types = measurement_metadata.filter(pl.col('value_type').is_null())[vocab_keys_col]
            for_val_type_inference = source_df.filter(
                (~pl.col(vocab_keys_col).is_in(measurement_metadata[vocab_keys_col])) |
                pl.col(vocab_keys_col).is_in(missing_val_types)
            )
        else:
            for_val_type_inference = source_df

        # a. Convert to integeres where appropriate.
        if self.config.min_true_float_frequency is not None:
            is_int_expr = (
                (vals_col == vals_col.round(0)).mean() > (1 - self.config.min_true_float_frequency)
            ).cast(pl.Boolean).alias('is_int')
            int_keys = for_val_type_inference.groupby(vocab_keys_col).agg(is_int_expr)

            measurement_metadata = measurement_metadata.join(int_keys, on=vocab_keys_col, how='outer')

            key_is_int = pl.col(vocab_keys_col).is_in(int_keys.filter('is_int')[vocab_keys_col])
            for_val_type_inference = for_val_type_inference.with_columns(
                pl.when(key_is_int).then(vals_col.round(0)).otherwise(vals_col)
            )
        else:
            measurement_metadata = measurement_metadata.with_columns(pl.lit(False).alias('is_int'))

        # b. Drop if only has a single observed numerical value.
        dropped_keys = for_val_type_inference.groupby(vocab_keys_col).agg(
            (vals_col.n_unique() == 1).cast(pl.Boolean).alias('should_drop')
        ).filter('should_drop')
        keep_key_expr = ~pl.col(vocab_keys_col).is_in(dropped_keys[vocab_keys_col])
        measurement_metadata = measurement_metadata.with_columns(
            pl.when(
                keep_key_expr
            ).then(
                pl.col('value_type')
            ).otherwise(
                pl.lit(NumericDataModalitySubtype.DROPPED)
            ).alias('value_type')
        )
        for_val_type_inference = for_val_type_inference.filter(keep_key_expr)

        # c. Convert to categorical if too few unique observations are seen.
        if self.config.min_unique_numerical_observations is not None:
            is_cat_expr = lt_count_or_proportion(
                vals_col.n_unique(), self.config.min_unique_numerical_observations, vals_col.len()
            ).cast(pl.Boolean).alias('is_categorical')

            categorical_keys = for_val_type_inference.groupby(vocab_keys_col).agg(
                is_cat_expr
            )

            measurement_metadata = measurement_metadata.join(categorical_keys, on=vocab_keys_col, how='outer')
        else:
            measurement_metadata = measurement_metadata.with_columns(pl.lit(False).alias('is_categorical'))

        inferred_value_type = pl.when(
            pl.col('is_int') & pl.col('is_categorical')
        ).then(
            pl.lit(NumericDataModalitySubtype.CATEGORICAL_INTEGER)
        ).when(
            pl.col('is_categorical')
        ).then(
            pl.lit(NumericDataModalitySubtype.CATEGORICAL_FLOAT)
        ).when(
            pl.col('is_int')
        ).then(
            pl.lit(NumericDataModalitySubtype.INTEGER)
        ).otherwise(
            pl.lit(NumericDataModalitySubtype.FLOAT)
        )

        return measurement_metadata.with_columns(
            pl.coalesce(['value_type', inferred_value_type]).alias('value_type')
        ).drop(['is_int', 'is_categorical'])

    @TimeableMixin.TimeAs
    def _fit_measurement_metadata(
        self, measure: str, config: MeasurementConfig, source_df: DF_T
    ) -> pd.DataFrame:
        """
        Pre-processes the numerical measurement `measure`.

        Performs the following steps:
            1. Drops any vocabulary elements that would be removed for insufficiently frequent occurrences.
            2. Eliminates hard outliers and performs censoring via specified config.
            3. Infers value types as needed and converts values to the appropriate types.
            4. Learns an outlier detector as directed.
            5. Learns a normalizer model as directed.

        Args:
            `measure` (`str`): The name of the measurement.
            `config` (`MeasurementConfig`): The configuration object governing this measure.
            `source_df`
        """

        source_df, vocab_keys_col, vals_col, _, measurement_metadata = self._prep_numerical_source(
            measure, config, source_df
        )
        # 1. Determines which vocab elements should be dropped due to insufficient occurrences.
        if self.config.min_valid_vocab_element_observations is not None:
            if config.temporality == TemporalityType.DYNAMIC:
                num_possible = source_df.select(pl.col('event_id').n_unique()).item()
                num_non_null = pl.col('event_id').filter(pl.col(vocab_keys_col).is_not_null()).n_unique()
            else:
                num_possible = len(source_df)
                num_non_null = pl.col(vocab_keys_col).drop_nulls().len()

            should_drop_expr = lt_count_or_proportion(
                num_non_null, self.config.min_valid_vocab_element_observations, num_possible
            ).cast(pl.Boolean)

            dropped_keys = source_df.groupby(vocab_keys_col).agg(
                should_drop_expr.alias('should_drop')
            ).filter('should_drop').with_columns(
                pl.lit(NumericDataModalitySubtype.DROPPED).alias('value_type')
            ).drop('should_drop')

            measurement_metadata = measurement_metadata.join(
                dropped_keys, on=vocab_keys_col, how='outer', suffix='_right',
            ).with_columns(
                pl.coalesce(['value_type', 'value_type_right']).alias('value_type')
            ).drop('value_type_right')
            source_df = source_df.filter(~pl.col(vocab_keys_col).is_in(dropped_keys[vocab_keys_col]))

            if len(source_df) == 0:
                measurement_metadata = measurement_metadata.to_pandas()
                measurement_metadata = measurement_metadata.set_index(vocab_keys_col)

                if config.modality == DataModality.UNIVARIATE_REGRESSION:
                    assert len(measurement_metadata) == 1
                    return measurement_metadata.loc[measure]
                else: return measurement_metadata

        source_df = source_df.drop_nulls([vocab_keys_col, vals_col]).filter(pl.col(vals_col).is_not_nan())

        # 2. Eliminates hard outliers and performs censoring via specified config.
        bound_cols = {}
        for col in (
            'drop_upper_bound', 'drop_upper_bound_inclusive', 'drop_lower_bound',
            'drop_lower_bound_inclusive', 'censor_lower_bound', 'censor_upper_bound',
        ):
            if col in source_df: bound_cols[col] = pl.col(col)

        if bound_cols:
            source_df = source_df.with_columns(
                self.drop_or_censor(pl.col(vals_col), **bound_cols).alias(vals_col)
            )

        source_df = source_df.filter(pl.col(vals_col).is_not_nan())
        if len(source_df) == 0: return config.measurement_metadata

        # 3. Infer the value type and convert where necessary.
        measurement_metadata = self._add_inferred_val_types(
            measurement_metadata, source_df, vocab_keys_col, vals_col
        )

        source_df = source_df.update(
            measurement_metadata.select(vocab_keys_col, 'value_type'), on=vocab_keys_col
        ).with_columns(
            pl.when(
                pl.col('value_type') == NumericDataModalitySubtype.INTEGER
            ).then(
                pl.col(vals_col).round(0)
            ).when(
                pl.col('value_type') == NumericDataModalitySubtype.FLOAT
            ).then(
                pl.col(vals_col)
            ).otherwise(
                None
            ).alias(vals_col)
        ).drop_nulls(vals_col).filter(pl.col(vals_col).is_not_nan())

        # 4. Infer outlier detector and normalizer parameters.
        if self.config.outlier_detector_config is not None:
            with self._time_as('fit_outlier_detector'):
                outlier_config, M = self._get_metadata_model(
                    self.config.outlier_detector_config, for_fit=True
                )
                outlier_model_params = source_df.groupby(vocab_keys_col).agg(
                    M.fit_from_polars(pl.col(vals_col)).alias('outlier_model')
                )

                measurement_metadata = measurement_metadata.with_columns(
                    pl.col('outlier_model').cast(outlier_model_params['outlier_model'].dtype)
                )
                source_df = source_df.with_columns(
                    pl.col('outlier_model').cast(outlier_model_params['outlier_model'].dtype)
                )

                measurement_metadata = measurement_metadata.update(outlier_model_params, on=vocab_keys_col)
                source_df = source_df.update(
                    measurement_metadata.select(vocab_keys_col, 'outlier_model'), on=vocab_keys_col
                )

                is_inlier = ~M.predict_from_polars(pl.col(vals_col), pl.col('outlier_model'))
                source_df = source_df.filter(is_inlier)

        # 5. Fit a normalizer model.
        if self.config.normalizer_config is not None:
            with self._time_as('fit_normalizer'):
                normalizer_config, M = self._get_metadata_model(
                    self.config.normalizer_config, for_fit=True
                )
                normalizer_params = source_df.groupby(vocab_keys_col).agg(
                    M.fit_from_polars(pl.col(vals_col)).alias('normalizer')
                )
                measurement_metadata = measurement_metadata.with_columns(
                    pl.col('normalizer').cast(normalizer_params['normalizer'].dtype)
                )
                measurement_metadata = measurement_metadata.update(normalizer_params, on=vocab_keys_col)

        # 6. Convert to the appropriate type and return.
        measurement_metadata = measurement_metadata.to_pandas()
        measurement_metadata = measurement_metadata.set_index(vocab_keys_col)

        if config.modality == DataModality.UNIVARIATE_REGRESSION:
            assert len(measurement_metadata) == 1
            return measurement_metadata.loc[measure]
        else: return measurement_metadata

    @TimeableMixin.TimeAs
    def _fit_vocabulary(self, measure: str, config: MeasurementConfig, source_df: DF_T) -> Vocabulary:
        match config.modality:
            case DataModality.MULTIVARIATE_REGRESSION:
                val_types = pl.from_pandas(
                    config.measurement_metadata[['value_type']], include_index=True
                ).with_columns(
                    pl.col('value_type').cast(pl.Categorical), pl.col(measure).cast(pl.Categorical)
                )
                observations = source_df.join(val_types, on=measure).with_columns(
                    pl.when(
                        pl.col('value_type') == NumericDataModalitySubtype.CATEGORICAL_INTEGER
                    ).then(
                        pl.col(measure).cast(pl.Utf8) +
                        "__EQ_" +
                        pl.col(config.values_column).round(0).cast(int).cast(pl.Utf8)
                    ).when(
                        pl.col('value_type') == NumericDataModalitySubtype.CATEGORICAL_FLOAT
                    ).then(
                        pl.col(measure).cast(pl.Utf8) + "__EQ_" + pl.col(config.values_column).cast(pl.Utf8)
                    ).otherwise(
                        pl.col(measure)
                    ).alias(measure)
                ).get_column(measure)
            case DataModality.UNIVARIATE_REGRESSION:
                if config.measurement_metadata.value_type == NumericDataModalitySubtype.CATEGORICAL_INTEGER:
                    observations = source_df.with_columns(
                        (f"{measure}__EQ_" + pl.col(measure).round(0).cast(int).cast(pl.Utf8)).alias(measure)
                    ).get_column(measure)
                elif config.measurement_metadata.value_type == NumericDataModalitySubtype.CATEGORICAL_FLOAT:
                    observations = source_df.with_columns(
                        (f"{measure}__EQ_" + pl.col(measure).cast(pl.Utf8)).alias(measure)
                    ).get_column(measure)
                else: return
            case _: observations = source_df.get_column(measure)

        # 1. Set the overall observation frequency for the column.
        observations = observations.drop_nulls()
        N = len(observations)
        if N == 0: return None

        # 3. Fit metadata vocabularies on the trianing set.
        if config.vocabulary is None:
            try:
                value_counts = observations.value_counts()
                vocab_elements = value_counts.get_column(measure).to_list()
                el_counts = value_counts.get_column('counts')
                return Vocabulary(vocabulary=vocab_elements, obs_frequencies=el_counts)
            except AssertionError as e:
                raise AssertionError(f"Failed to build vocabulary for {measure}") from e

    @TimeableMixin.TimeAs
    def _transform_numerical_measurement(
        self, measure: str, config: MeasurementConfig, source_df: DF_T
    ) -> DF_T:
        """
        Transforms the numerical measurement `measure` according to config `config`.

        Performs the following steps:
            1. Transforms keys to categorical representations for categorical keys.
            2. Eliminates any values associated with dropped or categorical keys.
            3. Eliminates hard outliers and performs censoring via specified config.
            4. Converts values to desired types.
            5. Adds inlier/outlier indices and remove learned outliers.
            6. Normalizes values.

        Args:
            `measure` (`str`): The column name of the governing measurement to transform.
            `config` (`MeasurementConfig`): The configuration object governing this measure.
            `source_df` (`DF_T`): The dataframe object containing the measure to be transformed.
        """

        source_df, keys_col_name, vals_col_name, inliers_col_name, _ = self._prep_numerical_source(
            measure, config, source_df
        )
        keys_col = pl.col(keys_col_name)
        vals_col = pl.col(vals_col_name)

        cols_to_drop_at_end = []
        for col in config.measurement_metadata:
            if col != measure and col in source_df: cols_to_drop_at_end.append(col)

        bound_cols = {}
        for col in (
            'drop_upper_bound', 'drop_upper_bound_inclusive', 'drop_lower_bound',
            'drop_lower_bound_inclusive', 'censor_lower_bound', 'censor_upper_bound',
        ):
            if col in source_df: bound_cols[col] = pl.col(col)

        if bound_cols:
            vals_col = self.drop_or_censor(vals_col, **bound_cols)

        value_type = pl.col('value_type')
        keys_col = pl.when(
            value_type == NumericDataModalitySubtype.DROPPED
        ).then(
            keys_col
        ).when(
            value_type == NumericDataModalitySubtype.CATEGORICAL_INTEGER
        ).then(
            keys_col + "__EQ_" + vals_col.round(0).fill_nan(-1).cast(pl.Int64).cast(pl.Utf8)
        ).when(
            value_type == NumericDataModalitySubtype.CATEGORICAL_FLOAT
        ).then(
            keys_col + "__EQ_" + vals_col.cast(pl.Utf8)
        ).otherwise(keys_col).alias(keys_col_name)

        vals_col = pl.when(
            value_type.is_in([
                NumericDataModalitySubtype.DROPPED, NumericDataModalitySubtype.CATEGORICAL_INTEGER,
                NumericDataModalitySubtype.CATEGORICAL_FLOAT,
            ])
        ).then(
            np.NaN
        ).when(
            value_type == NumericDataModalitySubtype.INTEGER
        ).then(
            vals_col.round(0)
        ).otherwise(vals_col).alias(vals_col_name)

        source_df = source_df.with_columns(keys_col, vals_col)

        null_idx = keys_col.is_null() | vals_col.is_null() | vals_col.is_nan()

        null_source = source_df.filter(null_idx)
        present_source = source_df.filter(~null_idx)

        if len(present_source) == 0:
            if self.config.outlier_detector_config is not None:
                null_source = null_source.with_columns(pl.lit(None).cast(pl.Boolean).alias(inliers_col_name))
            return null_source.drop(cols_to_drop_at_end)

        # 5. Add inlier/outlier indices and remove learned outliers.
        if self.config.outlier_detector_config is not None:
            M = self._get_metadata_model(self.config.outlier_detector_config, for_fit=False)

            inliers_col = ~M.predict_from_polars(vals_col, pl.col('outlier_model')).alias(inliers_col_name)
            vals_col = pl.when(inliers_col).then(vals_col).otherwise(np.NaN)

            present_source = present_source.with_columns(inliers_col, vals_col)
            null_source = null_source.with_columns(pl.lit(None).cast(pl.Boolean).alias(inliers_col_name))

            new_nulls = present_source.filter(~pl.col(inliers_col_name))
            null_source = null_source.vstack(new_nulls)
            present_source = present_source.filter(inliers_col_name)

        if len(present_source) == 0:
            return null_source.drop(cols_to_drop_at_end)

        # 6. Normalize values.
        if self.config.normalizer_config is not None:
            M = self._get_metadata_model(self.config.normalizer_config, for_fit=False)

            vals_col = M.predict_from_polars(vals_col, pl.col('normalizer'))
            present_source = present_source.with_columns(vals_col)

        source_df = present_source.vstack(null_source)

        return source_df.drop((cols_to_drop_at_end))

    @TimeableMixin.TimeAs
    def _transform_categorical_measurement(
        self, measure: str, config: MeasurementConfig, source_df: DF_T
    ) -> DF_T:
        """
        Transforms the categorical measurement `measure` according to config `config`.

        Performs the following steps:
            1. Converts the elements to categorical column types according to the learned vocabularies.

        Args:
            `measure` (`str`): The column name of the governing measurement to transform.
            `config` (`MeasurementConfig`): The configuration object governing this measure.
            `source_df` (`DF_T`): The dataframe object containing the measure to be transformed.
        """

        if (
            (config.modality == DataModality.UNIVARIATE_REGRESSION) and
            (config.measurement_metadata.value_type not in (
                NumericDataModalitySubtype.CATEGORICAL_INTEGER,
                NumericDataModalitySubtype.CATEGORICAL_FLOAT,
            ))
        ): return source_df

        transform_expr = []
        if config.modality == DataModality.MULTIVARIATE_REGRESSION:
            transform_expr.append(pl.when(
                ~pl.col(measure).is_in(list(config.vocabulary.vocab_set))
            ).then(
                np.NaN
            ).otherwise(
                pl.col(config.values_column)
            ).alias(config.values_column))
            vocab_el_col = pl.col(measure)
        elif config.modality == DataModality.UNIVARIATE_REGRESSION:
            vocab_el_col = pl.col('const_key')
        else: vocab_el_col = pl.col(measure)

        transform_expr.append(
            pl.when(
                vocab_el_col.is_null()
            ).then(
                None
            ).when(
                ~vocab_el_col.is_in(list(config.vocabulary.vocab_set))
            ).then(
                'UNK'
            ).otherwise(
                vocab_el_col
            ).cast(
                pl.Categorical
            ).alias(measure)
        )

        return source_df.with_columns(transform_expr)

    @TimeableMixin.TimeAs
    def update_attr_df(self, attr: str, id_col: str, df: DF_T, cols_to_update: List[str]):
        """Updates the attribute `attr` with the dataframe `df`."""
        old_df = getattr(self, attr)

        old_df = old_df.with_columns(**{c: pl.lit(None).cast(df[c].dtype) for c in cols_to_update})
        new_df = df.select(id_col, *cols_to_update)

        setattr(self, attr, old_df.update(new_df, on=id_col))

    def melt_df(self, source_df: DF_T, id_cols: Sequence[str], measures: List[str]) -> pl.Expr:
        struct_exprs = []
        total_vocab_size = self.vocabulary_config.total_vocab_size
        idx_dt = self.get_smallest_valid_int_type(total_vocab_size)

        for m in measures:
            if m == 'event_type':
                cfg = None
                modality = DataModality.SINGLE_LABEL_CLASSIFICATION
            else:
                cfg = self.measurement_configs[m]
                modality = cfg.modality

            if m in self.measurement_vocabs:
                idx_present_expr = pl.col(m).is_not_null() & pl.col(m).is_in(self.measurement_vocabs[m])
                idx_value_expr = pl.col(m).map_dict(self.unified_vocabulary_idxmap[m], return_dtype=idx_dt)
            else:
                idx_present_expr = pl.col(m).is_not_null()
                idx_value_expr = pl.lit(self.unified_vocabulary_idxmap[m][m]).cast(idx_dt)

            idx_present_expr = idx_present_expr.cast(pl.Boolean).alias('present')
            idx_value_expr = idx_value_expr.alias('index')

            if (
                (modality == DataModality.UNIVARIATE_REGRESSION) and
                (cfg.measurement_metadata.value_type in (
                    NumericDataModalitySubtype.FLOAT, NumericDataModalitySubtype.INTEGER
                ))
            ): val_expr = pl.col(m)
            elif modality == DataModality.MULTIVARIATE_REGRESSION: val_expr = pl.col(cfg.values_column)
            else: val_expr = pl.lit(None).cast(pl.Float64)

            struct_exprs.append(
                pl.struct([idx_present_expr, idx_value_expr, val_expr.alias('value')]).alias(m)
            )

        measurements_idx_dt = self.get_smallest_valid_int_type(len(self.unified_measurements_idxmap))
        return source_df.select(
            *id_cols, *struct_exprs
        ).melt(
            id_vars=id_cols, value_vars=measures, variable_name='measurement', value_name='value',
        ).filter(
            pl.col('value').struct.field('present')
        ).select(
            *id_cols,
            pl.col('measurement').map_dict(
                self.unified_measurements_idxmap
            ).cast(measurements_idx_dt).alias('measurement_index'),
            pl.col('value').struct.field('index').alias('index'),
            pl.col('value').struct.field('value').alias('value'),
        )

    def build_DL_cached_representation(
        self, subject_ids: Optional[List[int]] = None, do_sort_outputs: bool = False
    ) -> DF_T:
        """
        Produces a format with the below syntax:

        ```
        subject_id | start_time | batched_representation
        1          | 2019-01-01 | batch_1,
        ...

        Batch Representaiton:
          N = number of time points
          M = maximum number of dynamic measurements at any time point
          K = number of static measurements
        batch_1 = {
          'time': [...] float, (N,), minutes since start_time of event. No missing values.
          'dynamic_indices': [[...]] int, (N, M), indices of dynamic measurements. 0 Iff missing.
          'dynamic_values': [[...]] float, (N, M), values of dynamic measurements. 0 If missing.
          'dynamic_measurement_indices': [[...]] int, (N, M), indices of dynamic measurements. 0 Iff missing.
          'static_indices': [...] int, (K,), indices of static measurements. No missing values.
          'static_measurement_indices': [...] int, (K,), indices of static measurements. No missing values.
        ```
        """
        # Identify the measurements sourced from each dataframe:
        subject_measures, event_measures, dynamic_measures = [], ['event_type'], []
        for m in self.unified_measurements_vocab[1:]:
            temporality = self.measurement_configs[m].temporality
            match temporality:
                case TemporalityType.STATIC: subject_measures.append(m)
                case TemporalityType.FUNCTIONAL_TIME_DEPENDENT: event_measures.append(m)
                case TemporalityType.DYNAMIC: dynamic_measures.append(m)
                case _: raise ValueError(f'Unknown temporality type {temporality} for {m}')

        # 1. Process subject data into the right format.
        if subject_ids:
            subjects_df = self._filter_col_inclusion(self.subjects_df, {'subject_id': subject_ids})
        else:
            subjects_df = self.subjects_df

        static_data = self.melt_df(subjects_df, ['subject_id'], subject_measures).groupby(
            'subject_id'
        ).agg(
            pl.col('measurement_index').alias('static_measurement_indices'),
            pl.col('index').alias('static_indices'),
        )

        # 2. Process event data into the right format.
        if subject_ids:
            events_df = self._filter_col_inclusion(self.events_df, {'subject_id': subject_ids})
            event_ids = list(events_df['event_id'])
        else:
            events_df = self.events_df
            event_ids = None
        event_data = self.melt_df(events_df, ['subject_id', 'timestamp', 'event_id'], event_measures)

        # 3. Process measurement data into the right base format:
        if event_ids:
            dynamic_measurements_df = self._filter_col_inclusion(
                self.dynamic_measurements_df, {'event_id': event_ids}
            )
        else:
            dynamic_measurements_df = self.dynamic_measurements_df

        dynamic_ids = ['event_id', 'measurement_id'] if do_sort_outputs else ['event_id']
        dynamic_data = self.melt_df(dynamic_measurements_df, dynamic_ids, dynamic_measures)

        if do_sort_outputs: dynamic_data = dynamic_data.sort('event_id', 'measurement_id')

        # 4. Join dynamic and event data.

        event_data = pl.concat([event_data, dynamic_data], how='diagonal')
        event_data = event_data.groupby('event_id').agg(
            pl.col('timestamp').drop_nulls().first().alias('timestamp'),
            pl.col('subject_id').drop_nulls().first().alias('subject_id'),
            pl.col('measurement_index').alias('dynamic_measurement_indices'),
            pl.col('index').alias('dynamic_indices'),
            pl.col('value').alias('dynamic_values'),
        ).sort('subject_id', 'timestamp').groupby('subject_id').agg(
            pl.col('timestamp').first().alias('start_time'),
            ((pl.col('timestamp') - pl.col('timestamp').min()).dt.nanoseconds() / (1e9 * 60)).alias('time'),
            pl.col('dynamic_measurement_indices'),
            pl.col('dynamic_indices'),
            pl.col('dynamic_values')
        )

        out = static_data.join(event_data, on='subject_id', how='outer')
        if do_sort_outputs: out = out.sort('subject_id')

        return out

    def denormalize(self, events_df: DF_T, col: str) -> DF_T:
        if self.config.normalizer_config is None: return events_df
        elif self.config.normalizer_config['cls'] != 'standard_scaler':
            raise ValueError(f"De-normalizing from {self.config.normalizer_config} not yet supported!")

        config = self.measurement_configs[col]
        if config.modality != DataModality.UNIVARIATE_REGRESSION:
            raise ValueError(f"De-normalizing {config.modality} is not currently supported.")

        normalizer_params = config.measurement_metadata.normalizer
        return events_df.with_columns(
            ((pl.col(col)*normalizer_params['std_']) + normalizer_params['mean_']).alias(col)
        )