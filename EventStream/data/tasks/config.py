MEASUREMENT_NAME = str

@dataclasses.dataclass
class MeasurementPredicate:
    vocab_el: str
    value: list[float | None] | None | bool

@dataclasses.dataclass
class EventPredicate:
    has: dict[MEASUREMENT_NAME, MeasurementPredicate | list[MeasurementPredicate]]
    lacks: dict[MEASUREMENT_NAME, MeasurementPredicate | list[MeasurementPredicate]]

@dataclasses.dataclass
class Window:
    start: str
    end: str
    includes:
    excludes:

@dataclasses.dataclass
class TaskConfig:
    name: str
