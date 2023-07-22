# Task Configuration Language

## Task Config Resolution

TODO

## Task Cohort Querying

A given task consists of:

- A set of a event predicate functions that are used to make determinations about the viability of certain
  task windows.
- A set of constraints about windows or events implied by start and end event predicate relationships. These
  can be arranged as a DAG based on the relative dependency between them.

For example

```yaml
in_hospital_mortality:
  events:
    admission:
      has:
        event_type: ADMISSION
    discharge:
      has:
        event_type: DISCHARGE
    death:
      has:
        event_type: DEATH

  trigger:
    event: admission
  constraints.gap:
    start: trigger.end # this is the default
    end: input.end + 24h
    excludes: [admission, discharge, death]
  input:
    end: trigger.end + 24h
    duration:
      min:
        time: 30d
  target:
    start: gap.end # this is the default
    end:
      any: [discharge, death]
      inclusive: true
  label:
    target:
      includes: [death]
```

The events here are an admission event, a discharge event, and a death event. The windows have relationships
accessible in the following DAG:

```
                                                             *(GAP END)
              ------ 24h ----- *(INPUT_END) ------ 24h ------[                     death or discharge
admit        /                                               *(TARGET_START) ----- *(TRIGGER_END)
*(TRIGGER) -[
            *(GAP_START)
```

To query this, we first produce a dataframe that has as columns subject IDs, event timestamps, and boolean
values indicating whether each event predicate is satisfied at that given timestamp (e.g., admission, death,
and discharge).

Next, we translate this dataframe into a dataframe with nulls where predicates aren't satisfied and the
current timestamps where predicates are satisfied.

Then, starting at the leaf nodes of this graph, we impute backwards to the next most prior incident where
their parent predicate is satisfied, collecting statistics about the implicit window along the way.
