import copy
from typing import Any


def safe_merge(
    container: dict[str, Any], updates_dict: dict[str, Any], prior_k_str: str | None = None
) -> dict[str, Any]:
    """A simple utility to safely merge two dictionaries in a nested fashion.

    Args:
        container: A dictionary with string keys into which the updates in updates_dict should be merged.
        updates_dict: A dictionary with string keys containing the updates to container.
        prior_k_str: A '.' separated string containing any prior keys if this is a nested update for
            intelligent error messages.

    Returns:
        An updated dictionary containing the merged values of container and updates_dict. If keys collide with
        identical values, warnings are printed; if they collide with different values, an error is raised.

    Raises:
        ValueError: If at any point in the nesting there are key collisions with differing values.

    Examples:
        >>> safe_merge(
        ...     {'foo': 32, 'bar': {'baz': 1, 'biz': 'a'}},
        ...     {'boo': 31, 'bar': {'blah': {'fuzz': 13}}},
        ... )
        {'foo': 32, 'bar': {'baz': 1, 'biz': 'a', 'blah': {'fuzz': 13}}, 'boo': 31}
        >>> safe_merge(
        ...     {'foo': 32, 'bar': {'baz': 1, 'biz': 'a'}},
        ...     {'bar': {'baz': 1}},
        ...     'prior',
        ... )
        WARNING: duplicate key prior.bar.baz encountered with shared value 1!
        {'foo': 32, 'bar': {'baz': 1, 'biz': 'a'}}
        >>> safe_merge(
        ...     {'foo': 32, 'bar': {'baz': 1, 'biz': 'a'}},
        ...     {'bar': {'baz': 12}},
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Duplicate key bar.baz encountered with differing values 12 vs. 1
    """
    for key, val in updates_dict.items():
        if key in container:
            old_val = container[key]
            k_str = key if prior_k_str is None else f"{prior_k_str}.{key}"
            if isinstance(val, dict) and isinstance(old_val, dict):
                safe_merge(old_val, val, k_str)
            elif val == old_val:
                print(f"WARNING: duplicate key {k_str} encountered with shared value {val}!")
            else:
                raise ValueError(
                    f"Duplicate key {k_str} encountered with differing values {val} vs. {old_val}"
                )
        else:
            container[key] = val
    return container


def expand_nested_dict(nested_dict: dict[str, Any]) -> dict[str, Any]:
    """A utility for expanding a collapsed multi-level config with dot-separated keys.

    This function converts a dictionary with collapsed nesting in the Hydra style (e.g., {'a.b': 3}) into a
    fully expanded nested dictionary (e.g., {'a': {'b': 3}}) while guaranteeing no collisions.

    Args:
        nested_dict: The input dictionary, in nested format. Must have string keys.

    Returns:
        The dictionary in expanded, hierarchical form.

    Raises:
        ValueError: If there are collisions in the nesting options.

    Examples:
        >>> expand_nested_dict(
        ...     {'foo.bar.baz': 3, 'foo.bar.biz': 4, 'foo.bang': 2, 'biz': {'fuzz': 13, 'foo': 1}}
        ... )
        {'foo': {'bar': {'baz': 3, 'biz': 4}, 'bang': 2}, 'biz': {'fuzz': 13, 'foo': 1}}
        >>> expand_nested_dict(
        ...     {'foo.bar.baz': 3, 'foo.bar.biz': 4, 'foo': {'bang': 2}, 'biz': {'fuzz': 13, 'foo': 1}}
        ... )
        {'foo': {'bar': {'baz': 3, 'biz': 4}, 'bang': 2}, 'biz': {'fuzz': 13, 'foo': 1}}
        >>> expand_nested_dict({'foo.bar.baz': 3, 'foo': {'bar': 2}})
        Traceback (most recent call last):
            ...
        ValueError: Duplicate key foo.bar encountered with differing values 2 vs. {'baz': 3}
    """

    out = {}
    for key, value in nested_dict.items():
        parts = key.split(".")
        key = parts[0]
        for part in parts[-1:0:-1]:
            value = {part: value}

        if isinstance(value, dict):
            value = expand_nested_dict(value)
        safe_merge(out, {key: value})
    return out


def resolve_referential_nested_dict(
    nested_dict: dict[str, Any], output_fields: set[str], separator: str = "/", prior_k_str: str | None = None
) -> dict[str, Any]:
    """Parses a nested dictionary containing a set of top-level output fields and nested sub-updates.

    Suppose we want to have a dictionary containing a mapping of names to output objects, such that each
    output object is a dictionary with a static set of keys. We can instantiate such an object hierarchically,
    such that shared properties are stored in a top-level dictionary, and then nested names and updates to
    those shared properties are stored in other dictionary members. For example, if our outputs are of the
    form ``{"foo": ???, "bar": ???, "baz": ???}``, then rather than writing
    ```
    {
        "obj_1/1/1": {"foo": 3, "bar": 4, "baz": 1},
        "obj_1/1/2": {"foo": 3, "bar": 4, "baz": 2},
        "obj_1/2": {"foo": 3, "bar": 5, "baz": 3},
        "obj_2": {"foo": 4, "bar": 6, "baz": 3},
    }
    ```
    we could instead write
    ```
    {
        "obj_1": {
            "foo": 3,
            "1": {
                "bar": 4,
                "1": {"baz": 1},
                "2": {"baz": 2},
            },
            "2": {"bar": 5, "baz": 3},
        },
        "obj_2": {"foo": 4, "bar": 6, "baz": 3},
    }
    ```
    Obviously in this example, the latter is actually longer than the former, but for more complex examples
    with more shared content, the balance shifts.

    This function translates from the latter representation to the former.

    Args:
        nested_dict: The nested dictionary to be resolved.
        output_fields: The names of the top-level output fields to be found in the output object schema.
        separator: The string separator that should be used to merge key names in the resolved object.

    Returns:
        A fully resolved, expanded mapping from names to objects (as dictionaries).

    Raises:
        ValueError: If at any point in the nesting there are key collisions with differing values, or if
            non-schema leaf (non-dict) fields are found.

    Examples:
        >>> output_fields = {"outk1", "outk2", "outk3", "outk4"}
        >>> resolve_referential_nested_dict({"outk1": 3, "outk2": 4}, output_fields)
        {'outk1': 3, 'outk2': 4}
        >>> nested_dict = {
        ...     "outk4": 5,
        ...     "obj_1": {
        ...         "outk1": 3,
        ...         "outk2": {"a": 4},
        ...         "opt_1": {"outk2": {"b": 5}},
        ...         "opt_2": {"outk2": {"b": 7}, "sub_a": {"outk3": "a"}, "sub_b": {"outk3": 1}},
        ...     },
        ...     "obj_2": {"outk2": "foo"},
        ...     "obj_1/opt_3": {"outk1": 4, "outk2": "bizz"},
        ... }
        >>> out = resolve_referential_nested_dict(nested_dict, output_fields)
        >>> for k, v in out.items():
        ...     print(f"{k}: {v}")
        obj_1/opt_1: {'outk4': 5, 'outk1': 3, 'outk2': {'a': 4, 'b': 5}}
        obj_1/opt_2/sub_a: {'outk4': 5, 'outk1': 3, 'outk2': {'a': 4, 'b': 7}, 'outk3': 'a'}
        obj_1/opt_2/sub_b: {'outk4': 5, 'outk1': 3, 'outk2': {'a': 4, 'b': 7}, 'outk3': 1}
        obj_2: {'outk4': 5, 'outk2': 'foo'}
        obj_1/opt_3: {'outk4': 5, 'outk1': 4, 'outk2': 'bizz'}
        >>> nested_dict = {
        ...     "obj_1": {
        ...         "outk2": {"a": 4},
        ...         "opt_1": {"outk2": {"b": 5}},
        ...         "opt_2": {"outk2": {"b": 7}},
        ...     },
        ... }
        >>> out = resolve_referential_nested_dict(nested_dict, output_fields, ".")
        >>> for k, v in out.items():
        ...     print(f"{k}: {v}")
        obj_1.opt_1: {'outk2': {'a': 4, 'b': 5}}
        obj_1.opt_2: {'outk2': {'a': 4, 'b': 7}}
        >>> nested_dict = {
        ...     "obj_1": {
        ...         "outk2": {"a": 4},
        ...         "opt_1": {"outk2": {"b": 5}},
        ...         "opt_2": {"outk2": {"b": 7}},
        ...     },
        ... }
        >>> resolve_referential_nested_dict(nested_dict, {"outk1"}, ".")
        Traceback (most recent call last):
            ...
        ValueError: Encountered a non-schema leaf field a!
        >>> nested_dict = {
        ...     "obj_1": {
        ...         "outk2": {"a": 4},
        ...         "opt_1": {"outk2": {"a": 7}},
        ...     },
        ... }
        >>> out = resolve_referential_nested_dict(nested_dict, output_fields)
        Traceback (most recent call last):
            ...
        ValueError: Duplicate key obj_1/opt_1.outk2.a encountered with differing values 7 vs. 4
    """

    direct_obj_kwargs = {k: v for k, v in nested_dict.items() if k in output_fields}
    extra_kwargs = {k: v for k, v in nested_dict.items() if k not in output_fields}

    if not extra_kwargs:
        return direct_obj_kwargs

    if prior_k_str is None:
        prior_k_str = ""

    out = {}
    for k, v in extra_kwargs.items():
        if not isinstance(v, dict):
            raise ValueError(f"Encountered a non-schema leaf field {k}!")

        if all(k2 in output_fields for k2 in v.keys()):
            out[k] = safe_merge(copy.deepcopy(direct_obj_kwargs), v, f"{prior_k_str}{separator}{k}")
        else:
            for k2, v2 in resolve_referential_nested_dict(v, output_fields, separator, k).items():
                new_k = f"{k}{separator}{k2}"
                new_prior_k_str = f"{prior_k_str}{separator}{new_k}"
                out[new_k] = safe_merge(copy.deepcopy(direct_obj_kwargs), v2, new_prior_k_str)
    return out
