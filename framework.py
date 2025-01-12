import numpy as np
from functools import wraps
import json
from pathlib import Path
from typing import Callable


def get_store_path(name: str, by_npz: bool = False):
    if by_npz:
        return Path(f"./_data/{name}.npz")
    else:
        return Path(f"./_data/{name}.jsonl")


def get_overwrite_path(name: str, by_npz: bool = False):
    if by_npz:
        return Path(f"./_overwrite/{name}.overwrite.npz")
    else:
        return Path(f"./_overwrite/{name}.overwrite.jsonl")


def save(name: str, data: dict, by_npz: bool = False):
    if by_npz:
        save_npz(name, data)
    else:
        save_jsonl(name, data)


def save_jsonl(name: str, data: dict):
    store_path = get_store_path(name, by_npz=False)
    store_path.parent.mkdir(parents=True, exist_ok=True)
    with store_path.open("w") as f:
        for key, value in data.items():
            print(json.dumps({"id": key, name: value}, ensure_ascii=False), file=f)
    print(f"Saved {name} to {store_path}")


def save_npz(name: str, data: dict):
    store_path = get_store_path(name, by_npz=True)
    store_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(store_path, **data)
    print(f"Saved {name} to {store_path}")


def load(name: str, only_overwrite: bool = False, id_key: str = "id"):
    store_path_jsonl = get_store_path(name, by_npz=False)
    overwrite_path_jsonl = get_overwrite_path(name, by_npz=False)
    store_path_npz = get_store_path(name, by_npz=True)
    overwrite_path_npz = get_overwrite_path(name, by_npz=True)
    result = {}

    if (not only_overwrite) and store_path_jsonl.exists():
        result.update(load_jsonl(name, id_key))

    if (not only_overwrite) and store_path_npz.exists():
        result.update(load_npz(name))

    if overwrite_path_jsonl.exists():
        result.update(load_jsonl(name))

    if overwrite_path_npz.exists():
        result.update(load_npz(name))

    return result


def load_jsonl(name: str, id_key: str):
    with open(get_store_path(name, by_npz=False)) as f_in:
        result = {}
        for line in f_in:
            data = json.loads(line)
            result[data[id_key]] = data[name]
    return result


def load_npz(name: str):
    return dict(np.load(get_store_path(name, by_npz=True)))


class ErrorWithDebug(Exception):
    pass


def source(name_or_func: str | Callable = None, by_npz: bool = False):
    if not isinstance(name_or_func, str):
        func = name_or_func
        return source(func.__name__)(func)
    name = name_or_func

    store_path = get_store_path(name)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, recreate: bool = False, **kwargs):
            if store_path.exists() and not recreate:
                return

            result = func(*args, **kwargs)
            save(name, result, by_npz=by_npz)

        return wrapper

    return decorator


def model(
    inputs: list[str],
    outputs: list[str] | None = None,
    by_npz: list[bool] | bool = False,
):
    def decorator(func):
        outputs_fix = outputs or [func.__name__]
        if isinstance(by_npz, bool):
            by_npz_fix = [by_npz] * len(outputs_fix)
        else:
            by_npz_fix = by_npz

        @wraps(func)
        def wrapper(
            *args, recreate: bool = False, skip_if_exist: bool = False, **kwargs
        ):
            if skip_if_exist:
                if all(
                    [
                        get_store_path(output, by_npz).exists()
                        for output, by_npz in zip(outputs_fix, by_npz_fix)
                    ]
                ):
                    return

            previous_values = [
                load(output, only_overwrite=recreate) for output in outputs_fix
            ]

            not_update_keys = set.intersection(
                *[set(value.keys()) for value in previous_values]
            )

            input_values = [load(input_) for input_ in inputs]
            common_keys = set.intersection(
                *[set(value.keys()) for value in input_values]
            )

            if (common_keys - not_update_keys) or recreate:
                try:
                    result = func(*input_values)
                except ErrorWithDebug as e:
                    print(f"Error: {e}")
                    print(f"Key: {key}")
                    raise Exception("Error in model")

                if len(outputs_fix) == 1:
                    result = [result]

                for output_name, data, by_npz_per_output in zip(
                    outputs_fix, result, by_npz_fix
                ):
                    save(output_name, data, by_npz=by_npz_per_output)

        return wrapper

    return decorator


def indexing(
    inputs: list[str],
    outputs: list[str] | None = None,
    by_npz: list[bool] | bool = False,
):
    def decorator(func):
        outputs_fix = outputs or [func.__name__]
        if isinstance(by_npz, bool):
            by_npz_fix = [by_npz] * len(outputs_fix)
        else:
            by_npz_fix = by_npz

        @wraps(func)
        def wrapper(*args, store_each: bool = False, recreate: bool = False, **kwargs):
            previous_values = [
                load(output, only_overwrite=recreate) for output in outputs_fix
            ]

            not_update_keys = set.intersection(
                *[set(value.keys()) for value in previous_values]
            )

            input_values = [load(input_) for input_ in inputs]
            common_keys = set.intersection(
                *[set(value.keys()) for value in input_values]
            )
            if not (common_keys - not_update_keys) and not recreate:
                return

            to_save = previous_values

            for key in common_keys - not_update_keys:
                try:
                    result = func(*[value[key] for value in input_values])
                except ErrorWithDebug as e:
                    print(f"Error: {e}")
                    print(f"Key: {key}")
                    continue

                if len(outputs_fix) == 1:
                    to_save[0][key] = result
                else:
                    assert len(outputs_fix) == len(result)
                    for i in range(len(outputs_fix)):
                        to_save[i][key] = result[i]

                if store_each:
                    for output_name, data, by_npz_per_output in zip(
                        outputs_fix, to_save, by_npz_fix
                    ):
                        save(output_name, data, by_npz=by_npz_per_output)
                    print(f"Done {key}({outputs_fix})")

            if not store_each:
                for output_name, data, by_npz_per_output in zip(
                    outputs_fix, to_save, by_npz_fix
                ):
                    save(output_name, data, by_npz=by_npz_per_output)

        return wrapper

    return decorator
