"""Load condition JSON keyed by a relative sample path."""

from pathlib import PurePosixPath


def path_key(value):
    relative = value.replace(chr(92), '/')
    path = PurePosixPath(relative)
    if path.is_absolute() or '..' in path.parts or ':' in relative:
        raise ValueError(f'Expected a relative sample path: {value!r}')
    return path.with_suffix('').as_posix()


def index_conditions(data):
    """Accept both path-to-object dictionaries and lists with a path field."""
    if isinstance(data, list):
        records = ((item['path'], item) for item in data)
    elif isinstance(data, dict):
        records = data.items()
    else:
        raise ValueError('Condition JSON must contain an object or a list of records')
    indexed = {}
    for path, record in records:
        key = path_key(path)
        if key in indexed:
            raise ValueError(f'Duplicate condition for {path!r}')
        indexed[key] = record
    return indexed
