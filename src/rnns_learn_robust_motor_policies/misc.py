
def dict_str(d):
    """A string representation of a dict that is more filename-friendly than `str` or `repr`."""
    return '-'.join(f"{k}-{v}" for k, v in d.items())