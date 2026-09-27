"""Preprocessing layer: raw TSV -> standardised records. See README.md in this folder."""
from .address_normalizer import normalize_address, parse_address
from .gazetteer import normalize_country
from .name_normalizer import normalize_name, parse_name

__all__ = ["normalize_address", "parse_address", "normalize_country", "normalize_name", "parse_name",
           "OUTPUT_COLUMNS", "load_processed", "preprocess_frame", "preprocess_record"]


def __getattr__(name: str):
    # Lazy so that `python -m src.preprocessing.preprocessing` does not import itself twice.
    if name in ("OUTPUT_COLUMNS", "load_processed", "preprocess_frame", "preprocess_record"):
        from . import preprocessing
        return getattr(preprocessing, name)
    raise AttributeError(name)
