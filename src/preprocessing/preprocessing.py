"""Raw TSV -> standardised records (the interface for blocking and matching).

    python -m src.preprocessing.preprocessing                      # everything
    python -m src.preprocessing.preprocessing --split test --sources 1
    python -m src.preprocessing.preprocessing --sample 50000       # quick run

Writes data/processed/{split}_source{n}.parquet with one row per record:

    entity_id, source                      "S1-123", "S1"
    business_name_raw, business_name_norm   original / canonical name
    business_address_raw, business_address_norm
    country, country_norm                   raw label / canonical open-set label
    name_core, name_legal, name_alt, name_is_domain, name_known_ratio, name_script, name_phonetic
    addr_street, addr_city, addr_state, addr_house_no, addr_numbers,
    addr_postcode, addr_is_missing

Downstream code should load with ``load_processed(split, source)``.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .address_normalizer import parse_address
from .gazetteer import normalize_country
from .io import PROCESSED_DIR, ROOT, iter_source, source_path
from .name_normalizer import parse_name
from .resources import indic_token_map, name_vocab, native_state_map

OUTPUT_COLUMNS = [
    "entity_id", "source",
    "business_name_raw", "business_name_norm", "business_address_raw", "business_address_norm",
    "country", "country_norm",
    "name_core", "name_legal", "name_alt", "name_is_domain", "name_known_ratio", "name_script",
    "name_phonetic",
    "addr_street", "addr_city", "addr_state", "addr_house_no", "addr_numbers", "addr_postcode",
    "addr_is_missing",
]
_BOOL_COLUMNS = {"name_is_domain", "addr_is_missing"}
_FLOAT_COLUMNS = {"name_known_ratio"}
SCHEMA = pa.schema([(c, pa.bool_() if c in _BOOL_COLUMNS else pa.float32() if c in _FLOAT_COLUMNS
                     else pa.string()) for c in OUTPUT_COLUMNS])


def preprocess_record(entity_id: str, business_name: str, business_address: str, country: str) -> dict:
    """One raw record -> the standard record dict (see module docstring)."""
    country_norm = normalize_country(country)
    name = parse_name(business_name, country_norm)
    addr = parse_address(business_address, country_norm)
    return {
        "entity_id": entity_id,
        "source": entity_id.split("-", 1)[0],
        "business_name_raw": business_name,
        "business_name_norm": name.name_norm,
        "business_address_raw": business_address,
        "business_address_norm": addr.address_norm,
        "country": country,
        "country_norm": country_norm,
        "name_core": name.name_core,
        "name_legal": name.name_legal,
        "name_alt": name.name_alt,
        "name_is_domain": name.name_is_domain,
        "name_known_ratio": name.name_known_ratio,
        "name_script": name.name_script,
        "name_phonetic": name.name_phonetic,
        "addr_street": addr.addr_street,
        "addr_city": addr.addr_city,
        "addr_state": addr.addr_state,
        "addr_house_no": addr.addr_house_no,
        "addr_numbers": addr.addr_numbers,
        "addr_postcode": addr.addr_postcode,
        "addr_is_missing": addr.addr_is_missing,
    }


def preprocess_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Raw source DataFrame (entity_id, business_name, business_address, country) -> standard records."""
    rows = [preprocess_record(*r) for r in zip(df["entity_id"], df["business_name"],
                                                df["business_address"], df["country"])]
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def _warm_resources() -> None:
    indic_token_map(), native_state_map(), name_vocab()


def _work(df: pd.DataFrame) -> pa.Table:
    return pa.Table.from_pandas(preprocess_frame(df), schema=SCHEMA, preserve_index=False)


def processed_path(split: str, source: int | str) -> "os.PathLike":
    return PROCESSED_DIR / f"{split}_source{str(source).upper().lstrip('S')}.parquet"


def process_file(split: str, source: int, workers: int, chunksize: int,
                 sample: int | None = None) -> None:
    out = processed_path(split, source)
    if sample:
        out = out.with_name(out.stem + f"_sample{sample}.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    chunks = iter_source(split, source, chunksize=min(chunksize, sample or chunksize))
    if sample:
        chunks = (c for i, c in enumerate(chunks) if i * chunksize < sample)
    n = 0
    tmp = out.with_suffix(".parquet.tmp")
    with pq.ParquetWriter(tmp, SCHEMA, compression="zstd", compression_level=9) as writer, \
            mp.get_context("fork").Pool(workers, initializer=_warm_resources) as pool:
        for table in pool.imap(_work, chunks):
            writer.write_table(table)
            n += table.num_rows
            print(f"\r  {source_path(split, source).name}: {n:,} records "
                  f"({n / (time.time() - t0):,.0f}/s)", end="", flush=True)
    os.replace(tmp, out)
    print(f"\n  -> {out.relative_to(ROOT)} ({out.stat().st_size / 1e6:,.0f} MB, {time.time() - t0:,.0f}s)")


def load_processed(split: str, source: int | str, columns: list[str] | None = None) -> pd.DataFrame:
    """Load preprocessed records written by this module."""
    return pd.read_parquet(processed_path(split, source), columns=columns)


# A third of the French S2/S3 addresses stop at the city ("77 AV LEON JOUHAUX, LILLE"); US/India
# copies almost always keep the state, so the matcher reads a missing state as a mismatch.
REGION_FILL_COUNTRIES = ("france",)


def _city_key(city: str) -> str:
    return " ".join({"st": "saint", "ste": "sainte"}.get(t, t) for t in city.split())


def fill_missing_regions(split: str, countries=REGION_FILL_COUNTRIES, min_count: int = 3,
                         min_share: float = 0.95) -> None:
    """Fill an empty addr_state of S2/S3 records from the city -> state table of the split's own
    Source-1 records (unambiguous cities only; no labels), and append the code to the normalised
    address as a stated region would be. Idempotent."""
    s1 = load_processed(split, 1, ["country_norm", "addr_city", "addr_state"])
    s1 = s1[s1["country_norm"].isin(countries) & s1["addr_city"].ne("") & s1["addr_state"].ne("")]
    counts = s1.assign(city=s1["addr_city"].map(_city_key)).groupby(["country_norm", "city"])["addr_state"] \
        .value_counts().rename("n").reset_index()
    tot = counts.groupby(["country_norm", "city"])["n"].transform("sum")
    counts = counts[(counts["n"] >= min_count) & (counts["n"] >= min_share * tot)]
    table = dict(zip(zip(counts["country_norm"], counts["city"]), counts["addr_state"]))
    for source in (2, 3):
        path = processed_path(split, source)
        df = pd.read_parquet(path)
        todo = df["country_norm"].isin(countries) & df["addr_state"].eq("") & df["addr_city"].ne("") \
            & ~df["addr_is_missing"]
        keys = zip(df.loc[todo, "country_norm"], df.loc[todo, "addr_city"].map(_city_key))
        code = pd.Series([table.get(k, "") for k in keys], index=df.index[todo], dtype=object)
        code = code[code.ne("")]
        df.loc[code.index, "addr_state"] = code
        df.loc[code.index, "business_address_norm"] = df.loc[code.index, "business_address_norm"] + " " + code
        tmp = path.with_suffix(".parquet.tmp")
        pa_table = pa.Table.from_pandas(df[OUTPUT_COLUMNS], schema=SCHEMA, preserve_index=False)
        pq.write_table(pa_table, tmp, compression="zstd", compression_level=9)
        os.replace(tmp, path)
        print(f"  {split}_source{source}: filled the state of {len(code):,} of {int(todo.sum()):,} "
              f"stateless {'/'.join(countries)} addresses")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", nargs="+", default=["train", "test"], choices=["train", "test"])
    ap.add_argument("--sources", nargs="+", type=int, default=[1, 2, 3], choices=[1, 2, 3])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--chunksize", type=int, default=100_000)
    ap.add_argument("--sample", type=int, help="only the first N rows of each file (quick check)")
    ap.add_argument("--fill-regions-only", action="store_true",
                    help="only run the stateless-address region fill on existing outputs")
    args = ap.parse_args()
    for split in args.split:
        if not args.fill_regions_only:
            for source in args.sources:
                process_file(split, source, args.workers, args.chunksize, args.sample)
        if not args.sample:
            fill_missing_regions(split)


if __name__ == "__main__":
    main()
