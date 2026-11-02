"""A minimal feature store (Feast-style concepts, zero infrastructure).

Concepts implemented
  Entity        : the key features are attached to (origin airport, carrier, route)
  FeatureView   : a named group of features for one entity, with a TTL
  Offline store : full feature *history* (Parquet) — used to build training sets
  Online store  : latest value per entity (SQLite key-value) — used at serving
  Point-in-time : training rows only see feature values whose timestamp is
                  <= the event time (merge_asof backward) — prevents leakage
  Materialize   : copy the latest offline values into the online store

Every feature row has an `event_timestamp` = the moment the value became
available. Daily aggregates for day D are computed from days < D, so they are
stamped D 00:00 — exactly what a nightly batch job would publish.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field

import pandas as pd

from .config import OFFLINE_DIR, ONLINE_DB


@dataclass
class FeatureView:
    name: str
    entity: str
    features: list[str]
    ttl_days: int
    description: str
    owner: str = "data-eng"
    tags: dict = field(default_factory=dict)


FEATURE_VIEWS = {
    "origin_stats": FeatureView(
        "origin_stats", "origin",
        ["origin_delay_rate_1d", "origin_delay_rate_7d", "origin_avg_delay_7d", "origin_flights_1d"], 3,
        "Departure-delay behaviour of the origin airport (yesterday and previous 7 days)"),
    "carrier_stats": FeatureView(
        "carrier_stats", "carrier", ["carrier_delay_rate_30d"], 7,
        "Carrier on-time performance over the previous 30 days"),
    "route_stats": FeatureView(
        "route_stats", "route", ["route_delay_rate_30d"], 7,
        "Origin-destination route delay rate over the previous 30 days"),
}


def _daily(df: pd.DataFrame, key: str) -> pd.DataFrame:
    d = df.assign(date=df["scheduled_departure"].dt.normalize())
    g = d.groupby([key, "date"]).agg(n=("is_delayed", "size"), delayed=("is_delayed", "sum"),
                                     delay_min=("delay_minutes", "sum")).reset_index()
    full = []
    all_dates = pd.date_range(d["date"].min(), d["date"].max() + pd.Timedelta(days=1), freq="D")
    for k, grp in g.groupby(key):
        grp = grp.drop(columns=key).set_index("date").reindex(all_dates, fill_value=0)
        grp[key] = k
        full.append(grp.rename_axis("date").reset_index())
    return pd.concat(full, ignore_index=True)


def _rolling(daily: pd.DataFrame, key: str, window: int, include_today: bool) -> pd.DataFrame:
    out = []
    for _, grp in daily.groupby(key):
        grp = grp.sort_values("date").copy()
        roll = grp[["n", "delayed", "delay_min"]].rolling(window, min_periods=1).sum()
        if not include_today:            # correct: only days strictly before D
            roll = roll.shift(1)
        grp[["r_n", "r_delayed", "r_min"]] = roll.values
        lag = 1 if not include_today else 0
        grp["prev_n"] = grp["n"].shift(lag)
        grp["prev_rate"] = (grp["delayed"] / grp["n"].replace(0, float("nan"))).shift(lag)
        out.append(grp)
    return pd.concat(out, ignore_index=True)


def compute_feature_tables(df: pd.DataFrame, leaky: bool = False) -> dict[str, pd.DataFrame]:
    """Batch feature engineering. `leaky=True` reproduces the classic mistake of
    aggregating *including the current day* (and therefore the label itself)."""
    df = df.dropna(subset=["is_delayed"]).copy()
    df["route"] = df["origin"] + "-" + df["dest"]
    tables = {}

    o = _rolling(_daily(df, "origin"), "origin", 7, include_today=leaky)
    o["origin_delay_rate_7d"] = o["r_delayed"] / o["r_n"]
    o["origin_avg_delay_7d"] = o["r_min"] / o["r_n"]
    o["origin_flights_1d"] = o["prev_n"]
    o["origin_delay_rate_1d"] = o["prev_rate"]
    tables["origin_stats"] = o

    c = _rolling(_daily(df, "carrier"), "carrier", 30, include_today=leaky)
    c["carrier_delay_rate_30d"] = c["r_delayed"] / c["r_n"]
    tables["carrier_stats"] = c

    r = _rolling(_daily(df, "route"), "route", 30, include_today=leaky)
    r["route_delay_rate_30d"] = r["r_delayed"] / r["r_n"]
    tables["route_stats"] = r

    for name, t in tables.items():
        fv = FEATURE_VIEWS[name]
        t = t.rename(columns={"date": "event_timestamp"})
        tables[name] = t[[fv.entity, "event_timestamp"] + fv.features].dropna().reset_index(drop=True)
    return tables


class FeatureStore:
    def __init__(self, offline_dir=OFFLINE_DIR, online_db=ONLINE_DB):
        self.offline_dir = offline_dir
        self.online_db = online_db
        self.views = FEATURE_VIEWS

    # ---------------- offline -----------------
    def write_offline(self, tables: dict[str, pd.DataFrame], data_version: str):
        self.offline_dir.mkdir(parents=True, exist_ok=True)
        for name, t in tables.items():
            t.to_parquet(self.offline_dir / f"{name}.parquet", index=False)
        (self.offline_dir / "_metadata.json").write_text(json.dumps(
            {"data_version": data_version, "rows": {n: len(t) for n, t in tables.items()}}))

    def read_offline(self, name: str) -> pd.DataFrame:
        return pd.read_parquet(self.offline_dir / f"{name}.parquet")

    def offline_metadata(self) -> dict:
        p = self.offline_dir / "_metadata.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def get_historical_features(self, entity_df: pd.DataFrame, tables=None,
                                timestamp_col="scheduled_departure") -> pd.DataFrame:
        """Point-in-time correct join: for each event, take the latest feature
        value with event_timestamp <= event time and within the view's TTL."""
        ent = entity_df.copy()
        ent["route"] = ent["origin"] + "-" + ent["dest"]
        ent["_row"] = range(len(ent))
        out = ent.sort_values(timestamp_col)
        for name, fv in self.views.items():
            t = tables[name] if tables is not None else self.read_offline(name)
            t = t.sort_values("event_timestamp")
            out = pd.merge_asof(
                out.sort_values(timestamp_col), t, left_on=timestamp_col, right_on="event_timestamp",
                by=fv.entity, direction="backward", tolerance=pd.Timedelta(days=fv.ttl_days),
            ).drop(columns="event_timestamp")
        return out.sort_values("_row").drop(columns=["_row", "route"]).reset_index(drop=True)

    @staticmethod
    def naive_join(entity_df: pd.DataFrame, leaky_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """The WRONG way: join on calendar date using aggregates that include the
        same day's outcomes (i.e. information from the future and the label)."""
        out = entity_df.copy()
        out["route"] = out["origin"] + "-" + out["dest"]
        out["_date"] = out["scheduled_departure"].dt.normalize()
        for name, fv in FEATURE_VIEWS.items():
            t = leaky_tables[name].rename(columns={"event_timestamp": "_date"})
            out = out.merge(t, on=[fv.entity, "_date"], how="left")
        return out.drop(columns=["_date", "route"])

    # ---------------- online ------------------
    def materialize(self, as_of: pd.Timestamp | None = None) -> dict:
        """Push the latest value per entity (as of `as_of`) to the online store."""
        con = sqlite3.connect(self.online_db)
        counts = {}
        for name, fv in self.views.items():
            t = self.read_offline(name)
            if as_of is not None:
                t = t[t["event_timestamp"] <= as_of]
            latest = t.sort_values("event_timestamp").groupby(fv.entity).tail(1)
            latest = latest.assign(event_timestamp=latest["event_timestamp"].astype(str))
            latest.to_sql(name, con, if_exists="replace", index=False)
            counts[name] = len(latest)
        con.close()
        return counts

    def get_online_features(self, origin: str, dest: str, carrier: str) -> dict:
        con = sqlite3.connect(self.online_db)
        keys = {"origin": origin, "carrier": carrier, "route": f"{origin}-{dest}"}
        result = {}
        for name, fv in self.views.items():
            row = con.execute(
                f"SELECT {', '.join(fv.features)}, event_timestamp FROM {name} WHERE {fv.entity}=?",
                (keys[fv.entity],)).fetchone()
            for i, f in enumerate(fv.features):
                result[f] = row[i] if row else None
            result[f"_{name}_ts"] = row[-1] if row else None
        con.close()
        return result

    def online_table(self, name: str) -> pd.DataFrame:
        con = sqlite3.connect(self.online_db)
        df = pd.read_sql(f"SELECT * FROM {name}", con)
        con.close()
        return df
