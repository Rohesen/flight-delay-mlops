"""Synthetic Indian domestic flight operations data.

The generator encodes realistic, *learnable* delay mechanics:
  * carrier and airport effects (hub congestion),
  * evening "knock-on" delays that build up through the day,
  * weather, plus winter fog at north-Indian airports (Dec-Jan),
  * a persistent airport-day "disruption" shock (AR(1) across days) — this is
    why recent airport delay history (a feature-store feature) is predictive,
    and why using *same-day* outcomes leaks the label.

Versions of the dataset (v1.0 … v3.0) are derived from one master table so
that every version is exactly reproducible from the seed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SEED

AIRPORTS = {  # code: (lat, lon, congestion effect, fog-prone)
    "DEL": (28.56, 77.10, 0.35, True),
    "BOM": (19.09, 72.87, 0.40, False),
    "BLR": (13.20, 77.71, 0.10, False),
    "MAA": (12.99, 80.17, 0.05, False),
    "CCU": (22.65, 88.45, 0.15, True),
    "HYD": (17.24, 78.43, 0.00, False),
    "VNS": (25.45, 82.86, 0.10, True),
    "GOI": (15.38, 73.83, -0.10, False),
}
CARRIERS = {"AI": 0.10, "6E": -0.35, "UK": -0.15, "SG": 0.35, "IX": 0.20}
NEW_CARRIER = ("QP", 0.25)      # enters the market in July 2025
AIRCRAFT = ["A320neo", "A321neo", "B737-8", "ATR72", "A320ceo"]


def _haversine(a: str, b: str) -> float:
    lat1, lon1 = np.radians(AIRPORTS[a][:2])
    lat2, lon2 = np.radians(AIRPORTS[b][:2])
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return float(2 * 6371 * np.arcsin(np.sqrt(h)))


def generate_master(start="2025-01-01", end="2025-12-31", flights_per_day=12, seed=SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    codes = list(AIRPORTS)

    # Airport-day disruption shock, AR(1) so disruption persists for days.
    shock = {}
    for ap in codes:
        s = 0.0
        for d in dates:
            s = 0.85 * s + rng.normal(0, 0.38)
            shock[(ap, d)] = s
    # Daily weather per airport (0 = clear … ~3 = severe)
    weather = {(ap, d): max(0.0, rng.gamma(1.4, 0.45)) for ap in codes for d in dates}

    rows = []
    fid = 100000
    for d in dates:
        carriers = dict(CARRIERS)
        if d >= pd.Timestamp("2025-07-01"):
            carriers[NEW_CARRIER[0]] = NEW_CARRIER[1]
        for origin in codes:
            n = rng.poisson(flights_per_day)
            for _ in range(n):
                dest = rng.choice([c for c in codes if c != origin])
                carrier = rng.choice(list(carriers))
                dep_hour = int(np.clip(rng.normal(14, 4.5), 5, 23))
                w = weather[(origin, d)]
                fog = 1.0 if (AIRPORTS[origin][3] and d.month in (12, 1) and dep_hour <= 10) else 0.0
                logit = (-1.75 + carriers[carrier] + AIRPORTS[origin][2]
                         + 0.07 * max(0, dep_hour - 8)
                         + 0.75 * w + 1.4 * fog
                         + 1.1 * shock[(origin, d)])
                p = 1 / (1 + np.exp(-logit))
                delayed = rng.random() < p
                delay = float(rng.exponential(42) + 16) if delayed else float(max(0.0, rng.normal(3, 5)))
                rows.append({
                    "flight_id": f"FL{fid}",
                    "scheduled_departure": d + pd.Timedelta(hours=dep_hour, minutes=int(rng.integers(0, 60))),
                    "carrier": carrier,
                    "origin": origin,
                    "dest": dest,
                    "dep_hour": dep_hour,
                    "day_of_week": d.dayofweek,
                    "month": d.month,
                    "distance_km": round(_haversine(origin, dest), 1),
                    "weather_index": round(w, 3),
                    "aircraft_type": rng.choice(AIRCRAFT),
                    "delay_minutes": round(delay, 1),
                    "is_delayed": int(delay > 15),
                })
                fid += 1
    df = pd.DataFrame(rows).sort_values("scheduled_departure").reset_index(drop=True)
    return df


VERSION_SPECS = {
    "v1.0": "Initial extract Jan-Jun 2025 (raw: duplicate rows + missing weather readings)",
    "v1.1": "Cleaned v1.0: duplicates removed, missing weather imputed by airport median",
    "v2.0": "Q3 refresh Jan-Sep 2025: new carrier QP + new column aircraft_type (schema change)",
    "v3.0": "Full year 2025: adds Oct-Dec incl. winter fog season (distribution drift)",
}


def build_version(master: pd.DataFrame, tag: str, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 7)
    base_cols = [c for c in master.columns if c != "aircraft_type"]
    if tag in ("v1.0", "v1.1"):
        df = master[master["scheduled_departure"] < "2025-07-01"][base_cols].copy()
        # Raw extract problems
        dup = df.sample(frac=0.03, random_state=seed)
        df = pd.concat([df, dup]).sort_values("scheduled_departure").reset_index(drop=True)
        miss = rng.random(len(df)) < 0.04
        df.loc[miss, "weather_index"] = np.nan
        if tag == "v1.1":
            df = df.drop_duplicates(subset="flight_id").reset_index(drop=True)
            df["weather_index"] = df["weather_index"].fillna(
                df.groupby("origin")["weather_index"].transform("median")).round(3)
        return df
    if tag == "v2.0":
        return master[master["scheduled_departure"] < "2025-10-01"].reset_index(drop=True).copy()
    if tag == "v3.0":
        return master.copy()
    raise ValueError(f"Unknown version {tag}")
