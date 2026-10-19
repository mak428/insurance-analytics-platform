"""
Central data access layer. All services pull data through here so the
statistical/actuarial logic never depends on file paths or raw column names.
Put the four source files in the ./data folder.
"""
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@lru_cache(maxsize=1)
def load_liability_claims() -> pd.DataFrame:
    """French motor liability portfolio: ClaimNb, ClaimAmount, Exposure + rating factors."""
    df = pd.read_csv(DATA_DIR / "liability_claims.csv")
    df["ClaimNb"] = df["ClaimNb"].fillna(0)
    df["ClaimAmount"] = df["ClaimAmount"].fillna(0)
    df = df.dropna(subset=["Exposure"])  # unknown exposure can't enter a rate model
    df = df[df["Exposure"] > 0]
    return df


@lru_cache(maxsize=2)
def load_severity_sample(source: str = "liability") -> np.ndarray:
    """Positive claim amounts only (severity is conditional on a claim occurring)."""
    if source == "liability":
        df = load_liability_claims()
        amounts = df.loc[df["ClaimNb"] > 0, "ClaimAmount"]
    elif source == "generic":
        amounts = load_generic_insurance()["Claim_Amount"]
    else:
        raise ValueError(f"Unknown severity source: {source}")
    amounts = amounts[amounts > 0].dropna()
    return amounts.to_numpy(dtype=float)


@lru_cache(maxsize=1)
def load_generic_insurance() -> pd.DataFrame:
    """insurance_dataset.csv: demographics + Claim_Amount (severity fitting + ML)."""
    df = pd.read_csv(DATA_DIR / "insurance_dataset.csv")
    return df.dropna(subset=["Claim_Amount"])


@lru_cache(maxsize=1)
def load_synthetic_policies() -> pd.DataFrame:
    """data_synthetic.csv: policy table with dates, used for the monthly time series."""
    df = pd.read_csv(DATA_DIR / "data_synthetic.csv")
    # "Claim History" is a numeric claim COUNT (0-5) despite the name, not a date.
    df["Claim History"] = pd.to_numeric(df["Claim History"], errors="coerce")
    for col in ["Policy Start Date", "Policy Renewal Date"]:
        df[col] = pd.to_datetime(df[col], format="mixed", errors="coerce")
    return df


@lru_cache(maxsize=1)
def load_health_panel() -> pd.DataFrame:
    """Health portfolio panel. NOTE: the real per-risk key is `ID` (policy_insured),
    NOT `ID_insured`, which repeats across policies."""
    df = pd.read_excel(DATA_DIR / "Dataset_of_health_insurance_portfolio.xlsx")
    df = df.dropna(subset=["ID", "exposure_time", "cost_claims_year"])
    return df[df["exposure_time"] > 0]
