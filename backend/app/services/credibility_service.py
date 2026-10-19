"""
Topic 4: Buhlmann-Straub credibility on the health panel.
Z_i = e_i / (e_i + k),  k = EPV / VHM,  premium_i = Z_i * Xbar_i + (1 - Z_i) * Xbar.
"""
import numpy as np
import pandas as pd


def fit_buhlmann_straub(df: pd.DataFrame, entity_col: str, exposure_col: str, loss_col: str) -> dict:
    d = df[[entity_col, exposure_col, loss_col]].dropna()
    d = d[d[exposure_col] > 0].copy()
    d["rate"] = d[loss_col] / d[exposure_col]
    d["wr"] = d["rate"] * d[exposure_col]

    g = d.groupby(entity_col)
    exp_i = g[exposure_col].sum()
    mean_i = g["wr"].sum() / exp_i
    overall = d["wr"].sum() / d[exposure_col].sum()

    # EPV: exposure-weighted within-entity variance (entities with >= 2 periods)
    d["mean_i"] = d[entity_col].map(mean_i)
    d["ss"] = d[exposure_col] * (d["rate"] - d["mean_i"]) ** 2
    cnt = g.size()
    ss_i = d.groupby(entity_col)["ss"].sum()
    multi = cnt[cnt >= 2].index
    epv = float(ss_i[multi].sum() / (cnt[multi] - 1).sum())

    # VHM: unbiased between-entity variance estimator
    n_ent, tot = len(exp_i), exp_i.sum()
    ss_between = float(np.sum(exp_i * (mean_i - overall) ** 2))
    denom = tot - float((exp_i ** 2).sum() / tot)
    vhm = max((ss_between - (n_ent - 1) * epv) / denom, 1e-8)

    k = epv / vhm
    z = exp_i / (exp_i + k)
    table = pd.DataFrame({
        "exposure": exp_i, "individual_mean_rate": mean_i,
        "credibility_z": z, "credibility_premium": z * mean_i + (1 - z) * overall,
    }).reset_index().rename(columns={entity_col: "entity_id"})
    return {"overall_mean_rate": float(overall), "epv": epv, "vhm": float(vhm), "k": float(k),
            "n_entities": int(n_ent), "entities": table.to_dict(orient="records")}


def validate_out_of_sample(df: pd.DataFrame, entity_col: str, exposure_col: str,
                           loss_col: str, period_col: str) -> dict:
    """Hold out each entity's latest period; compare credibility vs individual-only vs portfolio-only MSE."""
    df = df.dropna(subset=[entity_col, exposure_col, loss_col, period_col])
    idx = df.groupby(entity_col)[period_col].idxmax()
    holdout, train = df.loc[idx], df.drop(index=idx)
    fit = fit_buhlmann_straub(train, entity_col, exposure_col, loss_col)
    tbl = pd.DataFrame(fit["entities"]).set_index("entity_id")
    h = holdout[holdout[entity_col].isin(tbl.index)].copy()
    h["actual"] = h[loss_col] / h[exposure_col]
    h = h.join(tbl, on=entity_col)
    return {
        "credibility_mse": float(((h["actual"] - h["credibility_premium"]) ** 2).mean()),
        "individual_only_mse": float(((h["actual"] - h["individual_mean_rate"]) ** 2).mean()),
        "portfolio_only_mse": float(((h["actual"] - fit["overall_mean_rate"]) ** 2).mean()),
        "n_holdout_entities": int(len(h)),
        "interpretation": "Lower MSE is better; credibility should beat both pure estimates.",
    }
