from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import warnings
from collections import defaultdict
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import wilcoxon
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


# ----------------------------
# Utilities
# ----------------------------

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_output_dir() -> Path:
    p = Path("outputs")
    p.mkdir(parents=True, exist_ok=True)
    return p


# ----------------------------
# Data
# ----------------------------

def synthetic_data(n_skus: int, n_days: int, seed: int) -> pd.DataFrame:
    """Generate reproducible SKU-day retail demand with seasonality, price, promo, trend and shocks."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    rows: list[tuple] = []

    for i in range(n_skus):
        sku = f"SKU_{i + 1:03d}"
        base = rng.uniform(12, 55)
        trend = rng.uniform(-0.006, 0.018)
        weekly_amp = rng.uniform(0.05, 0.30)
        annual_amp = rng.uniform(0.03, 0.18)
        base_price = rng.uniform(8, 80)
        elasticity = rng.uniform(0.5, 1.8)

        for t, dt in enumerate(dates):
            promo = int(rng.random() < 0.10)
            discount = rng.uniform(0.08, 0.25) if promo else 0.0
            price = base_price * (1 + rng.normal(0, 0.025)) * (1 - discount)

            weekly = 1 + weekly_amp * np.sin(2 * np.pi * dt.dayofweek / 7)
            annual = 1 + annual_amp * np.sin(2 * np.pi * dt.dayofyear / 365.25)
            trend_factor = max(0.60, 1 + trend * t / 30)
            promo_lift = 1 + (rng.uniform(0.15, 0.65) if promo else 0.0)
            price_effect = (base_price / max(price, 0.1)) ** elasticity

            mean_demand = max(0.5, base * weekly * annual * trend_factor * promo_lift * price_effect)
            latent = rng.gamma(shape=12.0, scale=mean_demand / 12.0)
            demand = int(rng.poisson(max(latent, 0.01)))

            # Rare demand shocks create inventory risk.
            if rng.random() < 0.015:
                demand = int(round(demand * rng.uniform(1.5, 2.8)))

            rows.append((dt, sku, float(demand), float(price), promo))

    return pd.DataFrame(rows, columns=["date", "sku", "demand", "price", "promo"])


def load_data(cfg: dict, seed: int) -> pd.DataFrame:
    dc = cfg["data"]
    input_csv = dc.get("input_csv")

    if input_csv:
        raw = pd.read_csv(input_csv)
        mapping = {
            dc["date_col"]: "date",
            dc["sku_col"]: "sku",
            dc["demand_col"]: "demand",
        }
        if dc.get("price_col") in raw.columns:
            mapping[dc["price_col"]] = "price"
        if dc.get("promo_col") in raw.columns:
            mapping[dc["promo_col"]] = "promo"
        df = raw.rename(columns=mapping).copy()
    else:
        print("No input CSV configured -> using reproducible synthetic dataset.")
        df = synthetic_data(int(dc["synthetic_skus"]), int(dc["synthetic_days"]), seed)

    required = {"date", "sku", "demand"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df["sku"] = df["sku"].astype(str)
    df["demand"] = pd.to_numeric(df["demand"], errors="raise").clip(lower=0).astype(float)

    if "price" not in df.columns:
        df["price"] = 1.0
    else:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df["price"] = df.groupby("sku")["price"].transform(lambda x: x.ffill().bfill().fillna(1.0))

    if "promo" not in df.columns:
        df["promo"] = 0.0
    else:
        df["promo"] = pd.to_numeric(df["promo"], errors="coerce").fillna(0.0).clip(0, 1)

    df = (
        df[["date", "sku", "demand", "price", "promo"]]
        .sort_values(["sku", "date"])
        .drop_duplicates(["sku", "date"], keep="last")
        .reset_index(drop=True)
    )
    return df


# ----------------------------
# Leakage-safe features
# ----------------------------

def make_features(df: pd.DataFrame, lags: list[int], windows: list[int]) -> pd.DataFrame:
    x = df.copy().sort_values(["sku", "date"]).reset_index(drop=True)

    x["dow"] = x["date"].dt.dayofweek
    x["month"] = x["date"].dt.month
    x["dayofyear"] = x["date"].dt.dayofyear
    x["weekofyear"] = x["date"].dt.isocalendar().week.astype(int)
    x["is_weekend"] = (x["dow"] >= 5).astype(int)
    x["sin_dow"] = np.sin(2 * np.pi * x["dow"] / 7.0)
    x["cos_dow"] = np.cos(2 * np.pi * x["dow"] / 7.0)
    x["sin_year"] = np.sin(2 * np.pi * x["dayofyear"] / 365.25)
    x["cos_year"] = np.cos(2 * np.pi * x["dayofyear"] / 365.25)

    for lag in lags:
        x[f"lag_{lag}"] = x.groupby("sku")["demand"].shift(lag)

    # Shift before rolling => today's demand never enters today's feature.
    shifted = x.groupby("sku")["demand"].shift(1)
    for w in windows:
        x[f"roll_mean_{w}"] = shifted.groupby(x["sku"]).transform(
            lambda s: s.rolling(w, min_periods=max(2, w // 3)).mean()
        )
        x[f"roll_std_{w}"] = shifted.groupby(x["sku"]).transform(
            lambda s: s.rolling(w, min_periods=max(2, w // 3)).std()
        )

    sku_map = {sku: i for i, sku in enumerate(sorted(x["sku"].unique()))}
    x["sku_code"] = x["sku"].map(sku_map).astype(int)

    # Conventional comparator.
    x["SeasonalNaive"] = x["lag_7"] if 7 in lags else x[f"lag_{min(lags)}"]

    required_lag = max(lags)
    x = x.dropna(subset=[f"lag_{required_lag}"]).copy()
    feature_cols = [c for c in x.columns if c not in {"date", "sku", "demand", "SeasonalNaive"}]
    x[feature_cols] = x[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return x.reset_index(drop=True)


def chronological_split(df: pd.DataFrame, val_days: int, test_days: int):
    max_date = df["date"].max()
    test_start = max_date - pd.Timedelta(days=test_days - 1)
    val_start = test_start - pd.Timedelta(days=val_days)

    train = df[df["date"] < val_start].copy()
    val = df[(df["date"] >= val_start) & (df["date"] < test_start)].copy()
    test = df[df["date"] >= test_start].copy()

    if min(len(train), len(val), len(test)) == 0:
        raise ValueError("Empty split. Add more history or reduce validation/test days.")
    return train, val, test


# ----------------------------
# Forecasting models
# ----------------------------
FEATURE_EXCLUDE = {"date", "sku", "demand", "SeasonalNaive"}


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in FEATURE_EXCLUDE]


def xgboost_device() -> str:
    """Use CUDA only when an NVIDIA GPU is actually visible to the container."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5, check=False
        )
        if result.returncode == 0 and result.stdout.strip():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def fit_models(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, cfg: dict, seed: int):
    cols = feature_columns(train)
    X_train, y_train = train[cols], train["demand"]
    X_val, X_test = val[cols], test[cols]

    p_val: dict[str, np.ndarray] = {"SeasonalNaive": np.maximum(val["SeasonalNaive"].to_numpy(), 0)}
    p_test: dict[str, np.ndarray] = {"SeasonalNaive": np.maximum(test["SeasonalNaive"].to_numpy(), 0)}
    trained = {}

    rc = cfg["models"]["random_forest"]
    rf = RandomForestRegressor(
        n_estimators=int(rc["n_estimators"]),
        max_depth=int(rc["max_depth"]),
        min_samples_leaf=int(rc["min_samples_leaf"]),
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    p_val["RandomForest"] = np.maximum(rf.predict(X_val), 0)
    p_test["RandomForest"] = np.maximum(rf.predict(X_test), 0)
    trained["RandomForest"] = rf

    hc = cfg["models"]["hist_gradient_boosting"]
    hgb = HistGradientBoostingRegressor(
        max_iter=int(hc["max_iter"]),
        learning_rate=float(hc["learning_rate"]),
        max_leaf_nodes=int(hc["max_leaf_nodes"]),
        random_state=seed,
    )
    hgb.fit(X_train, y_train)
    p_val["HistGradientBoosting"] = np.maximum(hgb.predict(X_val), 0)
    p_test["HistGradientBoosting"] = np.maximum(hgb.predict(X_test), 0)
    trained["HistGradientBoosting"] = hgb

    xc = cfg["models"]["xgboost"]
    device = xgboost_device()
    print(f"XGBoost device: {device}")
    def make_xgb(dev: str):
        return XGBRegressor(
            n_estimators=int(xc["n_estimators"]),
            max_depth=int(xc["max_depth"]),
            learning_rate=float(xc["learning_rate"]),
            subsample=float(xc["subsample"]),
            colsample_bytree=float(xc["colsample_bytree"]),
            reg_alpha=float(xc["reg_alpha"]),
            reg_lambda=float(xc["reg_lambda"]),
            objective="reg:squarederror",
            tree_method="hist",
            device=dev,
            random_state=seed,
            n_jobs=-1,
        )

    xgb = make_xgb(device)
    try:
        xgb.fit(X_train, y_train, verbose=False)
        val_xgb = xgb.predict(X_val)
        test_xgb = xgb.predict(X_test)
    except Exception as exc:
        if device == "cuda":
            print(f"GPU XGBoost failed; retrying on CPU: {type(exc).__name__}: {exc}")
            xgb = make_xgb("cpu")
            xgb.fit(X_train, y_train, verbose=False)
            val_xgb = xgb.predict(X_val)
            test_xgb = xgb.predict(X_test)
        else:
            raise
    p_val["XGBoost"] = np.maximum(val_xgb, 0)
    p_test["XGBoost"] = np.maximum(test_xgb, 0)
    trained["XGBoost"] = xgb

    return p_val, p_test, trained, cols


# ----------------------------
# Forecast metrics + statistics
# ----------------------------

def forecast_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    err = p - y
    return {
        "MAE": mean_absolute_error(y, p),
        "RMSE": math.sqrt(mean_squared_error(y, p)),
        "WAPE_pct": 100 * np.abs(err).sum() / max(np.abs(y).sum(), 1e-12),
        "Bias": float(err.mean()),
        "R2": r2_score(y, p),
    }


def bootstrap_ci(values: np.ndarray, samples: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    boot = np.empty(samples, dtype=float)
    for i in range(samples):
        boot[i] = rng.choice(values, len(values), replace=True).mean()
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(values.mean()), float(lo), float(hi)


def forecast_statistics(test: pd.DataFrame, predictions: dict, samples: int, seed: int) -> pd.DataFrame:
    y = test["demand"].to_numpy()
    base_abs = np.abs(y - predictions["SeasonalNaive"])
    rows = []
    for model, pred in predictions.items():
        if model == "SeasonalNaive":
            continue
        model_abs = np.abs(y - pred)
        improvement = base_abs - model_abs
        try:
            stat, p = wilcoxon(base_abs, model_abs, alternative="greater", zero_method="zsplit")
        except ValueError:
            stat, p = np.nan, np.nan
        mean_diff, lo, hi = bootstrap_ci(improvement, samples, seed)
        rows.append({
            "domain": "forecast_accuracy",
            "comparison": f"{model} vs SeasonalNaive",
            "test": "paired Wilcoxon: baseline absolute error > model absolute error",
            "statistic": stat,
            "p_value": p,
            "mean_improvement": mean_diff,
            "bootstrap_95_CI_low": lo,
            "bootstrap_95_CI_high": hi,
        })
    return pd.DataFrame(rows)


# ----------------------------
# Decision intelligence layer
# ----------------------------

def residual_sigma(val: pd.DataFrame, pred: np.ndarray) -> dict[str, float]:
    temp = val[["sku", "demand"]].copy()
    temp["residual"] = temp["demand"].to_numpy() - pred
    global_sigma = max(float(temp["residual"].std(ddof=1)), 1.0)
    sigma = temp.groupby("sku")["residual"].std(ddof=1).fillna(global_sigma).clip(lower=1.0)
    return sigma.to_dict()


def simulate_inventory(
    frame: pd.DataFrame,
    pred: np.ndarray,
    sigma_by_sku: dict[str, float],
    z: float,
    inv: dict,
):
    """Periodic-review lost-sales simulation with risk-aware reorder and order-up-to thresholds."""
    data = frame[["date", "sku", "demand"]].copy()
    data["forecast"] = np.maximum(np.asarray(pred, dtype=float), 0)
    data = data.sort_values(["sku", "date"]).reset_index(drop=True)

    L = int(inv["lead_time_days"])
    init_days = float(inv["initial_stock_days"])
    h = float(inv["holding_cost"])
    shortage_penalty = float(inv["stockout_cost"])
    order_unit_cost = float(inv["ordering_cost"])

    sku_rows = []
    for sku, g in data.groupby("sku", sort=False):
        g = g.sort_values("date")
        sig = max(float(sigma_by_sku.get(sku, 1.0)), 1.0)
        on_hand = init_days * max(float(g["forecast"].mean()), 0.1)
        pipeline: dict[pd.Timestamp, float] = defaultdict(float)
        holding = shortage = ordering = demand_total = sales_total = 0.0
        stockout_days = 0

        for row in g.itertuples(index=False):
            dt = row.date
            if dt in pipeline:
                on_hand += pipeline.pop(dt)

            demand = max(float(row.demand), 0.0)
            fc = max(float(row.forecast), 0.0)

            sales = min(on_hand, demand)
            lost = demand - sales
            on_hand -= sales

            demand_total += demand
            sales_total += sales
            shortage += lost * shortage_penalty
            holding += on_hand * h
            if lost > 1e-12:
                stockout_days += 1

            in_transit = sum(pipeline.values())
            inventory_position = on_hand + in_transit

            reorder_point = L * fc + z * sig * math.sqrt(max(L, 1))
            order_up_to = (L + 1) * fc + z * sig * math.sqrt(L + 1)

            if inventory_position < reorder_point:
                qty = max(order_up_to - inventory_position, 0.0)
                pipeline[dt + pd.Timedelta(days=L)] += qty
                ordering += qty * order_unit_cost

        total_cost = holding + shortage + ordering
        sku_rows.append({
            "sku": sku,
            "total_cost": total_cost,
            "holding_cost": holding,
            "stockout_cost": shortage,
            "ordering_cost": ordering,
            "service_level": sales_total / max(demand_total, 1e-12),
            "stockout_day_rate": stockout_days / max(len(g), 1),
            "ending_inventory": on_hand,
        })

    sku_df = pd.DataFrame(sku_rows)
    agg = {
        "total_cost": sku_df["total_cost"].sum(),
        "holding_cost": sku_df["holding_cost"].sum(),
        "stockout_cost": sku_df["stockout_cost"].sum(),
        "ordering_cost": sku_df["ordering_cost"].sum(),
        "service_level": np.average(sku_df["service_level"]),
        "stockout_day_rate": np.average(sku_df["stockout_day_rate"]),
        "ending_inventory": sku_df["ending_inventory"].sum(),
    }
    return agg, sku_df


def tune_risk_z(val: pd.DataFrame, pred: np.ndarray, sigma: dict, inv: dict):
    candidates = []
    for z in inv["z_grid"]:
        agg, _ = simulate_inventory(val, pred, sigma, float(z), inv)
        candidates.append({"z": float(z), "validation_total_cost": agg["total_cost"]})
    best = min(candidates, key=lambda r: r["validation_total_cost"])["z"]
    return best, candidates


def inventory_statistics(base: pd.DataFrame, model_df: pd.DataFrame, model: str, samples: int, seed: int) -> dict:
    merged = base[["sku", "total_cost"]].merge(
        model_df[["sku", "total_cost"]], on="sku", suffixes=("_baseline", "_model")
    )
    improvement = merged["total_cost_baseline"].to_numpy() - merged["total_cost_model"].to_numpy()
    try:
        stat, p = wilcoxon(
            merged["total_cost_baseline"], merged["total_cost_model"],
            alternative="greater", zero_method="zsplit"
        )
    except ValueError:
        stat, p = np.nan, np.nan
    mean_diff, lo, hi = bootstrap_ci(improvement, samples, seed)
    return {
        "domain": "inventory_cost",
        "comparison": f"{model} vs SeasonalNaive",
        "test": "paired Wilcoxon: baseline per-SKU inventory cost > model cost",
        "statistic": stat,
        "p_value": p,
        "mean_improvement": mean_diff,
        "bootstrap_95_CI_low": lo,
        "bootstrap_95_CI_high": hi,
    }


# ----------------------------
# Reporting
# ----------------------------

def save_plots(forecast_df: pd.DataFrame, inventory_df: pd.DataFrame, out: Path) -> None:
    f = forecast_df.sort_values("RMSE")
    plt.figure(figsize=(9, 5))
    plt.bar(f["model"], f["RMSE"])
    plt.ylabel("RMSE")
    plt.title("Held-Out Demand Forecast RMSE")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out / "forecast_rmse.png", dpi=180)
    plt.close()

    i = inventory_df.sort_values("total_cost")
    plt.figure(figsize=(9, 5))
    plt.bar(i["model"], i["total_cost"])
    plt.ylabel("Total inventory cost")
    plt.title("Held-Out Inventory Decision Performance")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out / "inventory_cost.png", dpi=180)
    plt.close()


def save_summary(out: Path, raw: pd.DataFrame, train, val, test, fmetrics, imetrics) -> None:
    bf = fmetrics.sort_values("RMSE").iloc[0]
    bi = imetrics.sort_values("total_cost").iloc[0]
    baseline_cost = float(imetrics.loc[imetrics["model"] == "SeasonalNaive", "total_cost"].iloc[0])
    cost_reduction = 100 * (baseline_cost - float(bi["total_cost"])) / max(baseline_cost, 1e-12)

    text = f"""# Experiment Summary

## Research title
AI-Enabled Decision Intelligence for Demand Forecasting: A Data-Driven Inventory Risk Optimization Framework

## Dataset and split
- Raw rows: {len(raw):,}
- SKUs: {raw['sku'].nunique():,}
- Train: {train['date'].min().date()} to {train['date'].max().date()}
- Validation: {val['date'].min().date()} to {val['date'].max().date()}
- Test: {test['date'].min().date()} to {test['date'].max().date()}

## Best forecasting result
- Model: {bf['model']}
- RMSE: {bf['RMSE']:.4f}
- MAE: {bf['MAE']:.4f}
- WAPE: {bf['WAPE_pct']:.2f}%

## Best inventory decision result
- Model/policy: {bi['model']}
- Validation-selected risk z: {bi['tuned_z']:.2f}
- Test total cost: {bi['total_cost']:.2f}
- Service level: {100 * bi['service_level']:.2f}%
- Stockout-day rate: {100 * bi['stockout_day_rate']:.2f}%
- Cost change vs SeasonalNaive policy: {cost_reduction:.2f}% improvement

## Research interpretation
The experiment separates predictive performance from decision performance. Forecast models are evaluated on a chronological test set. Forecast uncertainty is calibrated using validation residuals only. The inventory risk parameter is selected using validation inventory cost only, then frozen before the held-out test simulation. This design reduces test-set leakage and directly evaluates whether AI-enabled forecasts improve downstream inventory-risk decisions.

See `statistical_tests.csv` before claiming that an observed improvement is statistically supported.
"""
    (out / "summary.md").write_text(text, encoding="utf-8")


# ----------------------------
# Main experiment
# ----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg["seed"])
    set_seed(seed)
    out = ensure_output_dir()

    print("[1/7] Loading data")
    raw = load_data(cfg, seed)

    print("[2/7] Engineering leakage-safe features")
    feat = make_features(
        raw,
        [int(v) for v in cfg["features"]["lags"]],
        [int(v) for v in cfg["features"]["rolling_windows"]],
    )
    train, val, test = chronological_split(
        feat,
        int(cfg["split"]["validation_days"]),
        int(cfg["split"]["test_days"]),
    )
    print(f"Rows train/val/test = {len(train):,}/{len(val):,}/{len(test):,}")

    print("[3/7] Training forecasting models")
    p_val, p_test, models, cols = fit_models(train, val, test, cfg, seed)

    print("[4/7] Evaluating forecast performance")
    fm_rows = []
    y_test = test["demand"].to_numpy()
    for name, pred in p_test.items():
        row = {"model": name}
        row.update(forecast_metrics(y_test, pred))
        fm_rows.append(row)
    fmetrics = pd.DataFrame(fm_rows).sort_values("RMSE").reset_index(drop=True)
    fmetrics.to_csv(out / "forecast_metrics.csv", index=False)

    predictions_export = test[["date", "sku", "demand"]].reset_index(drop=True).copy()
    for name, pred in p_test.items():
        predictions_export[name] = pred
    predictions_export.to_csv(out / "test_predictions.csv", index=False)

    print("[5/7] Calibrating decision intelligence layer on validation data")
    inv = cfg["inventory"]
    inventory_rows = []
    per_sku = {}
    tuning_rows = []

    for name in p_val:
        sigma = residual_sigma(val, p_val[name])
        best_z, candidates = tune_risk_z(val, p_val[name], sigma, inv)
        for r in candidates:
            tuning_rows.append({"model": name, **r})
        agg, sku_df = simulate_inventory(test, p_test[name], sigma, best_z, inv)
        inventory_rows.append({"model": name, "tuned_z": best_z, **agg})
        per_sku[name] = sku_df

    imetrics = pd.DataFrame(inventory_rows).sort_values("total_cost").reset_index(drop=True)
    imetrics.to_csv(out / "inventory_metrics.csv", index=False)
    pd.DataFrame(tuning_rows).to_csv(out / "z_tuning.csv", index=False)

    print("[6/7] Statistical tests and artifacts")
    samples = int(cfg["statistics"]["bootstrap_samples"])
    stats_parts = [forecast_statistics(test, p_test, samples, seed)]
    baseline_sku = per_sku["SeasonalNaive"]
    inv_stats = []
    for name, sku_df in per_sku.items():
        if name != "SeasonalNaive":
            inv_stats.append(inventory_statistics(baseline_sku, sku_df, name, samples, seed))
    stats_parts.append(pd.DataFrame(inv_stats))
    stats = pd.concat(stats_parts, ignore_index=True)
    stats.to_csv(out / "statistical_tests.csv", index=False)

    joblib.dump({"feature_columns": cols, "models": models}, out / "models.joblib")
    save_plots(fmetrics, imetrics, out)
    save_summary(out, raw, train, val, test, fmetrics, imetrics)

    print("[7/7] DONE")
    print("\nForecast metrics:\n", fmetrics.to_string(index=False))
    print("\nInventory metrics:\n", imetrics.to_string(index=False))
    print(f"\nResults written to: {out.resolve()}")


if __name__ == "__main__":
    main()
