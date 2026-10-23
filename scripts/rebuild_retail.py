"""Rebuild a transparent 30-SKU daily panel from UCI Online Retail.

This is a NEW deterministic preprocessing protocol, not a claim to reproduce
the exact SKU selection or promotional flags of the previous Colab run.
"""
from pathlib import Path
from io import BytesIO
import zipfile
import urllib.request
import numpy as np
import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
OUT = Path("data/real_retail_daily.csv")

def main():
    print("Downloading UCI Online Retail dataset...")
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0 research-reproducibility"})
    with urllib.request.urlopen(req, timeout=180) as response:
        payload = response.read()
    with zipfile.ZipFile(BytesIO(payload)) as z:
        files = [n for n in z.namelist() if n.lower().endswith(".xlsx")]
        if not files:
            raise RuntimeError("UCI archive did not contain an XLSX file")
        with z.open(files[0]) as f:
            raw = pd.read_excel(f, engine="openpyxl")
    raw["InvoiceDate"] = pd.to_datetime(raw["InvoiceDate"], errors="coerce")
    raw["Quantity"] = pd.to_numeric(raw["Quantity"], errors="coerce")
    raw["UnitPrice"] = pd.to_numeric(raw["UnitPrice"], errors="coerce")
    # Remove returns, cancellations and non-positive prices.
    valid = raw.loc[
        raw["InvoiceDate"].notna()
        & raw["StockCode"].notna()
        & raw["Quantity"].gt(0)
        & raw["UnitPrice"].gt(0)
        & ~raw["InvoiceNo"].astype(str).str.startswith("C")
    ].copy()
    valid["sku"] = valid["StockCode"].astype(str)
    valid["date"] = valid["InvoiceDate"].dt.normalize()
    # Choose SKUs on a prespecified reproducible rule. Selection across the
    # full observation window is a descriptive cohort choice, not a causal design.
    ranks = valid.groupby("sku")["Quantity"].sum().sort_values(ascending=False)
    skus = list(ranks.head(30).index)
    valid = valid.loc[valid["sku"].isin(skus)]
    daily = valid.groupby(["date", "sku"], as_index=False).agg(
        demand=("Quantity", "sum"),
        sales_value=("UnitPrice", lambda s: 0.0),
    )
    # Quantity-weighted daily unit price.
    valid["value"] = valid["Quantity"] * valid["UnitPrice"]
    price = valid.groupby(["date", "sku"], as_index=False).agg(
        total_value=("value", "sum"), total_quantity=("Quantity", "sum")
    )
    price["price"] = price["total_value"] / price["total_quantity"]
    daily = daily.drop(columns="sales_value").merge(price[["date", "sku", "price"]], on=["date", "sku"])
    dates = pd.date_range(valid["date"].min(), valid["date"].max(), freq="D")
    index = pd.MultiIndex.from_product([dates, skus], names=["date", "sku"])
    panel = daily.set_index(["date", "sku"]).reindex(index).reset_index()
    panel["demand"] = panel["demand"].fillna(0).astype(float)
    panel["price"] = panel.groupby("sku")["price"].transform(lambda x: x.ffill().bfill())
    panel["price"] = panel["price"].fillna(float(valid["UnitPrice"].median()))
    # No genuine promotion indicator is present in UCI Online Retail.
    panel["promo"] = 0
    panel = panel[["date", "sku", "demand", "price", "promo"]].sort_values(["sku", "date"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUT, index=False)
    print(f"Saved {OUT}: {len(panel):,} rows, {panel.sku.nunique()} SKUs, {len(dates)} days")
    print("NOTE: promo is 0 because the source has no observed promotion field.")
    print("NOTE: this is a new cohort and may not match prior experimental results.")

if __name__ == "__main__":
    main()
