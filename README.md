# AI-Enabled Decision Intelligence for Demand Forecasting

**Title:** AI-Enabled Decision Intelligence for Demand Forecasting: A Data-Driven Inventory Risk Optimization Framework

## One-command Runpod run

Use an official Runpod PyTorch Pod, upload this repository, then run:

```bash
cd ai_decision_intelligence_inventory
chmod +x run.sh
./run.sh
```

The repository runs with included synthetic retail demand data by default, so you can verify the experiment immediately.

## Real dataset

Place your CSV in `data/`, for example `data/demand.csv`, and edit `config.yaml`:

```yaml
data:
  input_csv: data/demand.csv
```

Minimum columns are `date,sku,demand`. Optional useful columns are `price,promo`. Column names can be mapped in `config.yaml`.

## Experimental design

1. Chronological train / validation / test split.
2. Leakage-safe lag and rolling features.
3. Seasonal-naive benchmark.
4. Random Forest, Histogram Gradient Boosting, and XGBoost.
5. Validation residuals estimate forecast uncertainty.
6. Inventory risk parameter `z` is tuned on validation data only.
7. The final inventory policy is evaluated once on held-out test data.
8. Forecast and inventory improvements are tested using paired Wilcoxon tests and bootstrap confidence intervals.

## Outputs

- `outputs/forecast_metrics.csv`
- `outputs/inventory_metrics.csv`
- `outputs/statistical_tests.csv`
- `outputs/test_predictions.csv`
- `outputs/z_tuning.csv`
- `outputs/forecast_rmse.png`
- `outputs/inventory_cost.png`
- `outputs/summary.md`
- `outputs/models.joblib`

## Interpretation

The core contribution is not only prediction accuracy. It asks whether AI forecasts, combined with uncertainty-aware prescriptive inventory logic, improve operational decisions compared with a conventional seasonal forecast policy.
