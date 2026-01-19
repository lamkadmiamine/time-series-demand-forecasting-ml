## 📊 Benchmark Results

All models were evaluated on the same **strict temporal test set** using
multiple error metrics and execution time indicators.

| Model                     | MAE   | RMSE  | sMAPE | WAPE | Train Time (s) | Predict Time (s) |
|---------------------------|-------|-------|-------|-------|---------------|------------------|
| LightGBM                  | 11.10 | 28.34 | 50.63 | 0.269 | 2.70          | 0.13             |
| RandomForest              | 11.19 | 29.24 | 31.98 | 0.271 | 243.86        | 0.60             |
| RandomForest (Clustered)  | 11.20 | 28.96 | 32.18 | 0.272 | 266.91        | 0.74             |
| LSTM + LightGBM (Hybrid)  | 11.22 | 29.90 | 42.85 | 0.272 | 355.19        | 0.27             |
| LightGBM (Clustered)      | 11.24 | 28.04 | 50.91 | 0.272 | 9.15          | 0.50             |
| XGBoost                   | 11.30 | 28.84 | 50.73 | 0.274 | 7.23          | 0.30             |
| XGBoost (Clustered)       | 11.71 | 31.81 | 51.21 | 0.284 | 10.91         | 0.31             |
| LSTM                      | 26.97 | 71.58 | 77.49 | 0.654 | 483.22        | 1.64             |


