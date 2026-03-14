Energy Consumption: Time Series Clustering & Forecasting (Baseline)

Baseline implementation for:

Task 1 – Clustering (2023 data)
Task 2 – Forecasting

Cluster level: one model per cluster, predict each household’s 2024 daily consumption (366 days)
Dataset level: single global baseline model trained on entire 2023 dataset

Project Structure
/
│
├── data/
│   ├── energy_2023.csv          # input (wide format)
│   ├── energy_2024.csv          # input (wide format, used as ground truth)
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py          # load, extract, scale
│   ├── clustering.py             # baseline clustering (TimeSeriesKMeans)
│   ├── forecasting.py            # Prophet-based forecasting (cluster & dataset level)
│   ├── evaluation.py             # MAE, RMSE
│   └── experiment_runner.py      # orchestrates the baseline run
│
├── notebooks/
│   └── results_visualization.ipynb  # optional: plots & report narrative
│
└── README.md

Package Installations

pip install --upgrade pip
pip install pandas numpy scikit-learn tslearn matplotlib
pip install prophet --use-pep517
pip install cmdstanpy
pip install plotly

How to Run

python3 src/experiment_runner.py

Methods 

Preprocessing: row-wise StandardScaler so clustering emphasizes shape (daily pattern) rather than absolute consumption level.
Clustering (Task 1): TimeSeriesKMeans with Euclidean distance.

Use Silhouette to estimate a good k (run multiple k values and pick the one with higher silhouette and meaningful cluster profiles).


Forecasting (Task 2):

Cluster-level: one Prophet model per cluster trained on 2023; generates 366 predictions for each household in 2024.
Dataset-level (baseline): one global Prophet model trained on the average customer; used as a simple baseline.


Evaluation: MAE, RMSE, and MAPE functions in evaluation.py.
(The runner prints MAE/RMSE; extend easily to include MAPE.)

How to Run Your Experiments 

For preprocessing & clustering  & forecasting

1. Add functions in src/preprocessing.py, src/clustering.py and src/forecasting.py
2. Replace the call functions in all of them

Compare Performance 

Clustering: Silhouette score, visual cluster means
Forecasting: MAE/RMSE per cluster and overall
Then Pick the one with lower error of forecasting



