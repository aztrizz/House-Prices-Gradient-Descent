# House Prices Regression

A linear regression project predicting house prices from structural and location features, implementing gradient descent **from scratch** with NumPy and validating results against scikit-learn's `LinearRegression`.

## Overview

This notebook walks through a full regression workflow:

1. Explore a raw housing dataset and visualize each feature against price
2. Implement the core regression machinery manually (hypothesis function, cost function, batch gradient descent)
3. Train a baseline model on the raw features
4. Engineer new features based on patterns found in binned scatter plots
5. Retrain on the engineered feature set and compare performance

The custom gradient descent implementation is cross-checked against scikit-learn's `Pipeline` + `LinearRegression` + `StandardScaler` at each stage to confirm correctness.

## Dataset

The dataset (`regression_dataset.csv`) contains the following columns:

| Column | Description |
|---|---|
| `house_size_sqft` | Size of the house in square feet |
| `house_age_years` | Age of the house in years |
| `num_bathrooms` | Number of bathrooms |
| `distance_to_center_miles` | Distance from the city center, in miles |
| `neighborhood_score` | A neighborhood desirability score |
| `price_thousands` | Target variable — sale price, in thousands |

> **Note:** The notebook currently loads the data from a local Windows path (`C:/Users/.../regression_dataset.csv`). Update this path to point to your own copy of the dataset before running.

## Approach

### 1. Baseline model
The raw features are scaled with `StandardScaler` and fit using a hand-written batch gradient descent implementation (50 iterations, learning rate 0.3).

### 2. Feature engineering
Binned scatter plots of price against each feature (and simple transformations/interactions) are used to identify better-behaved, more linear relationships. This leads to four engineered features:

- `sqrt_house_size` — square root of house size
- `age_per_score` — house age adjusted by neighborhood score
- `baths*size` — interaction between bathrooms and house size
- `distance/size` — distance to center normalized by house size

### 3. Final model
The engineered feature set is scaled and retrained with more iterations (1000) and a higher learning rate (0.5), improving fit over the baseline.

## Results

| Model | MSE | MAE | R² |
|---|---|---|---|
| Baseline (raw features) | 347.77 | 15.00 | 0.9165 |
| Final (engineered features) | 292.01 | 13.87 | 0.9299 |

In both cases, the custom gradient descent solution matches scikit-learn's closed-form `LinearRegression` R² to within floating-point precision, confirming the manual implementation is correct.

## Project Structure

```
.
├── House_Prices_Regression.ipynb   # Main notebook
└── regression_dataset.csv          # Dataset (not included — see Dataset section)
```

## Requirements

- Python 3.x
- numpy
- pandas
- scikit-learn
- matplotlib

Install with:

```bash
pip install numpy pandas scikit-learn matplotlib
```

## Usage

1. Place `regression_dataset.csv` in the project directory (or update the file path in the notebook).
2. Open `House_Prices_Regression.ipynb` in Jupyter.
3. Run all cells in order — the notebook progresses linearly from data exploration through to the final evaluated model.

## Key Functions

- `hypothesis(X, theta)` — computes linear predictions `X · theta`
- `compute_cost(X, y, theta)` — mean squared error cost function
- `batch_gradient_descent(X, y, theta, learning_rate, iterations)` — runs batch gradient descent and returns the optimized parameters plus cost history
- `binned_scatter(x, y, bins)` — plots mean(y) within equal-width bins of x, used to explore feature relationships
