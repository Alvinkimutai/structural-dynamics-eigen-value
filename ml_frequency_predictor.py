"""
ml_frequency_predictor.py
=========================
Predicts the fundamental natural frequency (Mode 1) of multi-storey
shear-frame buildings using three ML models:
  1. Linear Regression
  2. Decision Tree Regression
  3. Neural Network (MLPRegressor)

Dataset is synthetically generated using the shear-frame eigenvalue
method (same physics as eigen_analysis.py) over a parametric sweep
of realistic building configurations.
"""

import numpy as np
from scipy import linalg
from sklearn.linear_model    import LinearRegression
from sklearn.tree            import DecisionTreeRegressor
from sklearn.neural_network  import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. PHYSICS ENGINE — compute f1 for any building configuration
# =============================================================================

def compute_fundamental_frequency(n_storeys, storey_h, b_col, d_col,
                                   n_cols, floor_mass):
    """
    Returns fundamental frequency (Hz) for a uniform shear-frame building.
    Uses fixed-fixed column stiffness: k = 12EI/h^3
    """
    fck = 25e6
    E   = 9500 * (fck / 1e6 + 8) ** (1/3) * 1e6   # EC2 C25 concrete

    I_col    = (b_col * d_col**3) / 12
    k_col    = 12 * E * I_col / storey_h**3
    k_storey = n_cols * k_col

    M = np.diag([floor_mass] * n_storeys)

    K = np.zeros((n_storeys, n_storeys))
    for i in range(n_storeys):
        if i < n_storeys - 1:
            K[i,   i  ] += k_storey
            K[i+1, i+1] += k_storey
            K[i,   i+1] -= k_storey
            K[i+1, i  ] -= k_storey
    K[0, 0] += k_storey

    eigenvalues, _ = linalg.eigh(K, M)
    omega1 = np.sqrt(eigenvalues[0])
    return omega1 / (2 * np.pi)


# =============================================================================
# 2. SYNTHETIC DATASET GENERATION
# =============================================================================

rng = np.random.default_rng(42)
N   = 2000      # number of building samples

# Parameter ranges (physically realistic)
n_storeys_arr = rng.integers(3,  20,  N).astype(float)   # 3–19 storeys
storey_h_arr  = rng.uniform(2.8, 4.0, N)                 # m
b_col_arr     = rng.uniform(0.15, 0.40, N)               # m
d_col_arr     = rng.uniform(0.30, 0.80, N)               # m
n_cols_arr    = rng.integers(4,  25,  N).astype(float)   # columns per floor
floor_mass_arr = rng.uniform(50e3, 300e3, N)             # kg

# Derived features (physically meaningful combinations)
I_col_arr     = (b_col_arr * d_col_arr**3) / 12          # m⁴
k_storey_arr  = n_cols_arr * 12 * (9500 * 33**0.333 * 1e6) * I_col_arr / storey_h_arr**3
total_mass_arr = n_storeys_arr * floor_mass_arr           # kg
building_h_arr = n_storeys_arr * storey_h_arr             # m

# Compute true fundamental frequency for each sample
f1_arr = np.array([
    compute_fundamental_frequency(
        int(n_storeys_arr[i]), storey_h_arr[i],
        b_col_arr[i], d_col_arr[i],
        int(n_cols_arr[i]), floor_mass_arr[i]
    )
    for i in range(N)
])

# --- Feature matrix ---
# [n_storeys, building_height, b_col, d_col, I_col, n_cols, floor_mass, total_mass, k_storey]
X = np.column_stack([
    n_storeys_arr,
    building_h_arr,
    b_col_arr,
    d_col_arr,
    I_col_arr,
    n_cols_arr,
    floor_mass_arr,
    total_mass_arr,
    k_storey_arr,
])
y = f1_arr

feature_names = [
    "No. of Storeys", "Building Height (m)", "Col. Width b (m)",
    "Col. Depth d (m)", "Col. Inertia I (m⁴)", "No. of Columns",
    "Floor Mass (kg)", "Total Mass (kg)", "Storey Stiffness k (N/m)"
]

print("=" * 60)
print("SYNTHETIC DATASET SUMMARY")
print("=" * 60)
print(f"  Samples generated : {N}")
print(f"  f1 range          : {f1_arr.min():.3f} – {f1_arr.max():.3f} Hz")
print(f"  f1 mean           : {f1_arr.mean():.3f} Hz")
print()

# =============================================================================
# 3. TRAIN / TEST SPLIT + SCALING
# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler  = StandardScaler()
Xs_train = scaler.fit_transform(X_train)
Xs_test  = scaler.transform(X_test)

# =============================================================================
# 4. MODELS
# =============================================================================

models = {
    "Linear Regression" : LinearRegression(),
    "Decision Tree"     : DecisionTreeRegressor(max_depth=10, random_state=42),
    "Neural Network"    : MLPRegressor(
                            hidden_layer_sizes=(128, 64, 32),
                            activation='relu',
                            max_iter=1000,
                            random_state=42,
                            learning_rate_init=0.001
                          ),
}

results  = {}
y_preds  = {}

print("=" * 60)
print("MODEL TRAINING & EVALUATION")
print("=" * 60)
print(f"{'Model':<22} {'MAE (Hz)':<14} {'R² Score':<12} {'Max Err (Hz)'}")
print("-" * 60)

for name, model in models.items():
    # Neural network and Linear Regression use scaled features
    if name == "Decision Tree":
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    else:
        model.fit(Xs_train, y_train)
        y_pred = model.predict(Xs_test)

    mae     = mean_absolute_error(y_test, y_pred)
    r2      = r2_score(y_test, y_pred)
    max_err = np.max(np.abs(y_test - y_pred))

    results[name]  = {"MAE": mae, "R2": r2, "MaxErr": max_err}
    y_preds[name]  = y_pred

    print(f"  {name:<20} {mae:<14.4f} {r2:<12.4f} {max_err:.4f}")

print()

# =============================================================================
# 5. EXAMPLE PREDICTION — 10-storey Kenya building from eigen_analysis.py
# =============================================================================

kenya_building = np.array([[
    10,                     # n_storeys
    10 * 3.0,               # building height (m)
    0.200,                  # b_col (m)
    0.700,                  # d_col (m)
    (0.200 * 0.700**3)/12,  # I_col (m⁴)
    16,                     # n_cols
    153980,                 # floor_mass (kg)
    10 * 153980,            # total_mass (kg)
    16 * 12 * (9500*33**0.333*1e6) * (0.200*0.700**3/12) / 3.0**3  # k_storey
]])

true_f1 = compute_fundamental_frequency(10, 3.0, 0.200, 0.700, 16, 153980)

print("=" * 60)
print("EXAMPLE: 10-Storey Kenya Building Prediction")
print("=" * 60)
print(f"  True f1 (eigenvalue) : {true_f1:.4f} Hz")
print()
print(f"  {'Model':<22} {'Predicted f1 (Hz)':<20} {'Error (%)'}")
print(f"  {'-'*55}")
for name, model in models.items():
    if name == "Decision Tree":
        pred = model.predict(kenya_building)[0]
    else:
        pred = model.predict(scaler.transform(kenya_building))[0]
    err_pct = abs(pred - true_f1) / true_f1 * 100
    print(f"  {name:<22} {pred:<20.4f} {err_pct:.2f}%")

# =============================================================================
# 6. PLOTS
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("ML Model Performance — Predicted vs True Fundamental Frequency",
             fontsize=13)

colors = ['#2980b9', '#27ae60', '#c0392b']

for ax, (name, y_pred), color in zip(axes, y_preds.items(), colors):
    ax.scatter(y_test, y_pred, alpha=0.35, s=12, color=color, label='Samples')
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, 'k--', linewidth=1.2, label='Perfect fit')
    ax.set_xlabel("True f₁ (Hz)")
    ax.set_ylabel("Predicted f₁ (Hz)")
    ax.set_title(f"{name}\nR² = {results[name]['R2']:.4f}  |  MAE = {results[name]['MAE']:.4f} Hz",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("ml_predictions.png", dpi=150, bbox_inches='tight')
print("\nSaved: ml_predictions.png")

# --- Bar chart: model comparison ---
fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))
fig2.suptitle("Model Comparison", fontsize=12)

model_names = list(results.keys())
maes  = [results[m]["MAE"]    for m in model_names]
r2s   = [results[m]["R2"]     for m in model_names]

axes2[0].bar(model_names, maes, color=colors, edgecolor='k', linewidth=0.7)
axes2[0].set_ylabel("MAE (Hz)")
axes2[0].set_title("Mean Absolute Error (lower = better)")
axes2[0].grid(axis='y', alpha=0.3)

axes2[1].bar(model_names, r2s, color=colors, edgecolor='k', linewidth=0.7)
axes2[1].set_ylabel("R² Score")
axes2[1].set_ylim(0, 1.05)
axes2[1].set_title("R² Score (higher = better)")
axes2[1].axhline(1.0, color='k', linestyle='--', linewidth=0.8)
axes2[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("ml_model_comparison.png", dpi=150, bbox_inches='tight')
print("Saved: ml_model_comparison.png")

plt.show()