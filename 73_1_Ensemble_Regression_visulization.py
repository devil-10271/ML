import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor,
    HuberRegressor, BayesianRidge,
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
    AdaBoostRegressor, BaggingRegressor, VotingRegressor,
)
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor

st.set_page_config(page_title="Voting Regressor Explorer", layout="wide")

# ---------------- Regressor catalog (broad slice of sklearn) ----------------
ESTIMATOR_FACTORIES = {
    "Linear Regression": lambda: LinearRegression(),
    "Ridge": lambda: Ridge(alpha=1.0),
    "Lasso": lambda: Lasso(alpha=0.05),
    "ElasticNet": lambda: ElasticNet(alpha=0.05, l1_ratio=0.5),
    "SGD Regressor": lambda: make_pipeline(StandardScaler(), SGDRegressor(max_iter=2000, random_state=0)),
    "Huber Regressor": lambda: HuberRegressor(),
    "Bayesian Ridge": lambda: BayesianRidge(),
    "Polynomial Regression (deg 3)": lambda: make_pipeline(PolynomialFeatures(degree=3), LinearRegression()),
    "SVR": lambda: make_pipeline(StandardScaler(), SVR(kernel="rbf", C=10, gamma=0.5)),
    "KNN Regressor": lambda: KNeighborsRegressor(n_neighbors=7),
    "Decision Tree Regressor": lambda: DecisionTreeRegressor(max_depth=5, random_state=0),
    "Random Forest Regressor": lambda: RandomForestRegressor(n_estimators=150, max_depth=6, random_state=0),
    "Extra Trees Regressor": lambda: ExtraTreesRegressor(n_estimators=150, max_depth=6, random_state=0),
    "Gradient Boosting Regressor": lambda: GradientBoostingRegressor(n_estimators=150, max_depth=3, random_state=0),
    "AdaBoost Regressor": lambda: AdaBoostRegressor(n_estimators=100, random_state=0),
    "Bagging Regressor": lambda: BaggingRegressor(n_estimators=50, random_state=0),
    "MLP Regressor": lambda: make_pipeline(
        StandardScaler(), MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=3000, random_state=0)
    ),
    "Gaussian Process Regressor": lambda: make_pipeline(StandardScaler(), GaussianProcessRegressor(random_state=0)),
}

LINE_STYLES = ["--", "-.", ":", "--", "-.", ":", "--", "-.", ":", "--", "-.", ":", "--", "-.", ":", "--", "-."]
LINE_COLORS = [
    "#2CA02C", "#D62728", "#9467BD", "#8C564B", "#E377C2", "#7F7F7F",
    "#BCBD22", "#17BECF", "#FF7F0E", "#1F77B4", "#AEC7E8", "#FFBB78",
    "#98DF8A", "#FF9896", "#C5B0D5", "#C49C94", "#F7B6D2",
]


def make_regression_data(n=140, noise=0.28, random_state=0):
    rng = np.random.RandomState(random_state)
    X = np.sort(rng.uniform(0, 5, n))
    y = (
        1.6 * np.sin(0.9 * X) * np.exp(-0.12 * X)
        - 0.35 * np.tanh(X - 2.5)
        + rng.normal(scale=noise, size=n)
    )
    return X.reshape(-1, 1), y


# ---------------- Sidebar ----------------
with st.sidebar:
    st.title("Voting Regressor")
    st.caption("Ensemble regression curves, live.")

    estimator_labels = st.multiselect(
        "Estimators",
        list(ESTIMATOR_FACTORIES.keys()),
        default=["Linear Regression", "SVR", "Decision Tree Regressor"],
    )

    with st.expander("Data settings"):
        n_points = st.slider("Number of points", 40, 300, 140, step=10)
        noise_level = st.slider("Noise", 0.0, 1.0, 0.28, step=0.02)
        test_size = st.slider("Test size", 0.1, 0.4, 0.2, step=0.05)

    run = st.button("Run Algorithm", type="primary", use_container_width=True)

if "seed" not in st.session_state:
    st.session_state.seed = 0
if run:
    st.session_state.seed += 1

st.title("Voting Ensemble | Regression")

if not estimator_labels:
    st.info("Pick at least one estimator from the sidebar to run the ensemble.")
    st.stop()

# ---------------- Data + models ----------------
X, y = make_regression_data(n=n_points, noise=noise_level, random_state=st.session_state.seed)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=st.session_state.seed
)

fitted = []
metrics = {}
for label in estimator_labels:
    reg = ESTIMATOR_FACTORIES[label]()
    reg.fit(X_train, y_train)
    pred = reg.predict(X_test)
    metrics[label] = {"r2": r2_score(y_test, pred), "mae": mean_absolute_error(y_test, pred)}
    fitted.append((label, reg))

voting_reg = VotingRegressor(estimators=fitted)
voting_reg.fit(X_train, y_train)
voting_pred_test = voting_reg.predict(X_test)
voting_r2 = r2_score(y_test, voting_pred_test)
voting_mae = mean_absolute_error(y_test, voting_pred_test)

# ---------------- Metrics ----------------
st.subheader("Regression Metrics")
cols = st.columns(len(estimator_labels) + 1)
cols[0].metric("Voting Regressor R²", f"{voting_r2:.2f}", f"MAE {voting_mae:.2f}")
for col, label in zip(cols[1:], estimator_labels):
    col.metric(label, f"R² {metrics[label]['r2']:.2f}", f"MAE {metrics[label]['mae']:.2f}")

# ---------------- Plot ----------------
x_grid = np.linspace(X.min(), X.max(), 400).reshape(-1, 1)

fig, ax = plt.subplots(figsize=(11, 6.5))
ax.scatter(
    X, y, facecolor="#FFD447", edgecolor="black", linewidth=0.9, s=70, zorder=3, label="_nolegend_"
)

for i, (label, reg) in enumerate(fitted):
    y_line = reg.predict(x_grid)
    ax.plot(
        x_grid.ravel(), y_line,
        linestyle=LINE_STYLES[i % len(LINE_STYLES)],
        color=LINE_COLORS[i % len(LINE_COLORS)],
        linewidth=1.6, label=label, zorder=2,
    )

voting_line = voting_reg.predict(x_grid)
ax.plot(x_grid.ravel(), voting_line, color="#1F4FD8", linewidth=3.2, label="Voting Regressor", zorder=4)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend(loc="upper right", frameon=True, fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
st.pyplot(fig, use_container_width=True)

# ---------------- Individual estimator mini-plots ----------------
st.subheader("Individual estimators")
sub_cols = st.columns(min(len(fitted), 4)) if len(fitted) <= 4 else st.columns(4)
for i, (label, reg) in enumerate(fitted):
    col = sub_cols[i % len(sub_cols)]
    with col:
        fig_i, ax_i = plt.subplots(figsize=(4, 3))
        ax_i.scatter(X, y, facecolor="#FFD447", edgecolor="black", linewidth=0.6, s=22, zorder=2)
        ax_i.plot(x_grid.ravel(), reg.predict(x_grid), color=LINE_COLORS[i % len(LINE_COLORS)], linewidth=2, zorder=3)
        ax_i.set_title(f"{label}\nR² {metrics[label]['r2']:.2f} · MAE {metrics[label]['mae']:.2f}", fontsize=9)
        ax_i.set_xticks([])
        ax_i.set_yticks([])
        st.pyplot(fig_i, use_container_width=True)