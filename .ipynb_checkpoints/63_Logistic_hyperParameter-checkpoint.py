# streamlit run 63_Logistic_hyperParameter.py

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def load_initial_graph(dataset, ax):
    if dataset == "Binary":
        X, y = make_blobs(n_features=2, centers=2, random_state=6)
    else:
        X, y = make_blobs(n_features=2, centers=3, random_state=2)

    ax.scatter(X.T[0], X.T[1], c=y, cmap="rainbow", edgecolors="k")
    return X, y


def draw_meshgrid(X):
    a = np.arange(start=X[:, 0].min() - 1, stop=X[:, 0].max() + 1, step=0.01)
    b = np.arange(start=X[:, 1].min() - 1, stop=X[:, 1].max() + 1, step=0.01)

    XX, YY = np.meshgrid(a, b)
    input_array = np.array([XX.ravel(), YY.ravel()]).T

    return XX, YY, input_array


plt.style.use("fivethirtyeight")

st.sidebar.markdown("# Logistic Regression Classifier")

dataset = st.sidebar.selectbox("Select Dataset", ("Binary", "Multiclass"))

penalty = st.sidebar.selectbox(
    "Regularization", ("l2", "l1", "elasticnet", "none")
)

c_input = float(st.sidebar.number_input("C", value=1.0, min_value=0.001))

solver = st.sidebar.selectbox(
    "Solver", ("lbfgs", "newton-cg", "liblinear", "sag", "saga")
)

max_iter = int(st.sidebar.number_input("Max Iterations", value=100, min_value=1))

# l1_ratio is a float between 0 and 1, only used when penalty is 'elasticnet'
l1_ratio = None
if penalty == "elasticnet":
    l1_ratio = float(
        st.sidebar.slider("l1 Ratio", min_value=0.0, max_value=1.0, value=0.5)
    )

# Load and render initial graph
fig, ax = plt.subplots()
X, y = load_initial_graph(dataset, ax)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
graph_placeholder = st.pyplot(fig)

if st.sidebar.button("Run Algorithm"):
    penalty_param = None if penalty == "none" else penalty

    try:
        # Removed multi_class keyword argument to resolve TypeError
        clf = LogisticRegression(
            penalty=penalty_param,
            C=c_input,
            solver=solver,
            max_iter=max_iter,
            l1_ratio=l1_ratio,
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        # Clear and redraw decision boundary + scatter points
        ax.clear()
        XX, YY, input_array = draw_meshgrid(X)
        labels = clf.predict(input_array)

        ax.contourf(
            XX, YY, labels.reshape(XX.shape), alpha=0.4, cmap="rainbow"
        )
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap="rainbow", edgecolors="k")
        ax.set_xlabel("Col1")
        ax.set_ylabel("Col2")

        graph_placeholder.pyplot(fig)
        st.subheader(
            f"Accuracy for Logistic Regression: {accuracy_score(y_test, y_pred):.2f}"
        )

    except Exception as e:
        st.error(f"Invalid Hyperparameter Combination: {e}")