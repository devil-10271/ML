import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import streamlit as st
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Voting Classifier Explorer", layout="wide")

CMAP = mcolors.LinearSegmentedColormap.from_list("cls", ["#7C6FF0", "#FF6E5A"])

DATASETS = ["U-Shaped", "Circles", "Blobs", "XOR", "Linear"]

ESTIMATOR_FACTORIES = {
    "Logistic Regression": lambda: LogisticRegression(),
    "SVM": lambda: SVC(kernel="rbf", gamma=1.2, probability=True),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=150, max_depth=6),
    "KNN": lambda: KNeighborsClassifier(n_neighbors=9),
    "Decision Tree": lambda: DecisionTreeClassifier(max_depth=6),
}


def make_dataset(name, n=300, noise=0.25, random_state=0):
    rng = np.random.RandomState(random_state)
    if name == "U-Shaped":
        X, y = make_moons(n_samples=n, noise=noise, random_state=random_state)
    elif name == "Circles":
        X, y = make_circles(n_samples=n, noise=noise * 0.5, factor=0.4, random_state=random_state)
    elif name == "Blobs":
        X, y = make_blobs(
            n_samples=n, centers=[(-1.2, -1.0), (1.2, 1.0)], cluster_std=0.9, random_state=random_state
        )
    elif name == "XOR":
        X = rng.uniform(-2, 2, size=(n, 2))
        y = (X[:, 0] * X[:, 1] > 0).astype(int)
        X = X + rng.normal(scale=noise * 0.5, size=X.shape)
    elif name == "Linear":
        X = rng.uniform(-2.2, 2.2, size=(n, 2))
        y = (X[:, 1] - (0.5 * X[:, 0] + 0.1) + rng.normal(scale=noise, size=n) > 0).astype(int)
    else:
        raise ValueError(name)
    return X, y.astype(int)


def plot_boundary(ax, clf, X, y, title, acc=None, point_size=26):
    x_min, x_max = X[:, 0].min() - 0.6, X[:, 0].max() + 0.6
    y_min, y_max = X[:, 1].min() - 0.6, X[:, 1].max() + 0.6
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 220), np.linspace(y_min, y_max, 220))
    grid = np.c_[xx.ravel(), yy.ravel()]

    if hasattr(clf, "predict_proba"):
        Z = clf.predict_proba(grid)[:, 1]
    else:
        Z = clf.predict(grid).astype(float)
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, levels=60, cmap=CMAP, alpha=0.9, vmin=0, vmax=1)
    ax.scatter(
        X[:, 0], X[:, 1], c=y, cmap=CMAP, vmin=0, vmax=1,
        edgecolor="white", linewidth=0.6, s=point_size, zorder=3,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    subtitle = f"{title}" + (f"  ·  acc {acc:.2f}" if acc is not None else "")
    ax.set_title(subtitle, fontsize=11, fontweight="bold", pad=8)


# ---------------- Sidebar ----------------
with st.sidebar:
    st.title("🗳️ Voting Classifier")
    st.caption("Ensemble decision boundaries, live.")

    dataset_label = st.selectbox("Dataset", DATASETS, index=0)

    estimator_labels = st.multiselect(
        "Estimators",
        list(ESTIMATOR_FACTORIES.keys()),
        default=["Logistic Regression", "SVM", "Random Forest"],
    )

    voting_type = st.radio("Voting type", ["hard", "soft"], horizontal=True)

    run = st.button("Run Algorithm", type="primary", use_container_width=True)

if "seed" not in st.session_state:
    st.session_state.seed = 0
if run:
    st.session_state.seed += 1

st.title(dataset_label)

if not estimator_labels:
    st.info("Pick at least one estimator from the sidebar to run the ensemble.")
    st.stop()

# ---------------- Data + models ----------------
X, y = make_dataset(dataset_label, n=300, noise=0.25, random_state=st.session_state.seed)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=st.session_state.seed, stratify=y
)

fitted = []
accs = {}
for label in estimator_labels:
    clf = ESTIMATOR_FACTORIES[label]()
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    fitted.append((label, clf))
    accs[label] = acc

voting_clf = VotingClassifier(estimators=fitted, voting=voting_type)
voting_clf.fit(X_train, y_train)
voting_acc = accuracy_score(y_test, voting_clf.predict(X_test))

# ---------------- Metrics row ----------------
cols = st.columns(len(estimator_labels) + 1)
cols[0].metric("Voting Classifier", f"{voting_acc:.2f}")
for col, label in zip(cols[1:], estimator_labels):
    col.metric(label, f"{accs[label]:.2f}")

# ---------------- Main decision boundary ----------------
fig, ax = plt.subplots(figsize=(9, 6))
plot_boundary(ax, voting_clf, X, y, "Voting Classifier", voting_acc, point_size=32)
st.pyplot(fig, use_container_width=True)

# ---------------- Individual estimators ----------------
st.subheader("Individual estimators")
sub_cols = st.columns(len(fitted))
for col, (label, clf) in zip(sub_cols, fitted):
    with col:
        fig_i, ax_i = plt.subplots(figsize=(4, 3))
        plot_boundary(ax_i, clf, X, y, label, accs[label], point_size=14)
        st.pyplot(fig_i, use_container_width=True)