import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.inspection import PartialDependenceDisplay


# Load Dataset Kaggle

df = pd.read_csv("housing.csv")
print(df.head())

# Preprocessing

# Fitur input & target
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

# Pisahkan tipe fitur
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = ["ocean_proximity"]

# Preprocessor: OneHot untuk kategori
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
], remainder="passthrough")


# Split Dataset

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Baseline Models

# Model 1: Linear Regression
model_lr = Pipeline([
    ("prep", preprocessor),
    ("reg", LinearRegression())
])

# Model 2: Random Forest
model_rf = Pipeline([
    ("prep", preprocessor),
    ("rf", RandomForestRegressor(n_estimators=150, random_state=42))
])

# Train kedua model
model_lr.fit(X_train, y_train)
model_rf.fit(X_train, y_train)

# Memilih Fitur Penting untuk ICE

selected_features = ["median_income", "households", "housing_median_age"]


# ICE Plot — Linear Regression

print("Membuat ICE untuk Linear Regression...")

fig_lr = PartialDependenceDisplay.from_estimator(
    model_lr,
    X_train,
    features=selected_features,
    kind="individual",
    subsample=150,
    grid_resolution=20
)

plt.suptitle("ICE Plot — Linear Regression")
plt.tight_layout()

# *** Set batas Y agar konsisten ***
for ax in fig_lr.axes_.flat:
    ax.set_ylim(0, 500000)

plt.show()


# ICE Plot — Random Forest

print("Membuat ICE untuk Random Forest...")

fig_rf = PartialDependenceDisplay.from_estimator(
    model_rf,
    X_train,
    features=selected_features,
    kind="individual",
    subsample=150,
    grid_resolution=20
)

plt.suptitle("ICE Plot — Random Forest")
plt.tight_layout()

# *** Samakan batas Y ***
for ax in fig_rf.axes_.flat:
    ax.set_ylim(0, 500000)

plt.show()
