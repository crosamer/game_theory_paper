# =============================
# 1. Import Library
# =============================
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


# =============================
# 2. Load Dataset Kaggle
# =============================

df = pd.read_csv("housing.csv")
print(df.head())


# =============================
# 3. Preprocessing
# =============================

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


# =============================
# 4. Split Dataset
# =============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =============================
# 5. Baseline Models
# =============================

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


# =============================
# 6. Memilih Fitur Penting untuk ICE
#    (gunakan fitur paling relevan di literatur)
# =============================

selected_features = ["median_income", "households", "housing_median_age"]


# =============================
# 7. ICE Plot — Linear Regression
# =============================

print("Membuat ICE untuk Linear Regression...")

fig_lr = PartialDependenceDisplay.from_estimator(
    model_lr,
    X_train,
    features=selected_features,
    kind="individual",      # ICE
    subsample=150,
    grid_resolution=20
)
plt.suptitle("ICE Plot — Linear Regression (Kaggle California Housing)")
plt.tight_layout()
plt.show()


# =============================
# 8. ICE Plot — Random Forest
# =============================

print("Membuat ICE untuk Random Forest...")

fig_rf = PartialDependenceDisplay.from_estimator(
    model_rf,
    X_train,
    features=selected_features,
    kind="individual",      # ICE
    subsample=150,
    grid_resolution=20
)
plt.suptitle("ICE Plot — Random Forest (Kaggle California Housing)")
plt.tight_layout()
plt.show()
