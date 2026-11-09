import os
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample

# Ensure output directory exists
os.makedirs("outputs", exist_ok=True)

print("--- STARTING PART B ---")

# ==========================================
# LOAD DATA & MERGE PRODUCTS FOR CATEGORY
# ==========================================
orders = pd.read_csv("data/clean/orders.csv")
products = pd.read_csv("data/clean/products.csv")
customers = pd.read_csv("data/clean/customers.csv")

orders["order_ts"] = pd.to_datetime(orders["order_ts"])
orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce")
customers["signup_date"] = pd.to_datetime(customers["signup_date"])

if "category" not in orders.columns:
    orders = orders.merge(products[["product_id", "price_inr", "category"]], on="product_id", how="left")


# ==========================================
# TASK 1: RFM Table Calculation
# ==========================================
completed_orders = orders[orders["status"] == "completed"].copy()
snapshot_date = pd.to_datetime("2025-10-01 00:00:00")

rfm = completed_orders.groupby("customer_id").agg(
    recency=("order_ts", lambda x: int((snapshot_date - x.max()).days)),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", lambda x: float(x.astype(float).sum()))
).reset_index()

print(f"Task 1 - RFM Customers count: {len(rfm)} (Reference: 1175)")


# ==========================================
# TASK 2 & 3: K-Means Clustering & Segments
# ==========================================
X_rfm = rfm[["recency", "frequency", "monetary"]].astype(float)
scaler_rfm = StandardScaler()
X_rfm_scaled = scaler_rfm.fit_transform(X_rfm)

print("\n--- K-MEANS TUNING (K=2 to 8) ---")
for k in range(2, 9):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_rfm_scaled)
    inertia = kmeans.inertia_
    sil = silhouette_score(X_rfm_scaled, labels)
    print(f"K={k} | Inertia: {inertia:.1f} | Silhouette: {sil:.3f}")

optimal_k = 4
kmeans_final = KMeans(n_clusters=optimal_k, n_init=10, random_state=42)
rfm["segment_id"] = kmeans_final.fit_predict(X_rfm_scaled)

profile = rfm.groupby("segment_id")[["recency", "frequency", "monetary"]].mean()
profile["size"] = rfm.groupby("segment_id")["customer_id"].count()
print("\nSegment Profiles (K=4):")
print(profile)

premium_cluster_id = profile["monetary"].idxmax()

def assign_segment_metadata(row):
    sid = row["segment_id"]
    is_prem = (sid == premium_cluster_id)
    if sid == premium_cluster_id:
        name = "High-Value Champions"
    elif row["recency"] > 60:
        name = "At-Risk Churners"
    elif row["frequency"] > 12:
        name = "Frequent Loyalists"
    else:
        name = "Low-Activity Occasionals"
    return pd.Series([name, is_prem])

rfm[["segment_name", "is_premium"]] = rfm.apply(assign_segment_metadata, axis=1)

pca = PCA(n_components=2, random_state=42)
pca.fit_transform(X_rfm_scaled)
print(f"\nTask 3 - PCA Explained Variance Ratio: {pca.explained_variance_ratio_}")

rfm.to_csv("outputs/customer_segments.csv", index=False)


# ==========================================
# TASK 4: Define Lapse Problem & Features
# ==========================================
cutoff_date = pd.to_datetime("2025-09-01 00:00:00")
pre_orders = orders[orders["order_ts"] < cutoff_date].copy()

cohort_customers = pre_orders[pre_orders["status"] == "completed"]["customer_id"].unique()
cohort_df = customers[customers["customer_id"].isin(cohort_customers)].copy()

completed_pre = pre_orders[pre_orders["status"] == "completed"]

recency_feat = completed_pre.groupby("customer_id")["order_ts"].max().apply(lambda x: (cutoff_date - x).days).rename("recency")
freq_feat = completed_pre.groupby("customer_id")["order_id"].nunique().rename("frequency")
monetary_feat = completed_pre.groupby("customer_id")["order_amount"].sum().rename("monetary")
disc_feat = completed_pre.groupby("customer_id")["discount_pct"].mean().rename("avg_discount")
deliv_feat = completed_pre.groupby("customer_id")["delivery_minutes"].mean().rename("avg_delivery_minutes")

cat_counts = completed_pre.groupby(["customer_id", "category"]).size().reset_index(name="count")
fav_cat = cat_counts.sort_values(["customer_id", "count"], ascending=[True, False]).drop_duplicates("customer_id").set_index("customer_id")["category"].rename("favourite_category")

all_status_counts = pre_orders[pre_orders["status"].isin(["completed", "cancelled"])].groupby("customer_id")["status"].value_counts().unstack(fill_value=0)
if "cancelled" not in all_status_counts.columns:
    all_status_counts["cancelled"] = 0
if "completed" not in all_status_counts.columns:
    all_status_counts["completed"] = 0
total_final_orders = all_status_counts["completed"] + all_status_counts["cancelled"]
cancel_rate = (all_status_counts["cancelled"] / total_final_orders.replace(0, 1)).rename("cancellation_rate")

tenure = (cutoff_date - cohort_df.set_index("customer_id")["signup_date"]).dt.days.rename("tenure_days")

features = pd.DataFrame({"customer_id": cohort_customers}).set_index("customer_id")
features = features.join([recency_feat, freq_feat, monetary_feat, disc_feat, deliv_feat, fav_cat, cancel_rate, tenure])
features = features.reset_index().merge(cohort_df[["customer_id", "city", "age", "gender"]], on="customer_id")

features["favourite_category"] = features["favourite_category"].fillna("Grocery")

sept_completed = orders[(orders["order_ts"] >= cutoff_date) & (orders["order_ts"] <= pd.to_datetime("2025-09-30 23:59:59")) & (orders["status"] == "completed")]["customer_id"].unique()
features["lapsed"] = features["customer_id"].apply(lambda cid: 0 if cid in sept_completed else 1)

X = features.drop(columns=["customer_id", "lapsed"])
y = features["lapsed"]

print(f"\nTask 4 - Cohort size: {len(features)}, Lapsed count: {y.sum()} ({y.mean()*100:.2f}%)")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


# ==========================================
# TASK 5: Preprocessing Pipeline (No Leakage)
# ==========================================
numeric_features = ["recency", "frequency", "monetary", "avg_discount", "avg_delivery_minutes", "cancellation_rate", "tenure_days", "age"]
categorical_features = ["city", "gender", "favourite_category"]

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

assert not pd.DataFrame(X_train_transformed).isna().any().any(), "Missing values found in training features!"
assert not pd.DataFrame(X_test_transformed).isna().any().any(), "Missing values found in test features!"
print("\nTask 5 - Preprocessing complete. Zero missing values confirmed.")


# ==========================================
# TASK 6: Train, Tune & Evaluate Models
# ==========================================
baseline_preds = np.zeros_like(y_test)
_, _, baseline_f1, _ = precision_recall_fscore_support(y_test, baseline_preds, average="binary", zero_division=0)
print(f"\nTask 6 - Baseline Lapsed-Class F1: {baseline_f1:.3f}")

lr_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", LogisticRegression(max_iter=1000, random_state=42))])
lr_grid = GridSearchCV(lr_pipeline, {"classifier__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1")
lr_grid.fit(X_train, y_train)
print(f"Logistic Regression Best C: {lr_grid.best_params_}, CV F1: {lr_grid.best_score_:.3f}")

rf_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))])
rf_grid = GridSearchCV(rf_pipeline, {"classifier__max_depth": [3, 5, 8], "classifier__min_samples_leaf": [5, 10, 20]}, cv=5, scoring="f1")
rf_grid.fit(X_train, y_train)
print(f"Random Forest Best Params: {rf_grid.best_params_}, CV F1: {rf_grid.best_score_:.3f}")

best_model = rf_grid if rf_grid.best_score_ > lr_grid.best_score_ else lr_grid
print("Selected Best Model for production.")


# ==========================================
# TASK 7: Class-Balance Experiment
# ==========================================
train_df = X_train.copy()
train_df["lapsed"] = y_train

majority = train_df[train_df["lapsed"] == 0]
minority = train_df[train_df["lapsed"] == 1]
minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)
balanced_train = pd.concat([majority, minority_upsampled])

X_train_bal = balanced_train.drop(columns=["lapsed"])
y_train_bal = balanced_train["lapsed"]

lr_bal_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", LogisticRegression(max_iter=1000, random_state=42))])
lr_bal_pipeline.fit(X_train_bal, y_train_bal)
bal_preds = lr_bal_pipeline.predict(X_test)
p, r, f, _ = precision_recall_fscore_support(y_test, bal_preds, average="binary", zero_division=0)
print(f"\nTask 7 - Balanced LR Test Metrics -> Precision: {p:.3f}, Recall: {r:.3f}, F1: {f:.3f}")


# ==========================================
# TASK 8: Interpret, Persist & Retention List
# ==========================================
joblib.dump(best_model, "outputs/lapse_model.joblib")
loaded_model = joblib.load("outputs/lapse_model.joblib")
assert np.array_equal(best_model.predict(X_test), loaded_model.predict(X_test)), "Model reload check failed!"

test_probs = loaded_model.predict_proba(X_test)[:, 1]
retention_df = pd.DataFrame({
    "customer_id": features.loc[X_test.index, "customer_id"].values,
    "lapse_probability": test_probs,
    "actually_lapsed": y_test.values
})

premium_customers = rfm[rfm["is_premium"] == True]["customer_id"]
premium_retention = retention_df[retention_df["customer_id"].isin(premium_customers)].sort_values(by="lapse_probability", ascending=False)

top_10_retention = premium_retention.head(10)
top_10_retention.to_csv("outputs/retention_list.csv", index=False)
print(f"Retention list written with {len(top_10_retention)} rows. Truly lapsed among them: {top_10_retention['actually_lapsed'].sum()}")
print("PART B COMPLETED SUCCESSFULLY!")