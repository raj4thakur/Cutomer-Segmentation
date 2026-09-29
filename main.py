# customer_segmentation.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
import plotly.express as px

# -------------------------------
# 1️⃣ Load Data
# -------------------------------
df = pd.read_csv("customer_features.csv")

# Select numerical columns for clustering
features = ["frequency", "monetary", "avg_order_value", "recency_days", "orders_per_month", "clv_proxy"]

X = df[features].copy()

# -------------------------------
# 2️⃣ Scale Data
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# 3️⃣ Find Optimal Cluster Count (Elbow Method)
# -------------------------------
inertias = []
sil_scores = []
K_range = range(2, 8)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(X_scaled, kmeans.labels_))

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(K_range, inertias, marker='o')
plt.title("Elbow Method (Inertia)")
plt.xlabel("k")
plt.ylabel("Inertia")

plt.subplot(1,2,2)
plt.plot(K_range, sil_scores, marker='s')
plt.title("Silhouette Score")
plt.xlabel("k")
plt.ylabel("Score")
plt.tight_layout()
plt.show()

# Choose cluster number (based on elbow/silhouette)
optimal_k = 4  # typically around 3–5
print(f"✅ Using k = {optimal_k}")

# -------------------------------
# 4️⃣ Train K-Means Model
# -------------------------------
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_scaled)

# -------------------------------
# 5️⃣ Analyze Segment Profiles
# -------------------------------
cluster_summary = df.groupby("cluster")[features].mean().round(2)
cluster_summary["count"] = df["cluster"].value_counts().sort_index().values
cluster_summary = cluster_summary.sort_values("monetary", ascending=False)
print("\n📊 Cluster Summary:")
print(cluster_summary)

# -------------------------------
# 6️⃣ Visualize with Plotly
# -------------------------------
fig = px.scatter(
    df,
    x="monetary",
    y="frequency",
    color=df["cluster"].astype(str),
    size="avg_order_value",
    hover_data=["customer_id", "recency_days"],
    title="Customer Segmentation: Frequency vs Monetary",
    template="plotly_white"
)
fig.show()

# 3D visualization example (optional)
fig3d = px.scatter_3d(
    df,
    x="monetary",
    y="recency_days",
    z="frequency",
    color=df["cluster"].astype(str),
    title="3D Cluster Visualization",
    template="plotly_white"
)
fig3d.show()

# -------------------------------
# 7️⃣ Save Clustered Data
# -------------------------------
df.to_csv("customer_segments_output.csv", index=False)
print("\n💾 Saved clustered data to customer_segments_output.csv")

# -------------------------------
# 8️⃣ Optional Insight: Cluster Descriptions
# -------------------------------
desc = cluster_summary.copy()
desc["segment_name"] = [
    "High-Value Loyalists",
    "Growing Customers",
    "Price-Sensitive Shoppers",
    "Inactive or New Customers"
][:len(desc)]

print("\n🧠 Segment Interpretation:")
print(desc[["segment_name", "frequency", "monetary", "avg_order_value", "recency_days"]])