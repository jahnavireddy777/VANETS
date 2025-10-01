import matplotlib.pyplot as plt
import pandas as pd

# Example: Replace with your real results after running each algorithm
metrics_rso = {
    "Avg_CH_Lifetime": 3.93,
    "Recluster_Count": 2.00,
    "Avg_Cluster_Size": 1.01,
    "Energy_Fairness": 0.26,
    "Orphan_Ratio": 0.00
}

metrics_anfc = {
    "Avg_CH_Lifetime": 4.00,
    "Recluster_Count": 0.00,
    "Avg_Cluster_Size": 1.00,
    "Energy_Fairness": 0.00,
    "Orphan_Ratio": 0.00
}

metrics_fitness = {
    "Avg_CH_Lifetime": 1.26,
    "Recluster_Count": 4.36,
    "Avg_Cluster_Size": 15.71,
    "Energy_Fairness": 0.98,
    "Orphan_Ratio": 0.00
}

# Put into DataFrame
df = pd.DataFrame([metrics_rso, metrics_anfc, metrics_fitness],
                  index=["RSO", "ANFC", "Fitness"])

print(df)

# Plot grouped bar chart
df.plot(kind="bar", figsize=(10,6))
plt.title("Comparison of Clustering Algorithms")
plt.ylabel("Metric Value")
plt.xticks(rotation=0)
plt.legend(title="Metrics")
plt.tight_layout()
plt.show()

# ---- Optional Radar Plot ----
import numpy as np

metrics = list(df.columns)
labels = np.array(metrics)
angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]  # close the circle

fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))

for algo, row in df.iterrows():
    values = row.tolist()
    values += values[:1]
    ax.plot(angles, values, label=algo)
    ax.fill(angles, values, alpha=0.25)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)
plt.title("Radar Plot: Algorithm Comparison")
plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
plt.show()
