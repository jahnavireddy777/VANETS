import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
import random

# --------------------------
# 1. Generate synthetic VANET data
# --------------------------
def generate_vanet_data(n_nodes=1000, area_size=5000, rsu_pos=None):
    if rsu_pos is None:
        rsu_pos = (area_size/2, area_size/2)  # place RSU in center

    data = []
    for node_id in range(1, n_nodes+1):
        x = random.uniform(0, area_size)
        y = random.uniform(0, area_size)
        speed = random.uniform(0, 30)       # m/s
        direction = random.uniform(0, 360)  # degrees
        re = random.uniform(50, 100)        # residual energy %
        jitter = random.uniform(0, 0.1)     # small variation
        lqi = random.uniform(0.5, 1.0)      # link quality
        d_rsu = np.sqrt((x-rsu_pos[0])**2 + (y-rsu_pos[1])**2)

        data.append([node_id, x, y, speed, direction, re, jitter, lqi, d_rsu])

    df = pd.DataFrame(data, columns=[
        "NodeID", "X", "Y", "Speed", "Direction", "RE", "Jitter", "LQI", "dRSU"
    ])
    return df

# --------------------------
# 2. Fitness-based clustering
# --------------------------
def fitness_based_clustering(df, comm_range=300):
    # Neighbor Degree
    positions = df[["X", "Y"]].to_numpy()
    dist_matrix = pairwise_distances(positions)
    ND = (dist_matrix <= comm_range).sum(axis=1) - 1
    df["ND"] = ND

    # Normalize values
    df_norm = df.copy()
    for col in ["ND","RE","Jitter","LQI","dRSU"]:
        df_norm[col] = (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min() + 1e-6)

    # Weights
    w1, w2, w3, w4, w5 = 0.3, 0.3, 0.1, 0.2, 0.1
    df["Fitness"] = (w1*df_norm["ND"] + w2*df_norm["RE"] 
                     - w3*df_norm["Jitter"] + w4*df_norm["LQI"] 
                     - w5*df_norm["dRSU"])

    # Assign CH
    cluster_id = {}
    for i, row in df.iterrows():
        neighbors = df[np.sqrt((df["X"]-row["X"])**2 + (df["Y"]-row["Y"])**2) <= comm_range]
        best_ch = neighbors.loc[neighbors["Fitness"].idxmax()]
        cluster_id[row["NodeID"]] = int(best_ch["NodeID"])

    df["ClusterHead"] = df["NodeID"].map(cluster_id)
    df["ClusterID"] = df["ClusterHead"]  # cluster id = CH id
    return df

# --------------------------
# 3. Metrics
# --------------------------
def evaluate_metrics(cluster_history, total_timesteps):
    metrics = {}

    # 1. CH Lifetime
    ch_lifetimes = {}
    for t in range(1, total_timesteps):
        prev = cluster_history[t-1]
        curr = cluster_history[t]
        for ch in set(curr.values()):
            if ch in prev.values():
                ch_lifetimes[ch] = ch_lifetimes.get(ch, 0) + 1
    metrics["Avg_CH_Lifetime"] = np.mean(list(ch_lifetimes.values())) if ch_lifetimes else 0

    # 2. Re-cluster Count
    re_clusters = 0
    for t in range(1, total_timesteps):
        if cluster_history[t] != cluster_history[t-1]:
            re_clusters += 1
    metrics["Recluster_Count"] = re_clusters

    # 3. Avg Cluster Size
    cluster_sizes = []
    for clusters in cluster_history.values():
        counts = {}
        for ch in clusters.values():
            counts[ch] = counts.get(ch, 0) + 1
        cluster_sizes.extend(counts.values())
    metrics["Avg_Cluster_Size"] = np.mean(cluster_sizes)

    # 4. Energy Fairness
    ch_counts = {}
    for clusters in cluster_history.values():
        for ch in clusters.values():
            ch_counts[ch] = ch_counts.get(ch, 0) + 1
    metrics["Energy_Fairness"] = np.std(list(ch_counts.values()))

    # 5. Orphan Ratio
    metrics["Orphan_Ratio"] = 0  # none in this algo

    return metrics

# --------------------------
# 4. Main
# --------------------------
if __name__ == "__main__":
    timesteps = 5
    n_nodes = 30
    cluster_history = {}
    last_df = None

    for t in range(timesteps):
        df = generate_vanet_data(n_nodes=n_nodes)
        df = fitness_based_clustering(df)
        cluster_map = dict(zip(df["NodeID"], df["ClusterHead"]))
        cluster_history[t] = cluster_map
        last_df = df  # store last timestep

    # Evaluate metrics
    metrics = evaluate_metrics(cluster_history, timesteps)

    print("\n=== Fitness-based Clustering Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}")

    print("\n=== Final Clustering Output (Last Timestep) ===")
    print(last_df[["NodeID", "ClusterID", "ClusterHead", "Fitness", "ND", "RE"]].head(15))

    # Save results
    last_df.to_csv("fitness_clustering_output.csv", index=False)
    print("\nSaved final clustering to fitness_clustering_last_output.csv")
