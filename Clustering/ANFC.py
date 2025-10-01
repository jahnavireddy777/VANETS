import numpy as np
import pandas as pd
import random
from sklearn.metrics import pairwise_distances

# --------------------------
# 1. Generate synthetic VANET data
# --------------------------
def generate_vanet_data(n_nodes=1000, area_size=5000, rsu_pos=None):
    if rsu_pos is None:
        rsu_pos = (area_size/2, area_size/2)

    data = []
    for node_id in range(1, n_nodes+1):
        x = random.uniform(0, area_size)
        y = random.uniform(0, area_size)
        speed = random.uniform(0, 30)
        direction = random.uniform(0, 360)
        re = random.uniform(50, 100)        # residual energy
        jitter = random.uniform(0, 0.1)
        lqi = random.uniform(0.5, 1.0)
        d_rsu = np.sqrt((x-rsu_pos[0])**2 + (y-rsu_pos[1])**2)

        data.append([node_id, x, y, speed, direction, re, jitter, lqi, d_rsu])

    df = pd.DataFrame(data, columns=[
        "NodeID", "X", "Y", "Speed", "Direction", "RE", "Jitter", "LQI", "dRSU"
    ])
    return df

# --------------------------
# 2. Fuzzy Membership Functions
# --------------------------
def fuzzy_low(x, min_val, max_val):
    return max(0, (max_val - x) / (max_val - min_val + 1e-6))

def fuzzy_high(x, min_val, max_val):
    return max(0, (x - min_val) / (max_val - min_val + 1e-6))

# --------------------------
# 3. ANFC Fitness Calculation
# --------------------------
def compute_anfc_fitness(df, comm_range=300):
    # Neighbor Degree
    positions = df[["X", "Y"]].to_numpy()
    dist_matrix = pairwise_distances(positions)
    ND = (dist_matrix <= comm_range).sum(axis=1) - 1
    df["ND"] = ND

    # Normalize values
    df_norm = df.copy()
    for col in ["ND","RE","dRSU"]:
        df_norm[col] = (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min() + 1e-6)

    # Apply fuzzy logic: IF ND high AND RE high AND dRSU low → high fitness
    fitness_scores = []
    for i, row in df_norm.iterrows():
        nd_high = fuzzy_high(row["ND"], 0, 1)
        re_high = fuzzy_high(row["RE"], 0, 1)
        drsu_low = fuzzy_low(row["dRSU"], 0, 1)

        # simple ANFIS-style weighted rule base
        score = (0.4*nd_high + 0.4*re_high + 0.2*drsu_low)
        fitness_scores.append(score)

    df["Fitness"] = fitness_scores
    return df

# --------------------------
# 4. Cluster Formation
# --------------------------
def form_clusters(df, comm_range=300):
    cluster_id = {}
    for i, row in df.iterrows():
        neighbors = df[np.sqrt((df["X"]-row["X"])**2 + (df["Y"]-row["Y"])**2) <= comm_range]
        best_ch = neighbors.loc[neighbors["Fitness"].idxmax()]
        cluster_id[row["NodeID"]] = int(best_ch["NodeID"])

    df["ClusterHead"] = df["NodeID"].map(cluster_id)
    df["ClusterID"] = df["ClusterHead"]
    return df

# --------------------------
# 5. Metrics
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
    metrics["Orphan_Ratio"] = 0

    return metrics

# --------------------------
# 6. Main Simulation
# --------------------------
if __name__ == "__main__":
    timesteps = 5
    n_nodes = 30
    cluster_history = {}
    last_df = None

    for t in range(timesteps):
        df = generate_vanet_data(n_nodes=n_nodes)
        df = compute_anfc_fitness(df)
        df = form_clusters(df)
        cluster_map = dict(zip(df["NodeID"], df["ClusterHead"]))
        cluster_history[t] = cluster_map
        last_df = df

    # Evaluate metrics
    metrics = evaluate_metrics(cluster_history, timesteps)

    print("\n=== ANFC-based Clustering Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}")

    print("\n=== Final Clustering Output (Last Timestep) ===")
    print(last_df[["NodeID", "ClusterID", "ClusterHead", "Fitness", "ND", "RE"]].head(15))

    # Save results
    last_df.to_csv("ANFC_clustering_output.csv", index=False)
    print("\nSaved final clustering to ANFC_clustering_output.csv")
