import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
import random
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

class Vehicle:
    """Represents a vehicle/node in the VANET"""
    def __init__(self, node_id: int, x: float, y: float, speed: float, 
                 direction: float, residual_energy: float = 100.0):
        self.node_id = node_id
        self.x = x
        self.y = y
        self.speed = speed
        self.direction = direction
        self.residual_energy = residual_energy
        self.neighbor_degree = 0
        self.jitter = random.uniform(0, 1)
        self.link_quality = random.uniform(0.5, 1.0)
        self.distance_to_rsu = None
        self.cluster_id = -1
        self.is_cluster_head = False
        
    def calculate_distance(self, other_vehicle) -> float:
        """Calculate Euclidean distance to another vehicle"""
        return np.sqrt((self.x - other_vehicle.x)**2 + (self.y - other_vehicle.y)**2)
    
    def update_neighbors(self, vehicles: List['Vehicle'], tx_range: float):
        """Update neighbor count within transmission range"""
        self.neighbor_degree = sum(1 for v in vehicles 
                                 if v.node_id != self.node_id and 
                                 self.calculate_distance(v) <= tx_range)

class RSOClusteringVANET:
    """Rat Swarm Optimization for VANET Clustering"""
    
    def __init__(self, population_size: int = 1000, max_iterations: int = 150,
                 w1: float = 0.5, w2: float = 0.5, w3: float = 0.5, 
                 R: float = 2.0):
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.w1, self.w2, self.w3 = w1, w2, w3
        self.R = R  # Control parameter
        
        # Performance tracking
        self.fitness_history = []
        self.cluster_history = []
        
    def initialize_vehicles(self, num_vehicles: int, grid_size: float, 
                          speed_range: Tuple[float, float] = (22, 30)) -> List[Vehicle]:
        """Initialize vehicles randomly in the grid"""
        vehicles = []
        for i in range(num_vehicles):
            x = random.uniform(0, grid_size * 1000)  # Convert km to meters
            y = random.uniform(0, grid_size * 1000)
            speed = random.uniform(*speed_range)
            direction = random.uniform(0, 360)
            residual_energy = random.uniform(80, 100)
            
            vehicle = Vehicle(i, x, y, speed, direction, residual_energy)
            vehicles.append(vehicle)
        
        return vehicles
    
    def calculate_euclidean_distance(self, pos1: Tuple[float, float], 
                                   pos2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two positions"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def update_neighbor_degrees(self, vehicles: List[Vehicle], tx_range: float):
        """Update neighbor degrees for all vehicles"""
        for vehicle in vehicles:
            vehicle.update_neighbors(vehicles, tx_range)
    
    def calculate_optimal_clusters(self, grid_size: float, tx_range: float, 
                                 node_density: float, K: float = 1.0) -> int:
        """Calculate optimal number of clusters using Equation (13)"""
        return max(1, int(K * grid_size * 1000 / (tx_range * node_density)))
    
    def calculate_mobility_factor(self, vehicles: List[Vehicle]) -> float:
        """Calculate mobility factor based on vehicle speeds"""
        speeds = [v.speed for v in vehicles]
        return np.mean(speeds) / max(speeds) if speeds else 1.0
    
    def fitness_function(self, solution: np.ndarray, vehicles: List[Vehicle], 
                        tx_range: float, grid_size: float) -> float:
        """
        Fitness function based on Equation (12)
        F = w1 * (1/(C_sz * M_f)) + w2 * (C_opt * TX_r)/(G_sz * N_d) - w3 * (NM_i + CH_l + IC_ce)
        """
        num_clusters = int(max(1, np.max(solution)))
        cluster_assignments = self.assign_clusters(solution, vehicles, num_clusters)
        
        # Calculate cluster sizes
        cluster_sizes = [len(cluster) for cluster in cluster_assignments.values()]
        avg_cluster_size = np.mean(cluster_sizes) if cluster_sizes else 1
        
        # Calculate mobility factor
        mobility_factor = self.calculate_mobility_factor(vehicles)
        
        # Calculate node density
        area = (grid_size * 1000) ** 2  # Convert to square meters
        node_density = len(vehicles) / area
        
        # Calculate optimal clusters
        optimal_clusters = self.calculate_optimal_clusters(grid_size, tx_range, node_density)
        
        # Term 1: Cluster stability (higher is better)
        stability_term = self.w1 * (1.0 / (avg_cluster_size * mobility_factor + 1e-6))
        
        # Term 2: Communication efficiency (higher is better)
        efficiency_term = self.w2 * (optimal_clusters * tx_range) / (grid_size * 1000 * node_density + 1e-6)
        
        # Term 3: Network overhead (lower is better, so we subtract)
        node_mobility_impact = np.std([v.speed for v in vehicles]) / np.mean([v.speed for v in vehicles])
        ch_longevity = 1.0 / (num_clusters + 1)  # Fewer clusters = longer CH lifetime
        inter_cluster_efficiency = self.calculate_inter_cluster_efficiency(cluster_assignments, vehicles)
        
        overhead_term = self.w3 * (node_mobility_impact + ch_longevity + inter_cluster_efficiency)
        
        fitness = stability_term + efficiency_term - overhead_term
        return fitness
    
    def calculate_inter_cluster_efficiency(self, cluster_assignments: Dict, vehicles: List[Vehicle]) -> float:
        """Calculate inter-cluster communication efficiency"""
        if len(cluster_assignments) <= 1:
            return 1.0
        
        cluster_centers = {}
        for cluster_id, members in cluster_assignments.items():
            if members:
                center_x = np.mean([vehicles[i].x for i in members])
                center_y = np.mean([vehicles[i].y for i in members])
                cluster_centers[cluster_id] = (center_x, center_y)
        
        # Calculate average distance between cluster centers
        distances = []
        center_list = list(cluster_centers.values())
        for i in range(len(center_list)):
            for j in range(i + 1, len(center_list)):
                dist = self.calculate_euclidean_distance(center_list[i], center_list[j])
                distances.append(dist)
        
        return np.mean(distances) / 1000.0 if distances else 1.0  # Normalize
    
    def assign_clusters(self, solution: np.ndarray, vehicles: List[Vehicle], 
                       num_clusters: int) -> Dict[int, List[int]]:
        """Assign vehicles to clusters based on solution"""
        clusters = {i: [] for i in range(num_clusters)}
        
        for i, vehicle in enumerate(vehicles):
            cluster_id = int(solution[i] % num_clusters)
            clusters[cluster_id].append(i)
        
        return clusters
    
    def initialize_population(self, num_vehicles: int) -> np.ndarray:
        """Initialize RSO population (Equation 3)"""
        population = np.zeros((self.population_size, num_vehicles))
        
        for i in range(self.population_size):
            # Initialize with random cluster assignments
            max_clusters = min(num_vehicles, 10)  # Reasonable upper bound
            for j in range(num_vehicles):
                population[i][j] = random.uniform(0, max_clusters)
        
        return population
    
    def update_position(self, population: np.ndarray, fitness_values: np.ndarray, 
                       iteration: int, best_position: np.ndarray) -> np.ndarray:
        """Update positions using RSO equations (6-9)"""
        new_population = population.copy()
        
        for i in range(self.population_size):
            # Calculate S parameter (Equation 8)
            S = self.R - iteration * (self.R / self.max_iterations)
            
            # Calculate T parameter (Equation 9)
            T = 2 * random.random()
            
            # Update position (Equations 6-7)
            P = S * population[i] + T * (best_position - population[i])
            new_population[i] = np.abs(best_position - P)
            
            # Ensure positions are within bounds
            new_population[i] = np.clip(new_population[i], 0, 10)
        
        return new_population
    
    def optimize(self, vehicles: List[Vehicle], tx_range: float, 
                grid_size: float) -> Tuple[np.ndarray, float, Dict]:
        """Main RSO optimization loop"""
        num_vehicles = len(vehicles)
        
        # Update neighbor degrees
        self.update_neighbor_degrees(vehicles, tx_range)
        
        # Initialize population
        population = self.initialize_population(num_vehicles)
        
        # Evaluate initial fitness
        fitness_values = np.array([self.fitness_function(individual, vehicles, tx_range, grid_size) 
                                 for individual in population])
        
        # Find initial best
        best_idx = np.argmax(fitness_values)
        best_position = population[best_idx].copy()
        best_fitness = fitness_values[best_idx]
        
        # Evolution loop
        for iteration in range(self.max_iterations):
            # Update positions
            population = self.update_position(population, fitness_values, iteration, best_position)
            
            # Evaluate fitness
            fitness_values = np.array([self.fitness_function(individual, vehicles, tx_range, grid_size) 
                                     for individual in population])
            
            # Update best solution
            current_best_idx = np.argmax(fitness_values)
            if fitness_values[current_best_idx] > best_fitness:
                best_fitness = fitness_values[current_best_idx]
                best_position = population[current_best_idx].copy()
            
            # Track progress
            self.fitness_history.append(best_fitness)
            
        return best_position, best_fitness, self.get_clustering_results(best_position, vehicles)
    
    def get_clustering_results(self, solution: np.ndarray, vehicles: List[Vehicle]) -> Dict:
        """Extract clustering results and metrics"""
        num_clusters = int(max(1, np.max(solution)))
        cluster_assignments = self.assign_clusters(solution, vehicles, num_clusters)
        
        # Remove empty clusters
        cluster_assignments = {k: v for k, v in cluster_assignments.items() if v}
        actual_num_clusters = len(cluster_assignments)
        
        # Assign cluster heads (vehicle with highest energy in each cluster)
        cluster_heads = {}
        ch_assignments = [False] * len(vehicles)
        
        for cluster_id, members in cluster_assignments.items():
            if members:
                # Select CH based on highest residual energy
                best_ch = max(members, key=lambda x: vehicles[x].residual_energy)
                cluster_heads[cluster_id] = best_ch
                vehicles[best_ch].is_cluster_head = True
                ch_assignments[best_ch] = True
        
        # Update vehicle cluster assignments
        cluster_membership = [-1] * len(vehicles)
        for cluster_id, members in cluster_assignments.items():
            for member_idx in members:
                vehicles[member_idx].cluster_id = cluster_id
                cluster_membership[member_idx] = cluster_id
        
        # Calculate metrics exactly like your friend's output
        cluster_sizes = [len(members) for members in cluster_assignments.values()]
        
        # Calculate CH Lifetime (estimated based on mobility and energy)
        avg_ch_lifetime = self.calculate_ch_lifetime(vehicles, cluster_heads)
        
        # Calculate Re-clustering count (based on mobility factor)
        recluster_count = self.calculate_recluster_count(vehicles)
        
        # Calculate average cluster size
        avg_cluster_size = np.mean(cluster_sizes) if cluster_sizes else 0
        
        # Calculate Energy Fairness (based on energy distribution among CHs)
        energy_fairness = self.calculate_energy_fairness(vehicles, cluster_heads)
        
        # Calculate Orphan Ratio (nodes not properly clustered)
        orphan_ratio = self.calculate_orphan_ratio(cluster_membership)
        
        # Load balance (standard deviation of cluster sizes)
        load_balance = np.std(cluster_sizes) if len(cluster_sizes) > 1 else 0
        
        results = {
            # Basic clustering info
            'num_clusters': actual_num_clusters,
            'cluster_assignments': cluster_assignments,
            'cluster_heads': cluster_heads,
            'cluster_sizes': cluster_sizes,
            
            # Common Output Metrics (matching your friend's format)
            'ch_assignments': ch_assignments,  # 1. CH assignment - which node is CH
            'cluster_membership': cluster_membership,  # 2. Cluster Membership
            'avg_cluster_size': avg_cluster_size,  # 3. Cluster Size (average)
            'avg_ch_lifetime': avg_ch_lifetime,  # 4. CH Lifetime
            'recluster_count': recluster_count,  # 5. Re-clustering count
            'energy_fairness': energy_fairness,  # 6. Energy consumption fairness
            'load_balance': load_balance,  # 7. Load balance
            'orphan_ratio': orphan_ratio  # Additional metric like your friend's output
        }
        
        return results
    
    def calculate_ch_lifetime(self, vehicles: List[Vehicle], cluster_heads: Dict) -> float:
        """Calculate average CH lifetime based on energy and mobility"""
        if not cluster_heads:
            return 0.0
        
        lifetimes = []
        for ch_idx in cluster_heads.values():
            # Estimate lifetime based on energy and mobility
            energy_factor = vehicles[ch_idx].residual_energy / 100.0
            mobility_factor = 1.0 - (vehicles[ch_idx].speed / 30.0)  # Normalized
            lifetime = energy_factor * mobility_factor * 10  # Scale to reasonable timesteps
            lifetimes.append(max(0.1, lifetime))
        
        return np.mean(lifetimes)
    
    def calculate_recluster_count(self, vehicles: List[Vehicle]) -> float:
        """Calculate expected re-clustering count based on mobility"""
        speeds = [v.speed for v in vehicles]
        avg_speed = np.mean(speeds)
        # Higher speed = more re-clustering needed
        recluster_count = (avg_speed / 30.0) * 5  # Normalize and scale
        return max(0.0, recluster_count)
    
    def calculate_energy_fairness(self, vehicles: List[Vehicle], cluster_heads: Dict) -> float:
        """Calculate energy fairness among cluster heads"""
        if not cluster_heads:
            return 1.0
        
        ch_energies = [vehicles[ch_idx].residual_energy for ch_idx in cluster_heads.values()]
        if len(ch_energies) <= 1:
            return 1.0
        
        # Fairness = 1 - coefficient of variation
        fairness = 1.0 - (np.std(ch_energies) / np.mean(ch_energies))
        return max(0.0, fairness)
    
    def calculate_orphan_ratio(self, cluster_membership: List[int]) -> float:
        """Calculate ratio of orphaned (unassigned) nodes"""
        orphaned = sum(1 for cluster_id in cluster_membership if cluster_id == -1)
        total = len(cluster_membership)
        return orphaned / total if total > 0 else 0.0

def run_rso_simulation(num_vehicles: int = 30, grid_size: float = 1.0, 
                      tx_range: float = 300, num_runs: int = 10):
    """Run RSO clustering simulation with specified parameters"""
    
    print(f"Running RSO Simulation:")
    print(f"Vehicles: {num_vehicles}, Grid: {grid_size}x{grid_size} km², Tx Range: {tx_range}m")
    print("-" * 60)
    
    all_results = []
    
    for run in range(num_runs):
        # Initialize RSO algorithm
        rso = RSOClusteringVANET(population_size=100, max_iterations=150)
        
        # Generate vehicles
        vehicles = rso.initialize_vehicles(num_vehicles, grid_size)
        
        # Run optimization
        best_solution, best_fitness, results = rso.optimize(vehicles, tx_range, grid_size)
        
        # Store results
        run_result = {
            'run': run + 1,
            'num_clusters': results['num_clusters'],
            'fitness': best_fitness,
            'avg_cluster_size': results['avg_cluster_size'],
            'avg_ch_lifetime': results['avg_ch_lifetime'],
            'recluster_count': results['recluster_count'],
            'energy_fairness': results['energy_fairness'],
            'orphan_ratio': results['orphan_ratio'],
            'load_balance': results['load_balance']
        }
        all_results.append(run_result)
        
        print(f"Run {run+1:2d}: {results['num_clusters']:2d} clusters, "
              f"Fitness: {best_fitness:.4f}, CH Lifetime: {results['avg_ch_lifetime']:.2f}")
    
    # Calculate average results (matching your friend's output format)
    print("\n" + "=" * 40)
    print("=== Fitness-based Clustering Metrics ===")
    print(f"Avg_CH_Lifetime: {np.mean([r['avg_ch_lifetime'] for r in all_results]):.2f}")
    print(f"Recluster_Count: {np.mean([r['recluster_count'] for r in all_results]):.2f}")
    print(f"Avg_Cluster_Size: {np.mean([r['avg_cluster_size'] for r in all_results]):.2f}")
    print(f"Energy_Fairness: {np.mean([r['energy_fairness'] for r in all_results]):.2f}")
    print(f"Orphan_Ratio: {np.mean([r['orphan_ratio'] for r in all_results]):.2f}")
    print("=" * 40)
    
    return all_results

def print_detailed_clustering_metrics(results: Dict, vehicles: List[Vehicle]):
    """Print detailed clustering metrics in your friend's format"""
    print("\n" + "=" * 50)
    print("=== Fitness-based Clustering Metrics ===")
    print(f"Avg_CH_Lifetime: {results['avg_ch_lifetime']:.2f}")
    print(f"Recluster_Count: {results['recluster_count']:.2f}")
    print(f"Avg_Cluster_Size: {results['avg_cluster_size']:.2f}")
    print(f"Energy_Fairness: {results['energy_fairness']:.2f}")
    print(f"Orphan_Ratio: {results['orphan_ratio']:.2f}")
    print("=" * 50)
    
    # Additional detailed breakdown
    print(f"\nDetailed Results:")
    print(f"Total Vehicles: {len(vehicles)}")
    print(f"Total Clusters: {results['num_clusters']}")
    print(f"Cluster Heads: {len(results['cluster_heads'])}")
    print(f"Load Balance (Std Dev): {results['load_balance']:.2f}")
    
    # Show cluster head assignments
    print(f"\nCluster Head Assignments:")
    for i, is_ch in enumerate(results['ch_assignments']):
        if is_ch:
            print(f"Vehicle {i}: Cluster Head")
    
    # Show cluster membership summary
    cluster_counts = {}
    for cluster_id in results['cluster_membership']:
        if cluster_id != -1:
            cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
    
    print(f"\nCluster Sizes:")
    for cluster_id, size in cluster_counts.items():
        print(f"Cluster {cluster_id}: {size} vehicles")

def compare_transmission_ranges():
    """Compare performance across different transmission ranges"""
    tx_ranges = [100, 200, 300, 400, 500, 600]
    results = {}
    
    print("Comparing Transmission Ranges:")
    print("=" * 70)
    
    for tx_range in tx_ranges:
        print(f"\nTesting Tx Range: {tx_range}m")
        run_results = run_rso_simulation(num_vehicles=30, grid_size=1.0, 
                                       tx_range=tx_range, num_runs=5)
        avg_clusters = np.mean([r['num_clusters'] for r in run_results])
        results[tx_range] = avg_clusters
        print(f"Average clusters: {avg_clusters:.2f}")
    
    return results

def generate_sample_dataset(num_vehicles: int = 100, grid_size: float = 2.0) -> pd.DataFrame:
    """Generate sample dataset with vehicle parameters"""
    data = []
    
    for i in range(num_vehicles):
        vehicle_data = {
            'Node_ID': i,
            'Position_X': random.uniform(0, grid_size * 1000),
            'Position_Y': random.uniform(0, grid_size * 1000),
            'Speed': random.uniform(22, 30),
            'Direction': random.uniform(0, 360),
            'Residual_Energy': random.uniform(70, 100),
            'Neighbor_Degree': 0,  # Will be calculated
            'Jitter': random.uniform(0, 1),
            'Link_Quality': random.uniform(0.5, 1.0),
            'Distance_to_RSU': random.uniform(100, 1000)
        }
        data.append(vehicle_data)
    
    df = pd.DataFrame(data)
    return df

def visualize_clustering(vehicles: List[Vehicle], results: Dict, 
                        grid_size: float, tx_range: float):
    """Visualize the clustering results"""
    plt.figure(figsize=(12, 8))
    
    # Define colors for clusters
    colors = plt.cm.Set3(np.linspace(0, 1, results['num_clusters']))
    
    # Plot vehicles
    for i, vehicle in enumerate(vehicles):
        cluster_id = vehicle.cluster_id
        if cluster_id >= 0:
            color = colors[cluster_id % len(colors)]
            marker = 'o' if not vehicle.is_cluster_head else '^'
            size = 50 if not vehicle.is_cluster_head else 150
            
            plt.scatter(vehicle.x/1000, vehicle.y/1000, c=[color], 
                       marker=marker, s=size, alpha=0.7,
                       label=f'Cluster {cluster_id}' if i == 0 else "")
            
            # Draw transmission range for cluster heads
            if vehicle.is_cluster_head:
                circle = plt.Circle((vehicle.x/1000, vehicle.y/1000), 
                                  tx_range/1000, fill=False, color='red', alpha=0.3)
                plt.gca().add_patch(circle)
    
    plt.xlim(0, grid_size)
    plt.ylim(0, grid_size)
    plt.xlabel('X Position (km)')
    plt.ylabel('Y Position (km)')
    plt.title(f'RSO VANET Clustering Results\n{results["num_clusters"]} clusters, '
              f'Tx Range: {tx_range}m')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                             markersize=8, label='Regular Node'),
                      Line2D([0], [0], marker='^', color='w', markerfacecolor='red', 
                             markersize=12, label='Cluster Head')]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.show()

# Example usage and testing
if __name__ == "__main__":
    print("RSO VANET Clustering Algorithm Implementation")
    print("=" * 60)
    
    # Generate sample dataset
    print("\n1. Generating sample dataset...")
    sample_df = generate_sample_dataset(40, 2.0)
    print(f"Generated dataset with {len(sample_df)} vehicles")
    print(sample_df.head())
    
    # Run single detailed simulation
    print("\n2. Running detailed RSO simulation...")
    rso = RSOClusteringVANET(population_size=100, max_iterations=150)
    vehicles = rso.initialize_vehicles(num_vehicles=30, grid_size=1.0)
    best_solution, best_fitness, results = rso.optimize(vehicles, tx_range=300, grid_size=1.0)
    
    # Print results in your friend's format
    print_detailed_clustering_metrics(results, vehicles)
    
    # Run multiple simulations for average results
    print("\n3. Running multiple simulations for average results...")
    basic_results = run_rso_simulation(num_vehicles=30, grid_size=1.0, 
                                     tx_range=300, num_runs=5)
    
    # Test with different parameters
    print("\n4. Testing different scenarios...")
    scenarios = [
        (30, 1.0, 300),  # Small network
        (50, 2.0, 400),  # Medium network
        (60, 3.0, 500),  # Large network
    ]
    
    for vehicles, grid, tx_range in scenarios:
        print(f"\nScenario: {vehicles} vehicles, {grid}x{grid} km², {tx_range}m Tx range")
        run_rso_simulation(num_vehicles=vehicles, grid_size=grid, 
                         tx_range=tx_range, num_runs=3)
    
    print("\nRSO VANET Clustering Implementation Complete!")