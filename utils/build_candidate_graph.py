import networkx as nx
import numpy as np
from shapely.geometry import Point, LineString
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import joblib
import os
from geopy.distance import geodesic
from scipy.spatial import cKDTree
import pickle
import warnings
from utils.graph_features import compute_graph_features
warnings.filterwarnings("ignore", message=".*ScriptRunContext.*")

MAX_EDGE_LENGTH = 500  # meters

def generate_points_between_stations(station_coords, samples_per_edge=10):
    points = set()
    for i in range(len(station_coords)):
        for j in range(i + 1, len(station_coords)):
            a = station_coords[i]
            b = station_coords[j]
            lat1, lon1 = float(a["lat"]), float(a["lon"])
            lat2, lon2 = float(b["lat"]), float(b["lon"])
            for k in range(samples_per_edge):
                frac = k / samples_per_edge
                lat = lat1 + frac * (lat2 - lat1)
                lon = lon1 + frac * (lon2 - lon1)
                points.add((lat, lon))
    return list(points)

def build_features(f):
    # Ensure we have valid values for calculations
    length = max(f["length_m"], 1)  # Avoid division by zero
    elev_change = f["avg_elevation_change"]
    pop_density = f["avg_population_density"]
    
    # Calculate derived features
    f["elevation_per_meter"] = elev_change / length
    f["curvature_deg"] = 0  # Placeholder for future implementation
    f["curvature_per_meter"] = 0  # Placeholder for future implementation
    f["closeness_avg"] = 0  # Placeholder for future implementation
    f["betweenness_avg"] = 0  # Placeholder for future implementation
    f["avg_population_density"] = pop_density
    
    # Print debug info for first few edges
    if len(build_features.debug_count) < 3:
        print(f"[DEBUG] Features for edge: {f}")
        build_features.debug_count.append(1)
    
    return f

# Initialize debug counter
build_features.debug_count = []

def build_candidate_graph(station_coords, city, country, model_path="models/enhanced_edge_classifier.pkl", k=1):
    print(f"[INFO] Building AI candidate graph for {city}, {country}...")

    safe_name = f"{city.lower().replace(',', '').replace(' ', '_')}_{country.lower().replace(' ', '_')}"
    gpickle_path = f"data/graphs/{safe_name}_raw.gpickle"
    if not os.path.exists(gpickle_path):
        raise FileNotFoundError(f"Missing enriched graph: {gpickle_path}")

    with open(gpickle_path, "rb") as f:
        G_real = pickle.load(f)

    enriched_nodes = []
    enriched_coords = []
    for n, data in G_real.nodes(data=True):
        if "elevation" in data and "population_density" in data:
            enriched_nodes.append((n, data))
            enriched_coords.append([float(data["y"]), float(data["x"])])

    if not enriched_coords:
        raise ValueError("No enriched nodes found in the graph")

    # Create KD-tree for efficient nearest neighbor search
    try:
        enriched_coords = np.array(enriched_coords)
        if enriched_coords.shape[1] != 2:
            raise ValueError(f"Invalid coordinate shape: {enriched_coords.shape}")
        kd_tree = cKDTree(enriched_coords)
    except Exception as e:
        raise ValueError(f"Failed to create KD-tree: {e}")

    # generate candidate points
    nodes = generate_points_between_stations(station_coords, samples_per_edge=10)
    G = nx.Graph()
    for i, (lat, lon) in enumerate(nodes):
        dist, idx = kd_tree.query([lat, lon])
        real_data = enriched_nodes[idx][1]
        G.add_node(i,
                   lat=lat,
                   lon=lon,
                   x=lon,
                   y=lat,
                   elevation=real_data.get("elevation", 0),
                   population_density=real_data.get("population_density", 0))

    # initial graph with all possible edges
    coords = np.array([[n["lat"], n["lon"]] for _, n in G.nodes(data=True)])
    knn = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(coords)
    distances, indices = knn.kneighbors(coords)

    for i, neighbors in enumerate(indices):
        for j in neighbors:
            if i == j or G.has_edge(i, j):
                continue
            dist = np.linalg.norm(coords[i] - coords[j]) * 111000  # meters
            if dist <= MAX_EDGE_LENGTH:
                G.add_edge(i, j, length_m=dist)

    print(f"[INFO] Candidate graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # Convert to MultiGraph for compute_graph_features (required by its implementation)
    G = nx.MultiGraph(G)
    
    # Compute graph features before scoring
    print("[INFO] Computing graph features...")
    G = compute_graph_features(G, city=city)

    # Load and apply ML model
    try:
        clf = joblib.load(model_path)
    except Exception as e:
        raise ValueError(f"Failed to load ML model: {e}")

    scores = []
    edge_rows = []

    for u, v, k, data in G.edges(keys=True, data=True):  # Using keys=True for MultiGraph
        n1, n2 = G.nodes[u], G.nodes[v]
        length = data["length_m"]
        elev1 = n1.get("elevation", 0)
        elev2 = n2.get("elevation", 0)
        elev_change = abs(elev1 - elev2)
        
        # Get population density from both nodes and average them
        pop1 = n1.get("population_density", 0)
        pop2 = n2.get("population_density", 0)
        pop_val = (pop1 + pop2) / 2

        # Get existing graph metrics
        closeness_avg = (n1.get("closeness", 0) + n2.get("closeness", 0)) / 2
        betweenness_avg = (n1.get("betweenness", 0) + n2.get("betweenness", 0)) / 2
        curvature = data.get("curvature_deg", 0)

        # Create feature dictionary with actual calculated values
        row = {
            "length_m": length,
            "avg_elevation_change": elev_change,
            "avg_population_density": pop_val,
            "elevation_per_meter": elev_change / max(length, 1),  # Add missing feature
            "closeness_avg": closeness_avg,  # Use calculated average
            "betweenness_avg": betweenness_avg,  # Use calculated average
            "curvature_deg": curvature,  # Use actual curvature
            "curvature_per_meter": curvature / max(length, 1)  # Calculate per meter
        }

        if len(scores) < 3:
            print(f"[DEBUG] Features for edge: {row}")

        X = pd.DataFrame([row])
        score = clf.predict_proba(X)[0][1]  # Get probability of positive class
        scores.append(score)
        edge_rows.append((u, v, k, score, data))

    # Determine percentile threshold
    threshold = np.percentile(scores, 15)
    print(f"[INFO] ML score threshold: {threshold:.4f}")
    print(f"[INFO] Score range: min={min(scores):.4f}, max={max(scores):.4f}")

    # Create filtered graph (as regular Graph for slime simulation)
    G_filtered = nx.Graph()
    
    # First add all nodes with their attributes
    for node, data in G.nodes(data=True):
        G_filtered.add_node(node, **data)
    
    # Then add only edges above threshold
    scored = 0
    for (u, v, k, score, data) in edge_rows:
        if score > threshold:  # Keep edges with scores above threshold
            edge_data = data.copy()  # Copy edge data
            edge_data["ml_score"] = score  # Add the ML score
            G_filtered.add_edge(u, v, **edge_data)
            scored += 1

    print(f"[DONE] Retained {scored} high-score AI edges.")
    
    # Map stations to nearest nodes
    station_node_ids = []
    for s in station_coords:
        lat = float(s["lat"]) if isinstance(s, dict) else float(s[0])
        lon = float(s["lon"]) if isinstance(s, dict) else float(s[1])
        closest = min(
            G_filtered.nodes,
            key=lambda n: geodesic((lat, lon), (float(G_filtered.nodes[n]["lat"]), float(G_filtered.nodes[n]["lon"]))).meters
        )
        station_node_ids.append(closest)
        G_filtered.nodes[closest]["is_station"] = True

    return G_filtered, station_node_ids
