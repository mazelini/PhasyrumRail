import random
import pickle
import pandas as pd
import networkx as nx
import osmnx as ox
import numpy as np
from shapely.geometry import LineString
from math import radians, cos, sin, sqrt, atan2
from utils.enrich_graph import enrich_graph_with_raster_features
from utils.graph_features import compute_graph_features
from joblib import Parallel, delayed

# City configurations
city_config = [
    {"name": "London, United Kingdom", "elevation": "data/elevation_rasters/elevation_london.tif", "population": "data/population_rasters/population_uk.tif"},
    {"name": "Paris, France", "elevation": "data/elevation_rasters/elevation_paris.tif", "population": "data/population_rasters/population_france.tif"},
    {"name": "Berlin, Germany", "elevation": "data/elevation_rasters/elevation_berlin.tif", "population": "data/population_rasters/population_germany.tif"},
    {"name": "Madrid, Spain", "elevation": "data/elevation_rasters/elevation_madrid.tif", "population": "data/population_rasters/population_spain.tif"},
    {"name": "Rome, Italy", "elevation": "data/elevation_rasters/elevation_rome.tif", "population": "data/population_rasters/population_italy.tif"},
    {"name": "Amsterdam, Netherlands", "elevation": "data/elevation_rasters/elevation_amsterdam.tif", "population": "data/population_rasters/population_netherlands.tif"},
    {"name": "Vienna, Austria", "elevation": "data/elevation_rasters/elevation_vienna.tif", "population": "data/population_rasters/population_austria.tif"},
    {"name": "Zurich, Switzerland", "elevation": "data/elevation_rasters/elevation_zurich.tif", "population": "data/population_rasters/population_switzerland.tif"},
    {"name": "Warsaw, Poland", "elevation": "data/elevation_rasters/elevation_warsaw.tif", "population": "data/population_rasters/population_poland.tif"},
    {"name": "Lisbon, Portugal", "elevation": "data/elevation_rasters/elevation_lisbon.tif", "population": "data/population_rasters/population_portugal.tif"},
    {"name": "Helsinki, Finland", "elevation": "data/elevation_rasters/elevation_helsinki.tif", "population": "data/population_rasters/population_finland.tif"},
    {"name": "Prague, Czech Republic", "elevation": "data/elevation_rasters/elevation_prague.tif", "population": "data/population_rasters/population_czechia.tif"},
    {"name": "Budapest, Hungary", "elevation": "data/elevation_rasters/elevation_budapest.tif", "population": "data/population_rasters/population_hungary.tif"},
]

# Geo utilities
# Haversine formula to calculate distance between two geographic coordinates
# Source: Stack Overflow answer by user 'mattd' (2010)
# https://stackoverflow.com/a/4913653
# Adapted for use in railway route optimization with graph-based models
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi / 2)**2 + cos(phi1) * cos(phi2) * sin(dlambda / 2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))

# Feature extraction
def extract_features(G, city, is_real):
    rows = []
    for u, v, k, data in G.edges(keys=True, data=True):
        try:
            n1, n2 = G.nodes[u], G.nodes[v]
            length = data.get("length", 0)
            elev = data.get("elevation_gain", 0)
            pop = data.get("population_density", 0)
            
            attrs = {
                "city": city,
                "u": u,
                "v": v,
                "k": k,
                "length_m": length,
                "avg_elevation_change": elev,
                "avg_population_density": pop,
                "elevation_per_meter": elev / max(length, 1),
                "pop_per_meter": pop / max(length, 1),
                "curvature_deg": data.get("curvature_deg", 0),
                "curvature_per_meter": data.get("curvature_deg", 0) / max(length, 0.000001),
                "betweenness_avg": (G.nodes[u].get("betweenness", 0) + G.nodes[v].get("betweenness", 0)) / 2,
                "closeness_avg": (G.nodes[u].get("closeness", 0) + G.nodes[v].get("closeness", 0)) / 2,
                "is_dense_area": int(pop * 100 >= 5000),  # Simplified density check
                "is_real_rail": int(is_real),
            }
            rows.append(attrs)
        except Exception as e:
            print(f"[WARN] Failed edge ({u}, {v}): {e}")
            continue
    return rows

def generate_fake_edges(G_real, num_samples=10000):
    """Generate fake edges using k-nearest neighbors."""
    from sklearn.neighbors import NearestNeighbors
    coords = np.array([[G_real.nodes[n]["y"], G_real.nodes[n]["x"]] for n in G_real.nodes])
    nodes = list(G_real.nodes)
    
    # Find 5 nearest neighbors (including self)
    knn = NearestNeighbors(n_neighbors=5).fit(coords)
    indices = knn.kneighbors(coords, return_distance=False)

    fake_edges = set()
    attempts = 0
    max_attempts = num_samples * 20

    while len(fake_edges) < num_samples and attempts < max_attempts:
        i = random.choice(range(len(nodes)))
        u = nodes[i]
        for j in indices[i][1:]:  # Skip first neighbor (self)
            v = nodes[j]
            if not G_real.has_edge(u, v):
                fake_edges.add(tuple(sorted((u, v))))
                if len(fake_edges) >= num_samples:
                    break
        attempts += 1

    if attempts >= max_attempts:
        print(f"[WARN] Only generated {len(fake_edges)} fake edges after {max_attempts} attempts.")
    return list(fake_edges)

# City pipeline
def process_city(entry):
    city = entry["name"]
    print(f"[INFO] Processing {city}...")
    try:
        #Railway network data fetching and processing.
        #This implementation is based on the OpenStreetMap data fetching approach from:
        #https://stackoverflow.com/questions/62067243/open-street-map-using-osmnx-how-to-retrieve-the-hannover-subway-network
        custom_filter = '["railway"~"rail|subway"]'
        G = ox.graph_from_place(city, custom_filter=custom_filter, simplify=True)
        G = nx.convert_node_labels_to_integers(G)
        G = nx.MultiDiGraph(G)

        # Process real edges
        G = enrich_graph_with_raster_features(G, entry["elevation"], entry["population"])
        G = compute_graph_features(G, city=city)
        real_features = extract_features(G, city, is_real=True)

        # Generate and process fake edges
        fake_pairs = generate_fake_edges(G, num_samples=len(real_features))
        for u, v in fake_pairs:
            try:
                n1, n2 = G.nodes[u], G.nodes[v]
                dist = haversine(n1["y"], n1["x"], n2["y"], n2["x"])
                line = LineString([(n1["x"], n1["y"]), (n2["x"], n2["y"])])
                G.add_edge(u, v, length=dist, geometry=line)
            except (KeyError, ValueError) as e:
                print(f"[WARN] Failed to add fake edge ({u}, {v}): {e}")
                continue

        # Process fake edges
        G = enrich_graph_with_raster_features(G, entry["elevation"], entry["population"])
        G = compute_graph_features(G, city=city)
        fake_features = extract_features(G, city, is_real=False)

        # Save enriched graph
        safe_name = city.lower().replace(",", "").replace(" ", "_")
        with open(f"data/graphs/{safe_name}_raw.gpickle", "wb") as f:
            pickle.dump(G, f)

        print(f"[DONE] {city}: {len(real_features)} real, {len(fake_features)} fake edges")
        return real_features + fake_features

    except Exception as e:
        print(f"[ERROR] {city} failed: {e}")
        return []

def main():
    n_jobs = 10
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(process_city)(entry) for entry in city_config
    )
    flat_data = [row for city_rows in results for row in city_rows]
    df = pd.DataFrame(flat_data)
    df.to_csv("data/ai_training_edges_full_enhanced.csv", index=False)
    print("[DONE] Enhanced training dataset saved.")

if __name__ == "__main__":
    main()
