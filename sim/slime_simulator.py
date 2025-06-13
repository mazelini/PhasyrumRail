import networkx as nx
from tqdm import tqdm
import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import random
import warnings
warnings.filterwarnings("ignore", message=".*ScriptRunContext.*")

def edge_weight_function(u,v,d):
    # compute edge weights based on ML score, curvature, and length.
    # lower weights are preferred in path finding.
    ml_penalty = (1 - d.get("ml_score", 0.5))  # Prefer edges with high ML scores
    curvature = d.get("curvature_deg", 0)
    curvature_penalty = max(curvature - 10, 0) / 45  # Penalize sharp turns
    length_penalty = d.get("length_m", 100) **2 / 1e6  # Quadratic length penalty
    return ml_penalty + curvature_penalty + length_penalty

def slime_cost_inverse(u,v,d):
    # convert slime flow to edge cost for final path finding.
    # higher flow results in lower cost.
    return 1 / max(d.get("slime_flow", 1e-6), 1e-6)

def run_path_flow(args):
    # Helper function for parallel processing - unpacks arguments to _compute_path_flows
    return _compute_path_flows(*args)

def _compute_path_flows(G_data, source, sinks):
    G = nx.node_link_graph(G_data)
    local_flows = {}

    try:
        # networkX's implementation of Dijkstra's algorithm
        # https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.weighted.single_source_dijkstra_path.html#networkx.algorithms.shortest_paths.weighted.single_source_dijkstra_path
        paths = nx.single_source_dijkstra_path(G, source, weight=edge_weight_function)
    except nx.NetworkXNoPath:
        return local_flows

    for sink in sinks:
        if source == sink or sink not in paths:
            continue
        path = paths[sink]
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            key = tuple(sorted((u, v)))
            local_flows[key] = local_flows.get(key, 0) + 1

    return local_flows

def simulate_slime_flow(G, sources, sinks, iterations=10, slime_multiplier=0.2, min_threshold=2):
    # inspired by GitHub implementations
    # https://github.com/fogleman/physarum/blob/main/pkg/physarum/model.go
    print("[INFO] Running slime simulation...")

    # Initialize edges with ML scores and zero slime flow
    for u, v, d in G.edges(data=True):
        d["ml_score"] = min(max(d.get("ml_score", 0.5), 0.0), 1.0)
        d["slime_flow"] = 0.0

    G_data = nx.node_link_data(G)

    # Simulation parameters
    MAX_SOURCES = 80
    MAX_SINKS_PER_SOURCE = 20
    random.seed(42)
    num_workers = min(20, multiprocessing.cpu_count())

    # Run simulation iterations
    for _ in tqdm(range(iterations), desc="Simulating slime flow"):
        edge_flows = {}
        sources_sample = random.sample(sources, min(len(sources), MAX_SOURCES))
        tasks = [(G_data, source, 
                 random.sample([s for s in sources if s != source],
                             min(len(sources) - 1, MAX_SINKS_PER_SOURCE)))
                for source in sources_sample]

        # Parallel execution using ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for local_flows in executor.map(run_path_flow, tasks):
                for edge, count in local_flows.items():
                    edge_flows[edge] = edge_flows.get(edge, 0) + count

        # Update flows
        for (u, v), count in edge_flows.items():
            if G.has_edge(u, v):
                G[u][v]["slime_flow"] += count
            elif G.has_edge(v, u):
                G[v][u]["slime_flow"] += count

    if G.number_of_edges() == 0:
        print("[WARN] Graph has no edges after simulation.")
        return nx.Graph()

    # Normalize and threshold flows
    max_flow = max(d.get("slime_flow", 0) for _, _, d in G.edges(data=True))
    if max_flow > 0:
        for _, _, d in G.edges(data=True):
            d["slime_flow_norm"] = d["slime_flow"] / max_flow

    # Get all flow values for threshold calculation
    flows = [d["slime_flow"] for _, _, d in G.edges(data=True)]
            
    # adaptive threshold based on flow distribution
    threshold = max(min(np.percentile(flows, 30), max_flow * slime_multiplier), min_threshold)
    
    # pruned graph
    strong_edges = [(u, v) for u, v, d in G.edges(data=True) if d["slime_flow"] >= threshold]
    G_pruned = G.edge_subgraph(strong_edges).copy()
    
    # Preserve stations
    for n, d in G.nodes(data=True):
        if d.get("is_station") and n not in G_pruned:
            G_pruned.add_node(n, **d)

    # Find paths between all stations - derived from NetworkX shortest path algorithms
    # https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.generic.shortest_path.html#networkx.algorithms.shortest_paths.generic.shortest_path
    final_edges = set()
    for i, source in enumerate(sources):
        for target in sources[i+1:]:
            try:
                # F=ind optimal paths between all station pairs
                path = nx.shortest_path(G_pruned, source, target, weight=slime_cost_inverse)
                final_edges.update(zip(path[:-1], path[1:]))
            except nx.NetworkXNoPath:
                continue

    if not final_edges:
        print("[WARN] No station-to-station paths formed. Using strongest slime edges.")
        return G_pruned

    #final graph
    G_final = nx.Graph()
    for u, v in final_edges:
        if G.has_edge(u, v):
            G_final.add_edge(u, v, **G[u][v])
            G_final.nodes[u].update(G.nodes[u])
            G_final.nodes[v].update(G.nodes[v])

    print(f"[DONE] Final AI-enhanced rail network has {G_final.number_of_edges()} edges.")
    return G_final
