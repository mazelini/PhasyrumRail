import networkx as nx
import time
import math
import numpy as np

def compute_graph_features(G, city="unspecified"):
    # Compute various graph metrics for a network.
    # - Betweenness Centrality: 
    #   NetworkX Documentation: betweenness_centrality
    #   https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.betweenness_centrality.html
    #   
    # - Closeness Centrality:
    #   NetworkX Documentation: closeness_centrality
    #   https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.closeness_centrality.html
    #   
    # - Curvature Calculation:
    #   StackOverflow: "Curve curvature in numpy"
    #   https://stackoverflow.com/questions/28269379/curve-curvature-in-numpy
    
    print(f"[INFO] ({city}) Computing graph features using NetworkX...")

    # Keep original MultiGraph structure (no conversion to simple Graph)
    graph_is_multigraph = isinstance(G, (nx.MultiGraph, nx.MultiDiGraph))

    try:
        print(f"[INFO] ({city}) Starting Calculation...")
        t0 = time.time()

        # Betweenness centrality
        # https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.betweenness_centrality.html
        btw = nx.betweenness_centrality(G, weight="length", normalized=True)
        nx.set_node_attributes(G, btw, "betweenness")

        # Closeness centrality
        # https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.closeness_centrality.html
        closeness = nx.closeness_centrality(G, distance="length")
        nx.set_node_attributes(G, closeness, "closeness")

        # Curvature calculation using derivatives
        # https://stackoverflow.com/questions/28269379/curve-curvature-in-numpy
        def calculate_curvature(p1, p2, p3):
            # Convert points to numpy arrays
            points = np.array([p1, p2, p3])
            x = points[:, 0]
            y = points[:, 1]
            
            # Calculate first derivatives
            dx = np.gradient(x)
            dy = np.gradient(y)
            
            # Calculate second derivatives
            d2x = np.gradient(dx)
            d2y = np.gradient(dy)
            
            # Calculate curvature using the formula
            curvature = np.abs(dx[1] * d2y[1] - dy[1] * d2x[1]) / (dx[1]**2 + dy[1]**2)**1.5
            
            # Convert to degrees for consistency with previous implementation
            # Scale to a reasonable range (0-180)
            curvature_deg = min(180, 180 * curvature)
            
            return curvature_deg

        # Calculate curvature for each edge
        for u, v, k, data in G.edges(keys=True, data=True):
            try:
                # Get coordinates for current edge
                p2 = (G.nodes[u]["x"], G.nodes[u]["y"])
                p3 = (G.nodes[v]["x"], G.nodes[v]["y"])
                
                # Find a previous neighbor to form three points
                prev_neighbors = [n for n in G.neighbors(u) if n != v]
                if prev_neighbors:
                    p1 = (G.nodes[prev_neighbors[0]]["x"], G.nodes[prev_neighbors[0]]["y"])
                    curvature = calculate_curvature(p1, p2, p3)
                else:
                    curvature = 0

                G[u][v][k]["curvature_deg"] = curvature
            except Exception as e:
                print(f"[WARN] Failed curvature for edge ({u}, {v}): {e}")
                G[u][v][k]["curvature_deg"] = 0

        print(f"[DONE] ({city}) Calculation done in {time.time() - t0:.2f} sec")

    except Exception as e:
        print(f"[WARN] Calculation failed: {e}")

    return G
