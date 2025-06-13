import os
import geopandas as gpd
import rasterio
import networkx as nx
from shapely.geometry import LineString
import rioxarray as rxr
import xarray as xr
import warnings

def get_raster_paths(city: str, country: str):
    elevation_path = f'data/elevation_rasters/elevation_{city.lower().replace(",", "").replace(" ", "_")}.tif'
    population_path = f'data/population_rasters/population_{country.lower()}.tif'
    
    if not os.path.exists(elevation_path):
        raise FileNotFoundError(f"Elevation raster not found: {elevation_path}")
    if not os.path.exists(population_path):
        raise FileNotFoundError(f"Population raster not found: {population_path}")
        
    return elevation_path, population_path

def enrich_graph_with_raster_features(G, elevation_path, population_path):
    # implement raster sampling for graph edges using geoPandas and rasterio.
    # https://geopandas.org/en/stable/gallery/geopandas_rasterio_sample.html
    
    # extract edge geometries
    edges = []
    for u, v, k, data in G.edges(keys=True, data=True):
        # both existing geometries and create new LineStrings from node coordinates
        if "geometry" in data:
            line = data["geometry"]
        else:
            n1, n2 = G.nodes[u], G.nodes[v]
            line = LineString([(n1["x"], n1["y"]), (n2["x"], n2["y"])])
        G.edges[u, v, k]["geometry"] = line
        edges.append({"u": u, "v": v, "k": k, "geometry": line})

    # GeoDataFrame from edges
    # EPSG:4326 is the standard geographic coordinate system
    # https://geopandas.org/en/stable/docs/user_guide/projections.html
    gdf = gpd.GeoDataFrame(edges, geometry="geometry", crs="EPSG:4326")

    # open raster files
    elev_raster = rxr.open_rasterio(elevation_path, masked=True).squeeze()
    pop_raster = rxr.open_rasterio(population_path, masked=True).squeeze()

    # https://geopandas.org/en/stable/gallery/geopandas_rasterio_sample.html
    # Convert GeoDataFrame CRS to match elevation raster CRS
    target_crs = elev_raster.rio.crs
    gdf = gdf.to_crs(target_crs)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message="Geometry is in a geographic CRS.*")
        centroids = gdf.geometry.centroid

    # raster values at node locations
    for i, (u, v, k) in enumerate(zip(gdf["u"], gdf["v"], gdf["k"])):
        node_u = G.nodes[u]
        node_v = G.nodes[v]
        
        try:
            # Sample elevation and population at node locations
            elev_u = float(elev_raster.sel(x=node_u["x"], y=node_u["y"], method="nearest").values)
            elev_v = float(elev_raster.sel(x=node_v["x"], y=node_v["y"], method="nearest").values)
            pop_u = float(pop_raster.sel(x=node_u["x"], y=node_u["y"], method="nearest").values)
            pop_v = float(pop_raster.sel(x=node_v["x"], y=node_v["y"], method="nearest").values)
        except:
            elev_u = elev_v = pop_u = pop_v = 0

        # average values for edges
        elev_val = (elev_u + elev_v) / 2.0
        pop_val = (pop_u + pop_v) / 2.0

        G.edges[u, v, k]["elevation_gain"] = elev_val
        G.edges[u, v, k]["population_density"] = pop_val
        G.nodes[u]["elevation"] = elev_u
        G.nodes[v]["elevation"] = elev_v
        G.nodes[u]["population_density"] = pop_u
        G.nodes[v]["population_density"] = pop_v

    # multiDiGraph
    if not isinstance(G, nx.MultiDiGraph):
        G = nx.MultiDiGraph(G)
    return G