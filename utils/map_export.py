import folium
import osmnx as ox
from branca.colormap import StepColormap  # https://python-visualization.github.io/folium/latest/advanced_guide/colormaps.html
from shapely.geometry import LineString, box, mapping
import cv2
import rasterio
import numpy as np
from matplotlib import cm
from folium.raster_layers import ImageOverlay
from rasterio.mask import mask
import os

def smooth_line(latlon_coords, tolerance=0.002):
    line = LineString(latlon_coords)
    return list(line.simplify(tolerance, preserve_topology=True).coords)

def generate_results_map(G, custom_ids, city_name, output_file="results_map.html", smoothing=0.002, colormap_name="viridis"):
    try:
        center_gdf = ox.geocode_to_gdf(city_name)
        center = center_gdf.geometry.iloc[0].centroid
        fmap = folium.Map(location=[center.y, center.x], zoom_start=11, tiles="CartoDB positron")
    except Exception as e:
        print(f"[WARN] Could not geocode city: {e}")
        # Use city's default coordinates from app.py
        fmap = folium.Map(location=[51.5074, -0.1278], zoom_start=11, tiles="CartoDB positron")

    # === Auto-detect raster file from city
    city_parts = city_name.split(", ")
    country_name = city_parts[1].strip().lower().replace(" ", "_") if len(city_parts) > 1 else "unknown"
    pop_raster_path = f"data/population_rasters/population_{country_name}.tif"
    
    if not os.path.exists(pop_raster_path):
        print(f"[WARN] Missing population raster: {pop_raster_path}")
    else:
        try:
            # population raster visualization approach adapted from:
            # https://www.linkedin.com/pulse/visualize-dem-interactive-map-chonghua-yin
            with rasterio.open(pop_raster_path) as src:
                bbox_geom = center_gdf.geometry.iloc[0].buffer(0.025).envelope
                bbox_bounds = bbox_geom.bounds
                geo_mask = [mapping(box(*bbox_bounds))]

                clipped_data = mask(dataset=src, shapes=geo_mask, crop=True)[0]
                data = clipped_data[0]

                nodata = src.nodata if src.nodata is not None else -99999
                data = np.where(data == nodata, np.nan, data)

                # Normalize
                data_clipped = np.clip(data, 1, 20000)
                norm = np.clip(np.log1p(data_clipped) / np.log1p(20000), 0, 1)
                norm = norm ** 0.2  # Optional contrast boost

                # Apply colormap
                cmap = cm.get_cmap(colormap_name)
                rgba_img = cmap(norm)
                rgb_img = (rgba_img[:, :, :3] * 255).astype(np.uint8)

                # Upscale for better display resolution
                upscale = 15
                h, w = rgb_img.shape[:2]
                rgb_img = cv2.resize(rgb_img, (w * upscale, h * upscale), interpolation=cv2.INTER_CUBIC)

                bounds = [
                    [bbox_bounds[1], bbox_bounds[0]],  # SW
                    [bbox_bounds[3], bbox_bounds[2]]   # NE
                ]

                ImageOverlay(
                    image=rgb_img,
                    bounds=bounds,
                    opacity=0.8,
                    name=f"Population Heatmap ({colormap_name})",
                    interactive=False,
                    zindex=1
                ).add_to(fmap)

                print(f"[INFO] Population clipped to city. Pop min: {np.nanmin(data):.2f}, max: {np.nanmax(data):.2f}")

        except Exception as e:
            print(f"[WARN] Could not render population heatmap: {e}")

    # === Slime Flow Edges ===
    slime_edges = [(u, v, d) for u, v, d in G.edges(data=True) if d.get("slime_flow", 0) > 0]
    slime_count = len(slime_edges)

    colormap = StepColormap(
        colors=['#fee5d9', '#fcae91', '#fb6a4a', '#de2d26', '#a50f15'],
        index=[0.1, 1, 2, 4, 8],
        vmin=0,
        vmax=10
    )
    colormap.caption = "Slime Flow Intensity"

    ai_layer = folium.FeatureGroup(name="AI Optimized Railway", show=True)
    for u, v, d in slime_edges:
        n1, n2 = G.nodes[u], G.nodes[v]
        coords = [(n1["y"], n1["x"]), (n2["y"], n2["x"])]
        smoothed = smooth_line(coords, tolerance=smoothing)
        flow = d["slime_flow"]
        norm = d.get("slime_flow_norm", 0)
        thickness = max(2, min(10, norm * 12))

        folium.PolyLine(
            smoothed,
            color=colormap(flow),
            weight=thickness,
            opacity=0.7
        ).add_to(ai_layer)

    fmap.add_child(ai_layer)
    colormap.add_to(fmap)

    # === Station Markers ===
    station_layer = folium.FeatureGroup(name="Chosen Stations", show=True)
    for i, nid in enumerate(custom_ids):
        if nid not in G.nodes:
            print(f"[WARN] Station {nid} not in final graph, skipping.")
            continue
        node = G.nodes[nid]
        folium.Marker(
            location=(node["y"], node["x"]),
            tooltip=f"Station {i+1}",
            icon=folium.Icon(color="blue", icon="star")
        ).add_to(station_layer)
    fmap.add_child(station_layer)

    # https://python-visualization.github.io/folium/modules.html
    folium.LayerControl(collapsed=False).add_to(fmap)
    print(f"[INFO] Map generated with {slime_count} AI-enhanced edges.")
    fmap.save(output_file)
