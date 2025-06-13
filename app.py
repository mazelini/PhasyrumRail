import streamlit as st
import osmnx as ox
import folium
from streamlit_folium import st_folium
from streamlit.components.v1 import html
from geopy.distance import geodesic
from utils.build_candidate_graph import build_candidate_graph
from sim.slime_simulator import simulate_slime_flow
from utils.map_export import generate_results_map
import warnings
warnings.filterwarnings("ignore", message=".*ScriptRunContext.*")

st.set_page_config(layout="wide")
st.title("PhysarumRail: Optimizing Railway Networks with Machine Learning and Bio-Inspired Reinforcement")

# Cities (only for centering the map)
city = st.selectbox("Choose a city:", [
    "London, United Kingdom",
    "Paris, France",
    "Berlin, Germany",
    "Madrid, Spain",
    "Rome, Italy",
    "Amsterdam, Netherlands",
    "Vienna, Austria",
    "Zurich, Switzerland",
    "Warsaw, Poland",
    "Lisbon, Portugal",
    "Helsinki, Finland",
    "Prague, Czech Republic",
    "Budapest, Hungary",
])

# Default map center
city_coords = {
    "London, United Kingdom": [51.5074, -0.1278],
    "Paris, France": [48.8566, 2.3522],
    "Berlin, Germany": [52.52, 13.405],
    "Madrid, Spain": [40.4168, -3.7038],
    "Rome, Italy": [41.9028, 12.4964],
    "Amsterdam, Netherlands": [52.3676, 4.9041],
    "Vienna, Austria": [48.2082, 16.3738],
    "Zurich, Switzerland": [47.3769, 8.5417],
    "Warsaw, Poland": [52.2297, 21.0122],
    "Lisbon, Portugal": [38.7169, -9.1399],
    "Helsinki, Finland": [60.1695, 24.9354],
    "Prague, Czech Republic": [50.0755, 14.4378],
    "Budapest, Hungary": [47.4979, 19.0402],
}
center = city_coords.get(city, [51.5074, -0.1278])

def fetch_osm_stations_for_city(city_name, radius_km=20, min_dist_m=300):
    """
    Railway network data fetching and processing.

    This implementation is based on the OpenStreetMap data fetching approach from:
    https://stackoverflow.com/questions/62067243/open-street-map-using-osmnx-how-to-retrieve-the-hannover-subway-network

    Uses ox.features_from_place()
    Filters stations within specified radius (radius_km) from city center
    Implements station deduplication by removing stations closer than min_dist_m
    Handles both rail and subway stations using railway=station|subway_station tags
    """
    # 1. Get city center
    gdf_place = ox.geocode_to_gdf(city_name)
    gdf_proj = gdf_place.to_crs(epsg=3857)  # Web Mercator projection (meters)
    city_center = gdf_proj.geometry.centroid.to_crs(epsg=4326).iloc[0]  # Back to lat/lon
    center_coords = (city_center.y, city_center.x)

    # 2. Fetch OSM railway station points
    tags = {"railway": ["station", "subway_station"]}
    gdf = ox.features_from_place(city_name, tags=tags)
    gdf = gdf[gdf.geometry.type == "Point"]
    gdf = gdf[gdf["railway"].isin(["station", "subway_station"])]
    gdf = gdf.to_crs("EPSG:4326")

    # 3. Filter to within radius_km
    def within_radius(row):
        pt = (row.geometry.y, row.geometry.x)
        return geodesic(center_coords, pt).km <= radius_km

    gdf = gdf[gdf.apply(within_radius, axis=1)]

    # 4. Deduplicate nearby stations
    coords = []
    for geom in gdf.geometry:
        candidate = (float(geom.y), float(geom.x))
        if not any(geodesic(candidate, existing).meters < min_dist_m for existing in coords):
            coords.append(candidate)

    return [{"lat": lat, "lon": lon} for lat, lon in coords]

# Session state
if "stations" not in st.session_state:
    st.session_state["stations"] = []
if "G_result" not in st.session_state:
    st.session_state["G_result"] = None
    
st.subheader("Real Station Fetch Settings:")
col1, col2 = st.columns(2)
with col1:
    radius_km = st.slider(
        "Search Radius (km)",
        min_value=1,
        max_value=15,
        value=5,
        step=1
    )
with col2:
    min_dist_m = st.slider(
        "Minimum Distance Between Stations (m)",
        min_value=50,
        max_value=1000,
        value=500,
        step=50
    )
    
col1, col2 = st.columns(2)
with col1:
    if st.button("ADD Real Stations"):
        with st.spinner("Fetching real railway stations from OpenStreetMap..."):
            real_stations = fetch_osm_stations_for_city(
                city, radius_km=radius_km, min_dist_m=min_dist_m)
            st.session_state["stations"] = real_stations
            st.success(f"Added {len(real_stations)} real stations for {city}")
with col2:
    if st.button("RESET All Stations"):
        st.session_state["stations"] = []
        st.session_state["G_result"] = None
        st.rerun()

# Map UI
fmap = folium.Map(location=center, zoom_start=11, tiles="CartoDB positron")
folium.Marker(center, tooltip="City Center", icon=folium.Icon(color="green")).add_to(fmap)

# Ensure all lat/lon values are floats
cleaned_stations = [
    {"lat": float(s["lat"]), "lon": float(s["lon"])} for s in st.session_state["stations"]
]

for idx, station in enumerate(cleaned_stations):
    lat = station["lat"]
    lon = station["lon"]
    folium.Marker([lat, lon], tooltip=f"Station {idx+1}",
                  icon=folium.Icon(color="blue")).add_to(fmap)


st.subheader("Click on the map to drop a station:")
clicked = st_folium(fmap, height=600, width=1000)

if clicked and clicked.get("last_clicked"):
    lat = clicked["last_clicked"]["lat"]
    lon = clicked["last_clicked"]["lng"]
    new_station = {"lat": lat, "lon": lon}
    if new_station not in st.session_state["stations"]:
        st.session_state["stations"].append(new_station)
        st.rerun()

if st.session_state["stations"]:
    st.success(f"{len(st.session_state['stations'])} station(s) placed:")

st.subheader("Graph Controls:")
col1, col2 = st.columns(2)
with col1:
    k_value = st.slider(
        "Possible Connections per Station",
        min_value=3,
        max_value=12,
        value=6
    )
with col2:
    smoothing_value = st.slider(
        "Path smoothing (visual only)", 
        min_value=0.001, 
        max_value=0.003, 
        value=0.002, 
        step=0.0001,
        format="%.4f"
    )

st.subheader("Slime Simulation Controls:")
col1, col2, col3 = st.columns(3)

with col1:
    slime_multiplier = st.slider(
        "Slime Flow Sensitivity", 
        min_value=0.1, max_value=1.0, value=0.3, step=0.05, format="%.2f"
    )
with col2:
    min_threshold = st.slider(
        "Minimum Slime Threshold", 
        min_value=2, max_value=8, value=4, step=1
    )
with col3:
    iterations = st.slider(
        "Iterations", 
        min_value=5, max_value=20, value=10, step=1
    )

if st.button("Run BIO-AI Optimization") and len(st.session_state["stations"]) >= 3:
    with st.spinner("Running ..."):
                
        # Extract city and country from dropdown value
        city_parts = city.split(", ")
        selected_city = city_parts[0]
        selected_country = city_parts[1].lower().replace(" ", "_") if len(city_parts) > 1 else "UK"  # fallback

        # Build AI-optimized candidate graph with enriched features
        G, custom_ids = build_candidate_graph(
        st.session_state["stations"],
        city=selected_city,
        country=selected_country,
        k=k_value
    )

        # Define slime source/sink nodes
        sources = custom_ids
        sinks = custom_ids

        existing_nodes = set(G.nodes)
        filtered_station_ids = [n for n in custom_ids if n in existing_nodes]

        # Simulate slime pathfinding
        G = simulate_slime_flow(
            G,
            sources=filtered_station_ids,
            sinks=filtered_station_ids,
            iterations=iterations,
            slime_multiplier=slime_multiplier,
            min_threshold=min_threshold
        )


        # Get all edges that have positive slime flow
        slime_edges = [(u, v) for u, v, d in G.edges(data=True) 
                       if d.get("slime_flow", 0) > 0]
        st.info(f"Slime-enhanced edges created: {len(slime_edges)}")

        if slime_edges:
            # Store the graph with slime flows in session state
            st.session_state["G_result"] = G
            
            # Generate an interactive map showing the suggested railway network
            generate_results_map(
                G, custom_ids=custom_ids, city_name=city, 
                smoothing=smoothing_value,
                colormap_name="viridis"
                )

            # Display the generated HTML map in the Streamlit interface
            with open("results_map.html", "r", encoding="utf-8") as f:
                html(f.read(), height=600, width=1000)
        else:
            st.warning("No slime-enhanced routes were generated.")
