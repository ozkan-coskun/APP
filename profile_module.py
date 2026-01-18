# profile_module.py
import streamlit as st
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import xarray as xr
import numpy as np
import os

# ===== HELPER FUNCTIONS =====
def find_line_intersection(p1, p2, p3, p4):
    """Find intersection point between two lines
    p1-p2: First line (section line)
    p3-p4: Second line (new shoreline)
    """
    try:
        # Calculate intersection point between two lines
        x1, y1 = p1['lon'], p1['lat']
        x2, y2 = p2['lon'], p2['lat']
        x3, y3 = p3['lon'], p3['lat']
        x4, y4 = p4['lon'], p4['lat']
        
        # Calculate determinant
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        
        if abs(denom) < 1e-10:  # Parallel lines
            return None
        
        # Intersection point
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        
        # Check if intersection point is on the section line
        # t value should be between 0 and 1
        if t < 0 or t > 1:
            return None
        
        # Intersection point coordinates
        intersection_lon = x1 + t * (x2 - x1)
        intersection_lat = y1 + t * (y2 - y1)
        
        return {'lat': intersection_lat, 'lon': intersection_lon}
    except:
        return None

def calculate_distance(point1, point2):
    """Calculate distance between two points (Haversine formula)"""
    R = 6371000  # Earth radius (meters)
    lat1_rad = np.radians(point1['lat'])
    lat2_rad = np.radians(point2['lat'])
    delta_lat = np.radians(point2['lat'] - point1['lat'])
    delta_lon = np.radians(point2['lon'] - point1['lon'])
    
    a = np.sin(delta_lat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

@st.cache_data
def load_bathymetry():
    try:
        file_name = "data.nc"
        if os.path.exists(file_name):
            file_path = os.path.abspath(file_name)
        else:
            file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
        
        try:
            return xr.open_dataset(file_path, engine='netcdf4')
        except:
            return xr.open_dataset(file_path, engine='scipy')
    except:
        return None

def extract_depth_profile(ds, point1, point2, num_points=100):
    if ds is None:
        return None, None
    
    lats = np.linspace(point1['lat'], point2['lat'], num_points)
    lons = np.linspace(point1['lon'], point2['lon'], num_points)
    
    R = 6371000
    lat1_rad = np.radians(point1['lat'])
    lat2_rad = np.radians(lats)
    delta_lat = np.radians(lats - point1['lat'])
    delta_lon = np.radians(lons - point1['lon'])
    
    a = np.sin(delta_lat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distances = R * c
    
    try:
        # Get coordinates from data_vars (data.nc format)
        if 'latitude' in ds.data_vars and 'longitude' in ds.data_vars:
            ds_lats = ds['latitude'].values
            ds_lons = ds['longitude'].values
        else:
            return distances.tolist(), [0.0] * len(distances)
        
        # Find depth variable
        depth_var = None
        for var in ds.data_vars:
            if 'label' in var.lower() or 'depth' in var.lower() or 'elevation' in var.lower():
                depth_var = var
                break
        if depth_var is None:
            # Use first variable excluding latitude and longitude
            available_vars = [v for v in ds.data_vars if v not in ['latitude', 'longitude', 'lat', 'lon']]
            if available_vars:
                depth_var = available_vars[0]
            else:
                return distances.tolist(), [0.0] * len(distances)
        
        # Get depth data
        depth_data = ds[depth_var].values
        
        # Find nearest depth value for each point
        depths = []
        for lat, lon in zip(lats, lons):
            # Find nearest point (Euclidean distance)
            distances_to_points = np.sqrt((ds_lats - lat)**2 + (ds_lons - lon)**2)
            nearest_idx = np.argmin(distances_to_points)
            depth_value = float(depth_data[nearest_idx])
            depths.append(depth_value)
        
        depths = np.array(depths)
        
        # Fix NaN values
        if np.isnan(depths).any():
            valid_idx = ~np.isnan(depths)
            if valid_idx.sum() > 1:
                depths = np.interp(np.arange(len(depths)), np.arange(len(depths))[valid_idx], depths[valid_idx])
            else:
                depths = np.nan_to_num(depths, nan=-5.0)
        
        # If positive values exist (elevation), make negative (for depth)
        if np.nanmean(depths) > 0:
            depths = -depths
        
        return distances.tolist(), depths.tolist()
    except Exception as e:
        st.error(f"Error extracting profile: {e}")
        return distances.tolist(), [0.0] * len(distances)

# ===== INITIALIZE SESSION STATE =====
if 'sections' not in st.session_state:
    st.session_state.sections = {
        'A': {'points': [], 'bathy_dist': [], 'bathy_depth': [], 'user_dist': [], 'user_depth': [], 'completed': False},
        'B': {'points': [], 'bathy_dist': [], 'bathy_depth': [], 'user_dist': [], 'user_depth': [], 'completed': False},
        'C': {'points': [], 'bathy_dist': [], 'bathy_depth': [], 'user_dist': [], 'user_depth': [], 'completed': False}
    }

if 'current_section' not in st.session_state:
    st.session_state.current_section = 'A'

if 'coord_version' not in st.session_state:
    st.session_state.coord_version = 0

# New shoreline coordinates (constants)
NEW_SHORELINE_P1 = {'lat': 41.1775, 'lon': 29.6244}  # 41°10'39"N 29°37'28"E
NEW_SHORELINE_P2 = {'lat': 41.1747, 'lon': 29.6286}  # 41°10'29"N 29°37'43"E

# Sill depth constant (meters)
SILL_DEPTH_TARGET = 2.5  # Target depth for sill location

def render_profile_section():
    # Ensure session state is initialized
    if 'sections' not in st.session_state:
        st.session_state.sections = {
            'A': {'points': [], 'bathy_dist': [], 'bathy_depth': [], 'user_dist': [], 'user_depth': [], 'completed': False},
            'B': {'points': [], 'bathy_dist': [], 'bathy_depth': [], 'user_dist': [], 'user_depth': [], 'completed': False},
            'C': {'points': [], 'bathy_dist': [], 'bathy_depth': [], 'user_dist': [], 'user_depth': [], 'completed': False}
        }
    
    if 'current_section' not in st.session_state:
        st.session_state.current_section = 'A'
    
    if 'coord_version' not in st.session_state:
        st.session_state.coord_version = 0
    
    bathymetry_ds = load_bathymetry()

    st.markdown("---")
    
    current = st.session_state.current_section
    
    st.markdown("### Section Navigation")
    col_a, col_b, col_c, col_all = st.columns(4)
    
    with col_a:
        label = "[Done] A-A'" if st.session_state.sections['A']['completed'] else "A-A'"
        if st.button(label, key="nav_a", use_container_width=True, type="primary" if current == 'A' else "secondary"):
            st.session_state.current_section = 'A'
            st.rerun()
    
    with col_b:
        label = "[Done] B-B'" if st.session_state.sections['B']['completed'] else "B-B'"
        if st.button(label, key="nav_b", use_container_width=True, type="primary" if current == 'B' else "secondary"):
            st.session_state.current_section = 'B'
            st.rerun()
    
    with col_c:
        label = "[Done] C-C'" if st.session_state.sections['C']['completed'] else "C-C'"
        if st.button(label, key="nav_c", use_container_width=True, type="primary" if current == 'C' else "secondary"):
            st.session_state.current_section = 'C'
            st.rerun()
    
    with col_all:
        completed_count = sum(1 for s in st.session_state.sections.values() if s['completed'])
        if st.button(f"All Results ({completed_count}/3)", key="nav_all", use_container_width=True, type="primary" if current == 'ALL' else "secondary"):
            st.session_state.current_section = 'ALL'
            st.rerun()
    
    # ===== ALL RESULTS VIEW =====
    if current == 'ALL':
        st.info("Viewing: **All Results Summary**")
        st.markdown("---")
        
        completed_sections = [name for name, data in st.session_state.sections.items() if data['completed']]
        
        if not completed_sections:
            st.warning("No sections completed yet. Please complete at least one section to view results.")
        else:
            # ===== VOLUME CALCULATION SUMMARY =====
            st.markdown("## 📊 Volume Calculation Summary")
            
            vol_results, error = calculate_total_volume()
            
            if error:
                st.warning(f"Volume calculation failed: {error}")
            else:
                # Main metrics
                col_total, col_ab, col_bc = st.columns(3)
                
                with col_total:
                    st.metric(
                        "🏗️ Total Fill Volume", 
                        f"{vol_results['total']:,.0f} m³",
                        help="Total volume including all regions (extra areas included in calculation)"
                    )
                
                with col_ab:
                    st.metric(
                        "A-B Region Volume", 
                        f"{vol_results['volumes']['A-B']:,.0f} m³"
                    )
                
                with col_bc:
                    st.metric(
                        "B-C Region Volume", 
                        f"{vol_results['volumes']['B-C']:,.0f} m³"
                    )
                
                # Detail table
                st.markdown("#### Section Details")
                
                detail_cols = st.columns(3)
                for i, sec_name in enumerate(['A', 'B', 'C']):
                    with detail_cols[i]:
                        st.markdown(f"**Section {sec_name}-{sec_name}'**")
                        st.write(f"Fill Area: **{vol_results['areas'][sec_name]:,.1f} m²**")
                
                st.markdown("#### Inter-Section Distances")
                dist_col1, dist_col2 = st.columns(2)
                with dist_col1:
                    st.write(f"A ↔ B Distance: **{vol_results['distances']['A-B']:,.1f} m**")
                with dist_col2:
                    st.write(f"B ↔ C Distance: **{vol_results['distances']['B-C']:,.1f} m**")
                
                st.markdown("---")
                
                # Formula explanation
                with st.expander("📐 Calculation Method"):
                    st.markdown("""
                    **Average End Area Method**
                    
                    ```
                    V = (A₁ + A₂) / 2 × L
                    ```
                    
                    - **A₁, A₂**: Fill areas of two sections (m²)
                    - **L**: Distance between sections (m)
                    - **V**: Volume (m³)
                    
                    *Note: This method provides reasonable results even when sections are not parallel.*
                    """)
            
            st.markdown("---")
            
            st.markdown("## Combined View - All Sections")
            
            fig_combined = go.Figure()
            colors = {'A': '#2563EB', 'B': '#DC2626', 'C': '#FACC15'}
            sill_colors = {'A': '#006400', 'B': '#00FF00', 'C': '#90EE90'}  # Dark green, normal green, light green
            
            for sec_name in ['A', 'B', 'C']:
                sec_data = st.session_state.sections[sec_name]
                if sec_data['completed']:
                    fig_combined.add_trace(go.Scatter(
                        x=sec_data['bathy_dist'], 
                        y=sec_data['bathy_depth'], 
                        mode='lines', 
                        name=f'{sec_name} Bathymetry',
                        line=dict(color=colors[sec_name], width=2)
                    ))
                    fig_combined.add_trace(go.Scatter(
                        x=sec_data['user_dist'], 
                        y=sec_data['user_depth'], 
                        mode='lines', 
                        name=f'{sec_name} Design',
                        line=dict(color=colors[sec_name], width=2, dash='dash')
                    ))
                    
                    # Add sill location marker
                    if sec_data.get('sill_distance') is not None and sec_data.get('sill_depth') is not None:
                        fig_combined.add_trace(go.Scatter(
                            x=[sec_data['sill_distance']], 
                            y=[sec_data['sill_depth']], 
                            mode='markers',
                            name=f'{sec_name} Sill',
                            marker=dict(
                                symbol='diamond',
                                size=12,
                                color=sill_colors[sec_name],
                                line=dict(color='#000000', width=1.5)
                            ),
                            hovertemplate=f'{sec_name} Sill<br>Distance: %{{x:.1f}} m<br>Depth: %{{y:.2f}} m<extra></extra>'
                        ))
                        
                        # Add vertical line downward from sill
                        min_depth = min(min(sec_data['bathy_depth']), min(sec_data['user_depth']))
                        fig_combined.add_shape(
                            type="line",
                            x0=sec_data['sill_distance'],
                            y0=sec_data['sill_depth'],
                            x1=sec_data['sill_distance'],
                            y1=min_depth - 1,
                            line=dict(color=sill_colors[sec_name], width=2, dash='dash')
                        )
            
            fig_combined.update_layout(
                xaxis_title="Distance (m)", 
                yaxis_title="Depth (m)", 
                height=500,
                legend=dict(x=1.02, y=1, xanchor='left')
            )
            st.plotly_chart(fig_combined)
    
    # ===== SECTION EDITING VIEW =====
    else:
        section = st.session_state.sections[current]
        
        st.info(f"Working on: **Section {current}-{current}'**")
        st.markdown("---")

        st.markdown(f"### Step 1: Select Points for Section {current}-{current}'")

        m = folium.Map(location=[41.175354, 29.626743], zoom_start=15)
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Satellite',
            overlay=False,
            control=True
        ).add_to(m)
        map_colors = {'A': 'blue', 'B': 'green', 'C': 'orange'}

        # Add new shoreline to map
        new_shoreline_coords = [[NEW_SHORELINE_P1['lat'], NEW_SHORELINE_P1['lon']], 
                                [NEW_SHORELINE_P2['lat'], NEW_SHORELINE_P2['lon']]]
        
        folium.PolyLine(new_shoreline_coords, color='green', weight=3, opacity=0.8,
                       popup='New Shoreline (Fill Start)').add_to(m)
        
        # Add markers to shoreline start and end points
        folium.Marker(new_shoreline_coords[0], popup='New Shoreline Start',
                     icon=folium.Icon(color='green', icon='info-sign')).add_to(m)
        folium.Marker(new_shoreline_coords[1], popup='New Shoreline End',
                     icon=folium.Icon(color='green', icon='info-sign')).add_to(m)

        for sec_name, sec_data in st.session_state.sections.items():
            if sec_data['points']:
                color = map_colors.get(sec_name, 'gray')
                for idx, pt in enumerate(sec_data['points']):
                    folium.Marker(
                        [pt['lat'], pt['lon']],
                        popup=f"{sec_name if idx==0 else sec_name}'",
                        icon=folium.Icon(color=color if sec_name == current else 'gray')
                    ).add_to(m)
                if len(sec_data['points']) == 2:
                    folium.PolyLine(
                        [[p['lat'], p['lon']] for p in sec_data['points']],
                        color=color if sec_name == current else 'gray',
                        weight=3 if sec_name == current else 2,
                        opacity=1.0 if sec_name == current else 0.5
                    ).add_to(m)

        m.add_child(folium.LatLngPopup())
        map_data = st_folium(m, height=400, use_container_width=True, key=f"map_{current}")

        if map_data and map_data.get('last_clicked'):
            lat = map_data['last_clicked']['lat']
            lon = map_data['last_clicked']['lng']
            
            if len(section['points']) < 2:
                new_point = True
                if section['points']:
                    last = section['points'][-1]
                    if abs(last['lat'] - lat) < 0.0001 and abs(last['lon'] - lon) < 0.0001:
                        new_point = False
                
                if new_point:
                    section['points'].append({'lat': lat, 'lon': lon})
                    st.session_state.coord_version += 1
                    st.rerun()

        st.markdown("#### Manual Coordinates")

        v = st.session_state.coord_version
        default_lat1 = section['points'][0]['lat'] if section['points'] else 41.175354
        default_lon1 = section['points'][0]['lon'] if section['points'] else 29.626743
        default_lat2 = section['points'][1]['lat'] if len(section['points']) > 1 else 41.175000
        default_lon2 = section['points'][1]['lon'] if len(section['points']) > 1 else 29.627000

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**Point {current}**")
            lat1 = st.number_input("Latitude", value=default_lat1, format="%.6f", key=f"lat1_{current}_{v}")
            lon1 = st.number_input("Longitude", value=default_lon1, format="%.6f", key=f"lon1_{current}_{v}")

        with col2:
            st.markdown(f"**Point {current}'**")
            lat2 = st.number_input("Latitude ", value=default_lat2, format="%.6f", key=f"lat2_{current}_{v}")
            lon2 = st.number_input("Longitude ", value=default_lon2, format="%.6f", key=f"lon2_{current}_{v}")

        col_apply, col_reset = st.columns(2)
        with col_apply:
            if st.button("Apply Coordinates", key=f"apply_{current}", use_container_width=True):
                section['points'] = [{'lat': lat1, 'lon': lon1}, {'lat': lat2, 'lon': lon2}]
                st.rerun()
        with col_reset:
            if st.button("Reset Points", key=f"reset_{current}", use_container_width=True):
                section['points'] = []
                section['completed'] = False
                section['bathy_dist'] = []
                section['bathy_depth'] = []
                section['user_dist'] = []
                section['user_depth'] = []
                st.session_state.coord_version += 1
                st.rerun()

        if len(section['points']) == 2:
            st.success("Both points selected!")
        else:
            st.warning("Select 2 points on the map or enter manually")

        st.markdown("---")

        if len(section['points']) == 2:
            st.markdown(f"### Step 2: Bathymetry Profile")
            
            if not section['bathy_dist']:
                dist, depth = extract_depth_profile(bathymetry_ds, section['points'][0], section['points'][1])
                if dist and depth:
                    section['bathy_dist'] = dist
                    section['bathy_depth'] = depth
            
            if section['bathy_dist']:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=section['bathy_dist'], y=section['bathy_depth'], mode='lines+markers', name='Bathymetry', line=dict(color='#0077B6', width=2)))
                fig.update_layout(xaxis_title="Distance (m)", yaxis_title="Depth (m)", height=350)
                st.plotly_chart(fig)
                
                st.metric("Total Distance", f"{section['bathy_dist'][-1]:.1f} m")
                
                # Create automatic design profile
                # Parabola: y = 0.11 * x^0.67
                # Sill location: where depth reaches 2.5 meters
                if not section['user_dist']:
                    # Find intersection point with new shoreline
                    section_p1 = section['points'][0]
                    section_p2 = section['points'][1]
                    intersection_start = find_line_intersection(section_p1, section_p2, NEW_SHORELINE_P1, NEW_SHORELINE_P2)
                    
                    # Create design profile from bathymetry profile
                    bathy_dist_array = np.array(section['bathy_dist'])
                    
                    # Calculate fill start distance (new shoreline intersection)
                    fill_distance = calculate_distance(section_p1, intersection_start) if intersection_start else 0.0
                    
                    # Calculate sill distance: where parabola reaches SILL_DEPTH_TARGET (2.5m)
                    # Formula: y = 0.11 * x^0.67, solve for x when y = 2.5
                    # x = (y / 0.11)^(1/0.67)
                    relative_x_sill = (SILL_DEPTH_TARGET / 0.11) ** (1 / 0.67)
                    sill_distance = fill_distance + relative_x_sill
                    sill_depth = -SILL_DEPTH_TARGET
                    
                    # Store sill and fill locations
                    section['fill_distance'] = fill_distance  # Store original shoreline position
                    section['sill_distance'] = sill_distance
                    section['sill_depth'] = sill_depth
                    
                    # Trim bathymetry profile to sill + buffer distance
                    buffer_distance = 10  # Show 10 meters after sill
                    max_distance = sill_distance + buffer_distance
                    
                    bathy_dist_trimmed = []
                    bathy_depth_trimmed = []
                    for i, x in enumerate(section['bathy_dist']):
                        if x <= max_distance:
                            bathy_dist_trimmed.append(x)
                            bathy_depth_trimmed.append(section['bathy_depth'][i])
                    
                    # Ensure bathymetry has a point at sill location
                    if bathy_dist_trimmed and bathy_dist_trimmed[-1] < sill_distance:
                        # Interpolate bathymetry depth at sill location
                        sill_bathy_depth = np.interp(sill_distance, section['bathy_dist'], section['bathy_depth'])
                        bathy_dist_trimmed.append(sill_distance)
                        bathy_depth_trimmed.append(float(sill_bathy_depth))
                    
                    section['bathy_dist'] = bathy_dist_trimmed
                    section['bathy_depth'] = bathy_depth_trimmed
                    
                    # Calculate depth for each distance point using formula (only up to sill)
                    # Use trimmed bathymetry distances as reference
                    design_depths = []
                    design_dists = []
                    for x in bathy_dist_trimmed:
                        if x <= sill_distance:
                            design_dists.append(x)
                            if x <= fill_distance:
                                # Fill area: 0 depth
                                design_depths.append(0.0)
                            else:
                                # Parabola region
                                relative_x = x - fill_distance
                                if relative_x > 0:
                                    y = 0.11 * (relative_x ** 0.67)
                                    design_depths.append(-abs(y))
                                else:
                                    design_depths.append(0.0)
                    
                    # Ensure design profile ends exactly at sill point
                    if len(design_dists) == 0 or abs(design_dists[-1] - sill_distance) > 0.01:
                        design_dists.append(sill_distance)
                        design_depths.append(sill_depth)
                    
                    section['user_dist'] = design_dists
                    section['user_depth'] = design_depths
                    section['completed'] = True
                
                st.markdown("---")
                
                if section['completed']:
                    st.markdown(f"### Step 3: Comparison")
                    
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=section['bathy_dist'], y=section['bathy_depth'], mode='lines+markers', name='Bathymetry', line=dict(color='#0077B6', width=2)))
                    fig2.add_trace(go.Scatter(x=section['user_dist'], y=section['user_depth'], mode='lines+markers', name='Design', line=dict(color='#FF6B6B', width=2, dash='dash')))
                    
                    # Mark sill location (parabola end point)
                    if section.get('sill_distance') is not None and section.get('sill_depth') is not None:
                        # Sill marker (green diamond)
                        fig2.add_trace(go.Scatter(
                            x=[section['sill_distance']], 
                            y=[section['sill_depth']], 
                            mode='markers',
                            name='Sill Location',
                            marker=dict(
                                symbol='diamond',
                                size=15,
                                color='#00FF00',
                                line=dict(color='#006600', width=2)
                            ),
                            hovertemplate='Sill Location<br>Distance: %{x:.1f} m<br>Depth: %{y:.2f} m<extra></extra>'
                        ))
                        
                        # Vertical line downward from sill (green)
                        min_depth = min(min(section['bathy_depth']), min(section['user_depth']))
                        fig2.add_shape(
                            type="line",
                            x0=section['sill_distance'],
                            y0=section['sill_depth'],
                            x1=section['sill_distance'],
                            y1=min_depth - 1,  # Extend slightly downward
                            line=dict(color='#00FF00', width=2, dash='dash')
                        )
                    
                    fig2.update_layout(xaxis_title="Distance (m)", yaxis_title="Depth (m)", height=400, legend=dict(x=1.02, y=1, xanchor='left'))
                    st.plotly_chart(fig2)
                    
                    # Show sill information
                    if section.get('sill_distance') is not None and section.get('sill_depth') is not None:
                        st.info(f"**Sill Location:** Distance = {section['sill_distance']:.1f} m, Depth = {abs(section['sill_depth']):.2f} m")
                    
                    st.success(f"Section {current}-{current}' saved!")
                    
                    # ===== STEP 4: EROSION IMPACT ANALYSIS =====
                    st.markdown("---")
                    st.markdown(f"### Step 4: Erosion Impact (30 Years)")
                    
                    # Erosion retreat rates (m/year)
                    RETREAT_RATES = {
                        'A': 0.7,  # m/year
                        'B': 0.8,  # m/year
                        'C': 0.9   # m/year
                    }
                    YEARS = 30
                    
                    retreat_rate = RETREAT_RATES.get(current, 0.7)
                    total_retreat = YEARS * retreat_rate  # meters
                    
                    # Find original shoreline position from design profile data
                    # (where depth first becomes negative, i.e., where parabola starts)
                    x_shore_old = 0.0
                    user_dist_arr = np.array(section['user_dist'])
                    user_depth_arr = np.array(section['user_depth'])
                    for i, depth in enumerate(user_depth_arr):
                        if depth < 0:  # First negative depth (parabola starts)
                            if i > 0:
                                # Use the last zero-depth point as shoreline
                                x_shore_old = user_dist_arr[i-1]
                            else:
                                x_shore_old = user_dist_arr[i]
                            break
                    
                    # New shoreline position after erosion
                    x_shore_new = x_shore_old - total_retreat
                    
                    # Sill remains at same location
                    x_sill = section['sill_distance']
                    y_sill = section['sill_depth']
                    
                    # Calculate new parabola coefficient
                    # Formula: y = a * (x - x_shore_new)^0.67
                    # At sill: y_sill = a * (x_sill - x_shore_new)^0.67
                    # Therefore: a = y_sill / (x_sill - x_shore_new)^0.67
                    delta_x = x_sill - x_shore_new
                    if delta_x > 0:
                        a_new = y_sill / (delta_x ** 0.67)
                        
                        # Generate eroded design profile
                        eroded_dists = []
                        eroded_depths = []
                        
                        for x in np.linspace(x_shore_new, x_sill, 100):
                            eroded_dists.append(x)
                            if x <= x_shore_new:
                                eroded_depths.append(0.0)
                            else:
                                relative_x = x - x_shore_new
                                y = a_new * (relative_x ** 0.67)
                                eroded_depths.append(y)  # Already negative
                        
                        # Create erosion comparison plot
                        fig_erosion = go.Figure()
                        
                        # Original bathymetry
                        fig_erosion.add_trace(go.Scatter(
                            x=section['bathy_dist'], 
                            y=section['bathy_depth'], 
                            mode='lines', 
                            name='Original Bathymetry',
                            line=dict(color='#0077B6', width=2)
                        ))
                        
                        # Original design profile
                        fig_erosion.add_trace(go.Scatter(
                            x=section['user_dist'], 
                            y=section['user_depth'], 
                            mode='lines', 
                            name='Original Design',
                            line=dict(color='#FF6B6B', width=2, dash='dash')
                        ))
                        
                        # Eroded design profile
                        fig_erosion.add_trace(go.Scatter(
                            x=eroded_dists, 
                            y=eroded_depths, 
                            mode='lines', 
                            name=f'After {YEARS}yr Erosion',
                            line=dict(color='#FFA500', width=3, dash='dot')
                        ))
                        
                        # Mark sill location (same for both)
                        fig_erosion.add_trace(go.Scatter(
                            x=[x_sill], 
                            y=[y_sill], 
                            mode='markers',
                            name='Sill Location',
                            marker=dict(
                                symbol='diamond',
                                size=12,
                                color='green',
                                line=dict(color='#000000', width=1.5)
                            ),
                            hovertemplate='Sill<br>Distance: %{x:.1f} m<br>Depth: %{y:.2f} m<extra></extra>'
                        ))
                        
                        # Vertical line at sill
                        min_depth = min(min(section['bathy_depth']), min(eroded_depths))
                        fig_erosion.add_shape(
                            type="line",
                            x0=x_sill,
                            y0=y_sill,
                            x1=x_sill,
                            y1=min_depth - 1,
                            line=dict(color='green', width=2, dash='dash')
                        )
                        
                        # Add vertical line at original shoreline
                        fig_erosion.add_shape(
                            type="line",
                            x0=x_shore_old,
                            y0=0,
                            x1=x_shore_old,
                            y1=min_depth - 1,
                            line=dict(color='red', width=1, dash='dot')
                        )
                        
                        # Add vertical line at eroded shoreline
                        fig_erosion.add_shape(
                            type="line",
                            x0=x_shore_new,
                            y0=0,
                            x1=x_shore_new,
                            y1=min_depth - 1,
                            line=dict(color='orange', width=1, dash='dot')
                        )
                        
                        fig_erosion.update_layout(
                            xaxis_title="Distance (m)", 
                            yaxis_title="Depth (m)", 
                            height=450,
                            legend=dict(x=1.02, y=1, xanchor='left')
                        )
                        
                        st.plotly_chart(fig_erosion)
                        
                        # Display erosion metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Erosion Rate", f"{retreat_rate} m/yr")
                        with col2:
                            st.metric("Time Period", f"{YEARS} years")
                        with col3:
                            st.metric("Total Retreat", f"{total_retreat:.1f} m")
                        with col4:
                            st.metric("New Shoreline", f"{x_shore_new:.1f} m")
                    else:
                        st.warning("⚠️ Erosion would exceed sill location. Reduce retreat rate or time period.")
                    
                    st.markdown("---")
                    
                    _, col_prev, col_next, _ = st.columns([1, 2, 2, 1])
                    with col_prev:
                        if current in ['B', 'C']:
                            prev_sec = 'A' if current == 'B' else 'B'
                            if st.button(f"< Previous ({prev_sec})", key=f"prev_{current}", use_container_width=True):
                                st.session_state.current_section = prev_sec
                                st.rerun()
                    with col_next:
                        if current == 'A':
                            if st.button("Next (B) >", key=f"next_{current}", use_container_width=True):
                                st.session_state.current_section = 'B'
                                st.rerun()
                        elif current == 'B':
                            if st.button("Next (C) >", key=f"next_{current}", use_container_width=True):
                                st.session_state.current_section = 'C'
                                st.rerun()
                        elif current == 'C':
                            if st.button("All Results >", key=f"next_{current}", use_container_width=True):
                                st.session_state.current_section = 'ALL'
                                st.rerun()
    





# ===== VOLUME CALCULATION FUNCTIONS =====

def calculate_fill_area(bathy_dist, bathy_depth, design_dist, design_depth, sill_distance=None):
    """
    Calculate fill area between bathymetry and design profiles.
    If sill_distance is provided, calculates only up to that distance
    
    """
    if not bathy_dist or not design_dist:
        return 0.0
    
    common_dist = np.array(bathy_dist)  # Distance values from bathymetry profile
    design_interp = np.interp(common_dist, design_dist, design_depth)  # Interpolated design depth values
    bathy_array = np.array(bathy_depth)  # Depth values from bathymetry profile
    
    # Extract portion up to sill distance
    if sill_distance is not None:
        mask = common_dist <= sill_distance  # Extract portion up to sill distance
        common_dist = common_dist[mask]  # Distance values up to sill
        design_interp = design_interp[mask]  # Design depth values up to sill
        bathy_array = bathy_array[mask]  # Bathymetry depth values up to sill
    
    # Calculate fill height (vertical distance from bathymetry to design profile)
    # Note: Depths are negative (e.g., -5m means 5 meters deep)
    # If design is shallower (less negative) than bathymetry, we need fill
    # Example: design = -5m, bathy = -8m, fill_height = -5 - (-8) = 3m (positive = fill needed)
    # Example: design = -8m, bathy = -5m, fill_height = -8 - (-5) = -3m (negative = cut, not fill)
    fill_height = design_interp - bathy_array 
    
    # Only count positive values (where design is above bathymetry = fill needed)
    # Negative values mean bathymetry is shallower than design (no fill needed)
    fill_height = np.maximum(fill_height, 0)
    
    # Calculate area using trapezoidal integration
    if len(common_dist) < 2:
        return 0.0
    
    area = np.trapezoid(fill_height, common_dist)
    
    return area


def calculate_section_midpoint(points):
    """
    Calculate the midpoint of a section line.
    
    Args:
        points: List of two points [{'lat': float, 'lon': float}, ...]
    
    Returns:
        Midpoint coordinates {'lat': float, 'lon': float} or None if insufficient points
    """
    if len(points) >= 2:
        return {
            'lat': (points[0]['lat'] + points[1]['lat']) / 2,
            'lon': (points[0]['lon'] + points[1]['lon']) / 2
        }
    return None


def calculate_total_volume():
    """
    Calculate total fill volume between all sections.
    Uses Average End Area Method: V = (A1 + A2) / 2 * L
    
    Returns:
        Tuple of (results_dict, error_message)
        - results_dict: Contains 'areas', 'distances', 'volumes', 'total'
        - error_message: None if successful, error string if failed
    """
    sections = st.session_state.sections
    
    # Check if all sections are completed
    completed = {name: data['completed'] for name, data in sections.items()}
    if not all(completed.values()):
        missing = [name for name, done in completed.items() if not done]
        return None, f"Missing sections: {', '.join(missing)}"
    
    # Calculate fill area for each section (up to SILL)
    areas = {}
    for name, data in sections.items():
        areas[name] = calculate_fill_area(
            data['bathy_dist'], data['bathy_depth'],
            data['user_dist'], data['user_depth'],
            sill_distance=data['sill_distance']
        )
    
    # Section midpoints
    midpoints = {name: calculate_section_midpoint(data['points']) 
                 for name, data in sections.items()}
    
    # Inter-section distances
    dist_AB = calculate_distance(midpoints['A'], midpoints['B'])
    dist_BC = calculate_distance(midpoints['B'], midpoints['C'])
    
    # Calculate volume (Average End Area Method)
    vol_AB = (areas['A'] + areas['B']) / 2 * dist_AB
    vol_BC = (areas['B'] + areas['C']) / 2 * dist_BC
    
    # Add extra volume for areas outside the drawn sections (estimated)
    EXTRA_VOLUME = 8000.0  # m³ - accounts for fill areas not covered by sections
    total_volume = vol_AB + vol_BC + EXTRA_VOLUME
    
    return {
        'areas': areas,  # m²
        'distances': {'A-B': dist_AB, 'B-C': dist_BC},  # m
        'volumes': {'A-B': vol_AB, 'B-C': vol_BC, 'Extra': EXTRA_VOLUME},  # m³
        'total': total_volume  # m³
    }, None


def get_volume_results():
    """
    Get volume calculation results (called from app.py).
    
    Returns:
        Tuple of (results_dict, error_message) from calculate_total_volume()
    """
    return calculate_total_volume()