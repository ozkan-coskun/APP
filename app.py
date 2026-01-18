# app.py
# Main application file for Beach Nourishment Design Tool

import streamlit as st
import profile_module as profile
import locale

# Set locale to use dot (.) as decimal separator
try:
    locale.setlocale(locale.LC_NUMERIC, 'C')
except:
    pass

st.set_page_config(
    page_title="Beach Nourishment Design Tool", 
    layout="wide", 
)

# Keep track of which page we're on (landing or project page)
if 'page' not in st.session_state:
    st.session_state.page = 'landing'  # Start on landing page

# Function to go to the project page
def switch_to_project():
    st.session_state.page = 'project'

# Function to go back to the landing page 
def reset_project():
    st.session_state.page = 'landing'

#  LANDING PAGE 
if st.session_state.page == 'landing':
    
    # Show the hero image at the top
    try:
        st.image("images/bg.jpg", width='stretch')
    except:
        st.warning("Background image not found.")
    
    # Main title and subtitle
    st.title("Beach Nourishment Design Tool")
    st.subheader("Professional solution for coastal engineering calculations, cross-section analysis and cost estimation")
    st.markdown("---")
    
    # Split into two columns: map on left, form on right
    col_map, col_form = st.columns([1, 1])
    
    with col_map:
        # Embed Google Maps showing the project location
        st.components.v1.iframe(
            "https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d3008.5!2d29.626743!3d41.175354!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x0%3A0x0!2zNDHCsDEwJzMxLjMiTiAyOcKwMzcnMzYuMyJF!5e1!3m2!1str!2str!4v1234567890",
            height=410
        )
    
    with col_form:
        st.markdown("### Start New Project")
        
        # Create a form for the user to enter their project name
        with st.form("entry_form"):
            project_name = st.text_input(
                "Enter Project Name:", 
                placeholder="e.g., Şile Ağlayankaya Beach Nourishment", 
                value="Şile Ağlayankaya Beach Nourishment"
            )
            st.markdown("")
            submitted = st.form_submit_button("Start Project", type="primary", use_container_width=True)
            
            # When they click submit
            if submitted:
                if project_name:
                    # Save the project name and go to the project page
                    st.session_state.project_name = project_name
                    switch_to_project()
                    st.rerun()
                else:
                    st.error("Please enter a project name to continue.")
    
    # Footer at the bottom
    st.markdown("---")
    st.caption("© 2025 Coastal Engineering Solutions | Ağlayankaya Beach Nourishment Project")

#  PROJECT DATA ENTRY PAGE 
elif st.session_state.page == 'project':
    
    # Top bar with back button and project title
    col_back, col_title = st.columns([1, 4])
    with col_back:
        if st.button("← Home", use_container_width=True):
            reset_project()
            st.rerun()
    with col_title:
        st.subheader(f"Project: {st.session_state.get('project_name', 'Untitled Project')}")
    
    st.divider()
    st.info("Please enter the required parameters for design calculations.")
    
    # Collect all required data from the user
    
    # Section 1: Wave and sediment properties
    st.markdown("### 1. Wave and Sediment Properties")
    
    # Split into two columns so it looks cleaner
    c1, c2 = st.columns(2)
    with c1:
        Hs = st.number_input("Significant Wave Height (Hs) [m]", value=2.0, step=0.1, help="Design wave height for the project area")
        T = st.number_input("Wave Period (T) [s]", value=7.0, step=0.1, help="Peak wave period")
        L_coast = st.number_input("Total Coastline Length [m]", value=480.0, step=10.0, help="Total length of beach nourishment")
    with c2:
        d50 = st.number_input("Median Grain Size (d₅₀) [mm]", value=0.25, step=0.01, help="Median sediment grain diameter")
        A_param = st.number_input("Sediment Scale Parameter (A)", value=0.09, step=0.01, help="Dean's parameter based on grain size")
        h_toe = st.number_input("Sill Depth (h) [m]", value=2.5, step=0.1, help="Target depth for sill placement")
    
    st.markdown("---")
    
    # Section 2: Cross-section analysis
    st.markdown("### 2. Cross-Section Analysis")
    profile.render_profile_section()
    st.markdown("---")
    
    # Section 3: Optional structural elements - groin and sill
    st.markdown("### 3. Structural Elements (Optional)")
    
    # Groin properties in an expandable section 
    with st.expander("Groin Properties"):
        use_groin = st.toggle("Include Groin in Project", value=False)
        if use_groin:  # Only show these inputs if they want a groin
            gc1, gc2 = st.columns(2)
            with gc1:
                groin_length = st.number_input("Groin Length (m)", value=28.3, key="gl")
                groin_width = st.number_input("Groin Width (m)", value=1.0, key="gw")
            with gc2:
                groin_depth = st.number_input("Groin Depth (m)", value=5.5, key="gd")
                groin_cost = st.number_input("Unit Cost ($/m³)", value=33.0, key="g_cost")
    
    # Sill properties (also in an expandable section)
    with st.expander("Sill (Submerged Breakwater) Properties", expanded=True):
        use_sill = st.toggle("Include Sill in Project", value=True)
        if use_sill:  # Only show these inputs if they want a sill
            sl1, sl2 = st.columns(2)
            with sl1:
                sill_length = st.number_input("Sill Length (m)", value=258.0, key="sl", help="Length along A-A section")
                sill_width = st.number_input("Sill Width (m)", value=1.5, key="sw")
            with sl2:
                sill_depth = st.number_input("Sill Height (m)", value=0.5, key="sd")
                sill_cost = st.number_input("Unit Cost ($/m³)", value=30.0, key="s_cost")
    
    st.markdown("---")
    
    # Section 4: Cost estimation
    st.markdown("### 4. Cost Estimation")
    cost1, cost2 = st.columns(2)
    with cost1:
        sand_cost = st.number_input("Sand Unit Cost ($/m³)", value=20.0, step=1.0, help="Cost per cubic meter of fill material")
    with cost2:
        transport_cost = st.number_input("Transport & Placement Cost ($/m³)", value=25.0, step=1.0, help="Additional costs for material placement")
    
    st.markdown("---")
    
    # The big calculate button
    if st.button("START CALCULATIONS", type="primary", use_container_width=True):
        # Get volume results from profile module
        vol_results, error = profile.get_volume_results()
        
        if error:
            st.error(f"Cannot calculate costs: {error}")
            st.warning("Please complete all cross-sections (A, B, C) first.")
        else:
            st.success("✓ Calculations completed successfully!")
            
            # Calculate costs
            total_fill_volume = vol_results['total']  # m³ (includes extra 8000 m³)
            
            # Fill material costs
            fill_material_cost = total_fill_volume * sand_cost
            fill_transport_cost = total_fill_volume * transport_cost
            total_fill_cost = fill_material_cost + fill_transport_cost
            
            # Groin costs (if applicable)
            groin_volume = 0
            groin_total_cost = 0
            if use_groin:
                groin_volume = groin_length * groin_width * groin_depth
                groin_total_cost = groin_volume * groin_cost
            
            # Sill costs (if applicable)
            sill_volume = 0
            sill_total_cost = 0
            if use_sill:
                sill_volume = sill_length * sill_width * sill_depth
                sill_total_cost = sill_volume * sill_cost
            
            # Grand total
            project_total_cost = total_fill_cost + groin_total_cost + sill_total_cost
            
            # Display results
            st.markdown("---")
            st.markdown("### 💰 Cost Analysis Results")
            
            # Main summary metrics
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("Fill Volume", f"{total_fill_volume:,.0f} m³")
            with col_m2:
                st.metric("Fill Cost", f"${total_fill_cost:,.0f}")
            with col_m3:
                if use_groin:
                    st.metric("Groin Cost", f"${groin_total_cost:,.0f}")
                else:
                    st.metric("Groin Cost", "N/A")
            with col_m4:
                if use_sill:
                    st.metric("Sill Cost", f"${sill_total_cost:,.0f}")
                else:
                    st.metric("Sill Cost", "N/A")
            
            st.markdown("---")
            
            # Grand total in a prominent box
            st.markdown(f"""
            <div style="background-color: #1e40af; padding: 20px; border-radius: 10px; text-align: center;">
                <h2 style="color: white; margin: 0;">PROJECT TOTAL COST</h2>
                <h1 style="color: white; margin: 10px 0;">${project_total_cost:,.0f}</h1>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Detailed breakdown
            with st.expander("📊 Detailed Cost Breakdown"):
                st.markdown("#### Fill Material")
                st.write(f"- Volume: **{total_fill_volume:,.0f} m³**")
                st.write(f"- Material Cost: **${fill_material_cost:,.0f}** ({sand_cost:,.2f} $/m³)")
                st.write(f"- Transport & Placement: **${fill_transport_cost:,.0f}** ({transport_cost:,.2f} $/m³)")
                st.write(f"- **Subtotal: ${total_fill_cost:,.0f}**")
                
                if use_groin:
                    st.markdown("#### Groin Structure")
                    st.write(f"- Dimensions: {groin_length:.1f}m × {groin_width:.1f}m × {groin_depth:.1f}m")
                    st.write(f"- Volume: **{groin_volume:,.2f} m³**")
                    st.write(f"- Unit Cost: **{groin_cost:.2f} $/m³**")
                    st.write(f"- **Subtotal: ${groin_total_cost:,.0f}**")
                
                if use_sill:
                    st.markdown("#### Sill (Submerged Breakwater)")
                    st.write(f"- Dimensions: {sill_length:.1f}m × {sill_width:.1f}m × {sill_depth:.1f}m")
                    st.write(f"- Volume: **{sill_volume:,.2f} m³**")
                    st.write(f"- Unit Cost: **{sill_cost:.2f} $/m³**")
                    st.write(f"- **Subtotal: ${sill_total_cost:,.0f}**")
