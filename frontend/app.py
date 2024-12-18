import streamlit as st
from modules.cli_interface import run_simulation
from modules.visualization import display_results
from modules.utils import load_config

# Set the page configuration
st.set_page_config(page_title="Trajectory Prediction Interface Design", page_icon="🚗", layout="wide")

# Display the application logo
st.image('assets/images/logo.png', use_column_width=True)

st.title("Trajectory Prediction Interface Design")


# Load default configuration
config = load_config('config/default_config.yaml')

# Sidebar inputs
st.sidebar.header("Simulation Parameters")
scenario = st.sidebar.selectbox("Select Scenario", options=config['scenarios'])
simulation_map = st.sidebar.selectbox("Select Simulation Map", options=config['simulation_map'])
config_file = st.sidebar.file_uploader("Upload Custom Configuration", type=['yaml', 'json'])

# horizontal tab ocnfiguration
tab1, tab2, tab3, tab4 = st.tabs(
    ["Home", "Metrics", "Settings", "Logs"]
)

with tab1:
    st.title("Home")
    st.write("This is the Home tab content.")

with tab2:
    st.title("Metrics")
    st.write("This is the Metrics tab content.")

with tab3:
    st.title("Settings")
    st.write("This is the Settings tab content.")

with tab4:
    st.title("Logs")
    st.write("This is the Logs tab content.")

# Run simulation button
if st.sidebar.button("Run Simulation"):
    with st.spinner("Running simulation..."):
        # Handle the uploaded configuration file
        if config_file is not None:
            custom_config_path = f"config/{config_file.name}"
            with open(custom_config_path, "wb") as f:
                f.write(config_file.getbuffer())
            simulation_config = custom_config_path
        else:
            simulation_config = 'config/default_config.yaml'

        # Run the simulation
        success, output_path = run_simulation(scenario, simulation_map, simulation_config)

        if success:
            st.success("Simulation completed successfully.")
            # Display results
            display_results(output_path)
        else:
            st.error("Simulation failed.")
            st.text(output_path)
