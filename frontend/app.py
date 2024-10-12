import streamlit as st
from modules.cli_interface import run_simulation
from modules.visualization import display_results
from modules.utils import load_config

# Set the page configuration
st.set_page_config(page_title="Trajectory Prediction Interface Design", page_icon="🚗", layout="wide")

# Display the application logo
st.image('assets/images/logo.png', width=200)

st.title("Trajectory Prediction Interface Design")

# Load default configuration
config = load_config('config/default_config.yaml')

# Sidebar inputs
st.sidebar.header("Simulation Parameters")
scenario = st.sidebar.selectbox("Select Scenario", options=config['scenarios'])
vehicle_model = st.sidebar.selectbox("Select Vehicle Model", options=config['vehicle_models'])
config_file = st.sidebar.file_uploader("Upload Custom Configuration", type=['yaml', 'json'])

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
        success, output_path = run_simulation(scenario, vehicle_model, simulation_config)

        if success:
            st.success("Simulation completed successfully.")
            # Display results
            display_results(output_path)
        else:
            st.error("Simulation failed.")
            st.text(output_path)
