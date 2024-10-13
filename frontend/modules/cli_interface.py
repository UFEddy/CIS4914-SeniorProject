import subprocess
import os

def run_simulation(scenario, vehicle_model, config_file):
    """
    Executes the NuPlan CLI simulation with the specified parameters.

    Args:
        scenario (str): The scenario to simulate.
        vehicle_model (str): The vehicle model to use.
        config_file (str): Path to the configuration file.

    Returns:
        tuple: (success (bool), output_path or error_message (str))
    """
    output_dir = 'data/simulations'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{scenario}_{simulation_map}_output")

    command = [
        "nuplan_cli_command",
        "--scenario", scenario,
        "--simulation_maps", simulation_map,
        "--config", config_file,
        "--output", output_path
    ]

    try:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        stdout, stderr = process.communicate()

        if process.returncode == 0:
            return True, output_path
        else:
            return False, stderr
    except Exception as e:
        return False, str(e)
