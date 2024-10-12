import yaml

def load_config(config_path):
    """
    Loads configuration parameters from a YAML file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Configuration parameters.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
