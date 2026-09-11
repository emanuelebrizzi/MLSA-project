import yaml

def load_config(config_path="configs/base.yaml"):
    """
    Loads the YAML configuration file.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

if __name__ == "__main__":
    # Test loading the config
    cfg = load_config()
    print(f"Running experiment: {cfg['experiment']['name']}")
    print(f"Batch size: {cfg['training']['batch_size']}")