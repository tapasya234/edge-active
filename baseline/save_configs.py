from pathlib import Path
import json


def save_config(args):
    """
    Save args, typically training configuration, as JSON for easy inspection.

    :param args: The args that need to be saved
    :param output_dir: The path where the config file will be saved.
    :type output_dir: Path
    """

    config = vars(args).copy()

    # Convert Path objects to strings for JSON serialization
    for key, value in config.items():
        if isinstance(value, Path):
            config[key] = str(value)

    if args.output_dir.is_dir():
        config_path = args.output_dir / "training_config.json"
    else:
        config_path = args.output_dir
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)

    print(f"Training config saved to {config_path}")
