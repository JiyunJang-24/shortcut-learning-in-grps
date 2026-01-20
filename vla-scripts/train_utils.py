from enum import Enum

class TrainingMode(Enum):
    VANILLA = "vanilla"
    PLUCKER = "plucker"
    BASIS = "basis"
    BASIS_RESCALED = "basis_rescale"
    BASIS_RESCALED_CONCAT = "basis_rescale_concat"
    PLUCKER_CONCAT = "plucker_concat"
def get_run_id(cfg) -> str:
    """
    Generates a unique run ID based on the training configuration.
    """
    # Define components in a dictionary as requested
    run_params = {
        "vla_id": cfg.vla.vla_id,
        "n_nodes": cfg.vla.expected_world_size // 8,
        "batch_size": cfg.per_device_batch_size,
        "mode": cfg.mode,
        "seed": cfg.seed,
    }

    # Modify vla_id based on training mode if it contains "dinosiglip"
    if "dinosiglip" in run_params["vla_id"]:
        # map cfg.mode string to Enum value
        mode_enum = TrainingMode(cfg.mode)
        run_params["vla_id"] = run_params["vla_id"].replace(f"dinosiglip-{run_params['mode']}", "dinosiglip")
        run_params["vla_id"] = run_params["vla_id"].replace("dinosiglip", f"dinosiglip-{mode_enum.value}")

    # Construct the run_id string
    run_id = "{vla_id}+n{n_nodes}+b{batch_size}+x{seed}".format(**run_params)

    return run_id
