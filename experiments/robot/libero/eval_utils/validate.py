import warnings

from .Config import GenerateConfig

def validate_ckpt(cfg: GenerateConfig) -> None:
    """Validate ckpt path & mode selection"""

    if "plucker" in str(cfg.pretrained_checkpoint):
        assert cfg.vla_mode == "plucker", f"[Error] Expected to be run in `plucker` mode, but invalid mode argument: {cfg.vla_mode}!"

    if "basis" in str(cfg.pretrained_checkpoint):
        assert "basis" in cfg.vla_mode, f"[Error] Expected to be run in `basis` mode, but invalid mode argument: {cfg.vla_mode}!"

    if "plucker" not in str(cfg.pretrained_checkpoint) and "basis" not in str(cfg.pretrained_checkpoint):
        assert cfg.vla_mode == "vanilla", f"[Error] Expected to be run in `vanilla` mode, but invalid mode argument: {cfg.vla_mode}!"

    print(f"CKPT PATH: {cfg.pretrained_checkpoint}")


def validate_modes(cfg: GenerateConfig) -> None:
    """Validate mode & vision backbone branching flags"""
    assert cfg.model_family in ["diffusion", "prismatic"], f"[Error] Invalid model family: {cfg.model_family}"


    if cfg.model_family == "prismatic":
        assert cfg.vla_mode in ["vanilla", "plucker", "basis", "basis-rescale"], f"[Error] Invalid mode argument: {cfg.vla_mode}!"

        print(f"Running {cfg.model_family} model with {cfg.vla_mode} mode.")

    elif cfg.model_family == "diffusion":
        if cfg.vla_mode is not None:
            warnings.warn(f"[WARNING] You are using a mode argument which is defined for VLA models")

        assert cfg.use_dynamics_basis ^ cfg.use_plucker, f"[ERROR] Both use_dynamics_basis and use_plucker are {cfg.use_dynamics_basis}."

        print(f"Running {cfg.model_family} model with {'plucker' if cfg.use_plucker else 'basis'}.")


def validate_cli_args(cfg: GenerateConfig) -> None:
    validate_modes(cfg)
    validate_ckpt(cfg)
