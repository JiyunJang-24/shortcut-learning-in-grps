import warnings

from .Config import GenerateConfig

def validate_ckpt(cfg: GenerateConfig) -> None:
    """Validate ckpt path & mode selection"""

    if cfg.model_family == "prismatic":
        ckpt_path = str(cfg.pretrained_checkpoint)
        if cfg.vla_mode == "vanilla":
            for mode in ["plucker", "basis", "basis-rescale"]:
                assert mode not in ckpt_path, f"[Error] Expected `vanilla` mode, but found `{mode}` in checkpoint path!"
        else:
            assert cfg.vla_mode in ckpt_path, (
                f"[Error] Expected `{cfg.vla_mode}` mode, but it was not found in checkpoint path!"
            )

    elif cfg.model_family == "diffusion":
        # Todo
        pass

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
