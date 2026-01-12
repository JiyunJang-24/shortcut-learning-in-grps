from .Config import GenerateConfig

def validate_ckpt(cfg: GenerateConfig) -> None:
    """Validate ckpt path & mode selection"""

    if "plucker" in str(cfg.pretrained_checkpoint):
        assert cfg.mode == "plucker", f"[Error] Expected to be run in `plucker` mode, but invalid mode argument: {cfg.mode}!"

    if "basis" in str(cfg.pretrained_checkpoint):
        assert cfg.mode == "basis", f"[Error] Expected to be run in `basis` mode, but invalid mode argument: {cfg.mode}!"

    if "plucker" not in str(cfg.pretrained_checkpoint) and "basis" not in str(cfg.pretrained_checkpoint):
        assert cfg.mode == "vanilla", f"[Error] Expected to be run in `vanilla` mode, but invalid mode argument: {cfg.mode}!"

    print(f"CKPT PATH: {cfg.pretrained_checkpoint}")


def validate_modes(cfg: GenerateConfig) -> None:
    """Validate mode & vision backbone branching flags"""

    assert cfg.mode in ["vanilla", "plucker", "basis"], f"[Error] Invalid mode argument: {cfg.mode}!"

    if cfg.mode == "vanilla":
        assert not (cfg.use_plucker or cfg.use_dynamics_basis), f"[Error] If the eval mode is `vanilla`, then use_plucker and use_dynamic_basis should be false!"

    if cfg.mode == "plucker":
        assert cfg.use_plucker, f"[Error] If the eval mode is `plucker`, then use_plucker should be True!"
        assert not cfg.use_dynamics_basis, f"[Error] If the eval mode is `plucker`, then use_dynamic_basis should be False!"


    print(f"EVAL MODE: {cfg.mode}")

def validate_cli_args(cfg: GenerateConfig) -> None:
    validate_modes(cfg)
    validate_ckpt(cfg)
