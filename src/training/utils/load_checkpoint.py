import os
import torch 

def load_checkpoint(model, cfg_data, train_cfg, region, device):
    """
    Load best model checkpoint based on selected polarizations.
    """
    polarizations = cfg_data["spatial"]["polarizations"]
    pol_tag = "_".join(sorted(polarizations))

    ckpt_cfg = train_cfg["checkpoint"]
    ckpt_path = (
        f"{ckpt_cfg['base_dir']}_{pol_tag}/"
        f"{region}/"
        f"{ckpt_cfg['best_filename']}"
    )

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    return model

