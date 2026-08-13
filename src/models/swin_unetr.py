# src/models/swin_unetr.py

import torch

try:
    from monai.networks.nets import SwinUNETR
except ImportError as e:
    raise ImportError(
        "MONAI is not installed. "
        "Please install it with: pip install monai"
    ) from e


def build_swin_unetr(
    img_size,
    in_channels,
    out_channels,
    feature_size=48,
    spatial_dims=2,
    device=None,
):
    try:
        model = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            use_checkpoint=False,
            spatial_dims=spatial_dims,
        )
    except TypeError:
        model = SwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            use_checkpoint=False,
            spatial_dims=spatial_dims,
        )

    if device is not None:
        model = model.to(device)

    return model

def main():
    # -----------------------------
    # Config
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    
    
    batch_size = 1
    t_len = 28          # temporal length (input channels)
    n_classes = 10          
    img_size = (256, 256)  # H, W (must be divisible by 32 for SwinUNETR)

    # -----------------------------
    # Build model
    # -----------------------------
    model = build_swin_unetr(
        img_size=img_size,
        in_channels=t_len,
        out_channels=n_classes,
        feature_size=48,
        spatial_dims=2,
        device=device,
    )

    model.eval()

    # -----------------------------
    # Dummy input
    # Shape: (B, C, H, W)
    # -----------------------------
    x = torch.randn(
        batch_size,
        t_len,
        img_size[0],
        img_size[1],
        device=device,
    )

    print("Input shape:", x.shape)

    # -----------------------------
    # Forward pass
    # -----------------------------
    with torch.no_grad():
        y = model(x)

    print("Output shape:", y.shape)

    # -----------------------------
    # Sanity checks
    # -----------------------------
    assert y.shape == (
        batch_size,
        n_classes,
        img_size[0],
        img_size[1],
    ), "Output shape mismatch!"

    print("Forward pass OK")


if __name__ == "__main__":
    main()