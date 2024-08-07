"""

"""
import torch
import utils.lorconlo as lo

def load_lorconlo(model_file: str,
                  batch_size: int = 8,
                  batch_norm: bool = True,
                  device: str = "cuda"):
    """

    Args:
        model_file:
        batch_size:
        batch_norm:
        device:

    Returns:

    """
    model = lo.LoRCoNLO(batch_size=batch_size, batchNorm=batch_norm, device=device)
    checkpoint = torch.load(model_file, map_location='cuda')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model


