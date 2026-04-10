"""FastDVDnet temporal video denoiser.
Vendored from https://github.com/m-tassano/fastdvdnet (MIT License).
"""
import logging
import os
from collections import OrderedDict

logger = logging.getLogger(__name__)

MODEL_URL = "https://github.com/m-tassano/fastdvdnet/raw/master/model_clipped_noise.pth"
MODEL_FILENAME = "fastdvdnet_clipped_noise.pth"

_cached_model = None


def _get_model_path() -> str:
    cache_dir = os.path.join(
        os.environ.get("LOCALAPPDATA", os.path.expanduser("~")),
        "prismasynth", "cache", "models",
    )
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, MODEL_FILENAME)


def _download_model(path: str):
    """Download the pretrained model from GitHub."""
    import urllib.request
    logger.info("Downloading FastDVDnet model (%s)...", MODEL_URL)
    urllib.request.urlretrieve(MODEL_URL, path)
    logger.info("Model saved to %s", path)


def load_model(device='cuda'):
    """Load FastDVDnet model. Downloads on first use. Cached across calls."""
    global _cached_model
    if _cached_model is not None:
        return _cached_model

    import torch
    from core.fastdvdnet.models import FastDVDnet

    model_path = _get_model_path()
    if not os.path.exists(model_path):
        _download_model(model_path)

    model = FastDVDnet(num_input_frames=5)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)

    # Strip 'module.' prefix from DataParallel-saved weights
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        new_sd[k.removeprefix('module.')] = v
    model.load_state_dict(new_sd)

    model.to(device).eval()
    _cached_model = model
    logger.info("FastDVDnet model loaded on %s", device)
    return model
