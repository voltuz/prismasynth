"""FastDVDnet inference helpers."""
import numpy as np
import torch
import torch.nn.functional as F


def denoise_frame(model, frame_buffer, noise_sigma=25.0, device='cuda'):
    """Denoise the center frame of a 5-frame window.

    Args:
        model: loaded FastDVDnet model (eval mode, on device)
        frame_buffer: list of 5 numpy arrays [H, W, 3] uint8 RGB
        noise_sigma: noise level 5-55 (in [0, 255] scale)
        device: 'cuda' or 'cpu'

    Returns:
        denoised: numpy array [H, W, 3] uint8 RGB
    """
    # Stack 5 frames → [1, 15, H, W] float32 [0, 1]
    frames = []
    for f in frame_buffer:
        t = torch.from_numpy(f).permute(2, 0, 1).float().div_(255.0)
        frames.append(t)
    x = torch.cat(frames, dim=0).unsqueeze(0).to(device)

    H, W = x.shape[2], x.shape[3]

    # Noise map [1, 1, H, W]
    sigma = torch.FloatTensor([noise_sigma / 255.0]).to(device)
    noise_map = sigma.expand(1, 1, H, W)

    # Pad to multiple of 4 (required by the U-Net downsampling)
    pad_h = (4 - H % 4) % 4
    pad_w = (4 - W % 4) % 4
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        noise_map = F.pad(noise_map, (0, pad_w, 0, pad_h), mode='constant', value=sigma.item())

    with torch.no_grad(), torch.amp.autocast('cuda'):
        out = model(x, noise_map).clamp_(0.0, 1.0)

    # Remove padding
    if pad_h:
        out = out[:, :, :H, :]
    if pad_w:
        out = out[:, :, :, :W]

    return (out[0].cpu().permute(1, 2, 0).mul_(255).byte().numpy())
