"""Generate src/ui/sounds/complete.wav — the notification ding played
when a long-running operation finishes.

Two-tone bell: 880 Hz then 660 Hz, each with a fast attack and exponential
decay envelope so it sounds like a chime rather than a beep. ~250 ms total,
44.1 kHz mono 16-bit PCM, ~25 KB.

Run once during implementation; commit both this script and the resulting
WAV. Re-run only if the sound changes.

    venv\\Scripts\\python scripts\\gen_completion_sound.py
"""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path


SAMPLE_RATE = 44100
TONE1_HZ = 880.0    # A5 — bright, attention-getting
TONE2_HZ = 660.0    # E5 — resolution note (perfect fifth below)
TONE1_DUR = 0.11    # seconds
TONE2_DUR = 0.16    # seconds — slightly longer for natural decay
ATTACK = 0.005      # seconds — short fade-in to avoid click
DECAY_TAU = 0.05    # exponential decay time constant
AMPLITUDE = 0.65    # peak (0..1) — headroom for any DAC overshoot


def _tone(freq: float, duration: float, sample_rate: int) -> list[float]:
    """Sine tone with attack + exponential decay envelope."""
    n = int(duration * sample_rate)
    samples: list[float] = []
    attack_n = max(1, int(ATTACK * sample_rate))
    two_pi_f = 2.0 * math.pi * freq
    for i in range(n):
        t = i / sample_rate
        # Linear attack ramp, then exponential decay.
        if i < attack_n:
            env = i / attack_n
        else:
            env = math.exp(-(t - ATTACK) / DECAY_TAU)
        samples.append(AMPLITUDE * env * math.sin(two_pi_f * t))
    return samples


def _generate() -> bytes:
    samples = _tone(TONE1_HZ, TONE1_DUR, SAMPLE_RATE)
    samples += _tone(TONE2_HZ, TONE2_DUR, SAMPLE_RATE)
    # Soft-clip safety + int16 conversion.
    out = bytearray()
    for s in samples:
        v = max(-1.0, min(1.0, s))
        out += struct.pack("<h", int(v * 32767))
    return bytes(out)


def main() -> None:
    out_path = Path(__file__).resolve().parents[1] / "src" / "ui" / "sounds" / "complete.wav"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pcm = _generate()
    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # 16-bit
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)
    print(f"Wrote {out_path}  ({out_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
