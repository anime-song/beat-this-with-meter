from pathlib import Path

import numpy as np
import torch
import torchaudio


def read_event_file(path: str | Path) -> tuple[np.ndarray | None, np.ndarray]:
    """
    Read an event file in either `.beats` format (`time<TAB>beat_number`) or a
    single-column downbeat list (`time`).
    """
    beats = []
    downbeats = []
    has_beat_numbers = False

    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            fields = stripped.split()
            time = float(fields[0])
            if len(fields) == 1:
                downbeats.append(time)
                continue

            beat_number = int(float(fields[1]))
            beats.append(time)
            if beat_number == 1:
                downbeats.append(time)
            has_beat_numbers = True

    if has_beat_numbers:
        return np.asarray(beats, dtype=np.float64), np.asarray(
            downbeats, dtype=np.float64
        )
    return None, np.asarray(downbeats, dtype=np.float64)


def synthesize_click(
    sample_rate: int,
    frequency: float,
    duration: float = 0.03,
    level: float = 0.3,
) -> np.ndarray:
    """
    Generate a short decaying sine click.
    """
    num_samples = max(1, int(round(duration * sample_rate)))
    times = np.arange(num_samples, dtype=np.float32) / sample_rate
    envelope = np.exp(-8.0 * times / max(duration, 1.0 / sample_rate))
    attack_samples = max(1, int(round(min(duration / 6.0, 0.002) * sample_rate)))
    envelope[:attack_samples] *= np.linspace(
        0.0, 1.0, attack_samples, endpoint=False, dtype=np.float32
    )
    click = np.sin(2 * np.pi * frequency * times) * envelope
    return (level * click).astype(np.float32)


def overlay_clicks(
    audio: np.ndarray, sample_rate: int, event_times: np.ndarray, click: np.ndarray
) -> np.ndarray:
    """
    Add a click waveform at each event time to mono or stereo audio.
    """
    mixed = np.asarray(audio, dtype=np.float32).copy()
    if event_times is None or len(event_times) == 0:
        return mixed

    num_samples = mixed.shape[0]
    for event_time in np.asarray(event_times, dtype=np.float64):
        start = int(round(event_time * sample_rate))
        if start < 0 or start >= num_samples:
            continue
        stop = min(num_samples, start + len(click))
        click_excerpt = click[: stop - start]
        if mixed.ndim == 1:
            mixed[start:stop] += click_excerpt
        else:
            mixed[start:stop, :] += click_excerpt[:, None]
    return mixed


def mix_click_track(
    audio: np.ndarray,
    sample_rate: int,
    beats: np.ndarray | None = None,
    downbeats: np.ndarray | None = None,
    beat_frequency: float = 1000.0,
    downbeat_frequency: float = 1500.0,
    click_duration: float = 0.03,
    beat_level: float = 0.3,
    downbeat_level: float = 1.0,
    audio_level: float = 0.1,
    normalize: bool = True,
) -> np.ndarray:
    """
    Mix beat and/or downbeat clicks into an audio waveform.
    """
    mixed = np.asarray(audio, dtype=np.float32) * audio_level
    downbeats = (
        np.asarray(downbeats, dtype=np.float64)
        if downbeats is not None
        else np.empty(0, dtype=np.float64)
    )

    if beats is not None:
        beats = np.asarray(beats, dtype=np.float64)
        if len(downbeats) > 0:
            beat_only_mask = ~np.isin(beats, downbeats)
            beats = beats[beat_only_mask]
    else:
        beats = np.empty(0, dtype=np.float64)

    if len(beats) > 0:
        beat_click = synthesize_click(
            sample_rate,
            frequency=beat_frequency,
            duration=click_duration,
            level=beat_level,
        )
        mixed = overlay_clicks(mixed, sample_rate, beats, beat_click)

    if len(downbeats) > 0:
        downbeat_click = synthesize_click(
            sample_rate,
            frequency=downbeat_frequency,
            duration=click_duration,
            level=downbeat_level,
        )
        mixed = overlay_clicks(mixed, sample_rate, downbeats, downbeat_click)

    if normalize:
        peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
        if peak > 1.0:
            mixed = mixed * (0.99 / peak)

    return mixed


def save_audio(path: str | Path, waveform: np.ndarray, sample_rate: int) -> None:
    """
    Save mono or stereo audio using torchaudio.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tensor = torch.as_tensor(np.asarray(waveform, dtype=np.float32))
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 2:
        tensor = tensor.transpose(0, 1)
    else:
        raise ValueError(f"Expected 1D or 2D waveform, got shape {tuple(tensor.shape)}")

    try:
        torchaudio.save(str(path), tensor, sample_rate, bits_per_sample=16)
    except KeyboardInterrupt:
        path.unlink(missing_ok=True)
        raise
