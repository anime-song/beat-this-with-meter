import json
from itertools import chain
from pathlib import Path

import numpy as np


def index_to_framewise(index, length):
    """Convert an index to a framewise sequence"""
    sequence = np.zeros(length, dtype=bool)
    sequence[index] = True
    return sequence


def filename_to_augmentation(filename):
    """Convert a filename to an augmentation factor."""
    parts = Path(filename).stem.split("_")
    augmentations = {}
    for part in parts[1:]:
        if part.startswith("ps"):
            augmentations["shift"] = int(part[2:])
        elif part.startswith("ts"):
            augmentations["stretch"] = int(part[2:])
    return augmentations


def infer_beat_numbers(beats: np.ndarray, downbeats: np.ndarray) -> np.ndarray:
    """
    From beat and downbeat times, infer a number for each beat such that each downbeat
    is associated with a 1 and beats in between are counted upwards.
    The function requires that all downbeats are also listed as beats.

    Args:
        beats (numpy.ndarray): Array of beat positions in seconds (including downbeats).
        downbeats (numpy.ndarray): Array of downbeat positions in seconds.

    Returns:
        numbers (numpy.ndarray): Array of integer beat numbers.
    """
    # check if all downbeats are beats
    if not np.all(np.isin(downbeats, beats)):
        raise ValueError("Not all downbeats are beats.")

    # handle pickup measure, by considering the beat count of the first full measure
    if len(downbeats) >= 2:
        # find the number of beats between the first two downbeats
        first_downbeat, second_downbeat = np.searchsorted(beats, downbeats[:2])
        beats_in_first_measure = second_downbeat - first_downbeat
        # find the number of beats before the first downbeat
        pickup_beats = first_downbeat
        # derive where to start counting
        if pickup_beats < beats_in_first_measure:
            start_counter = beats_in_first_measure - pickup_beats
        else:
            print(
                "WARNING: There are more beats in the pickup measure than in the first measure. The beat count will start from 2 without trying to estimate the length of the pickup measure."
            )
            start_counter = 1
    else:
        print(
            "WARNING: There are less than two downbeats in the predictions. Something may be wrong. The beat count will start from 2 without trying to estimate the length of the pickup measure."
        )
        start_counter = 1

    # assemble the beat numbers
    numbers = []
    counter = start_counter
    downbeats = chain(downbeats, [-1])
    next_downbeat = next(downbeats)
    for beat in beats:
        if beat == next_downbeat:
            counter = 1
            next_downbeat = next(downbeats)
        else:
            counter += 1
        numbers.append(counter)
    return np.asarray(numbers)


def save_beat_tsv(beats: np.ndarray, downbeats: np.ndarray, outpath: str) -> None:
    """
    Save beat information to a tab-separated file in the standard .beats format:
    each line has a time in seconds, a tab, and a beat number (1 = downbeat).
    The function requires that all downbeats are also listed as beats.

    Args:
        beats (numpy.ndarray): Array of beat positions in seconds (including downbeats).
        downbeats (numpy.ndarray): Array of downbeat positions in seconds.
        outpath (str): Path to the output TSV file.

    Returns:
        None
    """
    # infer beat numbers
    numbers = infer_beat_numbers(beats, downbeats)

    # write the beat file
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(outpath, "w") as f:
            f.writelines(f"{beat}\t{number}\n" for beat, number in zip(beats, numbers))
    except KeyboardInterrupt:
        outpath.unlink()  # avoid half-written files


def save_events_tsv(events: np.ndarray, outpath: str) -> None:
    """
    Save one event time per line.
    """
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(outpath, "w", encoding="utf-8") as handle:
            handle.writelines(f"{float(event)}\n" for event in events)
    except KeyboardInterrupt:
        outpath.unlink(missing_ok=True)


def save_meter_json(meter_prediction: dict, outpath: str) -> None:
    """
    Save decoded meter predictions as JSON.
    """
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(outpath, "w", encoding="utf-8") as handle:
            json.dump(meter_prediction, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    except KeyboardInterrupt:
        outpath.unlink(missing_ok=True)


def replace_state_dict_key(state_dict: dict, old: str, new: str):
    """Replaces `old` in all keys of `state_dict` with `new`."""
    keys = list(state_dict.keys())  # take snapshot of the keys
    for key in keys:
        if old in key:
            state_dict[key.replace(old, new)] = state_dict.pop(key)
    return state_dict


def resolve_annotation_paths(annotation_dir: str | Path, stem: str):
    """
    Resolve supported annotation file paths for a track stem.

    Supports the original `*.beats` / `*.beats.json` names and the
    `*.beat.beats.json` variant currently present in `meter_dataset`.
    """
    annotation_dir = Path(annotation_dir)

    txt_candidates = [annotation_dir / f"{stem}.beats"]
    json_candidates = [
        annotation_dir / f"{stem}.beats.json",
        annotation_dir / f"{stem}.beat.beats.json",
    ]

    txt_path = next((path for path in txt_candidates if path.exists()), None)
    json_path = next((path for path in json_candidates if path.exists()), None)
    return txt_path, json_path
