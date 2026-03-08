#!/usr/bin/env python3
import argparse
import inspect
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from beat_this.click_track import mix_click_track, read_event_file, save_audio
from beat_this.inference import File2Beats
from beat_this.preprocessing import load_audio

CLICK_DEFAULTS = {
    key: inspect.signature(mix_click_track).parameters[key].default
    for key in (
        "beat_frequency",
        "downbeat_frequency",
        "click_duration",
        "beat_level",
        "downbeat_level",
        "audio_level",
    )
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Add beat/downbeat clicks to an audio file and save it."
    )
    parser.add_argument("input", type=str, help="Input audio file.")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output audio file. Defaults to <input>.clicks.wav",
    )
    parser.add_argument(
        "--events",
        type=str,
        default=None,
        help="Optional .beats or .downbeats file. If omitted, the script runs inference.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="final0",
        help="Checkpoint path, shortname or URL used when --events is omitted.",
    )
    parser.add_argument(
        "--downbeats-only",
        action="store_true",
        help="Only overlay downbeat clicks.",
    )
    parser.add_argument(
        "--dbn",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Use DBN postprocessing during inference.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU id to use, or -1 for CPU.",
    )
    parser.add_argument(
        "--float16",
        action="store_true",
        help="Use float16 inference on GPU.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Data directory used by the inference stack.",
    )
    parser.add_argument(
        "--beat-frequency",
        type=float,
        default=CLICK_DEFAULTS["beat_frequency"],
        help="Beat click frequency in Hz.",
    )
    parser.add_argument(
        "--downbeat-frequency",
        type=float,
        default=CLICK_DEFAULTS["downbeat_frequency"],
        help="Downbeat click frequency in Hz.",
    )
    parser.add_argument(
        "--click-duration",
        type=float,
        default=CLICK_DEFAULTS["click_duration"],
        help="Click duration in seconds.",
    )
    parser.add_argument(
        "--beat-level",
        type=float,
        default=CLICK_DEFAULTS["beat_level"],
        help="Beat click amplitude.",
    )
    parser.add_argument(
        "--downbeat-level",
        type=float,
        default=CLICK_DEFAULTS["downbeat_level"],
        help="Downbeat click amplitude.",
    )
    parser.add_argument(
        "--audio-level",
        type=float,
        default=CLICK_DEFAULTS["audio_level"],
        help="Scale the original audio before mixing clicks.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Do not rescale the mixed signal if it clips.",
    )
    return parser.parse_args()


def default_output_path(input_path: Path) -> Path:
    return input_path.with_suffix("").with_name(input_path.stem + ".clicks.wav")


def run_inference(args, audio_path: Path):
    device = "cpu"
    if torch.cuda.is_available() and args.gpu >= 0:
        device = f"cuda:{args.gpu}"

    predictor = File2Beats(
        args.model,
        device=device,
        float16=args.float16,
        dbn=args.dbn,
        data_dir=args.data_dir,
    )
    prediction = predictor.predict_file(audio_path)

    downbeats_only = args.downbeats_only
    if not downbeats_only and not getattr(predictor.model, "train_beats", True):
        print(
            "Checkpoint was trained without beat supervision; overlaying downbeats only.",
            file=sys.stderr,
        )
        downbeats_only = True

    beats = None if downbeats_only else prediction["beats"]
    downbeats = prediction["downbeats"]
    return beats, downbeats


def main():
    args = parse_args()

    audio_path = Path(args.input)
    output_path = (
        Path(args.output)
        if args.output is not None
        else default_output_path(audio_path)
    )

    audio, sample_rate = load_audio(audio_path)
    if args.events is not None:
        beats, downbeats = read_event_file(args.events)
        if args.downbeats_only:
            beats = None
    else:
        beats, downbeats = run_inference(args, audio_path)

    mixed = mix_click_track(
        audio,
        sample_rate,
        beats=beats,
        downbeats=downbeats,
        beat_frequency=args.beat_frequency,
        downbeat_frequency=args.downbeat_frequency,
        click_duration=args.click_duration,
        beat_level=args.beat_level,
        downbeat_level=args.downbeat_level,
        audio_level=args.audio_level,
        normalize=not args.no_normalize,
    )
    save_audio(output_path, mixed, sample_rate)
    print(output_path)


if __name__ == "__main__":
    raise SystemExit(main())
