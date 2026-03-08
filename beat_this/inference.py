import inspect
import json
import pickle
from pathlib import Path

import numpy as np
import soxr
import torch
import torch.nn.functional as F

from beat_this.model.beat_tracker import BeatThis
from beat_this.model.postprocessor import Postprocessor
from beat_this.preprocessing import LogMelSpect, load_audio
from beat_this.utils import replace_state_dict_key, save_beat_tsv

CHECKPOINT_URL = "https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp"


def load_checkpoint(checkpoint_path: str, device: str | torch.device = "cpu") -> dict:
    """
    Load a BeatThis checkpoint as a dictionary.

    Args:
        checkpoint_path (str, optional): The path to the checkpoint. Can be a local path, a URL, or a shortname.
        device (torch.device or str): The device to load the model on.

    Returns:
        dict: The loaded checkpoint dictionary.
    """
    try:
        # try interpreting as local file name
        if torch.__version__ >= "2":
            try:
                return torch.load(
                    checkpoint_path, map_location=device, weights_only=True
                )
            except pickle.UnpicklingError as exc:
                # Fine-tuning checkpoints can contain non-tensor hparams such as
                # pathlib.Path objects, which are rejected by weights_only=True.
                if "Weights only load failed" not in str(exc):
                    raise
                return torch.load(
                    checkpoint_path, map_location=device, weights_only=False
                )
        return torch.load(checkpoint_path, map_location=device)
    except FileNotFoundError:
        try:
            if not (
                str(checkpoint_path).startswith("https://")
                or str(checkpoint_path).startswith("http://")
            ):
                # interpret it as a name of one of our checkpoints
                checkpoint_url = (
                    f"{CHECKPOINT_URL}/{checkpoint_path}.ckpt"
                )
                file_name = f"beat_this-{checkpoint_path}.ckpt"
            else:
                # try interpreting as a URL
                checkpoint_url = checkpoint_path
                file_name = None
            return torch.hub.load_state_dict_from_url(
                checkpoint_url,
                file_name=file_name,
                map_location=device,
            )
        except Exception:
            raise ValueError(
                "Could not load the checkpoint given the provided name",
                checkpoint_path,
            )


def load_model(
    checkpoint_path: str | None = "final0", device: str | torch.device = "cpu"
) -> BeatThis:
    """
    Load a BeatThis model from a checkpoint.

    Args:
        checkpoint_path (str, optional): The path to the checkpoint. Can be a local path, a URL, or a shortname.
        device (torch.device or str): The device to load the model on.

    Returns:
        BeatThis: The loaded model.
    """
    checkpoint_hparams = {}
    if checkpoint_path is not None:
        checkpoint = load_checkpoint(checkpoint_path, device)
        # Retrieve the model hyperparameters as it could be the small model
        checkpoint_hparams = checkpoint.get("hyper_parameters", {})
        hparams = checkpoint_hparams
        # Filter only those hyperparameters that apply to the model itself
        hparams = {
            k: v
            for k, v in hparams.items()
            if k in set(inspect.signature(BeatThis).parameters)
        }
        # Create the uninitialized model
        model = BeatThis(**hparams)
        # The PLBeatThis (LightningModule) checkpoint stores model weights
        # under the "model." prefix together with optimizer/loss state.
        raw_state_dict = checkpoint.get("state_dict", checkpoint)
        if any(key.startswith("model.") for key in raw_state_dict):
            state_dict = {
                key.removeprefix("model."): value
                for key, value in raw_state_dict.items()
                if key.startswith("model.")
            }
        else:
            state_dict = dict(raw_state_dict)
        state_dict = replace_state_dict_key(state_dict, "model.", "")
        model.load_state_dict(state_dict)
    else:
        model = BeatThis()
    model.checkpoint_hparams = checkpoint_hparams
    model.train_beats = checkpoint_hparams.get("train_beats", True)
    return model.to(device).eval()


def infer_meter_vocabulary(data_dir: str | Path = "data") -> dict[str, list]:
    """
    Reconstruct the meter vocabularies from locally available annotations.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        project_data_dir = Path(__file__).resolve().parents[1] / data_dir
        if project_data_dir.exists():
            data_dir = project_data_dir

    annotation_root = data_dir / "annotations"
    if not annotation_root.exists():
        return {}

    num_values = set()
    den_values = set()
    meter_values = set()
    for json_path in sorted(annotation_root.glob("*/annotations/beats/*.json")):
        try:
            with open(json_path, "r", encoding="utf-8") as handle:
                annotation = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        for measure in annotation.get("measures", []):
            num = int(measure["time_sig_num"])
            den = int(measure["time_sig_den"])
            num_values.add(num)
            den_values.add(den)
            meter_values.add(f"{num}/{den}")

    return {
        "num": sorted(num_values),
        "den": sorted(den_values),
        "meter": sorted(meter_values),
    }


def resolve_meter_vocabulary(
    checkpoint_hparams: dict | None, data_dir: str | Path = "data"
) -> dict[str, list]:
    """
    Resolve vocabularies for numerator, denominator and combined meter labels.
    """
    checkpoint_hparams = checkpoint_hparams or {}
    if all(key in checkpoint_hparams for key in ("num_vocab", "den_vocab", "meter_vocab")):
        return {
            "num": list(checkpoint_hparams["num_vocab"]),
            "den": list(checkpoint_hparams["den_vocab"]),
            "meter": list(checkpoint_hparams["meter_vocab"]),
        }

    inferred_vocab = infer_meter_vocabulary(data_dir)
    vocab = {}
    class_counts = {
        "num": checkpoint_hparams.get("num_classes"),
        "den": checkpoint_hparams.get("den_classes"),
        "meter": checkpoint_hparams.get("meter_classes"),
    }
    for key, class_count in class_counts.items():
        if class_count is None:
            continue
        labels = inferred_vocab.get(key, [])
        if len(labels) == class_count:
            vocab[key] = labels
        else:
            vocab[key] = [f"{key}_{idx}" for idx in range(class_count)]
    return vocab


def decode_class_prediction(
    logits: torch.Tensor, labels: list | None, fps: int
) -> dict | None:
    """
    Convert framewise class logits of shape (classes, time) into a majority label
    and contiguous time segments.
    """
    if logits.ndim != 2 or logits.shape[-1] == 0:
        return None

    class_ids = torch.argmax(logits.float(), dim=0).cpu().numpy()
    counts = np.bincount(class_ids)
    class_id = int(counts.argmax())
    label_values = labels if labels is not None else list(range(len(counts)))

    segments = []
    start = 0
    current = int(class_ids[0])
    for idx in range(1, len(class_ids) + 1):
        if idx == len(class_ids) or int(class_ids[idx]) != current:
            segments.append(
                {
                    "start": start / fps,
                    "end": idx / fps,
                    "class_id": current,
                    "value": label_values[current],
                }
            )
            if idx < len(class_ids):
                start = idx
                current = int(class_ids[idx])

    return {
        "class_id": class_id,
        "value": label_values[class_id],
        "confidence": float(counts[class_id] / len(class_ids)),
        "segments": segments,
    }


def decode_meter_prediction(
    model_prediction: dict,
    fps: int,
    checkpoint_hparams: dict | None = None,
    data_dir: str | Path = "data",
) -> dict | None:
    """
    Decode optional meter-related outputs from a model prediction dictionary.
    """
    if not any(key in model_prediction for key in ("num", "den", "meter")):
        return None

    vocabulary = resolve_meter_vocabulary(checkpoint_hparams, data_dir)
    result = {}
    if "num" in model_prediction:
        result["num"] = decode_class_prediction(
            model_prediction["num"], vocabulary.get("num"), fps
        )
    if "den" in model_prediction:
        result["den"] = decode_class_prediction(
            model_prediction["den"], vocabulary.get("den"), fps
        )
    if "meter" in model_prediction:
        result["meter"] = decode_class_prediction(
            model_prediction["meter"], vocabulary.get("meter"), fps
        )
    if result.get("num") is not None and result.get("den") is not None:
        result["combined_meter"] = {
            "value": f"{result['num']['value']}/{result['den']['value']}",
            "confidence": min(result["num"]["confidence"], result["den"]["confidence"]),
        }
        if "meter" not in result:
            result["meter"] = {
                "value": result["combined_meter"]["value"],
                "confidence": result["combined_meter"]["confidence"],
                "segments": [],
            }
    return result


def zeropad(spect: torch.Tensor, left: int = 0, right: int = 0):
    """
    Pads a tensor spectrogram matrix of shape (time x bins) with `left` frames in the beginning and `right` frames in the end.
    """
    if left == 0 and right == 0:
        return spect
    else:
        return F.pad(spect, (0, 0, left, right), "constant", 0)


def split_piece(
    spect: torch.Tensor,
    chunk_size: int,
    border_size: int = 6,
    avoid_short_end: bool = True,
):
    """
    Split a tensor spectrogram matrix of shape (time x bins) into time chunks of `chunk_size` and return the chunks and starting positions.
    The `border_size` is the number of frames assumed to be discarded in the predictions on either side (since the model was not trained on the input edges due to the max-pool in the loss).
    To cater for this, the first and last chunk are padded by `border_size` on the beginning and end, respectively, and consecutive chunks overlap by `border_size`.
    If `avoid_short_end` is true, the last chunk start is shifted left to ends at the end of the piece, therefore the last chunk can potentially overlap with previous chunks more than border_size, otherwise it will be a shorter segment.
    If the piece is shorter than `chunk_size`, avoid_short_end is ignored and the piece is returned as a single shorter chunk.

    Args:
        spect (torch.Tensor): The input spectrogram tensor of shape (time x bins).
        chunk_size (int): The size of the chunks to produce.
        border_size (int, optional): The size of the border to overlap between chunks. Defaults to 6.
        avoid_short_end (bool, optional): If True, the last chunk is shifted left to end at the end of the piece. Defaults to True.
    """
    # generate the start and end indices
    starts = np.arange(
        -border_size, len(spect) - border_size, chunk_size - 2 * border_size
    )
    if avoid_short_end and len(spect) > chunk_size - 2 * border_size:
        # if we avoid short ends, move the last index to the end of the piece - (chunk_size - border_size)
        starts[-1] = len(spect) - (chunk_size - border_size)
    # generate the chunks
    chunks = [
        zeropad(
            spect[max(start, 0) : min(start + chunk_size, len(spect))],
            left=max(0, -start),
            right=max(0, min(border_size, start + chunk_size - len(spect))),
        )
        for start in starts
    ]
    return chunks, starts


def aggregate_prediction(
    pred_chunks: list,
    starts: list,
    full_size: int,
    chunk_size: int,
    border_size: int,
    overlap_mode: str,
    device: str | torch.device,
) -> dict[str, torch.Tensor]:
    """
    Aggregates the predictions for the whole piece based on the given prediction chunks.

    Args:
        pred_chunks (list): List of prediction chunks, where each chunk is a dictionary containing 'beat' and 'downbeat' predictions.
        starts (list): List of start positions for each prediction chunk.
        full_size (int): Size of the full piece.
        chunk_size (int): Size of each prediction chunk.
        border_size (int): Size of the border to be discarded from each prediction chunk.
        overlap_mode (str): Mode for handling overlapping predictions. Can be 'keep_first' or 'keep_last'.
        device (torch.device): Device to be used for the predictions.

    Returns:
        dict: Dictionary containing the aggregated framewise predictions.
    """
    prediction_keys = list(pred_chunks[0].keys())
    if border_size > 0:
        # cut the predictions to discard the border
        pred_chunks = [
            {
                key: pchunk[key][..., border_size:-border_size]
                for key in prediction_keys
            }
            for pchunk in pred_chunks
        ]
    # aggregate the predictions for the whole piece
    aggregated_predictions = {
        key: torch.full(
            pred_chunks[0][key].shape[:-1] + (full_size,),
            -1000.0,
            device=device,
            dtype=pred_chunks[0][key].dtype,
        )
        for key in prediction_keys
    }
    if overlap_mode == "keep_first":
        # process in reverse order, so predictions of earlier excerpts overwrite later ones
        pred_chunks = reversed(list(pred_chunks))
        starts = reversed(list(starts))
    for start, pchunk in zip(starts, pred_chunks):
        for key in prediction_keys:
            aggregated_predictions[key][
                ..., start + border_size : start + chunk_size - border_size
            ] = pchunk[key]
    return aggregated_predictions


def split_predict_aggregate(
    spect: torch.Tensor,
    chunk_size: int,
    border_size: int,
    overlap_mode: str,
    model: torch.nn.Module,
) -> dict:
    """
    Function for pieces that are longer than the training length of the model.
    Split the input piece into chunks, run the model on them, and aggregate the predictions.
    The spect is supposed to be a torch tensor of shape (time x bins), i.e., unbatched, and the output is also unbatched.

    Args:
        spect (torch.Tensor): the input piece
        chunk_size (int): the length of the chunks
        border_size (int): the size of the border that is discarded from the predictions
        overlap_mode (str): how to handle overlaps between chunks
        model (torch.nn.Module): the model to run

    Returns:
        dict: the model framewise predictions for the hole piece as a dictionary containing 'beat' and 'downbeat' predictions.
    """
    # split the piece into chunks
    chunks, starts = split_piece(
        spect, chunk_size, border_size=border_size, avoid_short_end=True
    )
    # run the model
    pred_chunks = [model(chunk.unsqueeze(0)) for chunk in chunks]
    # remove the extra dimension in beat and downbeat prediction due to batch size 1
    pred_chunks = [{key: value[0] for key, value in p.items()} for p in pred_chunks]
    piece_prediction = aggregate_prediction(
        pred_chunks,
        starts,
        spect.shape[0],
        chunk_size,
        border_size,
        overlap_mode,
        spect.device,
    )
    return piece_prediction


class Spect2Frames:
    """
    Class for extracting framewise beat and downbeat predictions (logits) from a spectrogram.
    """

    def __init__(self, checkpoint_path="final0", device="cpu", float16=False):
        super().__init__()
        self.device = torch.device(device)
        self.float16 = float16
        self.model = load_model(checkpoint_path, self.device)

    def spect2predictions(self, spect):
        with torch.inference_mode():
            with torch.autocast(enabled=self.float16, device_type=self.device.type):
                model_prediction = split_predict_aggregate(
                    spect=spect,
                    chunk_size=1500,
                    overlap_mode="keep_first",
                    border_size=6,
                    model=self.model,
                )
        return {key: value.float() for key, value in model_prediction.items()}

    def spect2frames(self, spect):
        model_prediction = self.spect2predictions(spect)
        return model_prediction["beat"], model_prediction["downbeat"]

    def __call__(self, spect):
        return self.spect2frames(spect)


class Audio2Frames(Spect2Frames):
    """
    Class for extracting framewise beat and downbeat predictions (logits) from an audio tensor.
    """

    def __init__(self, checkpoint_path="final0", device="cpu", float16=False):
        super().__init__(checkpoint_path, device, float16)
        self.spect = LogMelSpect(device=self.device)
        self.fps = 50

    def signal2spect(self, signal, sr):
        if signal.ndim == 2:
            signal = signal.mean(1)
        elif signal.ndim != 1:
            raise ValueError(f"Expected 1D or 2D signal, got shape {signal.shape}")
        if sr != 22050:
            signal = soxr.resample(signal, in_rate=sr, out_rate=22050)
        signal = torch.tensor(signal, dtype=torch.float32, device=self.device)
        return self.spect(signal)

    def __call__(self, signal, sr):
        spect = self.signal2spect(signal, sr)
        return self.spect2frames(spect)


class Audio2Beats(Audio2Frames):
    """
    Class for extracting beat and downbeat positions (in seconds) from an audio tensor.

    Args:
        checkpoint_path (str): Path to the model checkpoint file. It can be a local path, a URL, or a key from the CHECKPOINT_URL dictionary. Default is "final0", which will load the model trained on all data except GTZAN with seed 0.
        device (str): Device to use for inference. Default is "cpu".
        float16 (bool): Whether to use half precision floating point arithmetic. Default is False.
        dbn (bool): Whether to use the madmom DBN for post-processing. Default is False.
    """

    def __init__(
        self,
        checkpoint_path="final0",
        device="cpu",
        float16=False,
        dbn=False,
        data_dir: str | Path = "data",
    ):
        super().__init__(checkpoint_path, device, float16)
        self.data_dir = data_dir
        checkpoint_hparams = getattr(self.model, "checkpoint_hparams", {})
        self.has_meter_predictions = any(
            checkpoint_hparams.get(key) is not None
            for key in ("num_classes", "den_classes", "meter_classes")
        )
        self.frames2beats = Postprocessor(
            type="dbn" if dbn else "minimal",
            use_beat_guidance=getattr(self.model, "train_beats", True),
        )

    def frames2meter(self, model_prediction: dict) -> dict | None:
        return decode_meter_prediction(
            model_prediction,
            fps=self.fps,
            checkpoint_hparams=getattr(self.model, "checkpoint_hparams", {}),
            data_dir=self.data_dir,
        )

    def predict(self, signal, sr) -> dict:
        spect = self.signal2spect(signal, sr)
        model_prediction = self.spect2predictions(spect)
        beats, downbeats = self.frames2beats(
            model_prediction["beat"], model_prediction["downbeat"]
        )
        return {
            "beats": beats,
            "downbeats": downbeats,
            "meter": self.frames2meter(model_prediction),
        }

    def __call__(self, signal, sr):
        prediction = self.predict(signal, sr)
        return prediction["beats"], prediction["downbeats"]


class File2Beats(Audio2Beats):
    def predict_file(self, audio_path) -> dict:
        signal, sr = load_audio(audio_path)
        return super().predict(signal, sr)

    def __call__(self, audio_path):
        prediction = self.predict_file(audio_path)
        return prediction["beats"], prediction["downbeats"]


class File2File(File2Beats):
    def __call__(self, audio_path, output_path):
        downbeats, beats = super().__call__(audio_path)
        save_beat_tsv(downbeats, beats, output_path)
