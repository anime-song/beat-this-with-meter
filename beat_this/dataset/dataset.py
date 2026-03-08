import concurrent.futures
import itertools
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from beat_this.dataset.augment import (
    augment_mask_,
    augment_pitchtempo,
    precomputed_augmentation_filenames,
)
from beat_this.utils import index_to_framewise, resolve_annotation_paths

from .mmnpz import MemmappedNpzFile


class BeatTrackingDataset(Dataset):
    """
    A PyTorch Dataset for beat tracking. This dataset loads preprocessed spectrograms and beat annotations
    from a given data folder and provides them for training or evaluation.

    Args:
        item_names (list of str): A list of dataset items such as "gtzan/gtzan_rock_00099".
        data_folder (Path or str): The base folder where the data is stored.
        spect_fps (int, optional): The frames per second of the spectrograms. Defaults to 50.
        train_length (int, optional): The length of the training sequences in frames. If None the entire piece is used. Defaults to 1500.
        deterministic (bool, optional): If True, the dataset always returns the same sequence for a given index.
            Defaults to False.
        augmentations (dict, optional): A dictionary of data augmentations to apply. Possible keys are "tempo", "pitch", and "mask". Defaults to an empty dictionary.
    """

    def __init__(
        self,
        item_names: list[str],
        data_folder,
        spect_fps=50,
        train_length=1500,
        deterministic=False,
        augmentations={},
        length_based_oversampling_factor=0,
        meter_to_idx=None,
        num_to_idx=None,
        den_to_idx=None,
    ):
        self.spect_basepath = data_folder / "audio" / "spectrograms"
        self.annotation_basepath = data_folder / "annotations"
        self.fps = spect_fps
        self.train_length = train_length
        self.deterministic = deterministic
        self.augmentations = augmentations
        self.length_based_oversampling_factor = length_based_oversampling_factor
        
        self.meter_to_idx = meter_to_idx
        self.num_to_idx = num_to_idx
        self.den_to_idx = den_to_idx
        
        datasets = sorted(set(name.split("/", 1)[0] for name in item_names))
        # load dataset info
        self.dataset_info = self._load_dataset_infos(datasets)
        # load .npz spectrogram bundles, if any
        self.spects = self._load_spect_bundles(datasets)
        # load the annotations in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            items = executor.map(self._load_dataset_item, item_names)
        items = [item for item in items if item is not None]
        if self.length_based_oversampling_factor and self.train_length is not None:
            # oversample the dataset according to the audio lengths, so that long pieces are sampled more often
            oversampled_items = []
            for item in items:
                oversampling_factor = np.round(
                    self.length_based_oversampling_factor
                    * len(self._get_spect(item))
                    / self.train_length
                ).astype(int)
                oversampling_factor = max(oversampling_factor, 1)
                oversampled_items.extend(itertools.repeat(item, oversampling_factor))
            print(
                f"Training set oversampled from {len(items)} to {len(oversampled_items)} excerpts."
            )
            items = oversampled_items
        self.items = items

    def _load_dataset_infos(self, datasets):
        dataset_info = {}
        for dataset in datasets:
            with open(self.annotation_basepath / dataset / "info.json") as f:
                dataset_info[dataset] = json.load(f)
        return dataset_info

    def _load_spect_bundles(self, datasets):
        spects = {}
        for dataset in datasets:
            npz_file = (self.spect_basepath / dataset).with_suffix(".npz")
            if npz_file.exists():
                spects[dataset] = MemmappedNpzFile(npz_file)
        return spects

    def _load_dataset_item(self, item_name):
        # stop if not all the augmented audio files are there
        dataset, remainder = item_name.split("/", 1)
        for aug_filename in precomputed_augmentation_filenames(self.augmentations):
            if (f"{remainder}/{aug_filename[:-4]}") not in self.spects.get(
                dataset, ()
            ) and not (self.spect_basepath / item_name / aug_filename).exists():
                print(
                    f"Skipping {item_name} because not all necessary spectrograms are there."
                )
                return

        # load beat and produce a default if beat values are not found
        dataset, stem = item_name.split("/", 1)
        annotation_base = (
            self.annotation_basepath
            / dataset
            / "annotations"
            / "beats"
        )
        
        txt_path, json_path = resolve_annotation_paths(annotation_base, stem)

        time_sig_num = None
        time_sig_den = None
        beat_annotation = None

        if json_path is not None:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            measures = data.get("measures", [])
            
            beat_times = []
            nums = []
            dens = []
            
            for m in measures:
                beat_times.append(m["downbeat_sec"])
                nums.append(m["time_sig_num"])
                dens.append(m["time_sig_den"])
                
            beat_time = np.array(beat_times, dtype=float)
            # all downbeats
            beat_value = np.ones_like(beat_time, dtype=np.int32)
            time_sig_num = np.array(nums, dtype=np.int32)
            time_sig_den = np.array(dens, dtype=np.int32)
            
            # Since this dataset only has downbeats, we set a flag to mask out normal beats loss
            has_only_downbeats = True
             
        elif txt_path is not None:
            beat_annotation = np.loadtxt(txt_path)
            if beat_annotation.ndim == 2:
                beat_time = beat_annotation[:, 0]
                beat_value = beat_annotation[:, 1].astype(int)
            else:
                beat_time = beat_annotation
                beat_value = np.zeros_like(beat_time, dtype=np.int32)
            has_only_downbeats = False
        else:
            print(f"Skipping {item_name} because no annotation file was found.")
            return

        # stop if the annotations that are supposed to be there are not there
        if self.dataset_info[dataset]["has_downbeats"]:
            if json_path is None and beat_annotation is not None and beat_annotation.ndim != 2:
                print(
                    f"Skipping {item_name} because it has {beat_annotation.ndim} columns but downbeat is supposed to be there."
                )
                return

        # create a downbeat mask to handle the case where the downbeat is not annotated
        downbeat_mask = self.dataset_info[dataset]["has_downbeats"]
        # take care of different subsections of rwc for the dataset name
        if dataset == "rwc":
            dataset = "rwc_" + stem.split("_", 2)[1]
            
        return {
            "spect_path": Path(item_name) / "track.npy",
            "beat_time": beat_time,
            "beat_value": beat_value,
            "downbeat_mask": downbeat_mask,
            "has_only_downbeats": has_only_downbeats,
            "time_sig_num": time_sig_num,
            "time_sig_den": time_sig_den,
            "dataset": dataset,
        }

    def _get_spect(self, item):
        try:
            dataset, filename = str(item["spect_path"]).split("/", 1)
            spect = self.spects[dataset][filename[:-4]]
        except KeyError:
            spect = np.load(self.spect_basepath / item["spect_path"], mmap_mode="r")
        return spect

    def get_frame_count(self, index):
        """Return number of frames of given item."""
        return len(self._get_spect(self.items[index]))

    def get_beat_count(self, index):
        """Return number of beats (including downbeats) of given item."""
        return len(self.items[index]["beat_time"])

    def get_downbeat_count(self, index):
        """Return number of downbeats of given item."""
        return (self.items[index]["beat_value"] == 1).sum()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        if isinstance(index, (int, np.int64)):  # when index is a single int
            item = self.items[index]

            # select a pitch shift and time stretch
            item = augment_pitchtempo(item, self.augmentations)

            # load spectrogram
            spect = self._get_spect(item)

            # define the excerpt to use
            original_length = len(spect)
            if self.train_length is not None:
                longer = original_length - self.train_length
            else:
                longer = 0
            if longer > 0:  # if the piece is longer than the desired length
                if self.deterministic:
                    # select the middle of the excerpt
                    start_frame = longer // 2
                else:
                    start_frame = np.random.randint(0, longer)
                end_frame = start_frame + self.train_length
            else:
                start_frame = 0
                end_frame = original_length

            # obtain a view of the excerpt
            spect = spect[start_frame:end_frame]

            if "mask" in self.augmentations:
                # copy the spectrogram and apply mask augmentation
                spect = np.copy(spect)
                spect = augment_mask_(spect, self.augmentations, self.fps)
            else:
                # only ensure we have a writeable array (so PyTorch is happy)
                spect = np.require(spect, requirements="W")

            # prepare annotations
            (
                framewise_truth_beat,
                framewise_truth_downbeat,
                truth_orig_beat,
                truth_orig_downbeat,
                framewise_time_sig_num,
                framewise_time_sig_den,
                has_only_downbeats,
            ) = prepare_annotations(item, start_frame, end_frame, self.fps)

            # restructure the item dict with the correct training information
            out_item = {
                "spect": spect,
                "spect_path": str(item["spect_path"]),
                "dataset": item["dataset"],
                "start_frame": start_frame,
                "truth_beat": framewise_truth_beat,
                "truth_downbeat": framewise_truth_downbeat,
                "downbeat_mask": torch.as_tensor(item["downbeat_mask"]),
                "padding_mask": (
                    np.ones(self.train_length, dtype=bool)
                    if self.train_length is not None
                    else np.ones(original_length, dtype=bool)
                ),
                "truth_orig_beat": truth_orig_beat,
                "truth_orig_downbeat": truth_orig_downbeat,
                "has_only_downbeats": torch.as_tensor(has_only_downbeats),
            }
            if framewise_time_sig_num is not None:
                # convert native values to class indices if vocab dictionaries are provided
                if self.num_to_idx is not None:
                    mapped_num = np.zeros_like(framewise_time_sig_num) - 1
                    for val, idx in self.num_to_idx.items():
                        mapped_num[framewise_time_sig_num == val] = idx
                    out_item["time_sig_num"] = mapped_num
                else:
                    out_item["time_sig_num"] = framewise_time_sig_num
                    
                if self.den_to_idx is not None:
                    mapped_den = np.zeros_like(framewise_time_sig_den) - 1
                    for val, idx in self.den_to_idx.items():
                        mapped_den[framewise_time_sig_den == val] = idx
                    out_item["time_sig_den"] = mapped_den
                else:
                    out_item["time_sig_den"] = framewise_time_sig_den
                    
                if self.meter_to_idx is not None:
                    # Create meter sequence from num and den
                    # we only iterate over non-ignore (-1) elements
                    meter_seq = np.zeros_like(framewise_time_sig_num) - 1
                    valid_mask = framewise_time_sig_num != -1
                    for i in np.where(valid_mask)[0]:
                        meter_str = f"{framewise_time_sig_num[i]}/{framewise_time_sig_den[i]}"
                        if meter_str in self.meter_to_idx:
                            meter_seq[i] = self.meter_to_idx[meter_str]
                    out_item["time_sig_meter"] = meter_seq

            # pad all framewise tensors if needed
            if longer < 0:
                out_item["spect"] = np.pad(
                    out_item["spect"], [(0, -longer), (0, 0)], constant_values=0
                )
                for k in "truth_beat", "truth_downbeat":
                    out_item[k] = np.pad(out_item[k], [(0, -longer)], constant_values=0)
                out_item["padding_mask"][longer:] = 0
                if framewise_time_sig_num is not None:
                    # use -1 as ignore label index for padding
                    out_item["time_sig_num"] = np.pad(out_item["time_sig_num"], [(0, -longer)], constant_values=-1)
                    out_item["time_sig_den"] = np.pad(out_item["time_sig_den"], [(0, -longer)], constant_values=-1)
                    if self.meter_to_idx is not None:
                        out_item["time_sig_meter"] = np.pad(out_item["time_sig_meter"], [(0, -longer)], constant_values=-1)
                    
            return out_item

        else:  # when index is a list of ints
            return [self[i] for i in index]


class BeatDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule for beat tracking. This DataModule handles the loading and preprocessing of the
    BeatTrackingDataset and prepares it for use with a PyTorch Lightning model.
    It can produce cross-validation or single  train/val/test splits.

    Args:
        data_dir (Path or str): The parent directory where the data (spectrograms and beat labels) is stored.
        batch_size (int, optional): The size of the batches to be generated by the DataLoader. Defaults to 8.
        train_length (int, optional): The length of the subsequences in frames. If None, the entire pieces are returner. Defaults to 1500.
        num_workers (int, optional): The number of worker processes to use for data loading. Defaults to 20.
        augmentations (dict, optional): A dictionary of data augmentations to apply. Defaults to {"pitch": {"min": -5, "max": 6}, "time": {"min": -20, "max": 20, "stride": 4}}.
        test_dataset (str, optional): The name of the dataset to use for testing. Defaults to "gtzan".
        hung_data (bool, optional): If True, only use the datasets from the Hung et al. paper for training; validation is still on all datasets. Defaults to False.
        no_val (bool, optional): If True, train on all train+val data and do not use a validation set; for compatibility reason, the validation metrics are still computed, but are not meaningful. Defaults to False.
        spect_fps (int, optional): The frames per second of the spectrograms. Defaults to 50.
        length_based_oversampling_factor (int, optional): The factor by which to oversample the train dataset based on sequence length. Defaults to 0.
        fold (int, optional): The fold number for cross-validation. If None, the single split is used. Defaults to None.
        predict_datasplit (str, optional): The split to use for prediction. Prediction dataset is always full pieces. Defaults to "test".
    """

    def __init__(
        self,
        data_dir,
        batch_size=8,
        train_length=1500,
        num_workers=20,
        augmentations={
            "pitch": {"min": -5, "max": 6},
            "tempo": {"min": -20, "max": 20, "stride": 4},
        },
        test_dataset="gtzan",
        hung_data=False,
        no_val=False,
        spect_fps=50,
        length_based_oversampling_factor=0,
        fold=None,
        predict_datasplit="test",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.initialized = {}
        # remember all arguments
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.train_length = train_length
        self.num_workers = num_workers
        if not set(augmentations.keys()).issubset({"mask", "pitch", "tempo"}):
            raise ValueError(f"Unsupported augmentations: {augmentations.keys()}")
        self.augmentations = augmentations
        self.test_set_name = test_dataset
        self.hung_data = hung_data
        self.no_val = no_val
        self.spect_fps = spect_fps
        self.length_based_oversampling_factor = length_based_oversampling_factor
        self.fold = fold
        self.predict_datasplit = predict_datasplit

    def setup(self, stage):
        if self.initialized.get(stage, False):
            return

        # set up the paths
        annotation_dir = self.data_dir / "annotations"

        # load train/val splits
        if stage in ("fit", "validate"):
            self.val_items = []
            self.train_items = []
            split_file = "8-folds.split" if self.fold is not None else "single.split"
            for dataset_dir in annotation_dir.iterdir():
                if not dataset_dir.is_dir() or not (dataset_dir / split_file).exists():
                    continue
                dataset = dataset_dir.name
                if dataset == self.test_set_name:
                    continue
                split = pd.read_csv(
                    dataset_dir / split_file,
                    header=None,
                    names=["piece", "part"],
                    sep="\t",
                )
                if self.fold is not None:
                    # CV: use given fold for validation, rest for training
                    self.val_items.extend(
                        f"{dataset}/{stem}"
                        for stem in split.piece[split.part == self.fold]
                    )
                    self.train_items.extend(
                        f"{dataset}/{stem}"
                        for stem in split.piece[split.part != self.fold]
                    )
                else:
                    # single split: marked as val and train
                    self.val_items.extend(
                        f"{dataset}/{stem}" for stem in split.piece[split.part == "val"]
                    )
                    self.train_items.extend(
                        f"{dataset}/{stem}"
                        for stem in split.piece[split.part == "train"]
                    )
            if self.no_val:
                # Train on all available data (excluding the test set).
                # For compatibility, validation metrics are still computed
                # on the original validation set now included in training.
                self.train_items.extend(self.val_items)
            if self.hung_data:
                # Use the training datasets from MODELING BEATS AND DOWNBEATS
                # WITH A TIME-FREQUENCY TRANSFORMER (for comparability, the
                # validation set stays the same, with all datasets).
                regexp = re.compile(
                    "^(hainsworth/|ballroom/|hjdb/|beatles/|rwc/rwc_popular|simac/|smc/|harmonix/|).*$"
                )
                self.train_items = [
                    item for item in self.train_items if regexp.match(item)
                ]
            self.val_items.sort()
            self.train_items.sort()

        # load validation set
        if stage in ("fit", "validate"):
            self.val_dataset = BeatTrackingDataset(
                self.val_items,
                deterministic=True,
                augmentations={},
                train_length=self.train_length,
                data_folder=self.data_dir,
                spect_fps=self.spect_fps,
            )
            print(
                "Validation set:",
                len(self.val_dataset),
                "items from:",
                *sorted(set(item.split("/", 1)[0] for item in self.val_items)),
            )
            self.initialized["validate"] = True

        # load training set
        if stage == "fit":
            self.train_dataset = BeatTrackingDataset(
                self.train_items,
                deterministic=False,
                augmentations=self.augmentations,
                train_length=self.train_length,
                data_folder=self.data_dir,
                spect_fps=self.spect_fps,
                length_based_oversampling_factor=self.length_based_oversampling_factor,
            )
            print(
                "Training set:",
                len(self.train_dataset),
                "items from:",
                *sorted(set(item.split("/", 1)[0] for item in self.train_items)),
            )
            self.initialized["fit"] = True

        # load test set
        if stage == "test":
            test_annotations_dir = (
                annotation_dir / self.test_set_name / "annotations" / "beats"
            )
            self.test_items = sorted(
                f"{self.test_set_name}/{item.stem}"
                for item in test_annotations_dir.glob("*.beats")
            )
            self.test_dataset = BeatTrackingDataset(
                self.test_items,
                deterministic=True,
                augmentations={},
                train_length=None,
                data_folder=self.data_dir,
                spect_fps=self.spect_fps,
            )
            print(
                "Test set:", len(self.test_dataset), "items from:", self.test_set_name
            )
            self.initialized["test"] = True

        # load prediction set
        if stage == "predict":
            if self.predict_datasplit == "test":
                self.setup("test")
                # we can directly use the test dataset for predictions
                self.predict_dataset = self.test_dataset
            else:
                if self.predict_datasplit == "train":
                    self.setup("fit")
                    items = self.train_items
                elif self.predict_datasplit == "val":
                    self.setup("validate")
                    items = self.val_items
                # for prediction, we want to use full items (train_length=None)
                self.predict_dataset = BeatTrackingDataset(
                    items,
                    deterministic=True,
                    augmentations={},
                    train_length=None,
                    data_folder=self.data_dir,
                    spect_fps=self.spect_fps,
                )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        # Warning: for performances, this only runs on the middle excerpt of the long pieces
        # The paper results are computed after training in the predict script
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset, batch_size=1, num_workers=self.num_workers
        )

    def get_train_positive_weights(self, widen_target_mask=3):
        """
        Computes the relation of negative targets to positive targets.
        `widen_target_mask` reduces the number of negative targets by the given
        factor times the number of positive targets (for ignoring a number of
        frames around each positive label).
        For example a `widen_target_mask` of 3 will ignore 7 frames, 3 for each side plus the central.
        """
        # find the positive weight for the loss as a ratio between (down)beat and non-(down)beat annotation
        dataset = self.train_dataset
        all_frames = all_frames_db = 0
        for item in dataset.items:
            frames = len(dataset._get_spect(item))
            all_frames += frames
            if item["downbeat_mask"]:
                all_frames_db += frames
        beat_frames = sum(len(item["beat_value"]) for item in dataset.items)
        downbeat_frames = sum(
            (item["beat_value"] == 1).sum()
            for item in dataset.items
            if item["downbeat_mask"]
        )

        return {
            "beat": int(
                np.round(
                    (all_frames - beat_frames * (widen_target_mask * 2 + 1))
                    / beat_frames
                )
            ),
            "downbeat": int(
                np.round(
                    (all_frames_db - downbeat_frames * (widen_target_mask * 2 + 1))
                    / downbeat_frames
                )
            ),
        }


def prepare_annotations(item, start_frame, end_frame, fps):
    truth_bdb_time = item["beat_time"]
    truth_bdb_value = item["beat_value"]
    # convert beat time from seconds to frame
    truth_bdb_frame = (truth_bdb_time * fps).round().astype(int)
    # form annotations excerpt
    # filter out the annotations that are earlier than the start and shift left
    truth_bdb_frame -= start_frame
    idx_start = np.searchsorted(truth_bdb_frame, 0)
    truth_bdb_frame = truth_bdb_frame[idx_start:]
    truth_bdb_value = truth_bdb_value[idx_start:]
    
    # filter out the annotations that are later than the end
    idx_end = np.searchsorted(truth_bdb_frame, end_frame - start_frame)
    truth_bdb_frame = truth_bdb_frame[:idx_end]
    truth_bdb_value = truth_bdb_value[:idx_end]

    # create beat and downbeat separated annotations
    truth_beat = truth_bdb_frame
    truth_downbeat = truth_bdb_frame[truth_bdb_value == 1]
    # transform beat downbeat to frame-wise annotations
    framewise_truth_beat = index_to_framewise(truth_beat, end_frame - start_frame)
    framewise_truth_downbeat = index_to_framewise(
        truth_downbeat, end_frame - start_frame
    )
    
    # prepare time_sig_num and time_sig_den if they are presented
    framewise_time_sig_num = None
    framewise_time_sig_den = None
    
    if item.get("time_sig_num") is not None and item.get("time_sig_den") is not None:
        idx_end_sig = idx_start + idx_end # recover original offset end index
        time_sig_num_seq = item["time_sig_num"][idx_start:idx_end_sig]
        time_sig_den_seq = item["time_sig_den"][idx_start:idx_end_sig]
        
        # transform to framewise annotations (similar to beat, but set categorical labels at the downbeat frames)
        framewise_time_sig_num = np.zeros(end_frame - start_frame, dtype=np.int32) - 1 # Use -1 as ignore index
        framewise_time_sig_num[truth_downbeat] = time_sig_num_seq
        
        framewise_time_sig_den = np.zeros(end_frame - start_frame, dtype=np.int32) - 1
        framewise_time_sig_den[truth_downbeat] = time_sig_den_seq

    # create orig beat, downbeat annotations for unquantized evaluation
    truth_orig_beat = item["beat_time"]
    truth_orig_downbeat = truth_bdb_time[
        item["beat_value"] == 1
    ]  # (use the full beat_value)
    # filter out the annotations that are outside the excerpt, and shift them left to the excerpt time
    truth_orig_beat = truth_orig_beat[
        (truth_orig_beat >= start_frame / fps) & (truth_orig_beat < end_frame / fps)
    ] - (start_frame / fps)
    truth_orig_downbeat = truth_orig_downbeat[
        (truth_orig_downbeat >= start_frame / fps)
        & (truth_orig_downbeat < end_frame / fps)
    ] - (start_frame / fps)
    # convert to strings (trick to collate sequences of different lengths)
    truth_orig_beat = truth_orig_beat.tobytes()
    truth_orig_downbeat = truth_orig_downbeat.tobytes()
    return (
        framewise_truth_beat,
        framewise_truth_downbeat,
        truth_orig_beat,
        truth_orig_downbeat,
        framewise_time_sig_num,
        framewise_time_sig_den,
        item.get("has_only_downbeats", False)
    )
