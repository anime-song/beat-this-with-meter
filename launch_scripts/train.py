import argparse
import gc
from pathlib import Path

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from beat_this.dataset import BeatDataModule
from beat_this.inference import load_checkpoint
from beat_this.model.pl_module import PLBeatThis
from beat_this.utils import replace_state_dict_key, resolve_annotation_paths


def clear_gpu_cache():
    if not torch.cuda.is_available():
        return
    gc.collect()
    torch.cuda.empty_cache()
    print("Cleared CUDA memory cache before training.")


def main(args):
    # for repeatability
    seed_everything(args.seed, workers=True)

    print("Starting a new run with the following parameters:")
    print(args)

    params_str = f"{'noval ' if not args.val else ''}{'hung ' if args.hung_data else ''}{'fold' + str(args.fold) + ' ' if args.fold is not None else ''}{args.loss}-h{args.transformer_dim}-aug{args.tempo_augmentation}{args.pitch_augmentation}{args.mask_augmentation}{' specAug ' if args.spec_augment else ''}{' phaseAux ' if args.phase_prediction else ''}{' cons ' if args.phase_prediction and args.consistency_loss_weight > 0 else ''}{f' p2db{args.phase_downbeat_coupling:g} ' if args.phase_prediction and args.phase_downbeat_coupling > 0 else ''}{' pseudoBeat ' if args.pseudo_beats_from_meter else ''}{' nosumH ' if not args.sum_head else ''}{' nopartialT ' if not args.partial_transformers else ''}{' downbeatOnly ' if args.downbeat_only else ''}"
    if args.logger == "wandb":
        if args.resume_checkpoint and args.resume_id:
            wandb_args = dict(id=args.resume_id, resume="must")
        else:
            wandb_args = {}
        logger = WandbLogger(
            project="beat_this", name=f"{args.name} {params_str}".strip(), **wandb_args
        )
    else:
        logger = None

    if args.force_flash_attention:
        print("Forcing the use of the flash attention.")
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)

    data_dir = Path(__file__).parent.parent.relative_to(Path.cwd()) / "data"
    checkpoint_dir = (
        Path(__file__).parent.parent.relative_to(Path.cwd()) / "checkpoints"
    )
    augmentations = {}
    if args.tempo_augmentation:
        augmentations["tempo"] = {"min": -20, "max": 20, "stride": 4}
    if args.pitch_augmentation:
        augmentations["pitch"] = {"min": -5, "max": 6}
    if args.mask_augmentation:
        # kind, min_count, max_count, min_len, max_len, min_parts, max_parts
        augmentations["mask"] = {
            "kind": "permute",
            "min_count": 1,
            "max_count": 6,
            "min_len": 0.1,
            "max_len": 2,
            "min_parts": 5,
            "max_parts": 9,
        }
    spec_augment = None
    if args.spec_augment:
        spec_augment = {
            "freq_mask_ratio": 0.25,
            "time_mask_ratio": 0.25,
            "num_freq_masks": 1,
            "num_time_masks": 1,
            "p": 1.0,
        }

    datamodule = BeatDataModule(
        data_dir,
        batch_size=args.batch_size,
        train_length=args.train_length,
        spect_fps=args.fps,
        num_workers=args.num_workers,
        test_dataset="gtzan",
        length_based_oversampling_factor=args.length_based_oversampling_factor,
        augmentations=augmentations,
        hung_data=args.hung_data,
        no_val=not args.val,
        fold=args.fold,
        pseudo_beats_from_meter=args.pseudo_beats_from_meter,
    )
    datamodule.setup(stage="fit")

    # Meter prediction logic
    num_classes = None
    num_class_counts = None
    den_classes = None
    den_class_counts = None
    meter_classes = None
    meter_class_counts = None
    phase_classes = None
    phase_class_counts = None
    num_vocab = None
    den_vocab = None
    meter_vocab = None
    phase_vocab = None

    if args.meter_prediction:
        import json

        num_counts = {}
        den_counts = {}
        meter_counts = {}

        print("Scanning training data for meter vocabulary...")
        # read the raw jsons corresponding to the training items
        for item in datamodule.train_dataset.items:
            spect_path = Path(item["spect_path"])
            dataset = spect_path.parts[0]
            stem = spect_path.parent.name
            annotation_base = (
                data_dir / "annotations" / dataset / "annotations" / "beats"
            )
            _, json_path = resolve_annotation_paths(annotation_base, stem)
            if json_path is not None:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for measure in data.get("measures", []):
                    num = measure["time_sig_num"]
                    den = measure["time_sig_den"]
                    meter = f"{num}/{den}"

                    num_counts[num] = num_counts.get(num, 0) + 1
                    den_counts[den] = den_counts.get(den, 0) + 1
                    meter_counts[meter] = meter_counts.get(meter, 0) + 1

        num_vocab = sorted(list(num_counts.keys()))
        den_vocab = sorted(list(den_counts.keys()))
        meter_vocab = sorted(list(meter_counts.keys()))

        num_to_idx = {k: v for v, k in enumerate(num_vocab)}
        den_to_idx = {k: v for v, k in enumerate(den_vocab)}
        meter_to_idx = {k: v for v, k in enumerate(meter_vocab)}

        num_classes = len(num_vocab)
        den_classes = len(den_vocab)
        meter_classes = len(meter_vocab)
        num_class_counts = [num_counts[num] for num in num_vocab]
        den_class_counts = [den_counts[den] for den in den_vocab]
        meter_class_counts = [meter_counts[meter] for meter in meter_vocab]

        # assign vocab to datasets
        datamodule.train_dataset.num_to_idx = num_to_idx
        datamodule.train_dataset.den_to_idx = den_to_idx
        datamodule.train_dataset.meter_to_idx = meter_to_idx

        if hasattr(datamodule, "val_dataset") and datamodule.val_dataset is not None:
            datamodule.val_dataset.num_to_idx = num_to_idx
            datamodule.val_dataset.den_to_idx = den_to_idx
            datamodule.val_dataset.meter_to_idx = meter_to_idx

        print(
            f"Meter Vocabulary - num_classes: {num_classes}, den_classes: {den_classes}, meter_classes: {meter_classes}"
        )

    if args.phase_prediction:
        phase_counts = {}
        print("Scanning training data for beat phase vocabulary...")
        for item in datamodule.train_dataset.items:
            for phase in item["beat_value"]:
                phase = int(phase)
                if phase <= 0:
                    continue
                phase_counts[phase] = phase_counts.get(phase, 0) + 1

        phase_vocab = sorted(list(phase_counts.keys()))
        phase_to_idx = {k: v for v, k in enumerate(phase_vocab)}
        phase_classes = len(phase_vocab)
        phase_class_counts = [phase_counts[phase] for phase in phase_vocab]

        datamodule.train_dataset.phase_to_idx = phase_to_idx
        if hasattr(datamodule, "val_dataset") and datamodule.val_dataset is not None:
            datamodule.val_dataset.phase_to_idx = phase_to_idx

        print(f"Beat Phase Vocabulary - phase_classes: {phase_classes}")

    # compute positive weights
    pos_weights = datamodule.get_train_positive_weights(widen_target_mask=3)
    if args.beat_pos_weight is not None:
        pos_weights["beat"] = args.beat_pos_weight
    if args.downbeat_pos_weight is not None:
        pos_weights["downbeat"] = args.downbeat_pos_weight
    print("Using positive weights: ", pos_weights)
    dropout = {
        "frontend": args.frontend_dropout,
        "transformer": args.transformer_dropout,
    }
    pl_model = PLBeatThis(
        spect_dim=128,
        fps=50,
        transformer_dim=args.transformer_dim,
        ff_mult=4,
        n_layers=args.n_layers,
        stem_dim=32,
        dropout=dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        pos_weights=pos_weights,
        head_dim=32,
        loss_type=args.loss,
        warmup_steps=args.warmup_steps,
        max_epochs=args.max_epochs,
        use_dbn=args.dbn,
        eval_trim_beats=args.eval_trim_beats,
        sum_head=args.sum_head,
        partial_transformers=args.partial_transformers,
        train_beats=not args.downbeat_only,
        num_classes=num_classes,
        num_class_counts=num_class_counts,
        den_classes=den_classes,
        den_class_counts=den_class_counts,
        meter_classes=meter_classes,
        meter_class_counts=meter_class_counts,
        phase_classes=phase_classes,
        phase_class_counts=phase_class_counts,
        num_vocab=num_vocab,
        den_vocab=den_vocab,
        meter_vocab=meter_vocab,
        phase_vocab=phase_vocab,
        loss_weights={
            "num": args.num_loss_weight,
            "den": args.den_loss_weight,
            "meter": args.meter_loss_weight,
            "phase": args.phase_loss_weight,
            "consistency": args.consistency_loss_weight,
        },
        freeze_backbone_epochs=args.freeze_backbone_epochs,
        phase_downbeat_coupling=args.phase_downbeat_coupling,
        spec_augment=spec_augment,
    )

    if args.pretrained_checkpoint:
        print(f"Loading pretrained weights from {args.pretrained_checkpoint}")
        checkpoint = load_checkpoint(args.pretrained_checkpoint, device="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        # PLBeatThis saves model weights with "model." prefix, remove it to load into BeatThis instance
        state_dict = replace_state_dict_key(state_dict, "model.", "")
        try:
            pl_model.model.load_state_dict(state_dict, strict=False)
            print("Successfully loaded pretrained weights (strict=False).")
        except Exception as e:
            print(f"Error loading state_dict: {e}")

    for part in args.compile:
        if hasattr(pl_model.model, part):
            setattr(pl_model.model, part, torch.compile(getattr(pl_model.model, part)))
            print("Will compile model", part)
        else:
            raise ValueError("The model is missing the part", part, "to compile")

    callbacks = [LearningRateMonitor(logging_interval="step")]
    # save the best downbeat model plus the latest checkpoint
    callbacks.append(
        ModelCheckpoint(
            monitor="val_F-measure_downbeat",
            mode="max",
            save_top_k=1,
            save_last=True,
            every_n_epochs=1,
            dirpath=str(checkpoint_dir),
            filename=f"{args.name} S{args.seed} {params_str}".strip(),
        )
    )

    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=[args.gpu],
        num_sanity_val_steps=1,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=1,
        precision="16-mixed",
        accumulate_grad_batches=args.accumulate_grad_batches,
        check_val_every_n_epoch=args.val_frequency,
    )

    clear_gpu_cache()
    trainer.fit(pl_model, datamodule, ckpt_path=args.resume_checkpoint)
    trainer.test(pl_model, datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--force-flash-attention", default=False, action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "--compile",
        action="store",
        nargs="*",
        type=str,
        default=["frontend", "transformer_blocks", "task_heads"],
        help="Which model parts to compile, among frontend, transformer_encoder, task_heads",
    )
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--transformer-dim", type=int, default=512)
    parser.add_argument(
        "--frontend-dropout",
        type=float,
        default=0.1,
        help="dropout rate to apply in the frontend",
    )
    parser.add_argument(
        "--transformer-dropout",
        type=float,
        default=0.2,
        help="dropout rate to apply in the main transformer blocks",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--logger", type=str, choices=["wandb", "none"], default="none")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--n-heads", type=int, default=16)
    parser.add_argument("--fps", type=int, default=50, help="The spectrograms fps.")
    parser.add_argument(
        "--loss",
        type=str,
        default="shift_tolerant_weighted_bce",
        choices=[
            "shift_tolerant_weighted_bce",
            "fast_shift_tolerant_weighted_bce",
            "weighted_bce",
            "bce",
        ],
        help="The loss to use",
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=50, help="warmup steps for optimizer"
    )
    parser.add_argument(
        "--max-epochs", type=int, default=100, help="max epochs for training"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="batch size for training"
    )
    parser.add_argument("--accumulate-grad-batches", type=int, default=8)
    parser.add_argument(
        "--train-length",
        type=int,
        default=1500,
        help="maximum seq length for training in frames",
    )
    parser.add_argument(
        "--dbn",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="use madmom postprocessing DBN",
    )
    parser.add_argument(
        "--eval-trim-beats",
        metavar="SECONDS",
        type=float,
        default=5,
        help="Skip the first given seconds per piece in evaluating (default: %(default)s)",
    )
    parser.add_argument(
        "--val-frequency",
        metavar="N",
        type=int,
        default=1,
        help="validate every N epochs (default: %(default)s)",
    )
    parser.add_argument(
        "--tempo-augmentation",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use precomputed tempo aumentation",
    )
    parser.add_argument(
        "--pitch-augmentation",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use precomputed pitch aumentation",
    )
    parser.add_argument(
        "--mask-augmentation",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use online mask aumentation",
    )
    parser.add_argument(
        "--spec-augment",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Apply model-side SpecAugment with 25%% masking on both time and frequency axes during training.",
    )
    parser.add_argument(
        "--sum-head",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use SumHead instead of two separate Linear heads",
    )
    parser.add_argument(
        "--partial-transformers",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use Partial transformers in the frontend",
    )
    parser.add_argument(
        "--length-based-oversampling-factor",
        type=float,
        default=0.65,
        help="The factor to oversample the long pieces in the dataset. Set to 0 to only take one excerpt for each piece.",
    )
    parser.add_argument(
        "--val",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Train on all data, including validation data, escluding test data. The validation metrics will still be computed, but they won't carry any meaning.",
    )
    parser.add_argument(
        "--hung-data",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Limit the training to Hung et al. data. The validation will still be computed on all datasets.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="If given, the CV fold number to *not* train on (0-based).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the random number generators.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help="Resume training from a local checkpoint.",
    )
    parser.add_argument(
        "--resume-id",
        type=str,
        default=None,
        help="When resuming with --resume-checkpoint, optionally provide the wandb id to continue logging to.",
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        type=str,
        default=None,
        help="Path or name of a pretrained checkpoint to use as initial weights for fine-tuning.",
    )
    parser.add_argument(
        "--meter-prediction",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enable meter (time signature) prediction heads and losses.",
    )
    parser.add_argument(
        "--phase-prediction",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enable auxiliary beat phase prediction from beat indices within each bar.",
    )
    parser.add_argument(
        "--downbeat-only",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Disable beat loss and beat metrics; train only downbeat and optional meter targets.",
    )
    parser.add_argument(
        "--pseudo-beats-from-meter",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Infer per-measure beat positions from time signatures in JSON downbeat annotations.",
    )
    parser.add_argument(
        "--num-loss-weight",
        type=float,
        default=0.1,
        help="Weight applied to the numerator classification loss.",
    )
    parser.add_argument(
        "--den-loss-weight",
        type=float,
        default=0.1,
        help="Weight applied to the denominator classification loss.",
    )
    parser.add_argument(
        "--meter-loss-weight",
        type=float,
        default=0.1,
        help="Weight applied to the combined meter classification loss.",
    )
    parser.add_argument(
        "--phase-loss-weight",
        type=float,
        default=0.1,
        help="Weight applied to the auxiliary beat phase classification loss.",
    )
    parser.add_argument(
        "--consistency-loss-weight",
        type=float,
        default=0.0,
        help="Small weight for matching p(downbeat) with p(phase==1) on beat frames.",
    )
    parser.add_argument(
        "--phase-downbeat-coupling",
        type=float,
        default=0.05,
        help="Small residual weight from the phase==1 logit into the downbeat logit.",
    )
    parser.add_argument(
        "--freeze-backbone-epochs",
        type=int,
        default=0,
        help="Freeze the pretrained frontend and transformer blocks for the first N epochs.",
    )
    parser.add_argument(
        "--beat-pos-weight",
        type=float,
        default=5,
        help="Override the automatically computed positive weight for beat loss.",
    )
    parser.add_argument(
        "--downbeat-pos-weight",
        type=float,
        default=20,
        help="Override the automatically computed positive weight for downbeat loss.",
    )

    args = parser.parse_args()

    main(args)
