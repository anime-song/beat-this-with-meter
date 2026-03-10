"""
Pytorch Lightning module, wraps a BeatThis model along with losses, metrics and
optimizers for training.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import mir_eval
import numpy as np
import torch
from pytorch_lightning import LightningModule

import beat_this.model.loss
from beat_this.inference import split_predict_aggregate
from beat_this.model.beat_tracker import BeatThis
from beat_this.model.postprocessor import Postprocessor
from beat_this.utils import replace_state_dict_key


class PLBeatThis(LightningModule):
    def __init__(
        self,
        spect_dim=128,
        fps=50,
        transformer_dim=512,
        ff_mult=4,
        n_layers=6,
        stem_dim=32,
        dropout={"frontend": 0.1, "transformer": 0.2},
        lr=0.0008,
        weight_decay=0.01,
        pos_weights={"beat": 1, "downbeat": 1},
        head_dim=32,
        loss_type="shift_tolerant_weighted_bce",
        warmup_steps=1000,
        max_epochs=100,
        use_dbn=False,
        eval_trim_beats=5,
        sum_head=True,
        partial_transformers=True,
        train_beats=True,
        num_classes=None,
        num_class_counts=None,
        den_classes=None,
        den_class_counts=None,
        meter_classes=None,
        meter_class_counts=None,
        phase_classes=None,
        phase_class_counts=None,
        num_vocab=None,
        den_vocab=None,
        meter_vocab=None,
        phase_vocab=None,
        loss_weights=None,
        freeze_backbone_epochs=0,
        phase_downbeat_coupling=0.0,
        spec_augment=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.fps = fps
        self.train_beats = train_beats
        self.freeze_backbone_epochs = freeze_backbone_epochs
        self._backbone_frozen = None
        default_loss_weights = {
            "beat": 1.0,
            "downbeat": 1.0,
            "num": 1.0,
            "den": 1.0,
            "meter": 1.0,
            "phase": 1.0,
            "consistency": 0.0,
        }
        self.loss_weights = default_loss_weights | (loss_weights or {})
        self.phase_vocab = list(phase_vocab) if phase_vocab is not None else None
        self.phase_one_idx = None
        if phase_classes is not None:
            if self.phase_vocab is None:
                self.phase_one_idx = 0
            elif 1 in self.phase_vocab:
                self.phase_one_idx = self.phase_vocab.index(1)
            else:
                raise ValueError("phase_vocab must contain beat phase 1.")
        # create model
        self.model = BeatThis(
            spect_dim=spect_dim,
            transformer_dim=transformer_dim,
            ff_mult=ff_mult,
            stem_dim=stem_dim,
            n_layers=n_layers,
            head_dim=head_dim,
            dropout=dropout,
            sum_head=sum_head,
            partial_transformers=partial_transformers,
            num_classes=num_classes,
            den_classes=den_classes,
            meter_classes=meter_classes,
            phase_classes=phase_classes,
            phase_downbeat_coupling=phase_downbeat_coupling,
            spec_augment=spec_augment,
        )
        self.warmup_steps = warmup_steps
        self.max_epochs = max_epochs
        # set up the losses
        self.pos_weights = pos_weights
        if loss_type == "shift_tolerant_weighted_bce":
            self.beat_loss = beat_this.model.loss.ShiftTolerantBCELoss(
                pos_weight=pos_weights["beat"]
            )
            self.downbeat_loss = beat_this.model.loss.ShiftTolerantBCELoss(
                pos_weight=pos_weights["downbeat"]
            )
        elif loss_type == "weighted_bce":
            self.beat_loss = beat_this.model.loss.MaskedBCELoss(
                pos_weight=pos_weights["beat"]
            )
            self.downbeat_loss = beat_this.model.loss.MaskedBCELoss(
                pos_weight=pos_weights["downbeat"]
            )
        elif loss_type == "bce":
            self.beat_loss = beat_this.model.loss.MaskedBCELoss()
            self.downbeat_loss = beat_this.model.loss.MaskedBCELoss()
        elif loss_type == "splitted_shift_tolerant_weighted_bce":
            self.beat_loss = beat_this.model.loss.SplittedShiftTolerantBCELoss(
                pos_weight=pos_weights["beat"]
            )
            self.downbeat_loss = beat_this.model.loss.SplittedShiftTolerantBCELoss(
                pos_weight=pos_weights["downbeat"]
            )
        else:
            raise ValueError(
                "loss_type must be one of 'shift_tolerant_weighted_bce', 'weighted_bce', 'bce'"
            )
        if not self.train_beats:
            self.beat_loss = None

        if num_classes:
            if num_class_counts is not None:
                self.num_loss = beat_this.model.loss.BalancedSoftmaxLoss(
                    class_counts=num_class_counts, tau=0.5, ignore_index=-1
                )
            else:
                self.num_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
        else:
            self.num_loss = None

        if den_classes:
            if den_class_counts is not None:
                self.den_loss = beat_this.model.loss.BalancedSoftmaxLoss(
                    class_counts=den_class_counts, tau=0.5, ignore_index=-1
                )
            else:
                self.den_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
        else:
            self.den_loss = None

        if meter_classes:
            if meter_class_counts is not None:
                self.meter_loss = beat_this.model.loss.BalancedSoftmaxLoss(
                    class_counts=meter_class_counts, tau=0.5, ignore_index=-1
                )
            else:
                self.meter_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
        else:
            self.meter_loss = None

        if phase_classes:
            if phase_class_counts is not None:
                self.phase_loss = beat_this.model.loss.BalancedSoftmaxLoss(
                    class_counts=phase_class_counts, tau=0.5, ignore_index=-1
                )
            else:
                self.phase_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
        else:
            self.phase_loss = None

        self.postprocessor = Postprocessor(
            type="dbn" if use_dbn else "minimal",
            fps=fps,
            use_beat_guidance=self.train_beats,
        )
        self.eval_trim_beats = eval_trim_beats
        self.metrics = Metrics(eval_trim_beats=eval_trim_beats)

    def _compute_phase_consistency_loss(self, batch, model_prediction):
        if (
            self.phase_one_idx is None
            or "phase" not in model_prediction
            or "beat_phase" not in batch
        ):
            return None

        valid_mask = batch["padding_mask"] & (batch["beat_phase"] != -1)
        if not torch.any(valid_mask):
            return None

        downbeat_prob = model_prediction["downbeat"].float().sigmoid()
        phase_prob = torch.softmax(model_prediction["phase"].float(), dim=1)[
            :, self.phase_one_idx, :
        ]
        squared_error = (downbeat_prob - phase_prob).square()
        return squared_error.masked_select(valid_mask).mean()

    def _compute_loss(self, batch, model_prediction):
        losses = {}
        total_loss = model_prediction["downbeat"].new_tensor(0.0)

        if self.beat_loss is not None:
            beat_mask = batch["padding_mask"].clone()

            # Disable beat loss if the track only has downbeat annotations
            if "has_only_downbeats" in batch:
                only_downbeats = batch["has_only_downbeats"]
                beat_mask[only_downbeats] = False

            beat_loss = self.beat_loss(
                model_prediction["beat"], batch["truth_beat"].float(), beat_mask
            )
            losses["beat"] = beat_loss
            total_loss += self.loss_weights["beat"] * beat_loss

        # downbeat mask considers padding and also pieces which don't have downbeat annotations
        downbeat_mask = batch["padding_mask"] * batch["downbeat_mask"][:, None]
        downbeat_loss = self.downbeat_loss(
            model_prediction["downbeat"], batch["truth_downbeat"].float(), downbeat_mask
        )

        losses["downbeat"] = downbeat_loss
        total_loss += self.loss_weights["downbeat"] * downbeat_loss

        # Meter predictions
        if (
            self.num_loss is not None
            and "num" in model_prediction
            and "time_sig_num" in batch
        ):
            targets_num = batch["time_sig_num"].long()
            if torch.any(targets_num != -1):
                loss_num = self.num_loss(model_prediction["num"], targets_num)
                losses["num"] = loss_num
                total_loss += self.loss_weights["num"] * loss_num

        if (
            self.den_loss is not None
            and "den" in model_prediction
            and "time_sig_den" in batch
        ):
            targets_den = batch["time_sig_den"].long()
            if torch.any(targets_den != -1):
                loss_den = self.den_loss(model_prediction["den"], targets_den)
                losses["den"] = loss_den
                total_loss += self.loss_weights["den"] * loss_den

        if (
            self.meter_loss is not None
            and "meter" in model_prediction
            and "time_sig_meter" in batch
        ):
            targets_meter = batch["time_sig_meter"].long()
            if torch.any(targets_meter != -1):
                loss_meter = self.meter_loss(model_prediction["meter"], targets_meter)
                losses["meter"] = loss_meter
                total_loss += self.loss_weights["meter"] * loss_meter

        if (
            self.phase_loss is not None
            and "phase" in model_prediction
            and "beat_phase" in batch
        ):
            targets_phase = batch["beat_phase"].long()
            if torch.any(targets_phase != -1):
                loss_phase = self.phase_loss(model_prediction["phase"], targets_phase)
                losses["phase"] = loss_phase
                total_loss += self.loss_weights["phase"] * loss_phase

        if (
            self.loss_weights["consistency"] > 0
        ):
            consistency_loss = self._compute_phase_consistency_loss(
                batch, model_prediction
            )
            if consistency_loss is not None:
                losses["consistency"] = consistency_loss
                total_loss += (
                    self.loss_weights["consistency"] * consistency_loss
                )

        losses["total"] = total_loss
        # return them in a dictionary for logging
        return losses

    def _compute_classification_metrics(self, batch, model_prediction):
        metrics = {}
        for pred_key, target_key in (
            ("num", "time_sig_num"),
            ("den", "time_sig_den"),
            ("meter", "time_sig_meter"),
            ("phase", "beat_phase"),
        ):
            if pred_key not in model_prediction or target_key not in batch:
                continue
            targets = batch[target_key].long()
            valid = targets != -1
            if not torch.any(valid):
                continue
            preds = torch.argmax(model_prediction[pred_key].float(), dim=1)
            metrics[f"accuracy_{pred_key}"] = (
                (preds[valid] == targets[valid]).float().mean().item()
            )
        return metrics

    def _compute_metrics(self, batch, postp_beat, postp_downbeat, step="val"):
        """ """
        metrics = {}
        if self.train_beats:
            metrics.update(
                self._compute_metrics_target(
                    batch,
                    postp_beat,
                    target="beat",
                    step=step,
                    metric_mask=batch.get("beat_metric_mask"),
                )
            )
        # compute for downbeat
        metrics.update(
            self._compute_metrics_target(
                batch,
                postp_downbeat,
                target="downbeat",
                step=step,
                metric_mask=batch.get("downbeat_mask"),
            )
        )

        return metrics

    def _compute_metrics_target(
        self, batch, postp_target, target, step, metric_mask=None
    ):
        def compute_item(pospt_pred, truth_orig_target):
            # take the ground truth from the original version, so there are no quantization errors
            piece_truth_time = np.frombuffer(truth_orig_target)
            # run evaluation
            metrics = self.metrics(piece_truth_time, pospt_pred, step=step)

            return metrics

        # if the input was not batched, postp_target is an array instead of a tuple of arrays
        # make it a tuple for consistency
        if not isinstance(postp_target, tuple):
            postp_target = (postp_target,)

        truth_targets = batch[f"truth_orig_{target}"]
        if metric_mask is not None:
            if isinstance(metric_mask, torch.Tensor):
                metric_mask = metric_mask.detach().cpu().bool().tolist()
            else:
                metric_mask = [bool(mask) for mask in metric_mask]
            filtered_items = [
                (pred, truth)
                for pred, truth, keep in zip(postp_target, truth_targets, metric_mask)
                if keep
            ]
            if not filtered_items:
                return {}
            postp_target, truth_targets = zip(*filtered_items)

        with ThreadPoolExecutor() as executor:
            piecewise_metrics = list(
                executor.map(
                    compute_item,
                    postp_target,
                    truth_targets,
                )
            )

        # average the beat metrics across the dictionary
        batch_metric = {
            key + f"_{target}": np.mean([x[key] for x in piecewise_metrics])
            for key in piecewise_metrics[0].keys()
        }

        return batch_metric

    def log_losses(self, losses, batch_size, step="train"):
        # log for separate targets
        for target in (
            "beat",
            "downbeat",
            "num",
            "den",
            "meter",
            "phase",
            "consistency",
        ):
            if target not in losses:
                continue
            self.log(
                f"{step}_loss_{target}",
                losses[target].item(),
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
                sync_dist=True,
            )
        # log total loss
        self.log(
            f"{step}_loss",
            losses["total"].item(),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=True,
        )

    def log_metrics(self, metrics, batch_size, step="val"):
        for key, value in metrics.items():
            self.log(
                f"{step}_{key}",
                value,
                prog_bar=key.startswith("F-measure"),
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
                sync_dist=True,
            )

    def training_step(self, batch, batch_idx):
        # run the model
        model_prediction = self.model(batch["spect"])
        # compute loss
        losses = self._compute_loss(batch, model_prediction)
        self.log_losses(losses, len(batch["spect"]), "train")
        return losses["total"]

    def validation_step(self, batch, batch_idx):
        # run the model
        model_prediction = self.model(batch["spect"])
        # compute loss
        losses = self._compute_loss(batch, model_prediction)
        # postprocess the predictions
        postp_beat, postp_downbeat = self.postprocessor(
            model_prediction["beat"],
            model_prediction["downbeat"],
            batch["padding_mask"],
        )
        # compute the metrics
        metrics = self._compute_metrics(batch, postp_beat, postp_downbeat, step="val")
        metrics.update(self._compute_classification_metrics(batch, model_prediction))
        # log
        self.log_losses(losses, len(batch["spect"]), "val")
        self.log_metrics(metrics, batch["spect"].shape[0], "val")

    def test_step(self, batch, batch_idx):
        metrics, model_prediction, _, _ = self.predict_step(batch, batch_idx)
        losses = self._compute_loss(batch, model_prediction)
        metrics.update(self._compute_classification_metrics(batch, model_prediction))
        # log
        self.log_losses(losses, len(batch["spect"]), "test")
        self.log_metrics(metrics, batch["spect"].shape[0], "test")

    def predict_step(
        self,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
        chunk_size: int = 1500,
        overlap_mode: str = "keep_first",
    ) -> Any:
        """
        Compute predictions and metrics for a batch (a dictionary with an "spect" key).
        It splits up the audio into multiple chunks of chunk size,
         which should correspond to the length of the sequence the model was trained with.
        Potential overlaps between chunks can be handled in two ways:
        by keeping the predictions of the excerpt coming first (overlap_mode='keep_first'), or
        by keeping the predictions of the excerpt coming last (overlap_mode='keep_last').
        Note that overlaps appear as the last excerpt is moved backwards
        when it would extend over the end of the piece.
        """
        if batch["spect"].shape[0] != 1:
            raise ValueError(
                "When predicting full pieces, only `batch_size=1` is supported"
            )
        if torch.any(~batch["padding_mask"]):
            raise ValueError(
                "When predicting full pieces, the Dataset must not pad inputs"
            )
        # compute border size according to the loss type
        border_size = 0
        for loss in (self.downbeat_loss, self.beat_loss):
            if hasattr(loss, "tolerance"):
                border_size = 2 * loss.tolerance
                break
        model_prediction = split_predict_aggregate(
            batch["spect"][0], chunk_size, border_size, overlap_mode, self.model
        )
        # add the batch dimension back in the prediction for consistency
        model_prediction = {
            key: value.unsqueeze(0) for key, value in model_prediction.items()
        }
        # postprocess the predictions
        postp_beat, postp_downbeat = self.postprocessor(
            model_prediction["beat"], model_prediction["downbeat"], None
        )
        # compute the metrics
        metrics = self._compute_metrics(batch, postp_beat, postp_downbeat, step="test")
        return metrics, model_prediction, batch["dataset"], batch["spect_path"]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW
        # only decay 2+-dimensional tensors, to exclude biases and norms
        # (filtering on dimensionality idea taken from Kaparthy's nano-GPT)
        params = [
            {
                "params": (
                    p for p in self.parameters() if p.requires_grad and p.ndim >= 2
                ),
                "weight_decay": self.weight_decay,
            },
            {
                "params": (
                    p for p in self.parameters() if p.requires_grad and p.ndim <= 1
                ),
                "weight_decay": 0,
            },
        ]

        optimizer = optimizer(params, lr=self.lr)

        self.lr_scheduler = CosineWarmupScheduler(
            optimizer, self.warmup_steps, self.trainer.estimated_stepping_batches
        )

        result = dict(optimizer=optimizer)
        result["lr_scheduler"] = {"scheduler": self.lr_scheduler, "interval": "step"}
        return result

    def _set_backbone_trainable(self, trainable: bool):
        for module in (self.model.frontend, self.model.transformer_blocks):
            for param in module.parameters():
                param.requires_grad = trainable

    def _update_backbone_freeze_state(self):
        should_freeze = (
            self.freeze_backbone_epochs > 0
            and self.current_epoch < self.freeze_backbone_epochs
        )
        if self._backbone_frozen == should_freeze:
            return
        self._set_backbone_trainable(not should_freeze)
        self._backbone_frozen = should_freeze
        if should_freeze:
            self.print(
                f"Freezing frontend and transformer blocks for the first {self.freeze_backbone_epochs} epochs."
            )
        elif self.freeze_backbone_epochs > 0:
            self.print("Unfreezing frontend and transformer blocks.")

    def on_fit_start(self):
        self._update_backbone_freeze_state()

    def on_train_epoch_start(self):
        self._update_backbone_freeze_state()

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # remove _orig_mod prefixes for compiled models
        state_dict = replace_state_dict_key(state_dict, "_orig_mod.", "")
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        # remove _orig_mod prefixes for compiled models
        state_dict = replace_state_dict_key(state_dict, "_orig_mod.", "")
        return state_dict


class Metrics:
    def __init__(self, eval_trim_beats: int) -> None:
        self.min_beat_time = eval_trim_beats

    def __call__(self, truth, preds, step) -> Any:
        truth = mir_eval.beat.trim_beats(truth, min_beat_time=self.min_beat_time)
        preds = mir_eval.beat.trim_beats(preds, min_beat_time=self.min_beat_time)
        if (
            step == "val"
        ):  # limit the metrics that are computed during validation to speed up training
            fmeasure = mir_eval.beat.f_measure(truth, preds)
            cemgil = mir_eval.beat.cemgil(truth, preds)
            return {"F-measure": fmeasure, "Cemgil": cemgil}
        elif step == "test":  # compute all metrics during testing
            CMLc, CMLt, AMLc, AMLt = mir_eval.beat.continuity(truth, preds)
            fmeasure = mir_eval.beat.f_measure(truth, preds)
            cemgil = mir_eval.beat.cemgil(truth, preds)
            return {"F-measure": fmeasure, "Cemgil": cemgil, "CMLt": CMLt, "AMLt": AMLt}
        else:
            raise ValueError("step must be either val or test")


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Cosine annealing over `max_iters` steps with `warmup` linear warmup steps.
    Optionally re-raises the learning rate for the final `raise_last` fraction
    of total training time to `raise_to` of the full learning rate, again with
    a linear warmup (useful for stochastic weight averaging).
    """

    def __init__(self, optimizer, warmup, max_iters, raise_last=0, raise_to=0.5):
        self.warmup = warmup
        self.max_num_iters = int((1 - raise_last) * max_iters)
        self.raise_to = raise_to
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(step=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, step):
        if step < self.max_num_iters:
            progress = step / self.max_num_iters
            lr_factor = 0.5 * (1 + np.cos(np.pi * progress))
            if step <= self.warmup:
                lr_factor *= step / self.warmup
        else:
            progress = (step - self.max_num_iters) / self.warmup
            lr_factor = self.raise_to * min(progress, 1)
        return lr_factor
