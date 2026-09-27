"""
=============================================================
V5 LATENT WORLD MODEL — INFERENCE ENGINE
=============================================================

The V5 checkpoint (`models/latent_world_model_v5/latent_world_model_v5.pt`)
was produced off-repo and records `model_class = "LatentWorldModelV5"`, but
no such class existed here, so the weights were unloadable. This module
supplies that class and a thin, validated inference wrapper around it.

Scope: make the V5 model CALLABLE. Nothing else.
This module does not train, does not touch the frozen V4 baseline, does not
assemble datasets, and does not select thresholds.

-------------------------------------------------------------
ARCHITECTURE (reconstructed from the checkpoint's own tensors)
-------------------------------------------------------------
Every layer below is derived from the shapes in `model_state_dict`, not
copied from V4 on faith; `verify_state_dict_compatibility()` re-checks the
key set and every shape before the weights are loaded.

    x  [B, 10, 82]
      -> LSTM(82 -> 128, num_layers=2, batch_first, dropout=0.20)
      -> take final timestep of the top layer            [B, 128]
      -> LayerNorm(128)
      -> Linear(128 -> 32)                        = z_t  [B, 32]

    transition (gated residual):
        candidate = Linear(32->128) -> ReLU -> Linear(128->32)
        gate      = sigmoid(Linear(32->32))
        z_next    = LayerNorm(gate * candidate + (1 - gate) * z)

    decoder:      Linear(32->64) -> ReLU -> Linear(64->26)
    attack head:  Linear(32->32) -> ReLU -> Dropout(0.10) -> Linear(32->1)

The attack head emits a LOGIT. Sigmoid is applied exactly once, in the
inference layer, never inside the module.

-------------------------------------------------------------
TARGET SCALING
-------------------------------------------------------------
`src/prepare_multiday_benchmark.py` (L146-L153) fits a StandardScaler on the
26 state features and *transforms* the multi-step targets before training.
The V5 target scaler mirrors that artifact exactly (same 26 feature names,
same sample count as the V5 input scaler), so the decoder's output lives in
target-scaled space and `inverse_transform` is the correct — and only —
inverse operation. No new transformation is invented here.

-------------------------------------------------------------
THRESHOLD
-------------------------------------------------------------
Read verbatim from the checkpoint's `selected_threshold` (0.5). No
threshold search is performed, and no test data is consulted.

Usage:
    python src/infer_latent_world_model_v5.py --self-test

=============================================================
"""

from __future__ import annotations

import argparse
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

# =============================================================
# PATHS
# =============================================================

REPO_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = REPO_ROOT / "models" / "latent_world_model_v5" / "latent_world_model_v5.pt"
INPUT_SCALER_PATH = REPO_ROOT / "data" / "processed" / "v5_multistep_K5" / "v5_input_scaler.pkl"
TARGET_SCALER_PATH = REPO_ROOT / "data" / "processed" / "v5_multistep_K5" / "v5_target_scaler.pkl"

# Contract constants. These are ASSERTED against the checkpoint and the
# scalers at load time rather than trusted.
EXPECTED_INPUT_DIM = 82
EXPECTED_LATENT_DIM = 32
EXPECTED_OUTPUT_DIM = 26
EXPECTED_SEQUENCE_LENGTH = 10
EXPECTED_MAX_K = 5
WINDOW_SECONDS = 10.0

# Dropout probability inside the attack head. Dropout carries no
# parameters, so its rate cannot be recovered from the checkpoint; 0.10
# matches the V4 head this architecture descends from. It is inert under
# eval(), so inference is unaffected either way.
ATTACK_HEAD_DROPOUT = 0.10


class V5InferenceError(RuntimeError):
    """Raised when a checkpoint, scaler or input violates the V5 contract."""


# =============================================================
# MODEL
# =============================================================


class LatentWorldModelV5(nn.Module):
    """The V5 latent world model: LSTM encoder -> 32-D latent, gated
    residual transition, state decoder and attack head.

    Module and parameter names reproduce the checkpoint's `state_dict`
    keys exactly so that `load_state_dict(..., strict=True)` succeeds with
    no missing, unexpected or reshaped tensors.
    """

    def __init__(
        self,
        input_size: int = EXPECTED_INPUT_DIM,
        hidden_size: int = 128,
        latent_size: int = EXPECTED_LATENT_DIM,
        output_size: int = EXPECTED_OUTPUT_DIM,
        num_layers: int = 2,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.output_size = output_size
        self.num_layers = num_layers

        # ----- Encoder -----
        self.encoder_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=(dropout if num_layers > 1 else 0.0),
        )
        self.encoder_norm = nn.LayerNorm(hidden_size)
        self.encoder_fc = nn.Linear(hidden_size, latent_size)

        # ----- Transition (gated residual + LayerNorm) -----
        # transition_mlp.0 = Linear(32->128), .2 = Linear(128->32); index 1
        # is the ReLU, which is why the checkpoint has no `.1` weights.
        self.transition_mlp = nn.Sequential(
            nn.Linear(latent_size, latent_size * 4),
            nn.ReLU(),
            nn.Linear(latent_size * 4, latent_size),
        )
        self.gate_fc = nn.Linear(latent_size, latent_size)
        self.transition_norm = nn.LayerNorm(latent_size)

        # ----- Decoder -----
        self.decoder_fc1 = nn.Linear(latent_size, 64)
        self.decoder_activation = nn.ReLU()
        self.decoder_fc2 = nn.Linear(64, output_size)

        # ----- Attack head -----
        # Params sit at .0 and .3; .1 (ReLU) and .2 (Dropout) are
        # parameter-free, matching the checkpoint's key layout.
        self.attack_head = nn.Sequential(
            nn.Linear(latent_size, 32),
            nn.ReLU(),
            nn.Dropout(ATTACK_HEAD_DROPOUT),
            nn.Linear(32, 1),
        )

    # ---------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, input_size] -> latent z_t [B, latent_size]."""
        output, _ = self.encoder_lstm(x)
        last_hidden = output[:, -1, :]
        last_hidden = self.encoder_norm(last_hidden)
        return self.encoder_fc(last_hidden)

    def transition_step(self, latent_state: torch.Tensor) -> torch.Tensor:
        """One learned step forward in latent space (gated residual)."""
        candidate = self.transition_mlp(latent_state)
        gate = torch.sigmoid(self.gate_fc(latent_state))
        z_next = gate * candidate + (1.0 - gate) * latent_state
        return self.transition_norm(z_next)

    def decode(self, latent_state: torch.Tensor) -> torch.Tensor:
        """Latent -> predicted 26-D network state, in TARGET-SCALED units."""
        hidden = self.decoder_fc1(latent_state)
        hidden = self.decoder_activation(hidden)
        return self.decoder_fc2(hidden)

    def attack_logit(self, latent_state: torch.Tensor) -> torch.Tensor:
        """Latent -> attack LOGIT [B]. Deliberately not a probability:
        sigmoid is applied once, by the inference layer."""
        return self.attack_head(latent_state).squeeze(-1)

    def rollout(
        self, x: torch.Tensor, steps: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode `x`, then recursively apply the transition `steps` times.

        Returns (current_latent, future_latents, future_states,
        future_logits) with shapes [B, L], [B, steps, L],
        [B, steps, output_size] and [B, steps]. Only FUTURE steps are
        returned; the nowcast latent is handed back separately as
        `current_latent`.
        """
        if steps < 1:
            raise V5InferenceError(f"steps must be >= 1, got {steps}")

        z_current = self.encode(x)

        latents: list[torch.Tensor] = []
        states: list[torch.Tensor] = []
        logits: list[torch.Tensor] = []

        z = z_current
        for _ in range(steps):
            z = self.transition_step(z)
            latents.append(z)
            states.append(self.decode(z))
            logits.append(self.attack_logit(z))

        return (
            z_current,
            torch.stack(latents, dim=1),
            torch.stack(states, dim=1),
            torch.stack(logits, dim=1),
        )


# =============================================================
# RESULT CONTAINER
# =============================================================


@dataclass
class V5Prediction:
    """A V5 forecast.

    Shapes (B = batch, K = rollout steps):
        current_latent          (B, 32)      latent encoding of the input
        future_latents          (B, K, 32)   rolled-out latents
        future_states           (B, K, 26)   ORIGINAL feature units
        future_states_scaled    (B, K, 26)   raw decoder output (debug)
        threat_probabilities    (B, K)       in [0, 1]
        attack_flags            (B, K)       bool, prob >= threshold
    """

    current_latent: np.ndarray
    future_latents: np.ndarray
    future_states: np.ndarray
    future_states_scaled: np.ndarray
    threat_probabilities: np.ndarray
    attack_flags: np.ndarray
    threshold: float
    state_feature_names: tuple[str, ...]
    target_inverse_applied: bool

    @property
    def batch_size(self) -> int:
        return int(self.current_latent.shape[0])

    @property
    def steps(self) -> int:
        return int(self.future_latents.shape[1])

    def shapes(self) -> dict[str, tuple[int, ...]]:
        """Shape map, for logging and tests."""
        return {
            "current_latent": tuple(self.current_latent.shape),
            "future_latents": tuple(self.future_latents.shape),
            "future_states": tuple(self.future_states.shape),
            "future_states_scaled": tuple(self.future_states_scaled.shape),
            "threat_probabilities": tuple(self.threat_probabilities.shape),
            "attack_flags": tuple(self.attack_flags.shape),
        }


# =============================================================
# CHECKPOINT COMPATIBILITY
# =============================================================


def verify_state_dict_compatibility(
    model: nn.Module, state_dict: dict[str, torch.Tensor]
) -> None:
    """Raise a descriptive error if `state_dict` does not match `model`
    key-for-key and shape-for-shape.

    `load_state_dict(strict=True)` would also fail, but with a message
    that is hard to act on; this reports exactly which keys are missing,
    unexpected or mis-shaped.
    """
    own = model.state_dict()
    own_keys, ckpt_keys = set(own), set(state_dict)

    missing = sorted(own_keys - ckpt_keys)
    unexpected = sorted(ckpt_keys - own_keys)
    mismatched = [
        (k, tuple(own[k].shape), tuple(state_dict[k].shape))
        for k in sorted(own_keys & ckpt_keys)
        if tuple(own[k].shape) != tuple(state_dict[k].shape)
    ]

    if missing or unexpected or mismatched:
        parts = ["V5 checkpoint is incompatible with LatentWorldModelV5:"]
        if missing:
            parts.append(f"  missing from checkpoint : {missing}")
        if unexpected:
            parts.append(f"  unexpected in checkpoint: {unexpected}")
        if mismatched:
            for key, expected, found in mismatched:
                parts.append(f"  shape mismatch {key}: model {expected} vs ckpt {found}")
        raise V5InferenceError("\n".join(parts))


def _scaler_feature_names(scaler: Any) -> tuple[str, ...]:
    """Feature order recorded on a fitted scaler, or () if it has none.

    `feature_names_in_` is a numpy array, so it must be tested for
    presence explicitly rather than for truthiness.
    """
    names = getattr(scaler, "feature_names_in_", None)
    if names is None:
        return ()
    return tuple(str(name) for name in np.asarray(names).ravel())


def _load_scaler(path: Path, expected_features: int, label: str) -> Any:
    """Unpickle a StandardScaler and check its width.

    The scalers were fitted under scikit-learn 1.6.1; unpickling under a
    newer build emits InconsistentVersionWarning. That warning is
    surfaced (never suppressed) as a non-fatal note.
    """
    if not path.exists():
        raise V5InferenceError(f"{label} scaler not found: {path}")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with open(path, "rb") as handle:
            scaler = pickle.load(handle)
        for entry in caught:
            if "InconsistentVersionWarning" in type(entry.message).__name__:
                warnings.warn(
                    f"{label} scaler ({path.name}) was pickled by a different "
                    f"scikit-learn build: {entry.message}",
                    RuntimeWarning,
                    stacklevel=2,
                )

    n_features = getattr(scaler, "n_features_in_", None)
    if n_features != expected_features:
        raise V5InferenceError(
            f"{label} scaler has n_features_in_={n_features}, "
            f"expected {expected_features} ({path})"
        )
    return scaler


# =============================================================
# INFERENCE WRAPPER
# =============================================================


class V5Inference:
    """Validated CPU inference for the V5 latent world model.

    Owns the checkpoint, both scalers and the frozen threshold. The model
    is placed in eval() mode once and every forward pass runs under
    `torch.no_grad()`, so weights are never modified.

        engine = V5Inference()
        engine.load_model()
        result = engine.predict(sequence)          # (10, 82) or (B, 10, 82)
    """

    def __init__(
        self,
        checkpoint_path: Path = CHECKPOINT_PATH,
        input_scaler_path: Path = INPUT_SCALER_PATH,
        target_scaler_path: Path = TARGET_SCALER_PATH,
        device: str = "cpu",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.input_scaler_path = Path(input_scaler_path)
        self.target_scaler_path = Path(target_scaler_path)
        self.device = torch.device(device)

        self.model: Optional[LatentWorldModelV5] = None
        self.checkpoint: Optional[dict] = None
        self.input_scaler: Any = None
        self.target_scaler: Any = None

        self.input_dim: int = EXPECTED_INPUT_DIM
        self.latent_dim: int = EXPECTED_LATENT_DIM
        self.output_dim: int = EXPECTED_OUTPUT_DIM
        self.sequence_length: int = EXPECTED_SEQUENCE_LENGTH
        self.max_k: int = EXPECTED_MAX_K
        self.threshold: float = 0.5

        self.input_feature_names: tuple[str, ...] = ()
        self.state_feature_names: tuple[str, ...] = ()

    # ---------------------------------------------------------
    # LOADING
    # ---------------------------------------------------------

    def load_model(self) -> "V5Inference":
        """Load checkpoint + scalers, rebuild the model, strict-load the
        weights and freeze into eval mode. Returns self for chaining."""
        if not self.checkpoint_path.exists():
            raise V5InferenceError(f"V5 checkpoint not found: {self.checkpoint_path}")

        checkpoint = torch.load(
            str(self.checkpoint_path), map_location="cpu", weights_only=False
        )
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise V5InferenceError(
                f"{self.checkpoint_path} is not a V5 training checkpoint "
                "(no 'model_state_dict' key)"
            )

        declared = checkpoint.get("model_class")
        if declared not in (None, "LatentWorldModelV5"):
            raise V5InferenceError(
                f"checkpoint model_class = {declared!r}, expected 'LatentWorldModelV5'"
            )

        self.input_dim = int(checkpoint.get("input_size", EXPECTED_INPUT_DIM))
        self.latent_dim = int(checkpoint.get("latent_size", EXPECTED_LATENT_DIM))
        self.output_dim = int(checkpoint.get("output_size", EXPECTED_OUTPUT_DIM))
        self.sequence_length = int(
            checkpoint.get("sequence_length", EXPECTED_SEQUENCE_LENGTH)
        )
        self.max_k = int(checkpoint.get("max_k", EXPECTED_MAX_K))
        # Frozen, taken verbatim. No threshold search happens in this module.
        self.threshold = float(checkpoint.get("selected_threshold", 0.5))

        for name, found, expected in (
            ("input_size", self.input_dim, EXPECTED_INPUT_DIM),
            ("latent_size", self.latent_dim, EXPECTED_LATENT_DIM),
            ("output_size", self.output_dim, EXPECTED_OUTPUT_DIM),
            ("sequence_length", self.sequence_length, EXPECTED_SEQUENCE_LENGTH),
        ):
            if found != expected:
                raise V5InferenceError(
                    f"checkpoint {name}={found} violates the V5 contract "
                    f"(expected {expected})"
                )

        model = LatentWorldModelV5(
            input_size=self.input_dim,
            hidden_size=int(checkpoint.get("hidden_size", 128)),
            latent_size=self.latent_dim,
            output_size=self.output_dim,
            num_layers=int(checkpoint.get("num_layers", 2)),
            dropout=float(checkpoint.get("dropout", 0.20)),
        )

        state_dict = checkpoint["model_state_dict"]
        verify_state_dict_compatibility(model, state_dict)
        model.load_state_dict(state_dict, strict=True)

        model.to(self.device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        self.model = model
        self.checkpoint = checkpoint

        # ----- scalers -----
        self.input_scaler = _load_scaler(
            self.input_scaler_path, self.input_dim, "V5 input"
        )
        self.target_scaler = _load_scaler(
            self.target_scaler_path, self.output_dim, "V5 target"
        )

        self.input_feature_names = _scaler_feature_names(self.input_scaler)
        self.state_feature_names = _scaler_feature_names(self.target_scaler)

        return self

    def _require_model(self) -> LatentWorldModelV5:
        if self.model is None:
            raise V5InferenceError("model not loaded — call load_model() first")
        return self.model

    @property
    def parameter_count(self) -> int:
        """Total parameter count of the loaded model."""
        return sum(p.numel() for p in self._require_model().parameters())

    # ---------------------------------------------------------
    # INPUT VALIDATION
    # ---------------------------------------------------------

    def _validate_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """Coerce to float64 [B, T, D] and reject anything off-contract.

        Nothing is imputed, clipped or reordered: a bad input is an error,
        never a silently repaired one.
        """
        array = np.asarray(sequence)

        if not np.issubdtype(array.dtype, np.number):
            raise V5InferenceError(
                f"sequence must be numeric, got dtype {array.dtype}"
            )

        if array.ndim == 2:
            array = array[None, ...]
        elif array.ndim != 3:
            raise V5InferenceError(
                f"sequence must have shape ({self.sequence_length}, {self.input_dim}) "
                f"or (B, {self.sequence_length}, {self.input_dim}); got ndim="
                f"{array.ndim} shape={tuple(np.shape(sequence))}"
            )

        _, timesteps, features = array.shape
        if timesteps != self.sequence_length:
            raise V5InferenceError(
                f"expected {self.sequence_length} timesteps, got {timesteps}"
            )
        if features != self.input_dim:
            raise V5InferenceError(
                f"expected {self.input_dim} features per timestep, got {features}"
            )

        array = array.astype(np.float64, copy=True)

        if not np.isfinite(array).all():
            n_nan = int(np.isnan(array).sum())
            n_inf = int(np.isinf(array).sum())
            bad = np.argwhere(~np.isfinite(array))
            first = tuple(int(i) for i in bad[0])
            raise V5InferenceError(
                f"sequence contains non-finite values ({n_nan} NaN, {n_inf} Inf); "
                f"first at (batch, timestep, feature)={first}. Missing packet "
                "features are not imputed — supply a complete window."
            )

        return array

    def _validate_feature_names(self, feature_names: Sequence[str]) -> None:
        """Assert the caller's feature order matches the V5 input scaler's."""
        if not self.input_feature_names:
            raise V5InferenceError(
                "input scaler carries no feature_names_in_; cannot validate order"
            )
        supplied = tuple(str(name) for name in feature_names)
        if supplied != self.input_feature_names:
            if len(supplied) != len(self.input_feature_names):
                raise V5InferenceError(
                    f"feature_names has {len(supplied)} entries, "
                    f"scaler expects {len(self.input_feature_names)}"
                )
            diffs = [
                f"    [{i}] got {got!r}, expected {want!r}"
                for i, (got, want) in enumerate(
                    zip(supplied, self.input_feature_names)
                )
                if got != want
            ]
            raise V5InferenceError(
                "feature order does not match the V5 input scaler "
                "(features are never reordered automatically):\n"
                + "\n".join(diffs[:10])
            )

    def _validate_timestamps(self, timestamps: Sequence[float]) -> None:
        """Reject a window that is not 10 s contiguous.

        Mirrors the contiguity rule the training sequences were built
        under: gaps are rejected, never stitched.
        """
        stamps = np.asarray(timestamps, dtype=np.float64).ravel()
        if stamps.size != self.sequence_length:
            raise V5InferenceError(
                f"timestamps has {stamps.size} entries, "
                f"expected {self.sequence_length}"
            )
        if not np.isfinite(stamps).all():
            raise V5InferenceError("timestamps contains non-finite values")

        diffs = np.diff(stamps)
        if not np.all(diffs == WINDOW_SECONDS):
            offenders = [
                f"    step {i}->{i + 1}: {float(d)}s"
                for i, d in enumerate(diffs)
                if d != WINDOW_SECONDS
            ]
            raise V5InferenceError(
                f"non-contiguous temporal window (expected {WINDOW_SECONDS}s "
                "between consecutive steps):\n" + "\n".join(offenders[:10])
            )

    # ---------------------------------------------------------
    # SCALING
    # ---------------------------------------------------------

    @staticmethod
    def _suppress_feature_name_warning() -> warnings.catch_warnings:
        """Silence sklearn's 'X does not have valid feature names' notice.

        The scalers were fitted from DataFrames, so passing a bare ndarray
        triggers it. Column order is enforced here instead — by
        `_validate_feature_names` when the caller supplies names, and by
        the fixed 82-D contract otherwise — so the warning is noise, not
        signal. Only this one message is filtered.
        """
        context = warnings.catch_warnings()
        context.__enter__()
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names",
            category=UserWarning,
        )
        return context

    def _scale_input(self, array: np.ndarray) -> np.ndarray:
        """Apply the V5 input scaler across [B, T, 82] without reordering."""
        batch, timesteps, features = array.shape
        flat = array.reshape(batch * timesteps, features)
        context = self._suppress_feature_name_warning()
        try:
            scaled = np.asarray(self.input_scaler.transform(flat), dtype=np.float64)
        finally:
            context.__exit__(None, None, None)
        return scaled.reshape(batch, timesteps, features)

    def _inverse_target(self, states_scaled: np.ndarray) -> tuple[np.ndarray, bool]:
        """Map decoder output back to original state units.

        Returns (states, applied). `applied` is False only if the target
        scaler exposes no inverse_transform, in which case the scaled
        values are passed through unchanged rather than guessed at.
        """
        if not hasattr(self.target_scaler, "inverse_transform"):
            return states_scaled.copy(), False

        batch, steps, features = states_scaled.shape
        flat = states_scaled.reshape(batch * steps, features)
        context = self._suppress_feature_name_warning()
        try:
            restored = np.asarray(
                self.target_scaler.inverse_transform(flat), dtype=np.float64
            )
        finally:
            context.__exit__(None, None, None)
        return restored.reshape(batch, steps, features), True

    # ---------------------------------------------------------
    # PUBLIC API
    # ---------------------------------------------------------

    def k_step_rollout(
        self,
        sequence: np.ndarray,
        steps: int = EXPECTED_MAX_K,
        feature_names: Optional[Sequence[str]] = None,
        timestamps: Optional[Sequence[float]] = None,
        already_scaled: bool = False,
    ) -> V5Prediction:
        """Roll the world model `steps` windows into the future.

        Args:
            sequence: raw 82-D state vectors, (10, 82) or (B, 10, 82), in
                the V5 input scaler's feature order.
            steps: rollout length. Beyond `max_k` (5) the checkpoint is
                outside its training curriculum, so a warning is emitted.
            feature_names: optional; if given, its order is checked
                against the scaler and a mismatch is an error.
            timestamps: optional per-window epoch seconds for the single
                -sequence case; checked for 10 s contiguity.
            already_scaled: set True only if `sequence` is already in
                scaler space; input scaling is then skipped.

        Returns:
            V5Prediction covering the FUTURE steps only.
        """
        model = self._require_model()

        if steps > self.max_k:
            warnings.warn(
                f"steps={steps} exceeds the checkpoint's trained max_k="
                f"{self.max_k}; forecasts beyond that horizon are "
                "extrapolation, not validated behaviour.",
                RuntimeWarning,
                stacklevel=2,
            )

        array = self._validate_sequence(sequence)

        if feature_names is not None:
            self._validate_feature_names(feature_names)

        if timestamps is not None:
            if array.shape[0] != 1:
                raise V5InferenceError(
                    "timestamps validation supports a single sequence only; "
                    f"got batch of {array.shape[0]}"
                )
            self._validate_timestamps(timestamps)

        scaled = array if already_scaled else self._scale_input(array)

        tensor = torch.from_numpy(scaled).to(dtype=torch.float32, device=self.device)

        with torch.no_grad():
            z0, latents, states_scaled, logits = model.rollout(tensor, steps)
            probabilities = torch.sigmoid(logits)

        z0_np = z0.cpu().numpy()
        latents_np = latents.cpu().numpy()
        states_scaled_np = states_scaled.cpu().numpy().astype(np.float64)
        probabilities_np = probabilities.cpu().numpy().astype(np.float64)

        if not np.isfinite(probabilities_np).all():
            raise V5InferenceError(
                "model produced non-finite threat probabilities — "
                "refusing to return a corrupt forecast"
            )
        if not np.isfinite(states_scaled_np).all():
            raise V5InferenceError(
                "model produced non-finite predicted states — "
                "refusing to return a corrupt forecast"
            )

        states_np, inverse_applied = self._inverse_target(states_scaled_np)

        return V5Prediction(
            current_latent=z0_np,
            future_latents=latents_np,
            future_states=states_np,
            future_states_scaled=states_scaled_np,
            threat_probabilities=probabilities_np,
            attack_flags=probabilities_np >= self.threshold,
            threshold=self.threshold,
            state_feature_names=self.state_feature_names,
            target_inverse_applied=inverse_applied,
        )

    def predict(
        self,
        sequence: np.ndarray,
        feature_names: Optional[Sequence[str]] = None,
        timestamps: Optional[Sequence[float]] = None,
        already_scaled: bool = False,
    ) -> V5Prediction:
        """Forecast at the checkpoint's trained horizon (K = max_k = 5).

        Thin wrapper over `k_step_rollout`; see that method for argument
        semantics.
        """
        return self.k_step_rollout(
            sequence,
            steps=self.max_k,
            feature_names=feature_names,
            timestamps=timestamps,
            already_scaled=already_scaled,
        )


def load_v5_model(
    checkpoint_path: Path = CHECKPOINT_PATH,
) -> tuple[LatentWorldModelV5, dict]:
    """Rebuild the V5 model and strict-load the checkpoint.

    Returns (model in eval mode, checkpoint dict). Mirrors the
    `load_v4_model` convention for callers that want the bare module
    without the scaler/validation layer.
    """
    engine = V5Inference(checkpoint_path=checkpoint_path)
    engine.load_model()
    return engine._require_model(), engine.checkpoint  # type: ignore[return-value]


# =============================================================
# SELF-TEST
# =============================================================
#
# SHAPE AND NUMERICAL SMOKE TEST ONLY.
#
# The input below is a random tensor. It carries no traffic semantics, so
# the probabilities it produces are NOT a measurement of anything. Nothing
# here is a performance claim; real evaluation needs real 10x82 data,
# which does not yet exist in this repository.
# =============================================================


def _self_test() -> int:
    """Load the real checkpoint, then run a random-input shape check."""
    engine = V5Inference()
    engine.load_model()

    print("V5 checkpoint loaded: OK")
    print(f"Input dimension: {engine.input_dim}")
    print(f"Latent dimension: {engine.latent_dim}")
    print(f"Rollout steps: {engine.max_k}")
    print(f"Parameter count: {engine.parameter_count}")
    print("Strict state_dict load: PASS")

    print("\n--- SHAPE / NUMERICAL SMOKE TEST (random input, NOT a metric) ---")
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 10, 82))

    result = engine.predict(x)

    checks: list[tuple[str, bool, str]] = []
    expected_shapes = {
        "current_latent": (4, 32),
        "future_latents": (4, 5, 32),
        "future_states": (4, 5, 26),
        "threat_probabilities": (4, 5),
        "attack_flags": (4, 5),
    }
    actual = result.shapes()
    for name, want in expected_shapes.items():
        got = actual.get(name, tuple(np.shape(getattr(result, name))))
        checks.append((f"{name} == {want}", got == want, f"got {got}"))

    probs = result.threat_probabilities
    checks.append(
        ("threat probabilities finite", bool(np.isfinite(probs).all()), "")
    )
    checks.append(
        (
            "threat probabilities in [0, 1]",
            bool((probs >= 0.0).all() and (probs <= 1.0).all()),
            f"range [{probs.min():.6f}, {probs.max():.6f}]",
        )
    )
    checks.append(
        (
            "future states finite",
            bool(np.isfinite(result.future_states).all()),
            "",
        )
    )
    checks.append(
        (
            "future latents finite",
            bool(np.isfinite(result.future_latents).all()),
            "",
        )
    )
    checks.append(
        (
            "attack_flags == (prob >= threshold)",
            bool(np.array_equal(result.attack_flags, probs >= result.threshold)),
            f"threshold {result.threshold}",
        )
    )
    checks.append(
        (
            "target inverse_transform applied",
            result.target_inverse_applied,
            "decoder output mapped to original state units",
        )
    )

    # Rejection paths: each of these MUST raise.
    for label, bad in (
        ("rejects wrong feature count", np.zeros((10, 52))),
        ("rejects wrong sequence length", np.zeros((7, 82))),
        ("rejects NaN input", np.full((10, 82), np.nan)),
        ("rejects Inf input", np.full((10, 82), np.inf)),
    ):
        try:
            engine.predict(bad)
        except V5InferenceError:
            checks.append((label, True, ""))
        else:
            checks.append((label, False, "no error raised"))

    try:
        engine.predict(
            rng.standard_normal((10, 82)),
            timestamps=[0, 10, 20, 30, 40, 50, 60, 80, 90, 100],
        )
    except V5InferenceError:
        checks.append(("rejects non-contiguous window", True, ""))
    else:
        checks.append(("rejects non-contiguous window", False, "no error raised"))

    width = max(len(name) for name, _, _ in checks)
    failures = 0
    for name, ok, note in checks:
        if not ok:
            failures += 1
        suffix = f"  ({note})" if note else ""
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<{width}}{suffix}")

    print(
        f"\n{len(checks) - failures}/{len(checks)} checks passed"
        + ("" if failures else " — shapes and numerics only, no performance claim")
    )
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="V5 latent world model inference engine."
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="load the real checkpoint and run the shape/numerical smoke test",
    )
    args = parser.parse_args()

    if not args.self_test:
        parser.print_help()
        return 0
    return _self_test()


if __name__ == "__main__":
    raise SystemExit(main())
