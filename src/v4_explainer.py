"""
=============================================================
V4 EXPLAINABILITY CORE -- provenance-locked attribution
=============================================================

This module is the ONLY sanctioned way to attribute a V4
future-threat prediction. It is a *consumer* of the frozen V4
baseline: it loads nothing that is not named in
``docs/v4_baseline_manifest.json`` and refuses to run if any
frozen artifact's SHA256 has drifted.

It supersedes ``src/shap_explainer.py`` for anything V4. That
module is retained for the historical V3 artifacts under
``models/latent_world_model_v3/shap_analysis/`` and is NOT
provenance-locked -- see ``docs/V4_EXPLAINABILITY.md``.

------------------------------------------------------------
WHAT IS BEING EXPLAINED
------------------------------------------------------------
The V4 rollout produces K+1 probabilities per input window:

    probs[0] = P(attack at t)      -- nowcast, no transition
    probs[k] = P(attack at t+k)    -- after k latent transitions

Two target families are supported, and every explanation
records which one it used:

  TARGET A  horizon_probability(k)
            F_k(x) = sigmoid(attack_head(z_k)),
            z_k = transition_step^k(encode(x))
            This is *exactly* ``probs[:, k]`` -- the quantity the
            frozen reference evaluation thresholds for horizon k.

  TARGET B  trajectory_max(K)
            displayed value = max(probs[1..K]).
            MAX is not differentiable at ties and the argmax can
            change along the Integrated-Gradients path. This
            module therefore attributes F_{k*} where k* = argmax
            evaluated AT THE INPUT, and records
            ``argmax_switches_on_path`` as a diagnostic. The
            attributed function value at x equals the displayed
            max exactly; the completeness reference
            F(baseline) is F_{k*}(baseline), not max at the
            baseline. Documented, not hidden.

------------------------------------------------------------
WHAT AN ATTRIBUTION IS NOT
------------------------------------------------------------
Attribution measures this model's sensitivity/contribution on
this input. It does not establish causality, attacker intent,
attack stage, or actual compromise. MITRE ATT&CK wording is
produced by a SEPARATE rule-based layer (``src/mitre_mapper.py``)
and is never derived from attribution values.
=============================================================
"""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from captum.attr import IntegratedGradients

# The canonical V4 model class + checkpoint loader. Importing it
# (rather than redefining the architecture) is deliberate: a
# second copy of the class is a contamination vector.
from pcap_ingest_v4 import (
    LatentWorldModelV4,
    load_v4_model,
    CHECKPOINT_PATH,
    INPUT_SCALER_PATH,
    INPUT_FEATURE_NAMES_PATH,
    INPUT_FEATURE_NAMES,
    CANON_DIR,
)


REPO_ROOT = Path(__file__).resolve().parent.parent

MANIFEST_PATH = REPO_ROOT / "docs" / "v4_baseline_manifest.json"
FEATURE_SCHEMA_PATH = REPO_ROOT / "docs" / "v4_feature_schema.json"

SEQ_LEN = 10
N_FEATURES = 52
MAX_K = 5

# Integration steps. Chosen from the measured convergence sweep in
# docs/V4_EXPLAINABILITY.md (50 -> 1024 steps), NOT for appearance:
# 1024 is the first setting whose worst-case relative completeness
# error is below 1% at every evaluated horizon.
IG_N_STEPS = 1024
IG_METHOD = "gausslegendre"

# Frozen, validation-selected thresholds. Copied from the manifest
# at load time -- these literals are only a fallback assertion.
EXPECTED_HORIZON_THRESHOLDS = {1: 0.45, 2: 0.54, 3: 0.65, 5: 0.72}
EXPECTED_TRAJECTORY_MAX_THRESHOLDS = {2: 0.51, 3: 0.68, 5: 0.69}

# Any path matching these is a hard failure inside an explanation.
FORBIDDEN_ARTIFACT_MARKERS = (
    "latent_world_model_v1",
    "latent_world_model_v2",
    "latent_world_model_v3",
    "world_model_v2",
    "temporal_benchmark_v2",
    "rotated_v4",
    "states_v3",
    "network_states_v3",
)


# =============================================================
# HASHING / MANIFEST
# =============================================================

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_manifest() -> dict:
    with open(MANIFEST_PATH) as fh:
        return json.load(fh)


class ProvenanceError(RuntimeError):
    """Raised when a frozen V4 artifact does not match the manifest.

    This is always fatal. A provenance failure is never downgraded
    to a warning -- an explanation from an unverified checkpoint is
    worse than no explanation.
    """


# =============================================================
# PROVENANCE RECORD
# =============================================================

@dataclass(frozen=True)
class V4Provenance:
    """Everything a consumer needs to prove which model produced
    an explanation. Serialised into every output artifact."""

    model_version: str
    model_class: str
    checkpoint_path: str
    checkpoint_hash: str
    checkpoint_hash_expected: str
    feature_schema_path: str
    feature_schema_hash: str
    feature_order_hash: str
    feature_order_hash_expected: str
    scaler_path: str
    scaler_hash: str
    scaler_hash_expected: str
    input_shape: tuple
    parameter_count: int
    torch_version: str
    manifest_path: str
    manifest_hash: str

    def as_dict(self) -> dict:
        d = asdict(self)
        d["input_shape"] = list(self.input_shape)
        return d


def build_provenance(manifest: dict | None = None) -> V4Provenance:
    """Hash every frozen artifact the explainer touches and compare
    against the frozen baseline manifest. Raises ProvenanceError on
    ANY mismatch -- there is no permissive mode."""

    manifest = manifest or load_manifest()

    ckpt_expected = manifest["checkpoint"]["sha256"]
    ckpt_actual = sha256_file(CHECKPOINT_PATH)
    if ckpt_actual != ckpt_expected:
        raise ProvenanceError(
            "V4 checkpoint hash mismatch -- refusing to explain.\n"
            f"  path     : {CHECKPOINT_PATH}\n"
            f"  expected : {ckpt_expected}\n"
            f"  actual   : {ckpt_actual}"
        )

    manifest_ckpt_path = (REPO_ROOT / manifest["checkpoint"]["path"]).resolve()
    if manifest_ckpt_path != CHECKPOINT_PATH.resolve():
        raise ProvenanceError(
            "checkpoint path disagrees with the manifest.\n"
            f"  manifest : {manifest_ckpt_path}\n"
            f"  loader   : {CHECKPOINT_PATH.resolve()}"
        )

    scaler_expected = manifest["scalers"]["input_scaler"]["sha256"]
    scaler_actual = sha256_file(INPUT_SCALER_PATH)
    if scaler_actual != scaler_expected:
        raise ProvenanceError(
            "V4 input scaler hash mismatch -- refusing to explain.\n"
            f"  path     : {INPUT_SCALER_PATH}\n"
            f"  expected : {scaler_expected}\n"
            f"  actual   : {scaler_actual}"
        )

    order_expected = manifest["input"]["feature_order_sha256"]
    order_actual = sha256_file(INPUT_FEATURE_NAMES_PATH)
    if order_actual != order_expected:
        raise ProvenanceError(
            "V4 feature-order file hash mismatch -- refusing to explain.\n"
            f"  path     : {INPUT_FEATURE_NAMES_PATH}\n"
            f"  expected : {order_expected}\n"
            f"  actual   : {order_actual}"
        )

    # The frozen schema copy must describe the same ordering.
    with open(FEATURE_SCHEMA_PATH) as fh:
        schema_text = fh.read()
    schema = json.loads(schema_text)
    if schema["source_sha256"] != order_actual:
        raise ProvenanceError(
            "docs/v4_feature_schema.json was frozen from a different "
            "feature-order file.\n"
            f"  schema.source_sha256 : {schema['source_sha256']}\n"
            f"  input_feature_names  : {order_actual}"
        )
    schema_names = [f["name"] for f in schema["features"]]
    if schema_names != INPUT_FEATURE_NAMES:
        raise ProvenanceError(
            "schema feature ORDER differs from "
            "input_feature_names.txt -- attribution indices would be "
            "mislabelled."
        )
    if [f["index"] for f in schema["features"]] != list(range(N_FEATURES)):
        raise ProvenanceError("schema feature indices are not 0..51 in order")
    if tuple(schema["input_shape"]) != (SEQ_LEN, N_FEATURES):
        raise ProvenanceError(
            f"schema input_shape {schema['input_shape']} != [10, 52]"
        )

    ckpt_meta = manifest["checkpoint"]

    return V4Provenance(
        model_version="V4",
        model_class=ckpt_meta["model_class"],
        checkpoint_path=str(CHECKPOINT_PATH.relative_to(REPO_ROOT)),
        checkpoint_hash=ckpt_actual,
        checkpoint_hash_expected=ckpt_expected,
        feature_schema_path=str(FEATURE_SCHEMA_PATH.relative_to(REPO_ROOT)),
        feature_schema_hash=sha256_text(schema_text),
        feature_order_hash=order_actual,
        feature_order_hash_expected=order_expected,
        scaler_path=str(INPUT_SCALER_PATH.relative_to(REPO_ROOT)),
        scaler_hash=scaler_actual,
        scaler_hash_expected=scaler_expected,
        input_shape=(SEQ_LEN, N_FEATURES),
        parameter_count=ckpt_meta["parameter_count"],
        torch_version=torch.__version__,
        manifest_path=str(MANIFEST_PATH.relative_to(REPO_ROOT)),
        manifest_hash=sha256_file(MANIFEST_PATH),
    )


# =============================================================
# TARGET DEFINITIONS
# =============================================================

@dataclass(frozen=True)
class ExplanationTarget:
    """A single, unambiguous scalar the explainer attributes."""

    target_type: str          # horizon_probability | trajectory_max
    target_horizon: int       # k for TARGET A; K for TARGET B
    target_definition: str    # human-readable formula
    threshold: float          # the frozen threshold for this target
    attributed_horizon: int   # the k actually differentiated

    def as_dict(self) -> dict:
        return asdict(self)


def horizon_target(k: int, manifest: dict) -> ExplanationTarget:
    thr = manifest["threshold"]["per_horizon_future_prediction"].get(str(k))
    if thr is None:
        raise ValueError(
            f"no frozen threshold for horizon k={k}; the reference "
            f"evaluation covers k in "
            f"{sorted(manifest['threshold']['per_horizon_future_prediction'])}"
        )
    return ExplanationTarget(
        target_type="horizon_probability",
        target_horizon=k,
        target_definition=(
            f"P(attack at t+{k}) = sigmoid(attack_head(z_{k})), "
            f"z_{k} = transition_step^{k}(encode(x)); "
            f"identical to k_step_rollout(x, K).probs[{k}]"
        ),
        threshold=float(thr),
        attributed_horizon=k,
    )


def trajectory_max_target(
    K: int, manifest: dict, argmax_k: int
) -> ExplanationTarget:
    thr = manifest["threshold"]["per_horizon_trajectory_max"].get(str(K))
    if thr is None:
        raise ValueError(f"no frozen trajectory-max threshold for K={K}")
    return ExplanationTarget(
        target_type="trajectory_max",
        target_horizon=K,
        target_definition=(
            f"MAX(P(t+1)..P(t+{K})); attributed as the fixed horizon "
            f"k*={argmax_k} selected by argmax AT THE INPUT. MAX is "
            f"non-differentiable at ties and the argmax may change "
            f"along the integration path (recorded per sample as "
            f"argmax_switches_on_path)."
        ),
        threshold=float(thr),
        attributed_horizon=argmax_k,
    )


# =============================================================
# DIFFERENTIABLE TARGET WRAPPER
# =============================================================

class HorizonProbability(nn.Module):
    """Input (B, 10, 52) -> (B, 1) probability at t+k.

    Built from the model's own encode / transition_step /
    predict_attack, so it is the SAME computation the rollout
    performs (verified bit-exact by ``check_prediction_identity``).
    """

    def __init__(self, model: LatentWorldModelV4, k: int):
        super().__init__()
        if k < 1:
            raise ValueError("horizon k must be >= 1 (k=0 is the nowcast)")
        self.model = model
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.model.encode(x)
        for _ in range(self.k):
            z = self.model.transition_step(z)
        return torch.sigmoid(self.model.predict_attack(z)).unsqueeze(-1)


# =============================================================
# BASELINES
# =============================================================

def zero_baseline(batch: torch.Tensor) -> torch.Tensor:
    """All-zeros in SCALED feature space.

    Principled, not cosmetic: the frozen StandardScaler subtracts
    the training-partition mean, so the zero vector IS the average
    training window, held constant across all 10 timesteps. It is
    label-free (no benign/attack label participates), exactly
    reproducible, and independent of the test set.

    Caveat recorded in the contract: a constant-over-time window is
    not itself a realisable observation -- a real 10-window history
    of the train mean would have zero relative change, which the
    zero vector also encodes (rel-change features are scaled about
    their own train mean, not exactly 0 raw). This is the standard
    IG reference-point compromise and is reported, not hidden.
    """
    return torch.zeros_like(batch)


def benign_mean_baseline_array() -> np.ndarray:
    """Mean over TRAINING sequences with no attack at any of t+1..t+5.

    Uses training labels only -- never test labels. Provided as a
    baseline-sensitivity check, not as the primary reference.
    """
    X = np.load(CANON_DIR / "X_train.npy")
    A = np.load(CANON_DIR / "A_binary_multistep_train.npy")
    benign = A.sum(axis=1) == 0
    if benign.sum() == 0:
        raise RuntimeError("no fully-benign training sequences found")
    return X[benign].mean(axis=0, keepdims=True).astype(np.float32)


# =============================================================
# THE EXPLAINER
# =============================================================

class V4Explainer:
    """Provenance-locked Integrated Gradients for the frozen V4
    checkpoint. Construction fails loudly if provenance fails."""

    def __init__(self, device: str = "cpu", n_steps: int = IG_N_STEPS):
        self.manifest = load_manifest()
        self.provenance = build_provenance(self.manifest)

        self.device = torch.device(device)
        self.n_steps = int(n_steps)
        self.method = IG_METHOD

        model, ckpt = load_v4_model(CHECKPOINT_PATH)
        if not isinstance(model, LatentWorldModelV4):
            raise ProvenanceError(
                f"loaded model is {type(model).__name__}, not "
                f"LatentWorldModelV4"
            )
        self.model = model.to(self.device)
        self.model.eval()          # dropout (0.20 LSTM, 0.10 head) disabled
        self.checkpoint = ckpt

        if ckpt["input_size"] != N_FEATURES:
            raise ProvenanceError(
                f"checkpoint input_size={ckpt['input_size']} != 52"
            )
        n_params = sum(p.numel() for p in self.model.parameters())
        if n_params != self.provenance.parameter_count:
            raise ProvenanceError(
                f"parameter count {n_params} != manifest "
                f"{self.provenance.parameter_count}"
            )
        for m in self.model.modules():
            if isinstance(m, nn.Dropout) and m.training:
                raise ProvenanceError("a Dropout module is still in train mode")

        self.feature_names = list(INPUT_FEATURE_NAMES)
        self.time_labels = [f"t-{SEQ_LEN - 1 - i}" for i in range(SEQ_LEN)]
        self.time_labels[-1] = "t"

        # Frozen thresholds, read from the manifest (asserted against
        # the literals so a silent manifest edit is caught).
        thr = self.manifest["threshold"]
        self.horizon_thresholds = {
            int(k): float(v)
            for k, v in thr["per_horizon_future_prediction"].items()
        }
        self.trajectory_max_thresholds = {
            int(k): float(v)
            for k, v in thr["per_horizon_trajectory_max"].items()
        }
        if self.horizon_thresholds != EXPECTED_HORIZON_THRESHOLDS:
            raise ProvenanceError(
                f"per-horizon thresholds changed: {self.horizon_thresholds}"
            )
        if self.trajectory_max_thresholds != EXPECTED_TRAJECTORY_MAX_THRESHOLDS:
            raise ProvenanceError(
                f"trajectory-max thresholds changed: "
                f"{self.trajectory_max_thresholds}"
            )

    # ---------------------------------------------------------
    # INFERENCE (the reference path)
    # ---------------------------------------------------------

    def _as_tensor(self, X: np.ndarray) -> torch.Tensor:
        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[None, ...]
        if arr.shape[1:] != (SEQ_LEN, N_FEATURES):
            raise ValueError(
                f"input shape {arr.shape} -- expected (batch, 10, 52)"
            )
        return torch.from_numpy(arr).to(self.device)

    def predict_rollout(self, X: np.ndarray, K: int = MAX_K) -> np.ndarray:
        """The V4 inference path. Returns (N, K+1) probabilities where
        column k is P(attack at t+k). Column 0 is the nowcast."""
        xb = self._as_tensor(X)
        with torch.no_grad():
            _, probs, _ = self.model.k_step_rollout(xb, K)
        return torch.stack(probs, dim=1).cpu().numpy()

    # ---------------------------------------------------------
    # ATTRIBUTION
    # ---------------------------------------------------------

    def attribute(
        self,
        X: np.ndarray,
        k: int,
        baseline: np.ndarray | None = None,
        n_steps: int | None = None,
    ) -> dict:
        """Integrated Gradients for the horizon-k probability.

        Returns a dict carrying the attribution tensor (N, 10, 52),
        the explained prediction, the baseline prediction, and the
        completeness error -- everything needed to audit the result.
        """
        xb = self._as_tensor(X)
        base = (
            zero_baseline(xb)
            if baseline is None
            else self._as_tensor(baseline).expand_as(xb)
        )
        n_steps = int(n_steps or self.n_steps)

        fn = HorizonProbability(self.model, k).to(self.device)
        fn.eval()
        ig = IntegratedGradients(fn)

        # Gradients must flow: the model is in eval() but grad is
        # required through the input. Captum handles requires_grad
        # on the input; parameters stay frozen (never stepped).
        attrs, delta = ig.attribute(
            xb,
            baselines=base,
            n_steps=n_steps,
            method=self.method,
            return_convergence_delta=True,
            internal_batch_size=max(1, 4096 // max(1, xb.shape[0])) * xb.shape[0],
        )

        with torch.no_grad():
            f_x = fn(xb).squeeze(-1)
            f_base = fn(base).squeeze(-1)

        attrs_np = attrs.detach().cpu().numpy()
        f_x_np = f_x.cpu().numpy()
        f_base_np = f_base.cpu().numpy()
        attr_sum = attrs_np.sum(axis=(1, 2))
        target_delta = f_x_np - f_base_np
        abs_err = np.abs(attr_sum - target_delta)
        rel_err = abs_err / np.maximum(np.abs(target_delta), 1e-12)

        if attrs_np.shape[1:] != (SEQ_LEN, N_FEATURES):
            raise RuntimeError(
                f"attribution shape {attrs_np.shape} != (N, 10, 52)"
            )

        return {
            "attributions": attrs_np,
            "prediction": f_x_np,
            "baseline_prediction": f_base_np,
            "target_delta": target_delta,
            "attribution_sum": attr_sum,
            "completeness_abs_error": abs_err,
            "completeness_rel_error": rel_err,
            "captum_convergence_delta": delta.detach().cpu().numpy(),
            "n_steps": n_steps,
            "method": self.method,
            "horizon": k,
        }

    # ---------------------------------------------------------
    # TRAJECTORY-MAX TARGET
    # ---------------------------------------------------------

    def trajectory_max_info(self, X: np.ndarray, K: int = MAX_K) -> dict:
        """argmax horizon, displayed max, tie count, and whether the
        argmax changes along the straight-line IG path."""
        probs = self.predict_rollout(X, K)
        fut = probs[:, 1:]
        argmax_k = fut.argmax(axis=1) + 1
        max_p = fut.max(axis=1)
        ties = (np.abs(fut - max_p[:, None]) == 0.0).sum(axis=1)

        xb = self._as_tensor(X)
        switches = np.zeros(xb.shape[0], dtype=bool)
        alphas = torch.linspace(0.0, 1.0, 64, device=self.device)
        with torch.no_grad():
            for i in range(xb.shape[0]):
                path = alphas[:, None, None] * xb[i][None]
                _, pp, _ = self.model.k_step_rollout(path, K)
                am = torch.stack(pp, dim=1)[:, 1:].argmax(dim=1)
                switches[i] = bool(len(torch.unique(am)) > 1)

        return {
            "argmax_horizon": argmax_k,
            "max_probability": max_p,
            "n_tied_horizons": ties,
            "argmax_switches_on_path": switches,
        }

    # ---------------------------------------------------------
    # AGGREGATIONS  (sign is never discarded)
    # ---------------------------------------------------------

    @staticmethod
    def feature_aggregate(attrs: np.ndarray) -> dict:
        """(N, 10, 52) -> per-feature signed and absolute importance."""
        return {
            "signed": attrs.sum(axis=1),          # (N, 52) sum over time
            "absolute": np.abs(attrs).sum(axis=1),
        }

    @staticmethod
    def temporal_aggregate(attrs: np.ndarray) -> dict:
        """(N, 10, 52) -> per-timestep signed and absolute importance."""
        return {
            "signed": attrs.sum(axis=2),          # (N, 10) sum over features
            "absolute": np.abs(attrs).sum(axis=2),
        }

    # ---------------------------------------------------------
    # FEATURE ABLATION CROSS-CHECK
    # ---------------------------------------------------------

    def ablate_feature(
        self,
        x: np.ndarray,
        feature_index: int,
        k: int,
        baseline: np.ndarray | None = None,
    ) -> dict:
        """Set one feature to its baseline value across all 10
        timesteps and re-run the SAME prediction function.

        Model-agnostic: uses no gradients. This is a directional
        sanity check on IG, not a replacement for it.
        """
        xb = self._as_tensor(x)
        base = (
            zero_baseline(xb)
            if baseline is None
            else self._as_tensor(baseline).expand_as(xb)
        )
        perturbed = xb.clone()
        perturbed[:, :, feature_index] = base[:, :, feature_index]

        fn = HorizonProbability(self.model, k).to(self.device)
        fn.eval()
        with torch.no_grad():
            p_orig = fn(xb).squeeze(-1).cpu().numpy()
            p_abl = fn(perturbed).squeeze(-1).cpu().numpy()
        return {
            "p_original": p_orig,
            "p_ablated": p_abl,
            "delta": p_orig - p_abl,   # +ve == feature was pushing prob UP
        }

    # ---------------------------------------------------------
    # PREDICTION-IDENTITY TEST
    # ---------------------------------------------------------

    def check_prediction_identity(
        self, X: np.ndarray, k: int, tolerance: float = 1e-9
    ) -> dict:
        """Compare the value the explainer differentiates against the
        value the V4 rollout produces, on the SAME tensor in the SAME
        batch. Any non-zero difference here means the explainer is
        explaining something other than the displayed prediction."""
        xb = self._as_tensor(X)
        fn = HorizonProbability(self.model, k).to(self.device)
        fn.eval()
        with torch.no_grad():
            p_explain = fn(xb).squeeze(-1)
            _, probs, _ = self.model.k_step_rollout(xb, MAX_K)
            p_ref = torch.stack(probs, dim=1)[:, k]
        diff = (p_ref - p_explain).abs()
        return {
            "p_reference": p_ref.cpu().numpy(),
            "p_explained": p_explain.cpu().numpy(),
            "abs_diff": diff.cpu().numpy(),
            "max_abs_diff": float(diff.max()),
            "bit_exact": bool(torch.equal(p_ref, p_explain)),
            "tolerance": tolerance,
            "passed": bool(diff.max() < tolerance),
        }

    # ---------------------------------------------------------
    # HUMAN-READABLE FORMATTER (deterministic)
    # ---------------------------------------------------------

    def format_explanation(
        self,
        attrs: np.ndarray,
        probability: float,
        target: ExplanationTarget,
        top_n: int = 5,
        input_source: str = "dataset",
    ) -> str:
        """Deterministic text rendering of one explanation.

        Descriptive only. It never names a tactic, a technique, or an
        attacker action -- that wording belongs to the separate
        rule-based MITRE layer.
        """
        if attrs.shape != (SEQ_LEN, N_FEATURES):
            raise ValueError(f"expected (10, 52), got {attrs.shape}")

        feat_signed = attrs.sum(axis=0)
        feat_abs = np.abs(attrs).sum(axis=0)
        # Deterministic tie-break: descending |attr|, then ascending index.
        order = sorted(
            range(N_FEATURES), key=lambda i: (-feat_abs[i], i)
        )[:top_n]

        step_abs = np.abs(attrs).sum(axis=1)
        top_step = int(np.argmax(step_abs))

        lines = [
            "V4 FUTURE THREAT EXPLANATION",
            "",
            f"Forecast:            {target.target_type} "
            f"K={target.target_horizon}",
            f"Threat probability:  {probability:.4f}",
            f"Frozen threshold:    {target.threshold:.2f}",
            f"Predicted label:     "
            f"{'ALERT' if probability >= target.threshold else 'no alert'}",
            f"Input source:        {input_source}",
            "",
            f"Top {top_n} contributing features "
            f"(signed attribution, summed over the 10-window history):",
        ]
        for rank, i in enumerate(order, 1):
            arrow = "↑" if feat_signed[i] > 0 else "↓"
            lines.append(
                f"  {rank}. {self.feature_names[i]:<38} {arrow} "
                f"{feat_signed[i]:+.6f}  (|attr| {feat_abs[i]:.6f})"
            )

        lines += [
            "",
            f"Most influential historical window: "
            f"{self.time_labels[top_step]} "
            f"(|attr| {step_abs[top_step]:.6f})",
            "",
            "MODEL ATTRIBUTION:",
        ]

        up = [self.feature_names[i] for i in order if feat_signed[i] > 0][:3]
        down = [self.feature_names[i] for i in order if feat_signed[i] <= 0][:3]
        if up:
            lines.append(
                f"  Increased predicted threat probability: "
                f"{', '.join(up)}."
            )
        if down:
            lines.append(
                f"  Decreased predicted threat probability: "
                f"{', '.join(down)}."
            )
        lines += [
            "",
            "  Attribution describes this model's sensitivity on this "
            "input.",
            "  It does not establish causality, attacker intent, attack "
            "stage,",
            "  or actual compromise.",
        ]
        if input_source == "pcap":
            lines += [
                "",
                "  PCAP CAVEAT: this explanation uses the same V4 feature "
                "contract",
                "  and transform chain, but flow-record production differs "
                "from the",
                "  training path, so the resulting feature values are "
                "approximate.",
            ]
        return "\n".join(lines)


# =============================================================
# CONTAMINATION AUDIT
# =============================================================

class FileAccessAudit:
    """Records every file the interpreter opens inside the block.

    Used to prove at RUNTIME -- not just by reading imports -- that no
    V1/V2/V3 checkpoint, scaler or feature list participates in an
    explanation.
    """

    def __init__(self):
        self.opened: list[str] = []
        self._installed = False

    def __enter__(self):
        def hook(event, args):
            if event == "open" and args and isinstance(args[0], (str, bytes)):
                p = args[0].decode() if isinstance(args[0], bytes) else args[0]
                self.opened.append(p)
        sys.addaudithook(hook)      # audit hooks cannot be removed
        self._installed = True
        return self

    def __exit__(self, *exc):
        return False

    def project_files(self) -> list[str]:
        out = []
        for p in self.opened:
            if "cyber-world-model" not in p:
                continue
            if "/venv/" in p or "site-packages" in p or "__pycache__" in p:
                continue
            if p not in out:
                out.append(p)
        return out

    def forbidden_hits(self) -> list[str]:
        return [
            p for p in self.project_files()
            if any(m in p for m in FORBIDDEN_ARTIFACT_MARKERS)
        ]
