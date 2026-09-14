"""
=============================================================
MITRE ATT&CK MAPPING — INTERPRETABLE DECISION SUPPORT
=============================================================

This module maps the V3 World Model's attack predictions
to the MITRE ATT&CK framework using observable network
telemetry indicators.

IMPORTANT DESIGN PRINCIPLES:
- The V3 model outputs an ATTACK PROBABILITY, not a
  MITRE technique. This layer is an INFERENCE layer.
- Mappings are based on DEFENSIBLE indicators from the
  52 network telemetry features.
- Where evidence is insufficient, the system returns
  "Unknown / Insufficient Evidence" rather than guessing.
- Evidence is always separated from inference.

MITRE ATT&CK Tactics relevant to network telemetry:
- TA0043: Reconnaissance
- TA0001: Initial Access
- TA0002: Execution
- TA0008: Lateral Movement
- TA0011: Command and Control (C2)
- TA0010: Exfiltration
- TA0040: Impact

=============================================================
"""

import argparse
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# =============================================================
# FEATURE INDEX MAPPING
# =============================================================
# These must match input_feature_names.txt EXACTLY.
# The first 26 are "base" features, the next 26 are
# their "_rel_change" variants.

INPUT_FEATURES = [
    "flow_count",                    # 0
    "log_flow_count",                # 1
    "log_total_packets",             # 2
    "log_total_bytes",               # 3
    "window_packet_rate",            # 4
    "window_byte_rate",              # 5
    "syn_rate",                      # 6
    "ack_rate",                      # 7
    "rst_rate",                      # 8
    "fin_rate",                      # 9
    "psh_rate",                      # 10
    "urg_rate",                      # 11
    "mean_pkt_len",                  # 12
    "std_pkt_len",                   # 13
    "log_mean_flow_duration",        # 14
    "log_mean_fwd_iat",              # 15
    "log_mean_bwd_iat",              # 16
    "fwd_packet_ratio",              # 17
    "bwd_packet_ratio",              # 18
    "fwd_byte_ratio",                # 19
    "bwd_byte_ratio",                # 20
    "mean_down_up_ratio",            # 21
    "log_unique_dst_ports",          # 22
    "unique_protocols",              # 23
    "dst_port_entropy",              # 24
    "protocol_entropy",              # 25
    "flow_count_rel_change",         # 26
    "log_flow_count_rel_change",     # 27
    "log_total_packets_rel_change",  # 28
    "log_total_bytes_rel_change",    # 29
    "window_packet_rate_rel_change", # 30
    "window_byte_rate_rel_change",   # 31
    "syn_rate_rel_change",           # 32
    "ack_rate_rel_change",           # 33
    "rst_rate_rel_change",           # 34
    "fin_rate_rel_change",           # 35
    "psh_rate_rel_change",           # 36
    "urg_rate_rel_change",           # 37
    "mean_pkt_len_rel_change",       # 38
    "std_pkt_len_rel_change",        # 39
    "log_mean_flow_duration_rel_change",  # 40
    "log_mean_fwd_iat_rel_change",   # 41
    "log_mean_bwd_iat_rel_change",   # 42
    "fwd_packet_ratio_rel_change",   # 43
    "bwd_packet_ratio_rel_change",   # 44
    "fwd_byte_ratio_rel_change",     # 45
    "bwd_byte_ratio_rel_change",     # 46
    "mean_down_up_ratio_rel_change", # 47
    "log_unique_dst_ports_rel_change",    # 48
    "unique_protocols_rel_change",   # 49
    "dst_port_entropy_rel_change",   # 50
    "protocol_entropy_rel_change",   # 51
]


# =============================================================
# STRUCTURED OUTPUT
# =============================================================

@dataclass
class MITREMapping:
    """
    Structured output for a single window's MITRE
    ATT&CK assessment. This is the contract consumed
    by the Streamlit dashboard, SHAP layer, and
    K-step rollout engine.
    """

    # From the V3 model
    attack_probability: float
    window_index: int
    timestamp: str

    # MITRE inference
    tactic_id: str          # e.g. "TA0008"
    tactic_name: str        # e.g. "Lateral Movement"
    technique_id: Optional[str] = None  # e.g. "T1046"
    technique_name: Optional[str] = None
    confidence: str = "low"  # "low", "medium", "high"

    # Evidence
    triggering_features: list = field(
        default_factory=list
    )
    feature_values: dict = field(
        default_factory=dict
    )

    # Temporal context
    temporal_phase: str = ""  # "early", "mid", "late"
    windows_since_first_alert: int = 0

    # Human-readable explanation
    analyst_explanation: str = ""

    def to_dict(self):
        return asdict(self)

    def to_json(self):
        return json.dumps(
            self.to_dict(), indent=2,
            default=str
        )


# =============================================================
# TELEMETRY-BASED MITRE RULES
# =============================================================
# These rules map OBSERVABLE telemetry signatures to
# defensible ATT&CK tactics/techniques.
#
# Each rule checks specific feature values from the
# LAST timestep of the input window (the most recent
# observation). A rule fires when its conditions are
# met AND the model's attack probability exceeds the
# threshold.
#
# Rules are ordered by priority (most specific first).
# The first matching rule wins.
#
# IMPORTANT: These are INFERRED mappings based on
# network telemetry patterns, NOT ground truth labels.
# =============================================================

@dataclass
class TelemetryRule:
    """A single evidence-based MITRE mapping rule."""
    rule_id: str
    tactic_id: str
    tactic_name: str
    technique_id: Optional[str]
    technique_name: Optional[str]
    description: str
    # Feature checks: list of (feature_name, operator, threshold)
    # operator is one of: ">", "<", ">=", "<=", "spike"
    checks: list
    min_confidence: str  # "low", "medium", "high"

    def evaluate(
        self, feature_snapshot: dict,
        rel_change_snapshot: dict,
    ) -> tuple:
        """
        Evaluate this rule against a feature snapshot.
        Returns (score: int, evidence: list).
        Score = number of checks that matched.
        """
        evidence = []

        for check in self.checks:
            feat_name = check["feature"]
            operator = check["operator"]
            threshold = check["threshold"]

            # Check relative changes for "spike"
            if operator == "spike":
                rel_key = feat_name + "_rel_change"
                if rel_key in rel_change_snapshot:
                    val = rel_change_snapshot[rel_key]
                    if abs(val) > threshold:
                        evidence.append({
                            "feature": feat_name,
                            "type": "spike",
                            "value": float(val),
                            "threshold": threshold,
                        })
                continue

            # Check absolute value
            if feat_name in feature_snapshot:
                val = feature_snapshot[feat_name]
                matched = False

                if operator == ">" and val > threshold:
                    matched = True
                elif operator == "<" and val < threshold:
                    matched = True
                elif operator == ">=" and val >= threshold:
                    matched = True
                elif operator == "<=" and val <= threshold:
                    matched = True

                if matched:
                    evidence.append({
                        "feature": feat_name,
                        "type": "threshold",
                        "operator": operator,
                        "value": float(val),
                        "threshold": threshold,
                    })

        # Score = fraction of checks that matched
        score = len(evidence)
        return score, evidence


def build_rule_library() -> list:
    """
    Build the library of telemetry-to-MITRE rules.

    IMPORTANT: All thresholds are in Z-SCORE SPACE
    because input features are StandardScaler-ed.
    A threshold of 0.5 means "0.5 std above the
    training mean."

    Actual feature statistics (test set, last timestep):
      log_unique_dst_ports: atk=0.34, ben=-0.38
      log_flow_count:       atk=0.38, ben=-0.13
      flow_count:           atk=0.21, ben=-0.22
      syn_rate:             atk=0.10, ben=-0.23
      fin_rate:             atk=0.17, ben=-0.16
      log_total_packets:    atk=0.31, ben=-0.11
      log_total_bytes:      atk=0.32, ben=-0.00
      std_pkt_len:          atk=0.12, ben=0.67
      mean_pkt_len:         atk=0.17, ben=0.56
      rst_rate:             atk=0.09, ben=0.28
    """
    rules = []

    # ----- RECONNAISSANCE (TA0043) -----
    # Port scanning: many unique dst ports + elevated
    # SYN rate. In z-score space, attack windows show
    # log_unique_dst_ports > 0.3 and syn_rate > 0.0

    rules.append(TelemetryRule(
        rule_id="RECON_PORT_SCAN",
        tactic_id="TA0043",
        tactic_name="Reconnaissance",
        technique_id="T1046",
        technique_name="Network Service Scanning",
        description=(
            "Elevated unique destination ports with "
            "above-average SYN rate — consistent with "
            "port scanning activity."
        ),
        checks=[
            {"feature": "log_unique_dst_ports",
             "operator": ">", "threshold": 0.5},
            {"feature": "syn_rate",
             "operator": ">", "threshold": 0.0},
        ],
        min_confidence="medium",
    ))

    rules.append(TelemetryRule(
        rule_id="RECON_HOST_DISCOVERY",
        tactic_id="TA0043",
        tactic_name="Reconnaissance",
        technique_id="T1018",
        technique_name="Remote System Discovery",
        description=(
            "High destination port entropy with "
            "above-average flow count — consistent "
            "with network discovery."
        ),
        checks=[
            {"feature": "dst_port_entropy",
             "operator": ">", "threshold": 0.3},
            {"feature": "log_flow_count",
             "operator": ">", "threshold": 0.3},
        ],
        min_confidence="low",
    ))

    # ----- INITIAL ACCESS (TA0001) -----
    # Infiltration attack: elevated flow count, many
    # new connections (SYN + FIN flags)

    rules.append(TelemetryRule(
        rule_id="INITIAL_ACCESS_INFILTRATION",
        tactic_id="TA0001",
        tactic_name="Initial Access",
        technique_id="T1190",
        technique_name="Exploit Public-Facing Application",
        description=(
            "Elevated flow count with SYN and FIN "
            "flag activity suggests active exploit "
            "attempts or infiltration."
        ),
        checks=[
            {"feature": "log_flow_count",
             "operator": ">", "threshold": 0.2},
            {"feature": "syn_rate",
             "operator": ">", "threshold": -0.1},
            {"feature": "fin_rate",
             "operator": ">", "threshold": 0.0},
        ],
        min_confidence="low",
    ))

    # ----- EXECUTION (TA0002) -----
    # High packet volume with data push (PSH flag)

    rules.append(TelemetryRule(
        rule_id="EXECUTION_PAYLOAD_DELIVERY",
        tactic_id="TA0002",
        tactic_name="Execution",
        technique_id="T1059",
        technique_name="Command and Scripting Interpreter",
        description=(
            "Elevated packet count with PSH flag "
            "activity suggests payload delivery "
            "or command execution."
        ),
        checks=[
            {"feature": "log_total_packets",
             "operator": ">", "threshold": 0.3},
            {"feature": "psh_rate",
             "operator": ">", "threshold": 0.0},
        ],
        min_confidence="low",
    ))

    # ----- COMMAND AND CONTROL (TA0011) -----
    # Beaconing: consistent flow patterns, low
    # packet size variance (std_pkt_len below avg)

    rules.append(TelemetryRule(
        rule_id="C2_BEACONING",
        tactic_id="TA0011",
        tactic_name="Command and Control",
        technique_id="T1071",
        technique_name="Application Layer Protocol",
        description=(
            "Low packet size variance with above-"
            "average download ratio suggests "
            "C2 beaconing or tunneling."
        ),
        checks=[
            {"feature": "std_pkt_len",
             "operator": "<", "threshold": -0.3},
            {"feature": "mean_down_up_ratio",
             "operator": ">", "threshold": 0.0},
        ],
        min_confidence="medium",
    ))

    rules.append(TelemetryRule(
        rule_id="C2_ENCRYPTED_CHANNEL",
        tactic_id="TA0011",
        tactic_name="Command and Control",
        technique_id="T1573",
        technique_name="Encrypted Channel",
        description=(
            "Low protocol entropy with uniform "
            "packet sizes suggests encrypted "
            "C2 channel."
        ),
        checks=[
            {"feature": "protocol_entropy",
             "operator": "<", "threshold": -0.5},
            {"feature": "std_pkt_len",
             "operator": "<", "threshold": -0.5},
        ],
        min_confidence="low",
    ))

    # ----- LATERAL MOVEMENT (TA0008) -----
    # High flow count + diverse ports + packets

    rules.append(TelemetryRule(
        rule_id="LATERAL_MOVEMENT_SPREAD",
        tactic_id="TA0008",
        tactic_name="Lateral Movement",
        technique_id="T1021",
        technique_name="Remote Services",
        description=(
            "High flow count with diverse destination "
            "ports and elevated packet volume suggests "
            "lateral movement."
        ),
        checks=[
            {"feature": "log_flow_count",
             "operator": ">", "threshold": 0.5},
            {"feature": "log_unique_dst_ports",
             "operator": ">", "threshold": 0.3},
            {"feature": "log_total_packets",
             "operator": ">", "threshold": 0.3},
        ],
        min_confidence="medium",
    ))

    # ----- EXFILTRATION (TA0010) -----
    # High byte volume, large packets, sustained

    rules.append(TelemetryRule(
        rule_id="EXFIL_DATA_TRANSFER",
        tactic_id="TA0010",
        tactic_name="Exfiltration",
        technique_id="T1048",
        technique_name=(
            "Exfiltration Over Alternative Protocol"
        ),
        description=(
            "High byte volume with large mean "
            "packet length suggests data transfer "
            "or exfiltration."
        ),
        checks=[
            {"feature": "log_total_bytes",
             "operator": ">", "threshold": 0.5},
            {"feature": "mean_pkt_len",
             "operator": ">", "threshold": 0.3},
        ],
        min_confidence="low",
    ))

    rules.append(TelemetryRule(
        rule_id="EXFIL_SUSTAINED_OUTBOUND",
        tactic_id="TA0010",
        tactic_name="Exfiltration",
        technique_id=None,
        technique_name=None,
        description=(
            "Sustained high byte rate with elevated "
            "forward byte ratio suggests ongoing "
            "outbound data transfer."
        ),
        checks=[
            {"feature": "window_byte_rate",
             "operator": ">", "threshold": 0.3},
            {"feature": "fwd_byte_ratio",
             "operator": ">", "threshold": 0.0},
        ],
        min_confidence="low",
    ))

    # ----- IMPACT (TA0040) -----

    rules.append(TelemetryRule(
        rule_id="IMPACT_RESOURCE_EXHAUSTION",
        tactic_id="TA0040",
        tactic_name="Impact",
        technique_id="T1499",
        technique_name="Endpoint Denial of Service",
        description=(
            "Massive spike in packet rate with "
            "high SYN rate suggests resource "
            "exhaustion or denial of service."
        ),
        checks=[
            {"feature": "window_packet_rate",
             "operator": ">", "threshold": 1.0},
            {"feature": "syn_rate",
             "operator": ">", "threshold": 0.5},
        ],
        min_confidence="medium",
    ))

    return rules


# =============================================================
# THE MITRE MAPPER CLASS
# =============================================================

class MITREMapper:
    """
    Maps V3 World Model predictions to MITRE ATT&CK
    framework using telemetry-based rules.

    Designed to be consumed by:
    - Streamlit dashboard
    - SHAP explainability layer
    - K-step rollout engine

    Usage:
        mapper = MITREMapper(model, threshold=0.10)
        results = mapper.map_batch(
            X_test, test_metadata
        )
    """

    def __init__(
        self,
        model,
        threshold: float = 0.10,
        device: str = "cpu",
    ):
        self.model = model
        self.model.eval()
        self.threshold = threshold
        self.device = torch.device(device)
        self.rules = build_rule_library()
        self.base_features = INPUT_FEATURES[:26]
        self.rel_features = INPUT_FEATURES[26:]

    def _extract_feature_snapshot(
        self, x_window: np.ndarray
    ) -> tuple:
        """
        Extract the feature values from the LAST
        timestep of the input window.

        Args:
            x_window: shape (seq_len, 52)

        Returns:
            (base_dict, rel_change_dict)
        """
        last_step = x_window[-1]  # shape: (52,)

        base_dict = {
            name: float(last_step[i])
            for i, name in enumerate(
                self.base_features
            )
        }

        rel_dict = {
            name: float(last_step[26 + i])
            for i, name in enumerate(
                self.rel_features
            )
        }

        return base_dict, rel_dict

    def _determine_temporal_phase(
        self,
        window_idx: int,
        total_windows: int,
        first_alert_idx: Optional[int],
    ) -> tuple:
        """
        Determine temporal phase and windows since
        first alert.
        """
        if first_alert_idx is None:
            phase = "pre-alert"
            since_first = 0
        elif window_idx == first_alert_idx:
            phase = "onset"
            since_first = 0
        else:
            since_first = (
                window_idx - first_alert_idx
            )
            progress = since_first / max(
                total_windows - first_alert_idx, 1
            )
            if progress < 0.25:
                phase = "early"
            elif progress < 0.75:
                phase = "mid"
            else:
                phase = "late"

        return phase, since_first

    def _boost_confidence(
        self,
        base_confidence: str,
        attack_prob: float,
        evidence_count: int,
        temporal_phase: str,
    ) -> str:
        """
        Adjust confidence based on attack probability,
        amount of evidence, and temporal phase.
        """
        score = {
            "low": 1, "medium": 2, "high": 3
        }[base_confidence]

        # High model confidence boosts
        if attack_prob > 0.5:
            score += 1
        if attack_prob > 0.7:
            score += 1

        # Multiple evidence points boost
        if evidence_count >= 3:
            score += 1

        # Sustained activity (mid/late phase) boosts
        if temporal_phase in ("mid", "late"):
            score += 1

        if score >= 4:
            return "high"
        elif score >= 2:
            return "medium"
        else:
            return "low"

    def _generate_explanation(
        self,
        mapping: MITREMapping,
    ) -> str:
        """
        Generate a human-readable analyst explanation.
        """
        prob_pct = mapping.attack_probability * 100

        if mapping.tactic_id == "NONE":
            return (
                f"Attack probability {prob_pct:.0f}% "
                f"detected at {mapping.timestamp}, "
                f"but insufficient telemetry evidence "
                f"to map to a specific ATT&CK tactic."
            )

        feat_list = ", ".join(
            mapping.triggering_features[:3]
        )

        technique_str = ""
        if mapping.technique_name:
            technique_str = (
                f" [{mapping.technique_id}: "
                f"{mapping.technique_name}]"
            )

        return (
            f"ALERT [{mapping.confidence.upper()}]: "
            f"{mapping.tactic_name}{technique_str} "
            f"— Attack probability {prob_pct:.0f}% "
            f"at {mapping.timestamp} "
            f"(phase: {mapping.temporal_phase}). "
            f"Triggered by: {feat_list}."
        )

    def map_single_window(
        self,
        x_window: np.ndarray,
        window_idx: int,
        timestamp: str,
        total_windows: int,
        first_alert_idx: Optional[int],
        attack_prob: Optional[float] = None,
    ) -> MITREMapping:
        """
        Map a single window to MITRE ATT&CK.

        Args:
            x_window: shape (seq_len, 52), scaled
            window_idx: index in the test set
            timestamp: ISO timestamp string
            total_windows: total test windows
            first_alert_idx: index of first alert
            attack_prob: if None, computed from model

        Returns:
            MITREMapping dataclass
        """
        # Get attack probability from model
        if attack_prob is None:
            with torch.no_grad():
                x_t = torch.tensor(
                    x_window[np.newaxis],
                    dtype=torch.float32,
                ).to(self.device)

                z = self.model.encode(x_t)
                z_next = self.model.transition_step(z)
                logit = self.model.predict_attack(
                    z_next
                )
                attack_prob = float(
                    torch.sigmoid(logit).cpu()
                )

        # If below threshold, no alert
        if attack_prob < self.threshold:
            return MITREMapping(
                attack_probability=attack_prob,
                window_index=window_idx,
                timestamp=timestamp,
                tactic_id="NONE",
                tactic_name="No Alert",
                confidence="n/a",
                temporal_phase="benign",
                analyst_explanation=(
                    f"No alert. Attack probability "
                    f"{attack_prob*100:.1f}% is below "
                    f"threshold {self.threshold*100:.0f}%."
                ),
            )

        # Extract telemetry snapshot
        base_snap, rel_snap = (
            self._extract_feature_snapshot(x_window)
        )

        # Temporal phase
        phase, since_first = (
            self._determine_temporal_phase(
                window_idx, total_windows,
                first_alert_idx,
            )
        )

        # Evaluate ALL rules — select best match
        best_rule = None
        best_score = 0
        best_evidence = []

        for rule in self.rules:
            score, evidence = rule.evaluate(
                base_snap, rel_snap
            )

            # Require at least 1 check to match
            if score > 0 and score > best_score:
                best_rule = rule
                best_score = score
                best_evidence = evidence

        if best_rule is not None:
            trigger_feats = [
                e["feature"] for e in best_evidence
            ]
            feat_vals = {
                e["feature"]: e["value"]
                for e in best_evidence
            }

            confidence = self._boost_confidence(
                best_rule.min_confidence,
                attack_prob,
                len(best_evidence),
                phase,
            )

            mapping = MITREMapping(
                attack_probability=attack_prob,
                window_index=window_idx,
                timestamp=timestamp,
                tactic_id=best_rule.tactic_id,
                tactic_name=best_rule.tactic_name,
                technique_id=best_rule.technique_id,
                technique_name=best_rule.technique_name,
                confidence=confidence,
                triggering_features=trigger_feats,
                feature_values=feat_vals,
                temporal_phase=phase,
                windows_since_first_alert=since_first,
            )
            mapping.analyst_explanation = (
                self._generate_explanation(mapping)
            )
            return mapping

        # No rule matched — report attack with
        # Unknown tactic
        mapping = MITREMapping(
            attack_probability=attack_prob,
            window_index=window_idx,
            timestamp=timestamp,
            tactic_id="UNKNOWN",
            tactic_name="Unknown / Insufficient Evidence",
            confidence="low",
            temporal_phase=phase,
            windows_since_first_alert=since_first,
            analyst_explanation="",
        )
        mapping.analyst_explanation = (
            self._generate_explanation(mapping)
        )
        return mapping

    def map_batch(
        self,
        X_test: np.ndarray,
        metadata: pd.DataFrame,
        batch_size: int = 64,
    ) -> list:
        """
        Map all test windows to MITRE ATT&CK.

        Args:
            X_test: shape (N, seq_len, 52)
            metadata: DataFrame with 'timestamp',
                      'attack_present', 'attack_ratio'

        Returns:
            List of MITREMapping objects
        """
        n = X_test.shape[0]

        # First pass: get all attack probabilities
        all_probs = np.zeros(n)

        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch = torch.tensor(
                    X_test[start:end],
                    dtype=torch.float32,
                ).to(self.device)

                z = self.model.encode(batch)
                z_next = (
                    self.model.transition_step(z)
                )
                logits = (
                    self.model.predict_attack(z_next)
                )
                probs = (
                    torch.sigmoid(logits).cpu().numpy()
                )
                all_probs[start:end] = probs

        # Find first alert index
        first_alert_idx = None
        for i in range(n):
            if all_probs[i] >= self.threshold:
                first_alert_idx = i
                break

        # Second pass: map each window
        mappings = []
        timestamps = metadata["timestamp"].values

        for i in range(n):
            mapping = self.map_single_window(
                x_window=X_test[i],
                window_idx=i,
                timestamp=str(timestamps[i]),
                total_windows=n,
                first_alert_idx=first_alert_idx,
                attack_prob=float(all_probs[i]),
            )
            mappings.append(mapping)

        return mappings


# =============================================================
# V3 MODEL DEFINITION (for standalone loading)
# =============================================================

class LatentWorldModelV3(nn.Module):

    def __init__(
        self, input_size, hidden_size,
        latent_size, output_size,
        num_layers, dropout
    ):
        super().__init__()

        self.encoder_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=(
                dropout if num_layers > 1
                else 0.0
            )
        )
        self.encoder_norm = nn.LayerNorm(
            hidden_size
        )
        self.encoder_fc = nn.Linear(
            hidden_size, latent_size
        )

        self.transition_delta = nn.Sequential(
            nn.Linear(latent_size, latent_size * 4),
            nn.ReLU(),
            nn.Linear(latent_size * 4, latent_size)
        )

        self.decoder_fc1 = nn.Linear(
            latent_size, 64
        )
        self.decoder_activation = nn.ReLU()
        self.decoder_fc2 = nn.Linear(
            64, output_size
        )

        self.attack_head = nn.Sequential(
            nn.Linear(latent_size, 32),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(32, 1)
        )

    def encode(self, x):
        output, _ = self.encoder_lstm(x)
        last_hidden = output[:, -1, :]
        last_hidden = self.encoder_norm(
            last_hidden
        )
        return self.encoder_fc(last_hidden)

    def transition_step(self, z):
        return z + self.transition_delta(z)

    def decode(self, z):
        h = self.decoder_fc1(z)
        h = self.decoder_activation(h)
        return self.decoder_fc2(h)

    def predict_attack(self, z):
        logits = self.attack_head(z)
        return logits.squeeze(-1)

    def forward(self, x):
        z = self.encode(x)
        z_next = self.transition_step(z)
        state = self.decode(z_next)
        attack = self.predict_attack(z_next)
        return state, z_next, attack, z


# =============================================================
# MAIN — RUN MITRE MAPPING ON TEST SET
# =============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Map World Model attack predictions to the "
            "MITRE ATT&CK framework."
        )
    )
    parser.add_argument(
        "--version",
        choices=["v3", "v4"],
        default="v4",
        help=(
            "Which World Model checkpoint to map "
            "(default: v4, the canonical model)."
        ),
    )
    args = parser.parse_args()
    version = args.version

    DATA_DIR = Path(
        "data/processed/temporal_benchmark_v2"
    )
    MODEL_DIR = Path(
        f"models/latent_world_model_{version}"
    )
    OUTPUT_DIR = MODEL_DIR / "mitre_mapping"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Device
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")

    print("=" * 80)
    print("MITRE ATT&CK MAPPING — INTERPRETABLE")
    print("DECISION SUPPORT LAYER")
    print("=" * 80)
    print(f"Model version: {version.upper()}")
    print(f"Device: {DEVICE}")

    # Load model
    print(f"\n--- Loading {version.upper()} model ---")
    if version == "v4":
        # Reuse the canonical V4 model/checkpoint loader instead of
        # duplicating the class definition (this file already carries
        # the V3 one below) -- see src/pcap_ingest_v4.py.
        from pcap_ingest_v4 import load_v4_model, CHECKPOINT_PATH

        model, checkpoint = load_v4_model(CHECKPOINT_PATH)
        model = model.to(DEVICE)
    else:
        checkpoint = torch.load(
            MODEL_DIR / "latent_world_model_v3.pt",
            map_location=DEVICE,
            weights_only=False,
        )

        model = LatentWorldModelV3(
            input_size=checkpoint["input_size"],
            hidden_size=checkpoint["hidden_size"],
            latent_size=checkpoint["latent_size"],
            output_size=checkpoint["output_size"],
            num_layers=checkpoint["num_layers"],
            dropout=0.20,
        )
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
        model = model.to(DEVICE)
    model.eval()

    threshold = checkpoint["selected_threshold"]
    print(f"Threshold: {threshold}")

    # Load data
    print("--- Loading test data ---")
    X_test = np.load(DATA_DIR / "X_test.npy")
    test_meta = pd.read_csv(
        DATA_DIR / "test_metadata.csv"
    )
    print(f"Test samples: {X_test.shape[0]}")

    # Run mapping
    print("\n--- Running MITRE mapping ---")
    mapper = MITREMapper(
        model=model,
        threshold=threshold,
        device=str(DEVICE),
    )

    mappings = mapper.map_batch(X_test, test_meta)

    # =========================================================
    # ANALYSIS 1: Summary statistics
    # =========================================================
    print("\n" + "=" * 80)
    print("MITRE MAPPING RESULTS")
    print("=" * 80)

    tactic_counts = {}
    alert_count = 0
    technique_counts = {}

    for m in mappings:
        if m.tactic_id not in ("NONE",):
            alert_count += 1
            tactic_counts[m.tactic_name] = (
                tactic_counts.get(m.tactic_name, 0)
                + 1
            )
            if m.technique_name:
                technique_counts[m.technique_name] = (
                    technique_counts.get(
                        m.technique_name, 0
                    ) + 1
                )

    total = len(mappings)
    print(f"\nTotal windows: {total}")
    print(f"Alert windows: {alert_count} "
          f"({alert_count/total*100:.1f}%)")
    print(f"No-alert windows: {total - alert_count}")

    print(f"\n--- Tactic Distribution ---")
    for tactic, count in sorted(
        tactic_counts.items(),
        key=lambda x: -x[1]
    ):
        print(f"  {tactic}: {count} windows")

    print(f"\n--- Technique Distribution ---")
    for tech, count in sorted(
        technique_counts.items(),
        key=lambda x: -x[1]
    ):
        print(f"  {tech}: {count} windows")

    # =========================================================
    # ANALYSIS 2: Confidence distribution
    # =========================================================

    confidence_counts = {"low": 0, "medium": 0,
                         "high": 0}
    for m in mappings:
        if m.tactic_id != "NONE":
            confidence_counts[m.confidence] = (
                confidence_counts.get(
                    m.confidence, 0
                ) + 1
            )

    print(f"\n--- Confidence Distribution ---")
    for conf, count in confidence_counts.items():
        print(f"  {conf}: {count} windows")

    # =========================================================
    # ANALYSIS 3: Sample alert explanations
    # =========================================================

    print(f"\n--- Sample Alert Explanations ---")
    alert_samples = [
        m for m in mappings
        if m.tactic_id != "NONE"
    ]

    for m in alert_samples[:10]:
        print(f"\n  Window {m.window_index}:")
        print(f"  {m.analyst_explanation}")

    # =========================================================
    # SAVE OUTPUTS
    # =========================================================

    # Save full mapping as JSON
    mapping_dicts = [m.to_dict() for m in mappings]
    with open(
        OUTPUT_DIR / "mitre_mappings.json", "w"
    ) as f:
        json.dump(mapping_dicts, f, indent=2,
                  default=str)

    # Save summary CSV
    summary_rows = []
    for m in mappings:
        summary_rows.append({
            "window_index": m.window_index,
            "timestamp": m.timestamp,
            "attack_probability": (
                m.attack_probability
            ),
            "tactic_id": m.tactic_id,
            "tactic_name": m.tactic_name,
            "technique_id": m.technique_id or "",
            "technique_name": (
                m.technique_name or ""
            ),
            "confidence": m.confidence,
            "temporal_phase": m.temporal_phase,
            "triggering_features": (
                "; ".join(m.triggering_features)
            ),
            "explanation": m.analyst_explanation,
            "ground_truth": int(
                test_meta["attack_present"]
                .iloc[m.window_index]
            ),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        OUTPUT_DIR / "mitre_summary.csv",
        index=False,
    )

    # =========================================================
    # VISUALIZATION: MITRE Timeline
    # =========================================================

    print("\n--- Generating timeline ---")

    fig, axes = plt.subplots(
        3, 1, figsize=(18, 12),
        gridspec_kw={"height_ratios": [1, 2, 1.5]},
    )

    n = len(mappings)
    gt = test_meta["attack_present"].values

    # Panel 1: Ground truth
    axes[0].fill_between(
        range(n), gt, alpha=0.4,
        color="crimson", label="Attack (ground truth)"
    )
    axes[0].set_ylabel("Ground Truth")
    axes[0].legend(loc="upper right")
    axes[0].set_title(
        "MITRE ATT&CK Mapping Timeline",
        fontsize=14, fontweight="bold"
    )

    # Panel 2: Attack probability + MITRE tactics
    probs = [m.attack_probability for m in mappings]
    axes[1].plot(
        range(n), probs,
        color="steelblue", linewidth=0.8,
        label="Attack probability"
    )
    axes[1].axhline(
        y=threshold, color="red",
        linestyle="--", alpha=0.5,
        label=f"Threshold={threshold:.2f}"
    )

    # Color-code by tactic
    tactic_colors = {
        "Reconnaissance": "#e74c3c",
        "Initial Access": "#e67e22",
        "Command and Control": "#9b59b6",
        "Lateral Movement": "#3498db",
        "Exfiltration": "#2ecc71",
        "Impact": "#1abc9c",
        "Unknown / Insufficient Evidence": "#95a5a6",
        "No Alert": "none",
    }

    for m in mappings:
        if m.tactic_id != "NONE":
            color = tactic_colors.get(
                m.tactic_name, "#95a5a6"
            )
            axes[1].axvspan(
                m.window_index - 0.5,
                m.window_index + 0.5,
                alpha=0.3, color=color,
            )

    # Legend for tactics
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(
            facecolor=color, alpha=0.5,
            label=tactic
        )
        for tactic, color in tactic_colors.items()
        if tactic != "No Alert"
        and tactic_counts.get(tactic, 0) > 0
    ]
    axes[1].legend(
        handles=legend_patches,
        loc="upper right", fontsize=8,
    )
    axes[1].set_ylabel("Attack Probability")

    # Panel 3: Confidence level over time
    conf_map = {"n/a": 0, "low": 1,
                "medium": 2, "high": 3}
    conf_values = [
        conf_map.get(m.confidence, 0)
        for m in mappings
    ]
    conf_colors_list = []
    for v in conf_values:
        if v == 0:
            conf_colors_list.append("#ecf0f1")
        elif v == 1:
            conf_colors_list.append("#f39c12")
        elif v == 2:
            conf_colors_list.append("#e67e22")
        else:
            conf_colors_list.append("#e74c3c")

    axes[2].bar(
        range(n), conf_values,
        color=conf_colors_list, width=1.0,
    )
    axes[2].set_ylabel("Confidence")
    axes[2].set_yticks([0, 1, 2, 3])
    axes[2].set_yticklabels(
        ["None", "Low", "Medium", "High"]
    )

    ts = pd.to_datetime(test_meta["timestamp"])
    axes[2].set_xlabel(
        f"Window index ({ts.iloc[0].strftime('%H:%M')}"
        f" to {ts.iloc[-1].strftime('%H:%M')})"
    )

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "mitre_timeline.png", dpi=150
    )
    plt.close(fig)
    print("  Saved: mitre_timeline.png")

    # =========================================================
    # VISUALIZATION: Tactic pie chart
    # =========================================================

    if tactic_counts:
        fig, ax = plt.subplots(figsize=(8, 8))
        labels = list(tactic_counts.keys())
        sizes = list(tactic_counts.values())
        colors = [
            tactic_colors.get(l, "#95a5a6")
            for l in labels
        ]

        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors,
            autopct="%1.0f%%", startangle=90,
            textprops={"fontsize": 10},
        )
        ax.set_title(
            "MITRE ATT&CK Tactic Distribution",
            fontsize=13, fontweight="bold",
        )

        fig.tight_layout()
        fig.savefig(
            OUTPUT_DIR / "tactic_distribution.png",
            dpi=150,
        )
        plt.close(fig)
        print("  Saved: tactic_distribution.png")

    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
    print("=" * 80)
