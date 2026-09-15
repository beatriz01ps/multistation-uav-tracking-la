"""Frozen pipeline configurations for the final experimental battery (OFAT + Lego matrix)"""

from __future__ import annotations

from pathlib import Path

import yaml

from config.models import AppConfig
from models.enums import AssociationMetric, AssociationMode, FilterType, FusionStrategyName, MotionModelName


def load_frozen_d_star(frozen_path: Path) -> float:

    """Reads D* (the calibrated Euclidean gate distance every config here
    is parameterized by) back from the bundled frozen config."""

    data = yaml.safe_load(frozen_path.read_text(encoding="utf-8"))
    return float(data["association"]["euclidean_gate_distance_m"])


def build_reference_config(euclidean_gate_distance: float) -> AppConfig:
    
    config = AppConfig()
    config.filter.type = FilterType.UKF
    config.filter.motion_model = MotionModelName.COORDINATED_TURN
    config.fusion.strategy = FusionStrategyName.INFORMATION
    config.association.metric = AssociationMetric.MAHALANOBIS
    config.association.mode = AssociationMode.POSITION_ONLY
    config.association.euclidean_gate_distance = euclidean_gate_distance
    config.duplicate_merger.enabled = True
    # NEVER follows fusion.strategy (an intentional decoupling)
    config.duplicate_merger.fusion_strategy = FusionStrategyName.INFORMATION
    return config


def build_ofat_configs(euclidean_gate_distance: float) -> dict[str, AppConfig]:

    reference = build_reference_config(euclidean_gate_distance)

    temporal_ekf = build_reference_config(euclidean_gate_distance)
    temporal_ekf.filter.type = FilterType.EKF

    t2tf_ci = build_reference_config(euclidean_gate_distance)
    t2tf_ci.fusion.strategy = FusionStrategyName.COVARIANCE_INTERSECTION

    association_euclidean = build_reference_config(euclidean_gate_distance)
    association_euclidean.association.metric = AssociationMetric.EUCLIDEAN

    identity_merger_off = build_reference_config(euclidean_gate_distance)
    identity_merger_off.duplicate_merger.enabled = False

    return {
        "reference": reference,
        "temporal_ekf": temporal_ekf,
        "t2tf_ci": t2tf_ci,
        "association_euclidean": association_euclidean,
        "identity_merger_off": identity_merger_off,
    }


def build_motion_model_ablation_configs(euclidean_gate_distance: float) -> dict[str, AppConfig]:

    """Supplementary motion-model ablation (CV vs CT)"""

    coordinated_turn = build_reference_config(euclidean_gate_distance)  # motion_model=CT 

    constant_velocity = build_reference_config(euclidean_gate_distance)
    constant_velocity.filter.motion_model = MotionModelName.CONSTANT_VELOCITY

    return {"coordinated_turn": coordinated_turn, "constant_velocity": constant_velocity}


def build_full_state_ablation_configs(euclidean_gate_distance: float) -> dict[str, AppConfig]:

    """Mahalanobis POSITION_ONLY vs FULL_STATE (RQ2)"""

    position_only = build_reference_config(euclidean_gate_distance)  # mode=POSITION_ONLY 

    full_state = build_reference_config(euclidean_gate_distance)
    full_state.association.mode = AssociationMode.FULL_STATE

    return {"position_only": position_only, "full_state": full_state}


def build_no_fusion_baseline_configs(euclidean_gate_distance: float) -> dict[str, AppConfig]:

    """local-only vs Information Fusion vs Covariance Intersection"""

    information = build_reference_config(euclidean_gate_distance)  # fusion=information 

    covariance_intersection = build_reference_config(euclidean_gate_distance)
    covariance_intersection.fusion.strategy = FusionStrategyName.COVARIANCE_INTERSECTION

    local_only = build_reference_config(euclidean_gate_distance)
    local_only.fusion.strategy = FusionStrategyName.LOCAL_ONLY

    return {"information": information, "covariance_intersection": covariance_intersection, "local_only": local_only}


def build_replay_contrast_configs(euclidean_gate_distance: float) -> dict[str, AppConfig]:

    """The union of the 6 main contrasts"""

    reference = build_reference_config(euclidean_gate_distance)

    association_euclidean = build_reference_config(euclidean_gate_distance)
    association_euclidean.association.metric = AssociationMetric.EUCLIDEAN

    full_state = build_reference_config(euclidean_gate_distance)
    full_state.association.mode = AssociationMode.FULL_STATE

    t2tf_ci = build_reference_config(euclidean_gate_distance)
    t2tf_ci.fusion.strategy = FusionStrategyName.COVARIANCE_INTERSECTION

    temporal_ekf = build_reference_config(euclidean_gate_distance)
    temporal_ekf.filter.type = FilterType.EKF

    identity_merger_off = build_reference_config(euclidean_gate_distance)
    identity_merger_off.duplicate_merger.enabled = False

    constant_velocity = build_reference_config(euclidean_gate_distance)
    constant_velocity.filter.motion_model = MotionModelName.CONSTANT_VELOCITY

    return {
        "reference": reference,
        "association_euclidean": association_euclidean,
        "full_state": full_state,
        "t2tf_ci": t2tf_ci,
        "temporal_ekf": temporal_ekf,
        "identity_merger_off": identity_merger_off,
        "constant_velocity": constant_velocity,
    }


def build_lego_matrix_configs(euclidean_gate_distance: float) -> dict[str, AppConfig]:

    """16 pipelines: {UKF,EKF} x {IF,CI} x {Mahalanobis,Euclidean} x {merger ON,OFF}, CT fixed throughout."""

    configs: dict[str, AppConfig] = {}
    for filter_type in (FilterType.UKF, FilterType.EKF):
        for fusion_strategy in (FusionStrategyName.INFORMATION, FusionStrategyName.COVARIANCE_INTERSECTION):
            for metric in (AssociationMetric.MAHALANOBIS, AssociationMetric.EUCLIDEAN):
                for merger_enabled in (True, False):
                    config = build_reference_config(euclidean_gate_distance)
                    config.filter.type = filter_type
                    config.fusion.strategy = fusion_strategy
                    config.association.metric = metric
                    config.duplicate_merger.enabled = merger_enabled
                    label = (
                        f"{filter_type.value}__{fusion_strategy.value}__{metric.value}"
                        f"__merger_{'on' if merger_enabled else 'off'}"
                    )
                    configs[label] = config
    assert len(configs) == 16
    return configs


def build_full_proposed_config(euclidean_gate_distance: float) -> AppConfig:

    """FULL_PROPOSED: UKF + Coordinated Turn + Covariance Intersection (T2TF) + Mahalanobis FULL_STATE (T2TA) + merger ON"""

    config = build_reference_config(euclidean_gate_distance)
    config.fusion.strategy = FusionStrategyName.COVARIANCE_INTERSECTION
    config.association.mode = AssociationMode.FULL_STATE
    return config
