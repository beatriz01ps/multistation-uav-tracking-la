"""Centralized system configuration."""

from __future__ import annotations

from pydantic import BaseModel, Field

from models.enums import (
    AssociationMetric,
    AssociationMode,
    FilterType,
    FusionStrategyName,
    LateMessagePolicy,
    MotionModelName,
)


class TrackingConfig(BaseModel):
    state_dimension: int = 6
    tentative_confirmation_hits: int = 3
    tentative_timeout_seconds: float = 5.0

    coasting_timeout_seconds: float = 10.0
    lost_timeout_seconds: float = 20.0
    deletion_timeout_seconds: float = 32.0


class AssociationConfig(BaseModel):
    metric: AssociationMetric = AssociationMetric.MAHALANOBIS
    mode: AssociationMode = AssociationMode.FULL_STATE
    chi_square_probability: float = 0.9983
    euclidean_gate_distance: float = 50.0
    synchronization_window_ms: float = 100.0
    late_message_policy: LateMessagePolicy = LateMessagePolicy.DROP


class DuplicateMergerConfig(BaseModel):
    
    """Configuration for duplicate-track merging."""

    enabled: bool = True
    fusion_strategy: FusionStrategyName = FusionStrategyName.INFORMATION


class FilterConfig(BaseModel):
    type: FilterType = FilterType.UKF
    motion_model: MotionModelName = MotionModelName.COORDINATED_TURN
    process_noise_acceleration_std: float = 2.0

    turn_rate_process_noise_std: float = 0.05
    initial_turn_rate_std: float = 0.3


class SamplingCovarianceIntersectionConfig(BaseModel):
    """Reserved configuration kept for compatibility with frozen config hashes."""

    sample_count: int = 1000
    fusion_weight: float = 0.5
    random_seed: int = 42


class FusionConfig(BaseModel):
    strategy: FusionStrategyName = FusionStrategyName.INFORMATION
    sci: SamplingCovarianceIntersectionConfig = Field(default_factory=SamplingCovarianceIntersectionConfig)


class TimeConfig(BaseModel):
    time_scale: float = 1.0


class AppConfig(BaseModel):
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    association: AssociationConfig = Field(default_factory=AssociationConfig)
    filter: FilterConfig = Field(default_factory=FilterConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    duplicate_merger: DuplicateMergerConfig = Field(default_factory=DuplicateMergerConfig)
    time: TimeConfig = Field(default_factory=TimeConfig)