"""
Configuration Management

YAML-based configuration classes for the pipeline framework.
"""

from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass
class OperatorConfig:
    """Configuration for a single operator."""

    name: str  # Operator name (e.g., "image_metadata_refiner") -> class name (e.g., "ImageMetadataRefiner")
    params: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    def get_class_name(self) -> str:
        """Convert snake_case name to PascalCase class name.

        Returns:
            Class name in PascalCase
        """
        parts = self.name.split("_")
        return "".join(word.capitalize() for word in parts)


@dataclass
class StageWorkerConfig:
    """Configuration for workers in a stage.

    Supports dynamic worker allocation with min/max replicas:
    - If only num_replicas is set, min_replicas = max_replicas = num_replicas (backward compatible)
    - If min_replicas and max_replicas are set, tries to create up to max_replicas workers
    - Job starts as soon as min_replicas workers are available
    """

    resources: dict[str, Any] = field(default_factory=dict)  # Ray resources: cpu, gpu, memory, etc.
    num_replicas: int | None = None  # Number of worker instances (backward compatible, sets min=max=num_replicas)
    min_replicas: int | None = None  # Minimum workers required to start (default: 1)
    max_replicas: int | None = None  # Maximum workers to create (default: num_replicas or min_replicas)

    def __post_init__(self):
        """Validate and normalize replica configuration."""
        # Backward compatibility: num_replicas sets both min and max
        if self.num_replicas is not None:
            if self.min_replicas is None:
                self.min_replicas = self.num_replicas
            if self.max_replicas is None:
                self.max_replicas = self.num_replicas

        # Set defaults if not specified
        if self.min_replicas is None:
            self.min_replicas = 1
        if self.max_replicas is None:
            self.max_replicas = self.min_replicas

        # Validation
        if self.min_replicas < 1:
            raise ValueError(f"min_replicas must be >= 1, got {self.min_replicas}")
        if self.max_replicas < self.min_replicas:
            raise ValueError(
                f"max_replicas ({self.max_replicas}) must be >= min_replicas ({self.min_replicas})"
            )


@dataclass
class StageConfig:
    """Configuration for a pipeline stage."""

    name: str  # Stage name
    operators: list[OperatorConfig]  # Operators to run in this stage (with params)
    worker: StageWorkerConfig = field(default_factory=StageWorkerConfig)  # Worker configuration
    output_path: str | None = None  # Output path for this stage (if None, uses base output path + stage name)


@dataclass
class DataLoaderConfig:
    """Configuration for data loader.

    Data loading is always distributed using Ray workers for mega-scale processing.
    Each worker loads a shard of the dataset in parallel.
    """

    type: str
    params: dict[str, Any] = field(default_factory=dict)
    num_workers: int = 8  # Number of parallel loader workers (distributed sharding)
    checkpoint_interval: int = 1000  # Save checkpoint every N records per worker


@dataclass
class DataWriterConfig:
    """Configuration for data writer."""

    type: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class RejectedSamplesConfig:
    """Configuration for rejected samples collection (for deep dive analysis)."""

    enabled: bool = False  # Whether to collect rejected samples
    output_path: str | None = (
        None  # Output path for rejected samples (uses data_writer.params.output_path + "_rejected" if not set)
    )
    writer_type: str | None = None  # Writer type (uses data_writer.type if not set)


@dataclass
class MetricsConfig:
    """Configuration for metrics collection and export."""

    enabled: bool = True  # Whether to collect metrics
    output_path: str = "./metrics"  # Base directory for metrics output
    collect_custom_metrics: bool = False  # Whether to collect custom metrics from operators
    write_on_completion: bool = True  # Whether to write metrics to Parquet on run completion
    generate_report: bool = False  # Whether to generate HTML visualization report
    huggingface_repo: str | None = None  # HuggingFace Space repo ID for publishing (e.g., "username/space-name")
    huggingface_token: str | None = None  # HuggingFace API token (if None, uses HF_TOKEN env var)


@dataclass
class ExecutorConfig:
    """Configuration for executor."""

    max_samples: int | None = None
    batch_size: int = 100
    use_ray: bool = True
    # Note: In distributed Ray clusters, CPU resources are managed by the cluster.
    # This is only used for local development to limit Ray's CPU usage.
    num_cpus: int | None = None
    dedup_num_buckets: int = 2  # Number of buckets for distributed deduplication
    rejected_samples: RejectedSamplesConfig = field(default_factory=RejectedSamplesConfig)  # Rejected samples config
    metrics: MetricsConfig = field(default_factory=MetricsConfig)  # Metrics collection config


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""

    data_loader: DataLoaderConfig
    stages: list[StageConfig]  # Pipeline stages (each contains operators)
    data_writer: DataWriterConfig
    executor: ExecutorConfig

    @classmethod
    def from_yaml(cls, config_path: str) -> "PipelineConfig":
        """Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            PipelineConfig instance
        """
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)

        # Handle backward compatibility: if 'operators' and 'workers' exist, convert to 'stages'
        if "operators" in config_dict and "workers" in config_dict and "stages" not in config_dict:
            # Convert old format to new format
            operators_dict = {op["name"]: op for op in config_dict["operators"]}
            stages = []
            for w in config_dict["workers"]:
                worker_config = w.pop("resources", {})
                num_replicas = w.pop("num_replicas", 1)
                # Map operator names to operator configs
                stage_operators = [
                    OperatorConfig(**operators_dict[name]) for name in w["operator_names"] if name in operators_dict
                ]
                stages.append(
                    {
                        "name": w["name"],
                        "operators": [op.__dict__ for op in stage_operators],
                        "worker": {"resources": worker_config, "num_replicas": num_replicas},
                    }
                )
            config_dict["stages"] = stages

        # Parse stage configs
        stage_configs = []
        for stage_dict in config_dict.get("stages", []):
            worker_config_dict = stage_dict.get("worker", {})
            # Parse operators in this stage
            stage_operators = [OperatorConfig(**op) for op in stage_dict.get("operators", [])]
            stage_configs.append(
                StageConfig(
                    name=stage_dict["name"],
                    operators=stage_operators,
                    worker=StageWorkerConfig(**worker_config_dict),
                    output_path=stage_dict.get("output_path"),
                )
            )

        # Parse executor config with nested rejected_samples and metrics
        executor_dict = config_dict.get("executor", {})
        rejected_samples_dict = executor_dict.pop("rejected_samples", {})
        rejected_samples_config = RejectedSamplesConfig(**rejected_samples_dict)
        metrics_dict = executor_dict.pop("metrics", {})
        metrics_config = MetricsConfig(**metrics_dict)
        executor_config = ExecutorConfig(
            **executor_dict, rejected_samples=rejected_samples_config, metrics=metrics_config
        )

        return cls(
            data_loader=DataLoaderConfig(**config_dict["data_loader"]),
            stages=stage_configs,
            data_writer=DataWriterConfig(**config_dict["data_writer"]),
            executor=executor_config,
        )
