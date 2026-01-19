"""
Configuration Management

YAML-based configuration classes for the pipeline framework.
"""

import yaml
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class OperatorConfig:
    """Configuration for a single operator."""
    name: str  # Operator name (e.g., "image_metadata_refiner") -> class name (e.g., "ImageMetadataRefiner")
    params: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    
    def get_class_name(self) -> str:
        """Convert snake_case name to PascalCase class name.
        
        Returns:
            Class name in PascalCase
        """
        parts = self.name.split('_')
        return ''.join(word.capitalize() for word in parts)


@dataclass
class StageWorkerConfig:
    """Configuration for workers in a stage."""
    resources: Dict[str, Any] = field(default_factory=dict)  # Ray resources: cpu, gpu, memory, etc.
    num_replicas: int = 1  # Number of worker instances to create


@dataclass
class StageConfig:
    """Configuration for a pipeline stage."""
    name: str  # Stage name
    operators: List[OperatorConfig]  # Operators to run in this stage (with params)
    worker: StageWorkerConfig = field(default_factory=StageWorkerConfig)  # Worker configuration
    output_path: Optional[str] = None  # Output path for this stage (if None, uses base output path + stage name)


@dataclass
class DataLoaderConfig:
    """Configuration for data loader."""
    type: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataWriterConfig:
    """Configuration for data writer."""
    type: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutorConfig:
    """Configuration for executor."""
    max_samples: Optional[int] = None
    batch_size: int = 100
    use_ray: bool = True
    # Note: In distributed Ray clusters, CPU resources are managed by the cluster.
    # This is only used for local development to limit Ray's CPU usage.
    num_cpus: Optional[int] = None
    dedup_num_buckets: int = 2  # Number of buckets for distributed deduplication


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""
    data_loader: DataLoaderConfig
    stages: List[StageConfig]  # Pipeline stages (each contains operators)
    data_writer: DataWriterConfig
    executor: ExecutorConfig
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'PipelineConfig':
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            PipelineConfig instance
        """
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Handle backward compatibility: if 'operators' and 'workers' exist, convert to 'stages'
        if 'operators' in config_dict and 'workers' in config_dict and 'stages' not in config_dict:
            # Convert old format to new format
            operators_dict = {op['name']: op for op in config_dict['operators']}
            stages = []
            for w in config_dict['workers']:
                worker_config = w.pop('resources', {})
                num_replicas = w.pop('num_replicas', 1)
                # Map operator names to operator configs
                stage_operators = [
                    OperatorConfig(**operators_dict[name])
                    for name in w['operator_names']
                    if name in operators_dict
                ]
                stages.append({
                    'name': w['name'],
                    'operators': [op.__dict__ for op in stage_operators],
                    'worker': {
                        'resources': worker_config,
                        'num_replicas': num_replicas
                    }
                })
            config_dict['stages'] = stages
        
        # Parse stage configs
        stage_configs = []
        for stage_dict in config_dict.get('stages', []):
            worker_config_dict = stage_dict.get('worker', {})
            # Parse operators in this stage
            stage_operators = [OperatorConfig(**op) for op in stage_dict.get('operators', [])]
            stage_configs.append(StageConfig(
                name=stage_dict['name'],
                operators=stage_operators,
                worker=StageWorkerConfig(**worker_config_dict),
                output_path=stage_dict.get('output_path')
            ))
        
        return cls(
            data_loader=DataLoaderConfig(**config_dict['data_loader']),
            stages=stage_configs,
            data_writer=DataWriterConfig(**config_dict['data_writer']),
            executor=ExecutorConfig(**config_dict.get('executor', {}))
        )
