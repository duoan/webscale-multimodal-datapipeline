"""
Registry: Component registries for dynamic instantiation

Provides registries for operators, loaders, and writers.
"""

from typing import Any

from .base import DataLoader, DataWriter
from .operator import Operator


class OperatorRegistry:
    """Registry for operator classes."""

    _operators: dict[str, type[Operator]] = {}

    @classmethod
    def register(cls, name: str, operator_class: type[Operator]):
        """Register an operator class.

        Args:
            name: Operator type name (used in config)
            operator_class: Operator class to register
        """
        cls._operators[name] = operator_class

    @classmethod
    def create(cls, name: str, params: dict[str, Any] = None) -> Operator:
        """Create an operator instance from registry.

        Args:
            name: Operator type name
            params: Parameters for operator initialization

        Returns:
            Operator instance

        Raises:
            ValueError: If operator name not found in registry
        """
        if name not in cls._operators:
            raise ValueError(
                f"Operator '{name}' not found in registry. Available operators: {list(cls._operators.keys())}"
            )

        params = params or {}
        return cls._operators[name](**params)

    @classmethod
    def list_operators(cls) -> list[str]:
        """List all registered operator names.

        Returns:
            List of registered operator type names
        """
        return list(cls._operators.keys())


class DataLoaderRegistry:
    """Registry for data loader classes."""

    _loaders: dict[str, type[DataLoader]] = {}

    @classmethod
    def register(cls, name: str, loader_class: type[DataLoader]):
        """Register a data loader class.

        Args:
            name: Loader type name (used in config)
            loader_class: Loader class to register
        """
        cls._loaders[name] = loader_class

    @classmethod
    def create(cls, name: str, params: dict[str, Any] = None) -> DataLoader:
        """Create a data loader instance from registry.

        Args:
            name: Loader type name
            params: Parameters for loader initialization

        Returns:
            DataLoader instance

        Raises:
            ValueError: If loader name not found in registry
        """
        if name not in cls._loaders:
            raise ValueError(
                f"DataLoader '{name}' not found in registry. Available loaders: {list(cls._loaders.keys())}"
            )

        params = params or {}
        return cls._loaders[name](**params)


class DataWriterRegistry:
    """Registry for data writer classes."""

    _writers: dict[str, type[DataWriter]] = {}

    @classmethod
    def register(cls, name: str, writer_class: type[DataWriter]):
        """Register a data writer class.

        Args:
            name: Writer type name (used in config)
            writer_class: Writer class to register
        """
        cls._writers[name] = writer_class

    @classmethod
    def create(cls, name: str, params: dict[str, Any] = None) -> DataWriter:
        """Create a data writer instance from registry.

        Args:
            name: Writer type name
            params: Parameters for writer initialization

        Returns:
            DataWriter instance

        Raises:
            ValueError: If writer name not found in registry
        """
        if name not in cls._writers:
            raise ValueError(
                f"DataWriter '{name}' not found in registry. Available writers: {list(cls._writers.keys())}"
            )

        params = params or {}
        return cls._writers[name](**params)
