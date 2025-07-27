"""
Provider Registry for LLM Meter

This module handles provider registration and discovery for cost calculation.
"""

from typing import Dict, List, Optional, Any, Type, Callable, Tuple
import importlib
import inspect
import pkgutil
import threading
from pathlib import Path

from .base import ProviderAdapter
from ..errors import enhance_provider_adapter


class ProviderAlreadyRegisteredError(Exception):
    """Exception raised when trying to register a provider that already exists."""

    pass


# Global registry for provider adapters
_PROVIDER_REGISTRY: Dict[str, Type[ProviderAdapter]] = {}

# Registry for calculator classes (for backward compatibility)
_CALCULATOR_REGISTRY: Dict[str, type] = {}

# Thread safety lock for registry operations
_REGISTRY_LOCK = threading.Lock()


def register_provider(
    name: str,
) -> Callable[[Type[ProviderAdapter]], Type[ProviderAdapter]]:
    """
    Decorator to register a provider adapter with the global registry.

    Args:
        name: Provider name identifier

    Returns:
        Decorator function that registers the provider class

    Raises:
        ProviderAlreadyRegisteredError: If provider name is already registered

    Example:
        @register_provider("custom")
        class CustomAdapter(ProviderAdapter):
            @property
            def provider_name(self) -> str:
                return "custom"
            # ... implement other methods
    """

    def decorator(cls: Type[ProviderAdapter]) -> Type[ProviderAdapter]:
        with _REGISTRY_LOCK:
            if name in _PROVIDER_REGISTRY:
                raise ProviderAlreadyRegisteredError(
                    f"Provider '{name}' is already registered with class {_PROVIDER_REGISTRY[name].__name__}"
                )

            # Validate that the class implements ProviderAdapter
            if not issubclass(cls, ProviderAdapter):
                raise TypeError(
                    f"Provider class {cls.__name__} must inherit from ProviderAdapter"
                )

            _PROVIDER_REGISTRY[name] = cls
        return cls

    return decorator


class ProviderRegistry:
    """Registry for LLM provider adapters."""

    _providers: Dict[str, ProviderAdapter] = {}
    _initialized = False

    @classmethod
    def register(cls, provider: ProviderAdapter) -> None:
        """
        Register a provider adapter instance.

        Args:
            provider: Provider adapter instance to register
        """
        # Apply enhanced error handling before registration
        enhanced_provider = enhance_provider_adapter(provider)
        cls._providers[provider.provider_name] = enhanced_provider

    @classmethod
    def get_provider(cls, name: str) -> Optional[ProviderAdapter]:
        """
        Get a provider adapter by name.

        Args:
            name: Provider name

        Returns:
            ProviderAdapter or None if not found
        """
        cls._ensure_initialized()

        # Check if we have an instance already
        if name in cls._providers:
            return cls._providers[name]

        # Check the global registry for class definitions
        if name in _PROVIDER_REGISTRY:
            provider_class = _PROVIDER_REGISTRY[name]
            # Instantiate and register the provider
            provider_instance = provider_class()
            cls.register(provider_instance)
            return cls._providers[name]

        return None

    @classmethod
    def get_calculator_class(cls, provider_name: str) -> Optional[type]:
        """
        Get the calculator class for a provider (for backward compatibility).

        Args:
            provider_name: Provider name

        Returns:
            Calculator class if registered, None otherwise
        """
        return _CALCULATOR_REGISTRY.get(provider_name)

    @classmethod
    def register_calculator_class(
        cls, provider_name: str, calculator_class: type
    ) -> None:
        """
        Register a calculator class for a provider.

        Args:
            provider_name: Provider name
            calculator_class: Calculator class to register
        """
        _CALCULATOR_REGISTRY[provider_name] = calculator_class

    @classmethod
    def get_all_providers(cls) -> List[ProviderAdapter]:
        """
        Get all registered provider adapters.

        Returns:
            List of provider adapters
        """
        cls._ensure_initialized()
        return list(cls._providers.values())

    @classmethod
    def detect_provider(
        cls, func: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Optional[ProviderAdapter]:
        """
        Auto-detect the provider based on the function and its arguments.

        Args:
            func: The function being called
            args: Function arguments
            kwargs: Function keyword arguments

        Returns:
            ProviderAdapter if detected, None otherwise
        """
        cls._ensure_initialized()
        for provider in cls._providers.values():
            if provider.matches_function(func, args, kwargs):
                return provider
        return None

    @classmethod
    def _ensure_initialized(cls) -> None:
        """
        Ensure the registry is initialized by auto-discovering providers.
        Uses double-checked locking pattern for thread safety.
        """
        if not cls._initialized:
            with _REGISTRY_LOCK:
                if not cls._initialized:  # Double check after acquiring lock
                    cls._discover_providers()
                    cls._initialized = True

    @classmethod
    def _discover_providers(cls) -> None:
        """
        Auto-discover and register provider adapters.
        Includes robust error handling for import failures.
        """
        import warnings

        # Get the path to the providers package
        providers_path = Path(__file__).parent

        # Iterate through all modules in the providers package
        for _, name, is_pkg in pkgutil.iter_modules([str(providers_path)]):
            if name != "base" and not is_pkg:
                try:
                    # Import the module
                    module = importlib.import_module(f".{name}", package=__name__)

                    # Find all provider adapter classes
                    for _, obj in inspect.getmembers(module):
                        try:
                            # Check if it's a class that inherits from ProviderAdapter
                            if (
                                inspect.isclass(obj)
                                and issubclass(obj, ProviderAdapter)
                                and obj is not ProviderAdapter
                            ):
                                # Register the class in the global registry if not already there
                                provider_name = name  # Use module name as provider name
                                if provider_name not in _PROVIDER_REGISTRY:
                                    _PROVIDER_REGISTRY[provider_name] = obj

                                # Check if there's a creator function available
                                creator_func_name = f"create_{name}_adapter"
                                if hasattr(module, creator_func_name):
                                    try:
                                        # Use the creator function to get an enhanced adapter
                                        creator_func = getattr(
                                            module, creator_func_name
                                        )
                                        adapter = creator_func()
                                        cls.register(adapter)
                                    except Exception as e:
                                        warnings.warn(
                                            f"Failed to create adapter for provider '{name}' using creator function: {e}",
                                            RuntimeWarning,
                                        )
                                        # Fall back to normal instantiation
                                        try:
                                            cls.register(obj())
                                        except Exception as fallback_e:
                                            warnings.warn(
                                                f"Failed to instantiate provider '{name}': {fallback_e}",
                                                RuntimeWarning,
                                            )
                                else:
                                    try:
                                        # Instantiate and register the provider normally
                                        cls.register(obj())
                                    except Exception as e:
                                        warnings.warn(
                                            f"Failed to instantiate provider '{name}': {e}",
                                            RuntimeWarning,
                                        )
                        except Exception as e:
                            # Continue processing other objects if one fails
                            warnings.warn(
                                f"Error processing object in provider module '{name}': {e}",
                                RuntimeWarning,
                            )
                            continue

                except ImportError as e:
                    # Warn about failed imports but continue discovery
                    warnings.warn(
                        f"Failed to import provider module '{name}': {e}",
                        RuntimeWarning,
                    )
                    continue
                except Exception as e:
                    # Catch any other unexpected errors during module processing
                    warnings.warn(
                        f"Unexpected error processing provider module '{name}': {e}",
                        RuntimeWarning,
                    )
                    continue


# Public exports
__all__ = ["ProviderRegistry", "register_provider", "ProviderAlreadyRegisteredError"]
