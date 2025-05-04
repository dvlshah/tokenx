"""
Provider Registry for LLM Meter

This module handles provider registration and discovery for cost calculation.
"""

from typing import Dict, List, Optional, Any
import importlib
import inspect
import pkgutil
from pathlib import Path

from .base import ProviderAdapter
from ..errors import enhance_provider_adapter


class ProviderRegistry:
    """Registry for LLM provider adapters."""

    _providers: Dict[str, ProviderAdapter] = {}
    _initialized = False

    @classmethod
    def register(cls, provider: ProviderAdapter) -> None:
        """
        Register a provider adapter.

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
        return cls._providers.get(name)

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
        cls, func: Any, args: tuple, kwargs: dict
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
        """
        if not cls._initialized:
            cls._discover_providers()
            cls._initialized = True

    @classmethod
    def _discover_providers(cls) -> None:
        """
        Auto-discover and register provider adapters.
        """
        # Get the path to the providers package
        providers_path = Path(__file__).parent

        # Iterate through all modules in the providers package
        for _, name, is_pkg in pkgutil.iter_modules([str(providers_path)]):
            if name != "base" and not is_pkg:
                # Import the module
                module = importlib.import_module(f".{name}", package=__name__)

                # Find all provider adapter classes
                for _, obj in inspect.getmembers(module):
                    # Check if it's a class that inherits from ProviderAdapter
                    if (
                        inspect.isclass(obj)
                        and issubclass(obj, ProviderAdapter)
                        and obj is not ProviderAdapter
                    ):
                        # Check if there's a creator function available
                        creator_func_name = f"create_{name}_adapter"
                        if hasattr(module, creator_func_name):
                            # Use the creator function to get an enhanced adapter
                            creator_func = getattr(module, creator_func_name)
                            adapter = creator_func()
                            cls.register(adapter)
                        else:
                            # Instantiate and register the provider normally
                            cls.register(obj())
