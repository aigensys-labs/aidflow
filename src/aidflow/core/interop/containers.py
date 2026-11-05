"""
Dictionary Containers — Deep Copy and Structural Extraction.

This module provides low-level utilities for:

- **Deep Copy**: Recursively copying nested dict structures
  while preserving `DictDataclass` type identity (sub-
  dataclasses are re-instantiated with their original class).
- **Flattened Extraction**: Extracting all leaf values from
  arbitrarily nested dictionaries into a flat tuple — useful
  for schema comparison and structural validation.
"""

from ...deps.types import Any
from ...utils.builtins import List
from ..factory import DictDataclass, DictClass


class Dict:
    """
    Dictionary structural utilities — deep copy and value
    extraction.

    Provides type-preserving deep copies of nested dict
    structures (maintaining `DictDataclass` identity at each
    level) and flat value extraction for schema inspection.
    """

    @classmethod
    def deep_copy(cls, org:dict) -> dict:
        """
        Create a deep copy of a nested dict structure, preserving
        `DictDataclass` type identity at each level.

        - **dict values**: Recursed into (deep-copied).
        - **DictDataclass values**: Re-instantiated with the
          same class after deep-copying their contents.
        - **Other values**: Referenced as-is (assumed immutable
          or intentionally shared).

        Parameters
        ----------
        org : dict | DictDataclass
            The source structure to copy.

        Returns
        -------
        dict | DictDataclass
            A new structure with the same shape and types,
            independent of the original.
        """        
        new = {}
        for key, value in org.items():
            if isinstance(value, dict):
                new[key] = cls.deep_copy(value)
            elif isinstance(value, DictDataclass):
                dict_cls = value.__class__
                new[key] = dict_cls(**cls.deep_copy(value))
            else:
                new[key] = value 
        if isinstance(org, DictDataclass):
            return org.__class__(**new)
        return new

    @staticmethod
    def get_flattened_values(dct: dict|Any) -> tuple:
        """
        Extract all leaf values from an arbitrarily nested
        dictionary structure into a single flat tuple.

        Handles:
        - **dict / DictClass / DictDataclass**: Recurses into
          nested values, collecting only non-dict leaves.
        - **Other types**: Wrapped via `List.ensure` and
          returned as a single-element tuple.

        Parameters
        ----------
        dct : dict | DictClass | DictDataclass | Any
            The structure to flatten.

        Returns
        -------
        tuple
            All leaf values in traversal order.
        """
        if not (
            isinstance(dct, (dict, DictClass, DictDataclass)) or
            (
                isinstance(dct, type) and 
                issubclass(dct, (DictClass, DictDataclass))
            )
        ):
            return tuple(List.ensure(dct))
        
        def _extract(nested_dict:dict):
            for value in nested_dict.values():
                if isinstance(value, dict):
                    yield from _extract(value)
                else:
                    yield value

        return tuple(_extract(dct))