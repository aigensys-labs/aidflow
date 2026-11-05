"""
AIDflow Production Exporter.

Exports a fitted champion agent to production-ready deployment
artifacts. The exporter serializes the self-contained scoring
object (model + preprocessing + feature lineage) and generates
an orchestration layer for scheduled inference.
"""

from .exporter import ProductionExporter, ExportConfig