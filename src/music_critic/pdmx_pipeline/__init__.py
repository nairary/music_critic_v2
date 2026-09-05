"""Explicit PDMX source manifests, window preprocessing and raw SSL loading."""

from .manifest import (
    LineageLink, PdmxPipelineError, SelectionConfig, SourceManifest, SourceRecord,
    load_source_manifest, plan_selection,
)
from .windowing import WindowConfig, WindowResult, WindowingError, iter_bar_windows
from .pipeline import PipelineConfig, PipelineResult, preprocess_pdmx, read_publication
from .loading import PdmxWindowDataset

__all__ = [
    "LineageLink", "PdmxPipelineError", "SelectionConfig", "SourceManifest", "SourceRecord",
    "load_source_manifest", "plan_selection", "WindowConfig", "WindowResult", "WindowingError",
    "iter_bar_windows", "PipelineConfig", "PipelineResult", "preprocess_pdmx", "read_publication",
    "PdmxWindowDataset",
]
