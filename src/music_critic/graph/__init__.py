"""Raw-inference-safe heterogeneous graph construction."""

from music_critic.graph.builder import GraphBuildError, build_raw_graph
from music_critic.graph.feature_registry import (
    FEATURE_REGISTRY_VERSION,
    RAW_FEATURE_REGISTRY,
    FeatureKind,
    FeatureRegistry,
    FeatureSpec,
    NodeType,
    Normalization,
)
from music_critic.graph.relations import (
    GRAPH_BUILDER_VERSION,
    GRAPH_SCHEMA_VERSION,
    MANDATORY_EDGE_TYPES,
    MANDATORY_NODE_TYPES,
    REVERSE_EDGE_TYPES,
)
from music_critic.graph.serialization import (
    MODEL_INPUT_FINGERPRINT_VERSION,
    dump_graph,
    dumps_graph,
    graph_fingerprint,
    graph_to_dict,
    model_input_fingerprint,
)
from music_critic.graph.validation import (
    ALLOWED_EDGE_ATTRIBUTES,
    ALLOWED_GLOBAL_ATTRIBUTES,
    BATCH_BASE_NODE_ATTRIBUTES,
    BATCH_CANDIDATE_NODE_ATTRIBUTES,
    BATCH_EDGE_ATTRIBUTES,
    BATCH_GLOBAL_ATTRIBUTES,
    BASE_NODE_ATTRIBUTES,
    CANDIDATE_NODE_ATTRIBUTES,
    GraphContractError,
    validate_raw_graph,
    validate_raw_graph_batch,
)


__all__ = [
    "FEATURE_REGISTRY_VERSION",
    "GRAPH_BUILDER_VERSION",
    "GRAPH_SCHEMA_VERSION",
    "MANDATORY_EDGE_TYPES",
    "MANDATORY_NODE_TYPES",
    "MODEL_INPUT_FINGERPRINT_VERSION",
    "RAW_FEATURE_REGISTRY",
    "REVERSE_EDGE_TYPES",
    "ALLOWED_EDGE_ATTRIBUTES",
    "ALLOWED_GLOBAL_ATTRIBUTES",
    "BATCH_BASE_NODE_ATTRIBUTES",
    "BATCH_CANDIDATE_NODE_ATTRIBUTES",
    "BATCH_EDGE_ATTRIBUTES",
    "BATCH_GLOBAL_ATTRIBUTES",
    "BASE_NODE_ATTRIBUTES",
    "CANDIDATE_NODE_ATTRIBUTES",
    "FeatureKind",
    "FeatureRegistry",
    "FeatureSpec",
    "GraphBuildError",
    "GraphContractError",
    "NodeType",
    "Normalization",
    "build_raw_graph",
    "dump_graph",
    "dumps_graph",
    "graph_fingerprint",
    "graph_to_dict",
    "model_input_fingerprint",
    "validate_raw_graph",
    "validate_raw_graph_batch",
]
