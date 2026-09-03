"""Extension interface for independently implemented source EDA adapters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from music_critic.eda.contracts import (
    CorpusId,
    EDA_ADAPTER_REGISTRY_VERSION,
    EDAContractError,
    RawCorpusEDA,
    SupervisionEDA,
    VersionedIdentity,
    corpus_eda_capability,
    source_extension_namespace_is_valid,
)


@runtime_checkable
class SourceEDAAdapter(Protocol):
    """Minimal raw-capable source-owned implementation surface.

    A child branch owns discovery and aggregation.  The shared registry owns
    capability checks and validates every returned report before exposing it.
    ``request`` is deliberately opaque so adding source-local configuration
    never changes the shared schema.
    """

    corpus: CorpusId
    adapter_identity: VersionedIdentity
    extension_namespaces: tuple[str, ...]

    def build_raw_eda(self, request: object) -> RawCorpusEDA:
        ...


@runtime_checkable
class SupervisionSourceEDAAdapter(SourceEDAAdapter, Protocol):
    """Additional surface required only from supervision-capable corpora."""

    def build_supervision_eda(self, request: object) -> SupervisionEDA:
        ...


@dataclass(frozen=True, slots=True)
class _AdapterRegistration:
    corpus: CorpusId
    adapter_identity: VersionedIdentity
    extension_namespaces: tuple[str, ...]
    adapter: SourceEDAAdapter


class EDAAdapterRegistry:
    """Process-local registry with deterministic, fail-closed registration."""

    contract_version = EDA_ADAPTER_REGISTRY_VERSION

    def __init__(self) -> None:
        self._registrations: dict[CorpusId, _AdapterRegistration] = {}

    def register(self, adapter: SourceEDAAdapter) -> None:
        if not isinstance(adapter, SourceEDAAdapter) or not callable(
            getattr(adapter, "build_raw_eda", None)
        ):
            raise EDAContractError(
                "eda.adapter.interface_invalid",
                "adapter does not implement the SourceEDAAdapter protocol",
            )
        try:
            corpus = CorpusId(adapter.corpus)
        except (TypeError, ValueError) as exc:
            raise EDAContractError(
                "eda.adapter.corpus_invalid", f"unknown adapter corpus {adapter.corpus!r}"
            ) from exc
        if corpus_eda_capability(corpus).supervision_eda and (
            not isinstance(adapter, SupervisionSourceEDAAdapter)
            or not callable(getattr(adapter, "build_supervision_eda", None))
        ):
            raise EDAContractError(
                "eda.adapter.interface_invalid",
                "supervision-capable corpus adapter must implement build_supervision_eda",
            )
        if not isinstance(adapter.adapter_identity, VersionedIdentity):
            raise EDAContractError(
                "eda.adapter.identity_invalid",
                "adapter_identity must be a VersionedIdentity",
            )
        if corpus in self._registrations:
            raise EDAContractError(
                "eda.adapter.duplicate", f"an adapter is already registered for {corpus.value}"
            )
        declared_namespaces = adapter.extension_namespaces
        if not isinstance(declared_namespaces, tuple) or any(
            not isinstance(value, str) for value in declared_namespaces
        ):
            raise EDAContractError(
                "eda.adapter.extension_namespace_invalid",
                "adapter extension_namespaces must be a tuple of strings",
            )
        namespaces = tuple(declared_namespaces)
        if (
            namespaces != tuple(sorted(namespaces))
            or len(namespaces) != len(set(namespaces))
            or any(
                not source_extension_namespace_is_valid(value, corpus)
                for value in namespaces
            )
        ):
            raise EDAContractError(
                "eda.adapter.extension_namespace_invalid",
                "adapter extension namespaces must be unique, sorted, and source-owned",
            )
        self._registrations[corpus] = _AdapterRegistration(
            corpus=corpus,
            adapter_identity=adapter.adapter_identity,
            extension_namespaces=namespaces,
            adapter=adapter,
        )

    def registrations(self) -> tuple[tuple[CorpusId, VersionedIdentity, tuple[str, ...]], ...]:
        """Return a deterministic immutable registry view."""

        return tuple(
            (
                corpus,
                self._registrations[corpus].adapter_identity,
                self._registrations[corpus].extension_namespaces,
            )
            for corpus in sorted(self._registrations, key=lambda item: item.value)
        )

    def build_raw(self, corpus: CorpusId | str, request: object) -> RawCorpusEDA:
        normalized, registration = self._resolve(corpus)
        adapter = registration.adapter
        report = adapter.build_raw_eda(request)
        if type(report) is not RawCorpusEDA:
            raise EDAContractError(
                "eda.adapter.raw_result_invalid", "raw adapter must return RawCorpusEDA"
            )
        self._validate_result(normalized, registration.adapter_identity, report)
        declared = {item.namespace for item in report.semantic_payload.extensions}
        if not declared <= set(registration.extension_namespaces):
            raise EDAContractError(
                "eda.adapter.extension_undeclared",
                "raw report uses an extension namespace absent from registration",
            )
        return report

    def build_supervision(
        self, corpus: CorpusId | str, request: object
    ) -> SupervisionEDA:
        normalized = self._corpus(corpus)
        if not corpus_eda_capability(normalized).supervision_eda:
            raise EDAContractError(
                "eda.capability.supervision_forbidden",
                "PDMX cannot create even an empty placeholder supervision report",
            )
        normalized, registration = self._resolve(normalized)
        adapter = registration.adapter
        if not isinstance(adapter, SupervisionSourceEDAAdapter):  # pragma: no cover
            raise EDAContractError(
                "eda.adapter.interface_invalid",
                "registered adapter has no supervision builder",
            )
        report = adapter.build_supervision_eda(request)
        if type(report) is not SupervisionEDA:
            raise EDAContractError(
                "eda.adapter.supervision_result_invalid",
                "supervision adapter must return SupervisionEDA",
            )
        self._validate_result(normalized, registration.adapter_identity, report)
        declared = {item.namespace for item in report.semantic_payload.extensions}
        if not declared <= set(registration.extension_namespaces):
            raise EDAContractError(
                "eda.adapter.extension_undeclared",
                "supervision report uses an extension namespace absent from registration",
            )
        return report

    @property
    def adapters(self) -> Mapping[CorpusId, SourceEDAAdapter]:
        """Read-only snapshot; mutation remains owned by ``register``."""

        return MappingProxyType(
            {
                corpus: registration.adapter
                for corpus, registration in self._registrations.items()
            }
        )

    def _resolve(
        self, corpus: CorpusId | str
    ) -> tuple[CorpusId, _AdapterRegistration]:
        normalized = self._corpus(corpus)
        try:
            return normalized, self._registrations[normalized]
        except KeyError as exc:
            raise EDAContractError(
                "eda.adapter.not_registered", f"no EDA adapter for {normalized.value}"
            ) from exc

    @staticmethod
    def _corpus(corpus: CorpusId | str) -> CorpusId:
        try:
            return CorpusId(corpus)
        except (TypeError, ValueError) as exc:
            raise EDAContractError(
                "eda.adapter.corpus_invalid", f"unknown corpus {corpus!r}"
            ) from exc

    @staticmethod
    def _validate_result(
        corpus: CorpusId,
        adapter_identity: VersionedIdentity,
        report: RawCorpusEDA | SupervisionEDA,
    ) -> None:
        if report.envelope.corpus != corpus:
            raise EDAContractError(
                "eda.adapter.result_corpus_mismatch",
                "adapter result corpus differs from its registration",
            )
        if report.envelope.producer_identity != adapter_identity:
            raise EDAContractError(
                "eda.adapter.result_producer_mismatch",
                "report producer identity differs from registered adapter identity",
            )


__all__ = [
    "EDAAdapterRegistry",
    "SourceEDAAdapter",
    "SupervisionSourceEDAAdapter",
]
