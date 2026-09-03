from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from music_critic.eda import (
    CorpusId,
    EDAAdapterRegistry,
    EDAContractError,
    RawCorpusEDA,
    SupervisionEDA,
    VersionedIdentity,
)


@dataclass
class DummyAdapter:
    corpus: CorpusId
    adapter_identity: VersionedIdentity
    extension_namespaces: tuple[str, ...]
    raw: RawCorpusEDA
    supervision: SupervisionEDA | None
    supervision_calls: int = 0

    def build_raw_eda(self, request: object) -> RawCorpusEDA:
        return self.raw

    def build_supervision_eda(self, request: object) -> SupervisionEDA:
        self.supervision_calls += 1
        if self.supervision is None:
            raise AssertionError("supervision builder must not be called")
        return self.supervision


@dataclass
class RawOnlyPDMXAdapter:
    corpus: CorpusId
    adapter_identity: VersionedIdentity
    extension_namespaces: tuple[str, ...]
    raw: RawCorpusEDA

    def build_raw_eda(self, request: object) -> RawCorpusEDA:
        return self.raw


def test_adapter_registration_and_result_validation(
    raw_reports, supervision_reports
) -> None:
    raw = raw_reports[CorpusId.HOOKTHEORY]
    supervision = supervision_reports[CorpusId.HOOKTHEORY]
    adapter = DummyAdapter(
        corpus=CorpusId.HOOKTHEORY,
        adapter_identity=raw.envelope.producer_identity,
        extension_namespaces=("hooktheory.fixture_supervision",),
        raw=raw,
        supervision=supervision,
    )
    registry = EDAAdapterRegistry()
    registry.register(adapter)
    assert registry.build_raw("hooktheory", object()) is raw
    assert registry.build_supervision("hooktheory", object()) is supervision
    assert registry.registrations() == (
        (
            CorpusId.HOOKTHEORY,
            adapter.adapter_identity,
            adapter.extension_namespaces,
        ),
    )
    with pytest.raises(EDAContractError, match="adapter.duplicate"):
        registry.register(adapter)


def test_pdmx_capability_rejects_before_adapter_supervision_method(raw_reports) -> None:
    raw = raw_reports[CorpusId.PDMX]
    adapter = DummyAdapter(
        corpus=CorpusId.PDMX,
        adapter_identity=raw.envelope.producer_identity,
        extension_namespaces=(),
        raw=raw,
        supervision=None,
    )
    registry = EDAAdapterRegistry()
    registry.register(adapter)
    with pytest.raises(EDAContractError, match="supervision_forbidden"):
        registry.build_supervision(CorpusId.PDMX, object())
    assert adapter.supervision_calls == 0


def test_pdmx_raw_only_adapter_needs_no_forbidden_supervision_stub(raw_reports) -> None:
    raw = raw_reports[CorpusId.PDMX]
    adapter = RawOnlyPDMXAdapter(
        corpus=CorpusId.PDMX,
        adapter_identity=raw.envelope.producer_identity,
        extension_namespaces=(),
        raw=raw,
    )
    registry = EDAAdapterRegistry()
    registry.register(adapter)

    assert registry.build_raw(CorpusId.PDMX, object()) is raw
    with pytest.raises(EDAContractError, match="supervision_forbidden"):
        registry.build_supervision(CorpusId.PDMX, object())


def test_registration_rejects_non_callable_raw_builder(raw_reports) -> None:
    raw = raw_reports[CorpusId.PDMX]

    class NonCallableRawAdapter:
        corpus = CorpusId.PDMX
        adapter_identity = raw.envelope.producer_identity
        extension_namespaces: tuple[str, ...] = ()
        build_raw_eda = 42

    with pytest.raises(EDAContractError, match="interface_invalid"):
        EDAAdapterRegistry().register(NonCallableRawAdapter())


def test_registration_rejects_non_callable_supervision_builder(
    raw_reports,
) -> None:
    raw = raw_reports[CorpusId.HOOKTHEORY]

    class NonCallableSupervisionAdapter:
        corpus = CorpusId.HOOKTHEORY
        adapter_identity = raw.envelope.producer_identity
        extension_namespaces: tuple[str, ...] = ()
        build_supervision_eda = 42

        def build_raw_eda(self, request: object) -> RawCorpusEDA:
            return raw

    with pytest.raises(EDAContractError, match="interface_invalid"):
        EDAAdapterRegistry().register(NonCallableSupervisionAdapter())


def test_registry_rejects_forged_raw_report_subclass(raw_reports) -> None:
    raw = raw_reports[CorpusId.PDMX]

    @dataclass(frozen=True, slots=True)
    class ExtendedRawCorpusEDA(RawCorpusEDA):
        extra: str = "outside-contract"

    forged = object.__new__(ExtendedRawCorpusEDA)
    object.__setattr__(forged, "envelope", raw.envelope)
    object.__setattr__(forged, "semantic_payload", raw.semantic_payload)
    object.__setattr__(forged, "semantic_fingerprint", raw.semantic_fingerprint)
    object.__setattr__(forged, "extra", "outside-contract")
    adapter = RawOnlyPDMXAdapter(
        corpus=CorpusId.PDMX,
        adapter_identity=raw.envelope.producer_identity,
        extension_namespaces=(),
        raw=forged,
    )
    registry = EDAAdapterRegistry()
    registry.register(adapter)
    with pytest.raises(EDAContractError, match="raw_result_invalid"):
        registry.build_raw(CorpusId.PDMX, object())


def test_adapter_result_must_match_registered_producer(raw_reports) -> None:
    raw = raw_reports[CorpusId.HOOKTHEORY]
    wrong = VersionedIdentity("hooktheory.other_adapter", "1.0.0", "f" * 64)
    adapter = DummyAdapter(
        corpus=CorpusId.HOOKTHEORY,
        adapter_identity=wrong,
        extension_namespaces=(),
        raw=raw,
        supervision=None,
    )
    registry = EDAAdapterRegistry()
    registry.register(adapter)
    with pytest.raises(EDAContractError, match="result_producer_mismatch"):
        registry.build_raw(CorpusId.HOOKTHEORY, object())


@pytest.mark.parametrize(
    "bad_namespace",
    ("other.bad", "pop909_cl.", "pop909_cl..x", "pop909_cl. x", "pop909_cl.\x00x"),
)
def test_registration_requires_source_owned_sorted_extension_namespaces(
    raw_reports,
    bad_namespace,
) -> None:
    raw = raw_reports[CorpusId.POP909_CL]
    adapter = DummyAdapter(
        corpus=CorpusId.POP909_CL,
        adapter_identity=raw.envelope.producer_identity,
        extension_namespaces=(bad_namespace,),
        raw=raw,
        supervision=None,
    )
    with pytest.raises(EDAContractError, match="extension_namespace_invalid"):
        EDAAdapterRegistry().register(adapter)


def test_registration_rejects_non_string_extension_namespace(raw_reports) -> None:
    raw = raw_reports[CorpusId.POP909_CL]
    adapter = DummyAdapter(
        corpus=CorpusId.POP909_CL,
        adapter_identity=raw.envelope.producer_identity,
        extension_namespaces=(1,),  # type: ignore[arg-type]
        raw=raw,
        supervision=None,
    )
    with pytest.raises(EDAContractError, match="extension_namespace_invalid"):
        EDAAdapterRegistry().register(adapter)


def test_registration_snapshots_identity_against_adapter_mutation(raw_reports) -> None:
    raw = raw_reports[CorpusId.HOOKTHEORY]
    registered_identity = raw.envelope.producer_identity
    adapter = DummyAdapter(
        corpus=CorpusId.HOOKTHEORY,
        adapter_identity=registered_identity,
        extension_namespaces=(),
        raw=raw,
        supervision=None,
    )
    registry = EDAAdapterRegistry()
    registry.register(adapter)

    adapter.adapter_identity = VersionedIdentity(
        "hooktheory.mutated_adapter", "1.0.0", "9" * 64
    )
    assert registry.registrations() == (
        (CorpusId.HOOKTHEORY, registered_identity, ()),
    )
    assert registry.build_raw(CorpusId.HOOKTHEORY, object()) is raw


def test_registration_snapshots_declared_namespaces(
    raw_reports,
    supervision_reports,
) -> None:
    raw = raw_reports[CorpusId.HOOKTHEORY]
    supervision = supervision_reports[CorpusId.HOOKTHEORY]
    adapter = DummyAdapter(
        corpus=CorpusId.HOOKTHEORY,
        adapter_identity=raw.envelope.producer_identity,
        extension_namespaces=(),
        raw=raw,
        supervision=supervision,
    )
    registry = EDAAdapterRegistry()
    registry.register(adapter)

    adapter.extension_namespaces = ("hooktheory.fixture_supervision",)
    with pytest.raises(EDAContractError, match="extension_undeclared"):
        registry.build_supervision(CorpusId.HOOKTHEORY, object())


def test_adapter_mapping_snapshot_is_read_only(raw_reports) -> None:
    raw = raw_reports[CorpusId.PDMX]
    adapter = RawOnlyPDMXAdapter(
        corpus=CorpusId.PDMX,
        adapter_identity=raw.envelope.producer_identity,
        extension_namespaces=(),
        raw=raw,
    )
    registry = EDAAdapterRegistry()
    registry.register(adapter)

    with pytest.raises(TypeError):
        registry.adapters[CorpusId.PDMX] = adapter
    assert registry.adapters[CorpusId.PDMX] is adapter
