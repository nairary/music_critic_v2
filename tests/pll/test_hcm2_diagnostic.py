from dataclasses import replace

import torch

from scripts.hcm2_pilot_diagnostic import ablate_motifs, variants, read_bound, bounded_generate, bindings_equal
from tests.pll.test_whole_model import small_model
from tests.pll.whole_fixtures import whole_example


def test_motif_removal_cuts_decoder_and_relation_paths_without_mutating_original():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        model = small_model().eval()
        ex = whole_example(8)
        with torch.no_grad():
            memory = model.encode(ex.context)
            before = memory.values.clone()
            cut = ablate_motifs(memory)
            assert 'motif' not in cut.branches
            assert not any(m.level.startswith('motif_') for m in cut.metadata)
            n = len(memory.branches['motif'])
            for key in memory.families:
                assert torch.equal(cut.families[key], memory.families[key][:-n])
            null = ablate_motifs(memory, null_value=model.motifs.null_value)
            assert null.values.shape == memory.values.shape
            assert torch.equal(null.branches['motif'][0], model.motifs.null_value)
            assert torch.count_nonzero(null.branches['motif'][1:]) == 0
            assert torch.equal(memory.values, before)
            assert model.decoder(ex.targets, cut, span=ex.span, tracks=ex.tracks).nll.isfinite()
    finally:
        torch.set_num_threads(old)


def test_target_edits_preserve_context_exact_time_and_grammar():
    model = small_model().eval()
    ex = whole_example(8)
    changes = variants(ex.targets, ex.span)
    assert changes['silence'] == ()
    assert len(changes['pitch']) == len(changes['rhythm']) == len(ex.targets)
    assert len(changes['delete_half']) < len(ex.targets)
    for events in changes.values():
        candidate = replace(ex, targets=events)
        assert candidate.context is ex.context
        tokens = model.codec.encode(events, span=ex.span, tracks=ex.tracks)
        assert model.codec.decode(tokens, span=ex.span, tracks=ex.tracks) == events


def test_bound_inputs_reject_escape_and_modified_bytes(tmp_path):
    import pytest
    from hashlib import sha256
    (tmp_path/'piece').write_bytes(b'abc')
    bound = dict(path='piece', sha256=sha256(b'abc').hexdigest(), size_bytes=3)
    assert read_bound(tmp_path, bound) == b'abc'
    with pytest.raises(ValueError, match='unsafe'):
        read_bound(tmp_path, dict(bound, path='../piece'))
    (tmp_path/'piece').write_bytes(b'abd')
    with pytest.raises(ValueError, match='integrity'):
        read_bound(tmp_path, bound)


def test_checkpoint_binding_survives_json_without_ignoring_changes():
    assert bindings_equal({'scales': (1, 2, 4)}, {'scales': [1, 2, 4]})
    assert not bindings_equal({'scales': (1, 2, 4)}, {'scales': [1, 2, 8]})


def test_summary_uses_event_weighting_and_keeps_deletion_counts_separate():
    from scripts.hcm2_pilot_diagnostic import summarize
    rows = []
    for i, (nll, events) in enumerate(((20., 2), (12., 6))):
        full = dict(nll=nll, events=events, changed=False)
        rows.append(dict(id=f'piece:one:{i}', split='validation', update=1000,
            scores={'full':full, 'no_motif':dict(nll=nll+1), 'null_only':dict(nll=nll)},
            diagnostics={'motifs':{'null_weight':1.}},
            variants={**{k:full for k in ('original','pitch','rhythm','delete_half')},
                      'silence':dict(nll=1., events=0, changed=True)}))
    result = summarize(dict(status='completed', elapsed_seconds=1, scores=rows, generations=[]))
    assert result['likelihood'][0]['nats_per_event'] == 4.
    assert result['likelihood'][0]['works'] == 1
    assert result['ablations'][0]['delta_nats_per_event'] == .25
    assert result['edits'][0]['lower_nll_for_edit'] == 2
    assert result['edits'][0]['mean_edited_events'] == 0


def test_bounded_generation_reports_empty_eos_and_time_limit():
    from music_critic.pll.codec import EOS
    old = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        model = small_model().eval()
        with torch.no_grad():
            model.decoder.output.weight.zero_(); model.decoder.output.bias.zero_()
            model.decoder.output.bias[EOS] = 100
            view = whole_example(4).context
            result = bounded_generate(model, view, seed=1, max_tokens=4, wall_seconds=10)
            assert result['status'] == 'completed' and result['events'] == []
            limited = bounded_generate(model, view, seed=1, max_tokens=4, wall_seconds=0)
            assert limited['status'] == 'incomplete' and limited['reason'] == 'wall_budget'
    finally:
        torch.set_num_threads(old)
