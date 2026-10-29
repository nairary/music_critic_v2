"""Synthetic mechanics, not scientific model-quality evidence."""
from __future__ import annotations

import importlib.util
import unittest

_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if _TORCH_AVAILABLE:
    import torch
    from music_critic.experiments.hierarchical_graph_analyzer.model import HAnalyzer


def synthetic_raw():
    """One held physical note has two raw fragments; a point note is retained."""
    torch.manual_seed(29)
    return {
        "x": torch.randn(9, 12),
        "node_type": torch.tensor([0, 0, 6, 6, 6, 6, 7, 7, 8]),
        "node_time": torch.tensor([[0., 1.], [.5, .5], [0., 0.], [1/3, 1/3],
                                   [2/3, 2/3], [1., 1.], [0., .5], [.5, 1.], [-1., 2.]]),
        "edge_index": torch.tensor([[0, 0, 6, 7, 2, 3, 4, 8, 8],
                                    [6, 7, 0, 0, 3, 4, 5, 0, 4]]),
        "edge_type": torch.tensor([0, 0, 1, 1, 2, 2, 2, 3, 3]),
        "time_index": torch.tensor([2, 3, 4, 5]),
        "time_positions": torch.tensor([0., 1/3, 2/3, 1.]),
        "bank_indices": torch.tensor([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]),
        "pool_mask": torch.tensor([False, False, True, True, True, True, True, True, False]),
    }


@unittest.skipUnless(_TORCH_AVAILABLE, "PyTorch is not installed; native mechanics unverified")
class HAnalyzerTests(unittest.TestCase):
    def model(self, kind):
        torch.manual_seed(17)
        return HAnalyzer(kind, {"H.function": 3, "H.chord.root_pc": 12},
                         input_width=12, hidden=16, rounds=2, max_events=3)

    def test_all_five_shapes_and_bounded_intervals(self):
        raw = synthetic_raw()
        for kind in ("E1", "E2", "E3", "E4", "E5"):
            with self.subTest(kind=kind):
                pred = self.model(kind)(raw)
                k = len(pred["spans"])
                self.assertGreater(k, 0)
                self.assertEqual(pred["spans"].shape, (k, 2))
                self.assertEqual(pred["object_logits"].shape, (k,))
                self.assertEqual(pred["states"].shape, (k, 16))
                self.assertEqual(pred["field_logits"]["H.function"].shape, (k, 3))
                self.assertTrue(torch.isfinite(pred["states"]).all())
                self.assertTrue((pred["spans"][:, 0] >= 0).all())
                self.assertTrue((pred["spans"][:, 1] <= 1).all())
                self.assertTrue((pred["spans"][:, 0] < pred["spans"][:, 1]).all())

@unittest.skipUnless(_TORCH_AVAILABLE, "PyTorch is not installed; native mechanics unverified")
class HMechanismTests(unittest.TestCase):
    model = HAnalyzerTests.model

    def assert_nonzero_grad(self, tensor):
        self.assertIsNotNone(tensor.grad)
        self.assertTrue(torch.isfinite(tensor.grad).all())
        self.assertGreater(float(tensor.grad.abs().sum()), 0.)

    def test_e1_uses_full_raw_bank_without_slot_cap(self):
        raw = synthetic_raw()
        pred = self.model("E1")(raw)
        self.assertTrue(torch.equal(pred["spans"], raw["time_positions"][raw["bank_indices"]]))
        self.assertEqual(len(pred["spans"]), 6)

    def test_e2_is_contiguous_and_learns_boundary_locations(self):
        raw = synthetic_raw()
        model = self.model("E2")
        with torch.no_grad():
            model.boundary_head[-1].weight.zero_()
            model.boundary_head[-1].bias.fill_(2.)
        pred = model(raw)
        spans = pred["spans"]
        self.assertEqual(len(spans), 3)
        self.assertEqual(float(spans[0, 0].detach()), 0.)
        self.assertEqual(float(spans[-1, 1].detach()), 1.)
        self.assertTrue(torch.equal(spans[:-1, 1], spans[1:, 0]))
        self.assertTrue(torch.equal(spans[:, 0], raw["time_positions"][:-1]))
        # Localization, rather than merely a diagnostic scalar, reaches logits.
        pred["boundary_logits"].retain_grad()
        spans[0, 1].backward()
        self.assert_nonzero_grad(pred["boundary_logits"])
        self.assert_nonzero_grad(model.boundary_head[-1].weight)

    def test_e2_no_cut_still_learns_from_h_fields(self):
        raw = synthetic_raw()
        model = self.model("E2")
        with torch.no_grad():
            model.boundary_head[-1].weight.zero_()
            model.boundary_head[-1].bias.fill_(-8.)
        pred = model(raw)
        self.assertTrue(torch.equal(pred["spans"], torch.tensor([[0., 1.]])))
        pred["boundary_logits"].retain_grad()
        pred["field_logits"]["H.function"].square().sum().backward()
        self.assert_nonzero_grad(pred["boundary_logits"])
        self.assert_nonzero_grad(model.boundary_head[-1].bias)

    def test_e2_cut_count_is_predicted_and_capped(self):
        raw = synthetic_raw()
        model = self.model("E2")
        with torch.no_grad():
            model.boundary_head[-1].weight.zero_()
            model.boundary_head[-1].bias.fill_(-8.)
        self.assertEqual(len(model(raw)["spans"]), 1)
        with torch.no_grad():
            model.boundary_head[-1].bias.fill_(8.)
        model.max_events = 2
        pred = model(raw)
        self.assertEqual(len(pred["spans"]), 2)
        self.assertEqual(float(pred["spans"][0, 1].detach()), float(raw["time_positions"][1]))

    def test_e3_anchor_free_offsets_have_endpoint_gradient(self):
        raw = synthetic_raw()
        model = self.model("E3")
        pred = model(raw)
        self.assertEqual(len(pred["spans"]), len(raw["time_positions"]))
        anchors = raw["time_positions"]
        self.assertTrue((pred["spans"][:, 0] <= anchors).all())
        self.assertTrue((pred["spans"][:, 1] >= anchors).all())
        self.assertFalse(torch.equal(pred["spans"][1], raw["time_positions"][:2]))
        pred["spans"].square().sum().backward()
        self.assert_nonzero_grad(model.endpoint_head.weight)

    def test_e4_is_query_cross_attention(self):
        raw = synthetic_raw()
        model = self.model("E4")
        pred = model(raw)
        self.assertIn("query_attention", pred)
        self.assertNotIn("assignment", pred)
        self.assertTrue(torch.allclose(pred["query_attention"].sum(-1), torch.ones(3)))
        pred["field_logits"]["H.function"].square().sum().backward()
        self.assert_nonzero_grad(model.event_queries)
        self.assert_nonzero_grad(model.query_key.weight)

    def test_e5_independent_sigmoid_assignment_and_h_gradient(self):
        raw = synthetic_raw()
        model = self.model("E5")
        # Set every sigmoid to one-half. Three recipients sum to 1.5, proving
        # many-recipient assignment rather than row-softmax or a query decoder.
        with torch.no_grad():
            model.assignment_head[-1].weight.zero_()
            model.assignment_head[-1].bias.zero_()
        pred = model(raw)
        self.assertFalse(hasattr(model, "event_queries"))
        assignment = pred["assignment"]
        assignment.retain_grad()
        self.assertTrue(torch.allclose(assignment.sum(-1), torch.full((6,), 1.5)))
        self.assertEqual(pred["pool_indices"].tolist(), [2, 3, 4, 5, 6, 7])
        self.assertTrue((assignment[-2:] > 0).all())  # Both held-note fragments, every H.
        pred["field_logits"]["H.function"].square().sum().backward()
        self.assert_nonzero_grad(assignment)
        self.assert_nonzero_grad(model.assignment_head[-1].weight)

    def test_e5_boundaries_are_independent_of_pool_minmax(self):
        raw = synthetic_raw()
        model = self.model("E5")
        with torch.no_grad():
            model.endpoint_head.weight.zero_()
            model.endpoint_head.bias.copy_(torch.tensor([-1., .5]))
        pred = model(raw)
        self.assertTrue((pred["spans"][:, 0] > .2).all())
        self.assertTrue((pred["spans"][:, 1] < .9).all())
        pred["spans"].square().sum().backward()
        self.assert_nonzero_grad(model.endpoint_head.bias)

    def test_no_target_keys_or_identity_lookup(self):
        raw = synthetic_raw()
        for kind in ("E1", "E2", "E3", "E4", "E5"):
            model = self.model(kind)
            before = model(raw)
            with self.assertRaisesRegex(ValueError, "target-free"):
                model({**raw, "targets": torch.ones(1)})
            after = model(raw)
            self.assertTrue(torch.equal(before["states"], after["states"]))
            self.assertFalse(any("identity" in n or "corpus" in n for n, _ in model.named_parameters()))

    def test_raw_node_permutation_is_equivariant(self):
        raw = synthetic_raw()
        permutation = torch.tensor([8, 7, 0, 4, 2, 6, 1, 5, 3])
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(len(permutation))
        reordered = dict(raw)
        for key in ("x", "node_type", "node_time", "pool_mask"):
            reordered[key] = raw[key][permutation]
        reordered["edge_index"] = inverse[raw["edge_index"]]
        reordered["time_index"] = inverse[raw["time_index"]]
        for kind in ("E1", "E2", "E3", "E4", "E5"):
            with self.subTest(kind=kind):
                model = self.model(kind)
                before, after = model(raw), model(reordered)
                for key in ("spans", "states", "object_logits"):
                    self.assertTrue(torch.allclose(before[key], after[key], atol=3e-6, rtol=3e-6), key)
                self.assertTrue(torch.allclose(before["raw_states"][permutation], after["raw_states"],
                                               atol=3e-6, rtol=3e-6))

    def test_e4_and_e5_slot_permutation(self):
        raw = synthetic_raw()
        order = torch.tensor([2, 0, 1])
        for kind in ("E4", "E5"):
            model = self.model(kind)
            before = model(raw)
            with torch.no_grad():
                if kind == "E4":
                    model.event_queries.copy_(model.event_queries[order].clone())
                else:
                    last = model.assignment_head[-1]
                    last.weight.copy_(last.weight[order].clone())
                    last.bias.copy_(last.bias[order].clone())
            after = model(raw)
            self.assertTrue(torch.allclose(before["spans"][order], after["spans"], atol=2e-6))
            self.assertTrue(torch.allclose(before["states"][order], after["states"], atol=2e-6))

    def test_common_backbone_and_head_parameters_match_across_arms(self):
        prefixes = ("projection.", "raw_rounds.", "event_geometry.", "refiners.",
                    "object_head.", "field_heads.")
        common = {k: v for k, v in self.model("E1").state_dict().items() if k.startswith(prefixes)}
        for kind in ("E2", "E3", "E4", "E5"):
            actual = {k: v for k, v in self.model(kind).state_dict().items() if k.startswith(prefixes)}
            self.assertEqual(actual.keys(), common.keys())
            for name in common:
                self.assertTrue(torch.equal(actual[name], common[name]), name)

    def test_raw_unpooled_note_still_influences_e5_h(self):
        raw = synthetic_raw()
        raw["x"].requires_grad_()
        model = self.model("E5")
        model(raw)["field_logits"]["H.function"].square().sum().backward()
        # Physical notes 0/1 are absent from E5 pool_mask, including the point
        # note and the hypothetical nonchord note; raw paths remain learnable.
        self.assert_nonzero_grad(raw["x"])
        self.assertGreater(float(raw["x"].grad[:2].abs().sum()), 0.)

    def test_edge_types_and_global_context_are_live_paths(self):
        raw = synthetic_raw()
        model = self.model("E1")
        first = model(raw)["states"]
        changed = dict(raw)
        changed["edge_type"] = (raw["edge_type"] + 1) % 64
        self.assertFalse(torch.allclose(first, model(changed)["states"]))
        changed = dict(raw)
        changed["x"] = raw["x"].clone()
        changed["x"][-1, 0] += 5
        self.assertFalse(torch.allclose(first, model(changed)["states"]))

    def test_silence_time_states_and_point_events_remain_finite(self):
        raw = synthetic_raw()
        raw.update({"x": torch.zeros(2, 12), "node_type": torch.tensor([6, 6]),
                    "node_time": torch.tensor([[0., 0.], [1., 1.]]),
                    "edge_index": torch.empty((2, 0), dtype=torch.long),
                    "edge_type": torch.empty((0,), dtype=torch.long),
                    "time_index": torch.tensor([0, 1]), "time_positions": torch.tensor([0., 1.]),
                    "bank_indices": torch.tensor([[0, 1]]), "pool_mask": torch.tensor([True, True])})
        for kind in ("E1", "E2", "E3", "E4", "E5"):
            pred = self.model(kind)(raw)
            self.assertTrue(torch.isfinite(pred["states"]).all())
            self.assertTrue((pred["spans"][:, 0] < pred["spans"][:, 1]).all())

    def test_empty_bank_and_repeated_time_indices(self):
        raw = synthetic_raw()
        raw["bank_indices"] = torch.empty((0, 2), dtype=torch.long)
        pred = self.model("E1")(raw)
        self.assertEqual(pred["spans"].shape, (0, 2))
        self.assertEqual(pred["field_logits"]["H.function"].shape, (0, 3))
        raw["time_index"] = torch.tensor([2, 2, 4, 4])
        self.assertTrue(torch.isfinite(self.model("E3")(raw)["states"]).all())

    def test_bad_geometry_indices_and_empty_pool_rejected(self):
        raw = synthetic_raw()
        invalid = [
            ("time_positions", torch.tensor([0., .5, .5, 1.])),
            ("bank_indices", torch.tensor([[2, 1]])),
            ("time_index", torch.tensor([2, 3, 4, 99])),
            ("pool_mask", torch.zeros(9, dtype=torch.bool)),
            ("x", torch.full((9, 12), float("nan"))),
        ]
        for key, value in invalid:
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.model("E5")({**raw, key: value})

    def test_checkpoint_replay_all_arms(self):
        raw = synthetic_raw()
        for kind in ("E1", "E2", "E3", "E4", "E5"):
            model = self.model(kind)
            before = model(raw)
            restored = self.model(kind)
            restored.load_state_dict(model.state_dict())
            after = restored(raw)
            self.assertTrue(torch.equal(before["states"], after["states"]))
            self.assertTrue(torch.equal(before["spans"], after["spans"]))


    def test_oracle_readout_is_diagnostic_and_uses_shared_paths(self):
        raw = synthetic_raw()
        snapshots = {name: value.clone() for name, value in raw.items()}
        oracle = torch.tensor([[.1, .4], [.4, .9]], requires_grad=True)
        common = None
        for kind in ("E1", "E2", "E3", "E4", "E5"):
            model = self.model(kind)
            pred = model(raw)
            diagnostic = model.oracle_readout(raw, oracle)
            self.assertTrue(torch.equal(diagnostic["spans"], oracle))
            self.assertTrue(torch.equal(pred["raw_states"], diagnostic["raw_states"]))
            self.assertFalse(diagnostic["states"].requires_grad)
            self.assertFalse(diagnostic["object_logits"].requires_grad)
            self.assertTrue(torch.equal(diagnostic["object_logits"],
                                        model.object_head(diagnostic["states"]).flatten()))
            for name, head in zip(model.field_names, model.field_heads):
                self.assertTrue(torch.equal(diagnostic["field_logits"][name],
                                            head(diagnostic["states"])))
            if common is None:
                common = diagnostic["states"]
            else:
                self.assertTrue(torch.equal(common, diagnostic["states"]))
            changed = model.oracle_readout(raw, torch.tensor([[.2, .6]]))
            self.assertTrue(torch.equal(diagnostic["raw_states"], changed["raw_states"]))
            self.assertEqual(changed["states"].shape, (1, 16))
        for name, original in snapshots.items():
            self.assertTrue(torch.equal(original, raw[name]), name)

    def test_oracle_geometry_validation(self):
        raw = synthetic_raw()
        model = self.model("E1")
        for spans in (torch.tensor([[.4, .4]]), torch.tensor([[-.1, .5]]),
                      torch.tensor([[0., float("nan")]]), torch.ones(3)):
            with self.assertRaises(ValueError):
                model.oracle_readout(raw, spans)
        self.assertEqual(model.oracle_readout(raw, torch.empty((0, 2)))["states"].shape, (0, 16))


if __name__ == "__main__":
    unittest.main()
