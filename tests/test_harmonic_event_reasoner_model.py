"""Synthetic head contracts; these tests do not establish musical quality."""
from __future__ import annotations

import unittest

import torch

from music_critic.experiments.harmonic_event_reasoner.model import (
    EventReadout,
    masked_membership_loss,
)


class MembershipMaskTests(unittest.TestCase):
    def test_unknown_target_has_no_loss_or_gradient(self):
        logits = torch.tensor([0.0, 17.0, -3.0], requires_grad=True)
        labels = torch.tensor([1.0, float("nan"), 0.0])
        mask = torch.tensor([True, False, True])
        loss = masked_membership_loss(logits, labels, mask)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertLess(logits.grad[0].item(), 0.0)
        self.assertEqual(logits.grad[1].item(), 0.0)
        self.assertGreater(logits.grad[2].item(), 0.0)

    def test_weighted_loss_uses_known_count_and_masks_unknowns(self):
        import math
        logits = torch.tensor([0., math.log(3.), float('nan')], dtype=torch.float64, requires_grad=True)
        labels = torch.tensor([1., 0., float('nan')], dtype=torch.float64)
        mask = torch.tensor([True, True, False])
        loss = masked_membership_loss(logits, labels, mask, class_weights=(2., .25))
        expected = (.25 * math.log(2.) + 2. * math.log(4.)) / 2
        self.assertAlmostEqual(loss.item(), expected, places=12)
        self.assertNotAlmostEqual(loss.item(), expected * 2 / 2.25, places=8)
        loss.backward()
        torch.testing.assert_close(logits.grad, torch.tensor([-.0625, .75, 0.], dtype=torch.float64), rtol=0, atol=1e-15)

    def test_default_and_unit_weights_preserve_unweighted_formula(self):
        logits = torch.tensor([-.7, 1.2, 8.], requires_grad=True)
        labels = torch.tensor([1., 0., float('nan')])
        mask = torch.tensor([True, True, False])
        expected = torch.nn.functional.binary_cross_entropy_with_logits(logits[:2], labels[:2])
        self.assertTrue(torch.equal(masked_membership_loss(logits, labels, mask), expected))
        self.assertTrue(torch.equal(masked_membership_loss(logits, labels, mask, class_weights=(1., 1.)), expected))

    def test_weighted_all_unknown_is_zero_and_invalid_weights_raise(self):
        logits = torch.tensor([1.], requires_grad=True)
        labels = torch.tensor([float('nan')]); mask = torch.tensor([False])
        loss = masked_membership_loss(logits, labels, mask, class_weights=(2., .5))
        loss.backward()
        self.assertEqual(loss.item(), 0.)
        self.assertEqual(logits.grad.item(), 0.)
        for weights in ((0., 1.), (-1., 1.), (float('nan'), 1.), (float('inf'), 1.),
                        (True, 1.), (1.,), 'bad'):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                masked_membership_loss(logits, labels, mask, class_weights=weights)

    def test_all_unknown_returns_differentiable_zero(self):
        logits = torch.tensor([2.0, -1.0], requires_grad=True)
        loss = masked_membership_loss(
            logits, torch.full((2,), float("nan")), torch.zeros(2, dtype=torch.bool)
        )
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits))

    def test_mask_shape_and_known_binary_values_are_checked(self):
        logits = torch.zeros(2, requires_grad=True)
        with self.assertRaises(ValueError):
            masked_membership_loss(logits, torch.ones(2), torch.ones(2))
        with self.assertRaises(ValueError):
            masked_membership_loss(logits, torch.tensor([1.0, 2.0]), torch.ones(2, dtype=torch.bool))


class EventReadoutTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.x = torch.randn(5, 7)
        self.types = torch.tensor([2, 0, 0, 1, 0])
        self.edges = torch.tensor([[0, 2, 3, 1, 4], [1, 1, 2, 4, 0]])
        self.edge_types = torch.tensor([0, 1, 2, 1, 0])
        self.candidates = torch.tensor([1, 2, 4])
        self.empty_edges = torch.empty((2, 0), dtype=torch.long)
        self.empty_edge_types = torch.empty(0, dtype=torch.long)

    def model(self, kind):
        return EventReadout(input_width=7, kind=kind, hidden=8, rounds=3, node_types=3, edge_types=3)

    def score(self, model, *, x=None, source=0, empty=False):
        return model(
            self.x if x is None else x, self.types,
            self.empty_edges if empty else self.edges,
            self.empty_edge_types if empty else self.edge_types,
            source, self.candidates,
        )

    def test_all_kinds_have_finite_logits_and_gradients(self):
        for kind in ("mlp", "transe", "self", "graph"):
            with self.subTest(kind=kind):
                model = self.model(kind)
                x = self.x.clone().requires_grad_(True)
                logits = self.score(model, x=x)
                self.assertEqual(logits.shape, (3,))
                self.assertTrue(torch.isfinite(logits).all())
                masked_membership_loss(
                    logits, torch.tensor([1.0, 0.0, float("nan")]), torch.tensor([True, True, False])
                ).backward()
                self.assertTrue(torch.isfinite(x.grad).all())
                self.assertGreater(x.grad.abs().sum().item(), 0)
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                if kind in {"self", "graph"}:
                    self.assertGreater(model.query[0].weight.grad.abs().sum().item(), 0)

    def test_identical_seed_gives_identical_common_projection(self):
        projections = []
        for kind in ("mlp", "transe", "self", "graph"):
            torch.manual_seed(101)
            projections.append(self.model(kind).projection.state_dict())
        for other in projections[1:]:
            for key, value in projections[0].items():
                torch.testing.assert_close(value, other[key], rtol=0, atol=0)

    def test_self_and_graph_match_exactly_when_inter_node_messages_are_absent(self):
        self_only, graph = self.model("self"), self.model("graph")
        graph.load_state_dict(self_only.state_dict(), strict=True)
        self.assertEqual(
            sum(p.numel() for p in self_only.parameters()), sum(p.numel() for p in graph.parameters())
        )
        torch.testing.assert_close(self.score(self_only, empty=True), self.score(graph, empty=True), rtol=0, atol=0)

    def test_self_ignores_edges_but_graph_can_use_a_neighbor(self):
        self_only, graph = self.model("self"), self.model("graph")
        graph.load_state_dict(self_only.state_dict(), strict=True)
        torch.testing.assert_close(self.score(self_only), self.score(self_only, empty=True), rtol=0, atol=0)
        # Node 3 is neither the query source nor an answer, but sends to node 2.
        changed = self.x.clone()
        changed[3] += torch.tensor([7.0, -3.0, 1.0, 2.0, 0.0, -4.0, 1.0])
        torch.testing.assert_close(self.score(self_only), self.score(self_only, x=changed), rtol=0, atol=0)
        self.assertFalse(torch.allclose(self.score(graph), self.score(graph, x=changed), rtol=1e-5, atol=1e-7))

    def test_source_features_reach_self_refinement_even_without_edges(self):
        model = self.model("self")
        changed = self.x.clone()
        changed[0] += torch.tensor([2.0, -3.0, 1.0, 2.0, 0.0, -4.0, 1.0])
        observed = []
        handle = model.refinement[0].register_forward_hook(lambda _m, _a, out: observed.append(out.detach().clone()))
        try:
            first = self.score(model, empty=True)
            second = self.score(model, x=changed, empty=True)
        finally:
            handle.remove()
        self.assertFalse(torch.allclose(first, second))
        # Inspecting the round verifies global conditioning, not merely the final pair head.
        self.assertFalse(torch.allclose(observed[0][self.candidates], observed[1][self.candidates]))

    def test_node_permutation_preserves_candidate_scores(self):
        order = torch.tensor([3, 0, 4, 1, 2])
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(order.numel())
        for kind in ("mlp", "transe", "self", "graph"):
            with self.subTest(kind=kind):
                model = self.model(kind)
                before = self.score(model)
                after = model(
                    self.x[order], self.types[order], inverse[self.edges], self.edge_types,
                    int(inverse[0]), inverse[self.candidates],
                )
                torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-6)

    def test_empty_candidate_list_is_supported(self):
        for kind in ("mlp", "transe", "self", "graph"):
            with self.subTest(kind=kind):
                score = self.model(kind)(
                    self.x, self.types, self.edges, self.edge_types, 0, torch.empty(0, dtype=torch.long)
                )
                self.assertEqual(score.shape, (0,))

    def test_translation_relation_can_break_source_target_symmetry(self):
        model = self.model("transe")
        with torch.no_grad():
            model.relation.zero_()
        forward = model(self.x, self.types, self.edges, self.edge_types, 0, torch.tensor([1]))
        reverse = model(self.x, self.types, self.edges, self.edge_types, 1, torch.tensor([0]))
        torch.testing.assert_close(forward, reverse)
        with torch.no_grad():
            model.relation[0] = 1.0
        forward = model(self.x, self.types, self.edges, self.edge_types, 0, torch.tensor([1]))
        reverse = model(self.x, self.types, self.edges, self.edge_types, 1, torch.tensor([0]))
        self.assertFalse(torch.allclose(forward, reverse))

    def test_invalid_routing_fails_explicitly(self):
        model = self.model("graph")
        with self.assertRaises(ValueError):
            self.score(model, source=99)
        with self.assertRaises(ValueError):
            model(self.x, self.types, self.edges, self.edge_types, 0, torch.tensor([-1]))
        with self.assertRaises(ValueError):
            model(self.x, self.types, self.edges, torch.full((5,), 3), 0, self.candidates)


if __name__ == "__main__":
    unittest.main()
