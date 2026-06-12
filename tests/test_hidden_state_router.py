import unittest

from safeprune.hidden_state_router import HiddenStateCentroidRouter


class HiddenStateRouterTests(unittest.TestCase):
    def test_routes_to_nearest_centroid(self):
        router = HiddenStateCentroidRouter.from_vectors(
            {
                "plan": [[1.0, 0.0], [0.9, 0.1]],
                "answer": [[0.0, 1.0], [0.1, 0.9]],
            }
        )

        self.assertEqual(router.route([0.8, 0.2]).stage, "plan")
        self.assertEqual(router.route([0.2, 0.8]).stage, "answer")

    def test_roundtrip(self):
        router = HiddenStateCentroidRouter.from_vectors({"plan": [[1.0, 0.0]]})
        loaded = HiddenStateCentroidRouter.from_dict(router.to_dict())
        self.assertEqual(loaded.route([1.0, 0.0]).stage, "plan")


if __name__ == "__main__":
    unittest.main()
