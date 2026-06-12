from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class CentroidRoute:
    stage: str
    score: float


@dataclass(frozen=True)
class HiddenStateCentroidRouter:
    centroids: dict[str, list[float]]

    @classmethod
    def from_vectors(
        cls,
        vectors_by_stage: dict[str, list[Iterable[float]]],
    ) -> "HiddenStateCentroidRouter":
        centroids = {}
        for stage, vectors in vectors_by_stage.items():
            vectors = [list(vector) for vector in vectors]
            if not vectors:
                continue
            width = len(vectors[0])
            sums = [0.0] * width
            for vector in vectors:
                if len(vector) != width:
                    raise ValueError(f"Inconsistent vector width for stage {stage!r}")
                for idx, value in enumerate(vector):
                    sums[idx] += float(value)
            centroid = [value / len(vectors) for value in sums]
            centroids[stage] = _normalize(centroid)
        if not centroids:
            raise ValueError("At least one centroid is required")
        return cls(centroids=centroids)

    def route(self, vector: Iterable[float]) -> CentroidRoute:
        query = _normalize([float(value) for value in vector])
        best_stage = None
        best_score = -math.inf
        for stage, centroid in self.centroids.items():
            if len(query) != len(centroid):
                raise ValueError("Query vector width does not match centroid width")
            score = sum(a * b for a, b in zip(query, centroid, strict=True))
            if score > best_score:
                best_stage = stage
                best_score = score
        if best_stage is None:
            raise ValueError("No centroids available")
        return CentroidRoute(stage=best_stage, score=best_score)

    def to_dict(self) -> dict:
        return {"format": "safeprune.hidden_state_centroid_router.v1", "centroids": self.centroids}

    @classmethod
    def from_dict(cls, raw: dict) -> "HiddenStateCentroidRouter":
        centroids = raw.get("centroids")
        if not isinstance(centroids, dict) or not centroids:
            raise ValueError("centroid router requires non-empty centroids")
        return cls(
            centroids={
                str(stage): [float(value) for value in vector]
                for stage, vector in centroids.items()
            }
        )


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]
