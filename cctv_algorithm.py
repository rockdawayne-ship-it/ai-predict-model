"""중대안전 지능형 CCTV 알고리즘 (참고 구현)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class BoundingBox:
    """사람/객체 탐지 결과를 표현하는 바운딩 박스."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float
    confidence: float

    @property
    def width(self) -> float:
        return max(0.0, self.x_max - self.x_min)

    @property
    def height(self) -> float:
        return max(0.0, self.y_max - self.y_min)

    @property
    def aspect_ratio(self) -> float:
        if self.height == 0:
            return 0.0
        return self.width / self.height

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0


@dataclass
class SafetyEvent:
    """위험 이벤트."""

    name: str
    score: float
    meta: Dict[str, float]


@dataclass
class SafetyConfig:
    """가중치 및 임계값 설정."""

    intrusion_weight: float = 0.35
    ppe_weight: float = 0.25
    fall_weight: float = 0.25
    smoke_weight: float = 0.15
    intrusion_threshold: float = 0.5
    ppe_threshold: float = 0.6
    fall_threshold: float = 0.55
    smoke_threshold: float = 0.4


class SafetyCCTVAlgorithm:
    """중대안전 지능형 CCTV 알고리즘의 핵심 파이프라인."""

    def __init__(self, config: Optional[SafetyConfig] = None) -> None:
        self.config = config or SafetyConfig()

    def analyze_frame(
        self,
        person_boxes: Iterable[BoundingBox],
        restricted_zones: List[List[Tuple[float, float]]],
        ppe_scores: Optional[Dict[int, float]] = None,
        posture_scores: Optional[Dict[int, float]] = None,
        smoke_score: Optional[float] = None,
    ) -> Tuple[float, List[SafetyEvent]]:
        """한 프레임에 대한 위험 점수와 이벤트 목록을 계산한다."""

        events: List[SafetyEvent] = []
        person_boxes = list(person_boxes)

        intrusion_score = self._score_intrusion(person_boxes, restricted_zones)
        if intrusion_score >= self.config.intrusion_threshold:
            events.append(
                SafetyEvent(
                    name="restricted_zone_intrusion",
                    score=intrusion_score,
                    meta={"zones": float(len(restricted_zones))},
                )
            )

        ppe_score = self._score_ppe(person_boxes, ppe_scores or {})
        if ppe_score >= self.config.ppe_threshold:
            events.append(
                SafetyEvent(
                    name="ppe_violation",
                    score=ppe_score,
                    meta={"persons": float(len(person_boxes))},
                )
            )

        fall_score = self._score_fall(person_boxes, posture_scores or {})
        if fall_score >= self.config.fall_threshold:
            events.append(
                SafetyEvent(
                    name="fall_risk",
                    score=fall_score,
                    meta={"persons": float(len(person_boxes))},
                )
            )

        smoke_score = float(smoke_score or 0.0)
        if smoke_score >= self.config.smoke_threshold:
            events.append(
                SafetyEvent(
                    name="smoke_or_fire",
                    score=smoke_score,
                    meta={},
                )
            )

        total_score = (
            intrusion_score * self.config.intrusion_weight
            + ppe_score * self.config.ppe_weight
            + fall_score * self.config.fall_weight
            + smoke_score * self.config.smoke_weight
        )

        return total_score, events

    def _score_intrusion(
        self, person_boxes: Iterable[BoundingBox], zones: List[List[Tuple[float, float]]]
    ) -> float:
        person_boxes = list(person_boxes)
        if not zones or not person_boxes:
            return 0.0
        intrusions = 0
        for box in person_boxes:
            if any(self._point_in_polygon(box.center, zone) for zone in zones):
                intrusions += 1
        return intrusions / len(person_boxes)

    def _score_ppe(
        self, person_boxes: Iterable[BoundingBox], ppe_scores: Dict[int, float]
    ) -> float:
        person_boxes = list(person_boxes)
        if not person_boxes:
            return 0.0
        violations = 0
        for idx, _box in enumerate(person_boxes):
            score = ppe_scores.get(idx, 1.0)
            if score < self.config.ppe_threshold:
                violations += 1
        return violations / len(person_boxes)

    def _score_fall(
        self, person_boxes: Iterable[BoundingBox], posture_scores: Dict[int, float]
    ) -> float:
        person_boxes = list(person_boxes)
        if not person_boxes:
            return 0.0
        fall_candidates = 0
        for idx, box in enumerate(person_boxes):
            posture_score = posture_scores.get(idx, 0.0)
            if posture_score >= self.config.fall_threshold or box.aspect_ratio > 1.1:
                fall_candidates += 1
        return fall_candidates / len(person_boxes)

    @staticmethod
    def _point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
        """Ray casting 알고리즘으로 점-다각형 포함 여부 확인."""

        x, y = point
        inside = False
        j = len(polygon) - 1
        for i, (xi, yi) in enumerate(polygon):
            xj, yj = polygon[j]
            intersect = (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi
            if intersect:
                inside = not inside
            j = i
        return inside


if __name__ == "__main__":
    sample_boxes = [
        BoundingBox(0.1, 0.2, 0.3, 0.8, 0.92),
        BoundingBox(0.5, 0.2, 0.7, 0.6, 0.88),
    ]
    sample_zone = [[(0.0, 0.5), (0.4, 0.5), (0.4, 1.0), (0.0, 1.0)]]
    algorithm = SafetyCCTVAlgorithm()
    total, events = algorithm.analyze_frame(
        sample_boxes,
        sample_zone,
        ppe_scores={0: 0.4, 1: 0.9},
        posture_scores={0: 0.2, 1: 0.8},
        smoke_score=0.1,
    )
    print("Total score:", round(total, 3))
    for event in events:
        print(event)
