from pathlib import Path

import cv2
import numpy as np
import pytest

from dashcam_ai.application.adapters import DetectorTrackerBackend
from dashcam_ai.application.analyzer import Analyzer
from dashcam_ai.detection.fake import FakeDetector
from dashcam_ai.domain.geometry import BBox
from dashcam_ai.domain.perception import Detection
from dashcam_ai.tracking.centroid import CentroidTracker

pytestmark = pytest.mark.cv


def test_opencv_pipeline_reads_and_annotates_synthetic_video(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(
        str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 240)
    )
    assert writer.isOpened()
    for _ in range(3):
        writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
    writer.release()

    observations = [
        [
            Detection(
                class_id=2,
                class_name="car",
                confidence=0.9,
                bbox=BBox(x1=50 + offset, y1=80, x2=130 + offset, y2=180),
            )
        ]
        for offset in (0, 5, 10)
    ]
    backend = DetectorTrackerBackend(
        FakeDetector(observations), CentroidTracker(maximum_distance=30)
    )
    output = tmp_path / "analysis"

    summary = Analyzer(backend, save_video=True).analyze(source, output)

    assert summary.frames_processed == 3
    assert summary.tracks_created == 1
    capture = cv2.VideoCapture(str(output / "annotated.mp4"))
    assert capture.isOpened()
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
    capture.release()

