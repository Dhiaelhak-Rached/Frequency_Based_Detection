"""Shared test fakes and helper builders for vision and tracking tests."""

import numpy as np

from frequency_id.vision.logic import Detection


class FakeTensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class FakeBoxes:
    def __init__(self, rows):
        rows = list(rows)
        self.xyxy = FakeTensor([row[:4] for row in rows])
        self.conf = FakeTensor([row[4] for row in rows])
        self.cls = FakeTensor([row[5] for row in rows])


class FakePrediction:
    def __init__(self, rows):
        self.boxes = FakeBoxes(rows)


class FakeModel:
    names = {0: "drone", 1: "other"}

    def __init__(self, rows=()):
        self.rows = rows
        self.calls = 0

    def predict(self, frame, conf, verbose):
        self.calls += 1
        return [FakePrediction(self.rows)]


class FakeCapture:
    def __init__(self, opened=True):
        self.opened = opened
        self.released = False
        self.frames = 0

    def isOpened(self):
        return self.opened

    def read(self):
        self.frames += 1
        return True, np.full((48, 64, 3), self.frames % 255, dtype=np.uint8)

    def release(self):
        self.released = True


def D(x, y, w=20, h=10, s=0.9):
    return Detection((x, y, w, h), s, "car")


class FakeTracker:
    """Moves its box 1 px to the right per update."""

    def __init__(self):
        self.box = None
        self.inits = []

    def init(self, frame, box):
        self.box = list(box)
        self.inits.append(tuple(box))

    def update(self, frame):
        self.box[0] += 1
        return True, tuple(self.box), None
