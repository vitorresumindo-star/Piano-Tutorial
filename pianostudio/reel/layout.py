"""Pixel layout of the vertical reel: footage panel on top, piano roll below.

    +---------------------------+  0
    |   cinematic footage       |
    |   + handwritten titles    |
    +---------------------------+  footage_h   (~38 % of the height)
    |                           |
    |   falling notes           |
    |                           |
    +===========================+  key_top
    |   keyboard (labelled)     |
    +---------------------------+  key_bottom
    |   piano body / palms      |
    +---------------------------+  height
"""
from __future__ import annotations

from dataclasses import dataclass

from ..geometry import is_black

# black-key centres in white-key widths from the left edge of C (shared with the 3D renderer)
_BLACK_CENTER = {1: 0.87, 3: 2.13, 6: 3.84, 8: 5.0, 10: 6.16}
_WHITE_IDX = {0: 0, 2: 1, 4: 2, 5: 3, 7: 4, 9: 5, 11: 6}


def white_index(m: int) -> int:
    return 7 * (m // 12) + _WHITE_IDX[m % 12]


def key_center_units(m: int) -> float:
    """Key centre in white-key units on an absolute axis (C-1 left edge = 0)."""
    if is_black(m):
        return 7 * (m // 12) + _BLACK_CENTER[m % 12]
    return white_index(m) + 0.5


def _down_to_white(m: int) -> int:
    while is_black(m):
        m -= 1
    return m


def _up_to_white(m: int) -> int:
    while is_black(m):
        m += 1
    return m


def fit_range(pitches, margin_whites: int = 1, min_whites: int = 15) -> tuple[int, int]:
    """Lowest/highest MIDI key (both white) covering `pitches` plus a margin."""
    lo, hi = _down_to_white(min(pitches)), _up_to_white(max(pitches))
    for _ in range(margin_whites):
        lo = _down_to_white(lo - 1)
        hi = _up_to_white(hi + 1)
    while white_index(hi) - white_index(lo) + 1 < min_whites:
        lo = _down_to_white(lo - 1)
        if white_index(hi) - white_index(lo) + 1 < min_whites:
            hi = _up_to_white(hi + 1)
    return lo, hi


@dataclass
class Layout:
    width: int
    height: int
    footage_h: int
    key_top: int
    key_h: int
    lo: int
    hi: int

    @classmethod
    def build(cls, width: int, height: int, lo: int, hi: int, footage_ratio: float = 0.38,
              key_ratio: float = 0.142, body_ratio: float = 0.112) -> "Layout":
        footage_h = int(round(height * footage_ratio))
        key_h = int(round(height * key_ratio))
        body = int(round(height * body_ratio))
        key_top = height - body - key_h
        return cls(width, height, footage_h, key_top, key_h, lo, hi)

    # ---- keyboard geometry --------------------------------------------------
    @property
    def n_white(self) -> int:
        return white_index(self.hi) - white_index(self.lo) + 1

    @property
    def white_w(self) -> float:
        return self.width / self.n_white

    @property
    def black_w(self) -> float:
        return self.white_w * 0.6

    @property
    def black_h(self) -> float:
        return self.key_h * 0.62

    @property
    def key_bottom(self) -> int:
        return self.key_top + self.key_h

    @property
    def notes_top(self) -> int:
        return self.footage_h

    def x_center(self, m: int) -> float:
        return (key_center_units(m) - white_index(self.lo)) * self.white_w

    def white_left(self, m: int) -> float:
        return (white_index(m) - white_index(self.lo)) * self.white_w

    def keys(self):
        return list(range(self.lo, self.hi + 1))

    # ---- where fingertips touch the keys --------------------------------------
    @property
    def tip_y_white(self) -> float:
        return self.key_top + self.key_h * 0.80

    @property
    def tip_y_black(self) -> float:
        return self.key_top + self.black_h * 0.78

    def tip_y(self, m: int) -> float:
        return self.tip_y_black if is_black(m) else self.tip_y_white

    @property
    def knuckle_y(self) -> float:
        """Rest y of the middle-finger knuckle (palms sit over the piano body)."""
        return self.tip_y_white + 2.12 * self.white_w
