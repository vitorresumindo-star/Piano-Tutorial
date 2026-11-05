"""Physical geometry shared by the animation solver (Python) and the renderer (JS).

Units are centimetres.  World axes:
  x  -> to the right along the keyboard (A0 on the left)
  y  -> up (white-key top surface is y = 0)
  z  -> towards the player (key back edge at z = 0, white key front at z = WHITE_LEN)

Everything the renderer needs to build the keyboard and the hand skeleton is
exported by :func:`scene_geometry` so the two sides can never disagree.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

LOWEST_MIDI = 21   # A0
HIGHEST_MIDI = 108  # C8

WHITE_W = 2.35        # white key pitch (incl. gap)
WHITE_GAP = 0.12
WHITE_LEN = 15.0
WHITE_H = 2.2         # visible front-face height
BLACK_W = 1.30
BLACK_LEN = 9.6
BLACK_TOP = 1.25      # black key top height above white top
KEY_DEPTH_WHITE = 1.0  # key travel at the front contact zone
KEY_DEPTH_BLACK = 0.9

# Black-key centre offsets (in white-key widths from the left edge of C)
_BLACK_OFFSETS = {1: 1.0 - 0.13, 3: 2.0 + 0.13, 6: 4.0 - 0.16, 8: 5.0, 10: 6.0 + 0.16}
_WHITE_INDEX_IN_OCT = {0: 0, 2: 1, 4: 2, 5: 3, 7: 4, 9: 5, 11: 6}

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def is_black(midi: int) -> bool:
    return (midi % 12) in _BLACK_OFFSETS


def note_name(midi: int) -> str:
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


@lru_cache(maxsize=None)
def _white_x_of_c(octave_midi_c: int) -> float:
    # x of the left edge of the C at/below midi value (A0 left edge = 0)
    # A0 is white index 0, B0 index 1, C1 index 2.
    whites_before = 2 + 7 * ((octave_midi_c - 24) // 12)
    return whites_before * WHITE_W


@dataclass(frozen=True)
class Key:
    midi: int
    black: bool
    x: float        # centre x
    width: float
    length: float
    top: float      # y of key top surface at rest

    @property
    def left(self) -> float:
        return self.x - self.width / 2

    @property
    def right(self) -> float:
        return self.x + self.width / 2


@lru_cache(maxsize=None)
def key(midi: int) -> Key:
    if not (LOWEST_MIDI <= midi <= HIGHEST_MIDI):
        raise ValueError(f"MIDI note {midi} is outside the 88-key range")
    pc = midi % 12
    c_midi = midi - pc
    base = _white_x_of_c(c_midi)  # A0..B0 use a virtual C0 at x = -5 white widths
    if pc in _BLACK_OFFSETS:
        x = base + _BLACK_OFFSETS[pc] * WHITE_W
        return Key(midi, True, x, BLACK_W, BLACK_LEN, BLACK_TOP)
    x = base + (_WHITE_INDEX_IN_OCT[pc] + 0.5) * WHITE_W
    return Key(midi, False, x, WHITE_W - WHITE_GAP, WHITE_LEN, 0.0)


def key_x(midi: int) -> float:
    return key(midi).x


def keyboard_extent() -> tuple[float, float]:
    return key(LOWEST_MIDI).left, key(HIGHEST_MIDI).right


def surface_top(x: float, z: float) -> float:
    """Height of the key surface under point (x, z) at rest."""
    if z < BLACK_LEN + 0.3:
        for k in _black_keys():
            if abs(x - k.x) < k.width / 2 + 0.25:
                return BLACK_TOP
    return 0.0


@lru_cache(maxsize=None)
def _black_keys() -> tuple[Key, ...]:
    return tuple(key(m) for m in range(LOWEST_MIDI, HIGHEST_MIDI + 1) if is_black(m))


def contact_z(midi: int, finger: int) -> float:
    """Where along the key a given finger naturally strikes it."""
    if is_black(midi):
        return {1: 8.4, 5: 8.6}.get(finger, 7.7)
    return {1: 13.2, 2: 11.4, 3: 10.8, 4: 11.4, 5: 13.0}[finger]


def key_depth_at_contact(midi: int) -> float:
    return KEY_DEPTH_BLACK if is_black(midi) else KEY_DEPTH_WHITE


# ---------------------------------------------------------------------------
# Hand skeleton (right hand; the left hand mirrors x).
# Hand local frame: origin at the wrist joint, fingers point along -z, y up.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FingerSpec:
    finger: int
    base: tuple[float, float, float]          # MCP (or thumb CMC) in hand frame (RH)
    base_euler: tuple[float, float, float]    # fixed base rotation (YXZ, radians, RH)
    lengths: tuple[float, float, float]       # phalanx lengths
    radii: tuple[float, float, float]         # mesh radii per segment (renderer)
    dip_ratio: float                          # distal flex = ratio * middle flex

    @property
    def tip_pad(self) -> float:                # tip-centre -> skin contact offset
        return self.radii[2]


HAND_SCALE = 1.0
FINGERS: dict[int, FingerSpec] = {
    1: FingerSpec(1, (-2.55, -1.3, -3.2), (math.radians(28), math.radians(-14), math.radians(38)),
                  (4.6, 3.3, 2.6), (1.15, 0.98, 0.86), 0.8),
    2: FingerSpec(2, (-2.05, 1.0, -8.7), (0.0, 0.0, 0.0), (4.1, 2.45, 2.0), (0.88, 0.80, 0.72), 0.72),
    3: FingerSpec(3, (0.0, 1.2, -9.0), (0.0, 0.0, 0.0), (4.55, 2.85, 2.15), (0.90, 0.82, 0.74), 0.72),
    4: FingerSpec(4, (1.9, 0.95, -8.6), (0.0, 0.0, 0.0), (4.25, 2.7, 2.1), (0.85, 0.77, 0.70), 0.72),
    5: FingerSpec(5, (3.6, 0.2, -7.9), (0.0, 0.0, 0.0), (3.65, 2.2, 1.9), (0.76, 0.69, 0.63), 0.72),
}

REST_FLEX = {1: (0.10, 0.25, 0.25), 2: (0.42, 0.95, 0.68), 3: (0.42, 0.98, 0.70),
             4: (0.42, 0.95, 0.68), 5: (0.42, 0.90, 0.65)}
WRIST_HEIGHT = 4.5          # wrist joint height above white-key tops
HOVER = 1.15                # idle fingertip clearance above the key surface

# Comfortable / maximum spans between finger pairs, in cm (RH ordering, absolute).
# Roughly: thumb-pinky comfortable octave (8 whites ~ 16.5cm), max ~ 9th/10th.
SPAN_COMFORT = {(1, 2): 9.5, (1, 3): 12.0, (1, 4): 14.0, (1, 5): 17.0, (2, 3): 4.8, (2, 4): 7.4,
                (2, 5): 10.5, (3, 4): 4.8, (3, 5): 7.6, (4, 5): 4.8}
SPAN_MAX = {(1, 2): 13.5, (1, 3): 15.5, (1, 4): 17.5, (1, 5): 21.5, (2, 3): 7.6, (2, 4): 10.2,
            (2, 5): 14.0, (3, 4): 6.6, (3, 5): 10.4, (4, 5): 7.2}


def scene_geometry() -> dict:
    """Everything the JS renderer needs, serialised once."""
    keys = []
    for m in range(LOWEST_MIDI, HIGHEST_MIDI + 1):
        k = key(m)
        keys.append({"midi": m, "black": k.black, "x": round(k.x, 4), "w": k.width,
                     "len": k.length, "top": k.top, "name": note_name(m)})
    return {
        "units": "cm",
        "keys": keys,
        "white": {"w": WHITE_W, "gap": WHITE_GAP, "len": WHITE_LEN, "h": WHITE_H,
                  "depth": KEY_DEPTH_WHITE},
        "black": {"w": BLACK_W, "len": BLACK_LEN, "top": BLACK_TOP, "depth": KEY_DEPTH_BLACK},
        "hand": {
            "fingers": {str(f): {"base": s.base, "baseEuler": s.base_euler, "lengths": s.lengths,
                                 "radii": s.radii, "tipPad": s.tip_pad}
                        for f, s in FINGERS.items()},
            "wristHeight": WRIST_HEIGHT,
        },
    }
