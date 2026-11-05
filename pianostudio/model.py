"""Core data structures."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

RIGHT = "R"
LEFT = "L"


@dataclass
class Note:
    id: int
    pitch: int
    start: float          # seconds (exact, from tick + tempo map)
    end: float
    velocity: int
    start_tick: int = 0
    end_tick: int = 0
    track: int = 0
    channel: int = 0
    hand: Optional[str] = None          # "L" / "R"
    hand_source: str = "unknown"        # midi-track | midi-channel | heuristic | user
    finger: Optional[int] = None        # 1..5
    finger_source: str = "none"         # midi | auto | user

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TempoChange:
    tick: int
    time: float
    tempo: int   # microseconds per beat

    @property
    def bpm(self) -> float:
        return 60_000_000 / self.tempo


@dataclass
class TimeSignature:
    tick: int
    time: float
    numerator: int
    denominator: int


@dataclass
class PedalEvent:
    time: float
    down: bool
    tick: int = 0


@dataclass
class Song:
    notes: list[Note]
    ticks_per_beat: int
    tempos: list[TempoChange]
    time_signatures: list[TimeSignature]
    pedal: list[PedalEvent]
    duration: float
    track_names: list[str] = field(default_factory=list)
    source_file: str = ""
    parse_log: list[dict] = field(default_factory=list)

    def notes_for(self, hand: str) -> list[Note]:
        return [n for n in self.notes if n.hand == hand]

    def pedal_down_at(self, t: float) -> bool:
        state = False
        for ev in self.pedal:
            if ev.time > t:
                break
            state = ev.down
        return state

    def beat_times(self, until: float | None = None) -> list[tuple[float, bool]]:
        """(time, is_downbeat) for each beat; follows tempo + time-signature maps exactly."""
        until = self.duration if until is None else until
        out: list[tuple[float, bool]] = []
        tpb = self.ticks_per_beat
        sigs = self.time_signatures or [TimeSignature(0, 0.0, 4, 4)]
        tick = 0
        sig_i = 0
        beat_in_bar = 0
        while True:
            while sig_i + 1 < len(sigs) and sigs[sig_i + 1].tick <= tick:
                sig_i += 1
                beat_in_bar = 0
            sig = sigs[sig_i]
            t = tick_to_seconds(tick, self.tempos, tpb)
            if t > until + 1e-9:
                break
            out.append((t, beat_in_bar == 0))
            step = int(tpb * 4 / sig.denominator)
            tick += step
            beat_in_bar = (beat_in_bar + 1) % sig.numerator
        return out


def tick_to_seconds(tick: int, tempos: list[TempoChange], tpb: int) -> float:
    """Exact conversion using the tempo map (no cumulative float drift: each
    segment is computed from its own anchor)."""
    seg = tempos[0]
    for tc in tempos:
        if tc.tick <= tick:
            seg = tc
        else:
            break
    return seg.time + (tick - seg.tick) * seg.tempo / (tpb * 1_000_000)
