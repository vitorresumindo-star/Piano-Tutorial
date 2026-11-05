"""Video program: the ordered segments (intro, performance, practice, outro)
and the exact mapping between video time and song time.

Both the audio renderer and the visual renderer consume this structure, which
is what guarantees audio/visual synchronisation: a note with song time ``s`` in
a segment is heard at sample ``round((seg.start + (s - seg.songStart)/rate) * sr)``
and seen at video frame ``(seg.start + (s - seg.songStart)/rate) * fps``.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from fractions import Fraction

from .model import Song


@dataclass
class Segment:
    type: str                 # intro | performance | countdown | practice | outro
    start: float              # video seconds
    end: float
    songStart: float | None = None
    rate: float = 1.0         # song seconds per video second (0 = frozen)
    songEnd: float | None = None
    audio: bool = False       # render notes in this segment
    showFingers: bool | None = None
    label: str = ""
    count: int = 0
    repeatIndex: int = 0
    repeatTotal: int = 0
    dipIn: bool = False

    def video_time(self, song_t: float) -> float:
        return self.start + (song_t - self.songStart) / self.rate

    def to_dict(self):
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}


@dataclass
class Program:
    fps: int
    width: int
    height: int
    segments: list[Segment] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.segments[-1].end if self.segments else 0.0

    @property
    def frame_count(self) -> int:
        return frame_count(self.duration, self.fps)

    def to_dict(self):
        return {"fps": self.fps, "width": self.width, "height": self.height, "duration": self.duration,
                "frames": self.frame_count, "segments": [s.to_dict() for s in self.segments]}


def frame_count(duration: float, fps: int) -> int:
    return int(Fraction(duration).limit_denominator(1_000_000) * fps)


def frame_time(i: int, fps: int) -> float:
    """Exact time of frame i (rational arithmetic; no accumulated drift)."""
    return float(Fraction(i, fps))


def default_practice_window(song: Song, bars: int = 8, max_seconds: float = 22.0) -> tuple[float, float]:
    if not song.notes:
        return 0.0, 0.0
    first = min(n.start for n in song.notes)
    downbeats = [t for t, db in song.beat_times() if db]
    start = max([t for t in downbeats if t <= first + 1e-6] or [first])
    later = [t for t in downbeats if t > start + 1e-6]
    end = later[bars - 1] if len(later) >= bars else song.duration
    end = min(end, start + max_seconds, song.duration)
    # snap end to a downbeat if possible
    cands = [t for t in downbeats if start + 4 < t <= end + 1e-6]
    if cands:
        end = cands[-1]
    return start, end


def build_program(song: Song, cfg: dict) -> Program:
    v = cfg["video"]
    st = cfg["structure"]
    travel = float(cfg["notes"]["travelTime"])
    prog = Program(int(v["fps"]), int(v["width"]), int(v["height"]))
    first = min((n.start for n in song.notes), default=0.0)
    last = max((n.end for n in song.notes), default=0.0)
    perf_song_start = first - travel - 1.0
    perf_song_end = last + 1.8
    t = 0.0
    if st.get("intro", True):
        d = float(st.get("introSeconds", 5.0))
        prog.segments.append(Segment("intro", t, t + d, songStart=perf_song_start, rate=0.0))
        t += d
    d = perf_song_end - perf_song_start
    prog.segments.append(Segment("performance", t, t + d, songStart=perf_song_start, songEnd=perf_song_end,
                                 rate=1.0, audio=True))
    t += d
    if st.get("practice", False) and song.notes:
        rate = float(st.get("practiceRate", 0.6))
        ps, pe = st.get("practiceStart"), st.get("practiceEnd")
        if ps is None or pe is None:
            ps, pe = default_practice_window(song, int(st.get("practiceBars", 8)))
        repeats = max(1, int(st.get("practiceRepeats", 1)))
        if st.get("countdown", True):
            d = travel / rate
            prog.segments.append(Segment("countdown", t, t + d, songStart=ps - travel, rate=rate,
                                         songEnd=ps, audio=False, count=3, label=st.get("practiceLabel", "Slow practice"),
                                         dipIn=True, showFingers=True))
            t += d
        for r in range(repeats):
            tail = 0.6
            sstart = ps if (r > 0 or st.get("countdown", True)) else ps - travel
            d = (pe + tail - sstart) / rate
            prog.segments.append(Segment("practice", t, t + d, songStart=sstart, songEnd=pe + tail, rate=rate,
                                         audio=True, showFingers=True, repeatIndex=r + 1, repeatTotal=repeats,
                                         label=st.get("practiceLabel", "Slow practice")))
            t += d
    if st.get("outro", True):
        d = float(st.get("outroSeconds", 4.5))
        prog.segments.append(Segment("outro", t, t + d, songStart=perf_song_end, rate=0.0, dipIn=False))
        t += d
    # snap total duration to a whole number of frames
    end = frame_count(t, prog.fps) / prog.fps
    prog.segments[-1].end = end
    return prog
