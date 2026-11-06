"""Bottom panel of the reel: white piano roll, outlined falling notes, labelled
keyboard with coloured key presses, and the illustrated hands."""
from __future__ import annotations

import math
from dataclasses import dataclass

import cairo
import numpy as np

from ..geometry import is_black
from ..model import LEFT, RIGHT, Note
from .hands2d import HandGeometry, HandStyle, HandTrack, draw_hand, draw_hand_shadow, hex_rgb
from .layout import Layout

# Warm greens / yellows / oranges with complementary teal, sky and violet.
DEFAULT_PALETTE = {
    "C": "#FF6B5B", "C#": "#FF8C42", "D": "#FFA630", "D#": "#FFC23C", "E": "#FFDA3D", "F": "#C2E851",
    "F#": "#8EDB4F", "G": "#4CCB72", "G#": "#2FC4A0", "A": "#35B8E0", "A#": "#6F8CF6", "B": "#B07CF0",
}
SHARP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
COMMON_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B"]


def display_name(name: str) -> str:
    return name.replace("#", "♯").replace("b", "♭") if len(name) > 1 else name


@dataclass
class RollStyle:
    fall_seconds: float = 2.2
    palette: dict = None
    color_mode: str = "pitch"            # "pitch" | "hand"
    hand_colors: tuple = ("#4CCB72", "#FFA630")   # right, left
    outline: float = 3.0
    radius: float = 8.0
    note_labels: bool = True
    key_labels: str = "all"              # "all" | "c" | "none"
    spelling: str = "common"             # "common" | "sharp" | "flat"
    bar_lines: bool = True
    background: str = "#FFFFFF"
    guide: str = "#EDEFF3"
    body: str = "#16161B"
    felt: str = "#8E1F2B"
    label_font: str = "Inter"


def _mix(a, b, t):
    return tuple(a[i] * (1 - t) + b[i] * t for i in range(3))


class PianoRoll:
    def __init__(self, lay: Layout, notes: list[Note], style: RollStyle, hand_style: HandStyle,
                 duration: float, bar_times: list[float] | None = None):
        self.lay, self.style, self.hand_style = lay, style, hand_style
        self.notes = sorted(notes, key=lambda n: n.start)
        self.starts = np.array([n.start for n in self.notes])
        self.max_dur = max((n.duration for n in self.notes), default=0.0)
        self.speed = (lay.key_top - lay.notes_top) / style.fall_seconds
        pal = dict(DEFAULT_PALETTE)
        pal.update(style.palette or {})
        self.pc_color = [hex_rgb(pal[SHARP_NAMES[i]]) for i in range(12)]
        self.hand_color = {RIGHT: hex_rgb(style.hand_colors[0]), LEFT: hex_rgb(style.hand_colors[1])}
        names = {"sharp": SHARP_NAMES, "flat": FLAT_NAMES}.get(style.spelling, COMMON_NAMES)
        self.pc_name = [display_name(n) for n in names]
        self.tracks = {}
        for hand in (RIGHT, LEFT):
            hn = [n for n in self.notes if n.hand == hand]
            if hn:
                self.tracks[hand] = HandTrack(hn, hand, lay, duration)
        self.visual_end = {}
        for tr in self.tracks.values():
            self.visual_end.update(tr.visual_end)
        self.bar_times = bar_times or []
        self.height = lay.height - lay.footage_h
        self.surface = cairo.ImageSurface(cairo.FORMAT_RGB24, lay.width, self.height)

    def color(self, n: Note):
        if self.style.color_mode == "hand":
            return self.hand_color.get(n.hand, self.pc_color[n.pitch % 12])
        return self.pc_color[n.pitch % 12]

    # ------------------------------------------------------------------ frame
    def render(self, t: float) -> np.ndarray:
        """RGB uint8 array of the bottom panel at video time t."""
        lay, st = self.lay, self.style
        ctx = cairo.Context(self.surface)
        ctx.translate(0, -lay.footage_h)
        ctx.set_antialias(cairo.ANTIALIAS_BEST)
        self._background(ctx, t)
        sounding = self._notes(ctx, t)
        self._body(ctx)
        pressed = self._keyboard(ctx, t, sounding)
        self._glow(ctx, t, sounding)
        states = {h: tr.state(t) for h, tr in self.tracks.items()}
        geos = [HandGeometry(s, lay.white_w) for s in states.values()]
        for g in geos:
            draw_hand_shadow(ctx, g, self.hand_style, lay.width, lay.height)
        for g in geos:
            draw_hand(ctx, g, self.hand_style)
        self._top_shadow(ctx)
        self.surface.flush()
        buf = np.ndarray((self.height, self.surface.get_stride() // 4, 4), np.uint8, self.surface.get_data())
        return buf[:, :lay.width, 2::-1].copy()   # BGRx -> RGB

    # ------------------------------------------------------------------ parts
    def _background(self, ctx, t):
        lay, st = self.lay, self.style
        ctx.set_source_rgb(*hex_rgb(st.background))
        ctx.rectangle(0, lay.notes_top, lay.width, lay.key_top - lay.notes_top)
        ctx.fill()
        gc = hex_rgb(st.guide)
        for m in lay.keys():
            if m % 12 in (0, 5):
                x = round(lay.white_left(m)) + 0.5
                ctx.set_source_rgb(*gc)
                ctx.set_line_width(2.0 if m % 12 == 0 else 1.0)
                ctx.move_to(x, lay.notes_top)
                ctx.line_to(x, lay.key_top)
                ctx.stroke()
        if st.bar_lines:
            ctx.set_line_width(1.5)
            for bt in self.bar_times:
                y = lay.key_top - (bt - t) * self.speed
                if lay.notes_top < y < lay.key_top:
                    ctx.set_source_rgba(0.55, 0.6, 0.7, 0.16)
                    ctx.move_to(0, round(y) + 0.5)
                    ctx.line_to(lay.width, round(y) + 0.5)
                    ctx.stroke()

    def _visible_notes(self, t):
        lo = t - self.max_dur - 0.1
        hi = t + self.style.fall_seconds + 0.1
        i0 = int(np.searchsorted(self.starts, lo))
        i1 = int(np.searchsorted(self.starts, hi))
        return [n for n in self.notes[i0:i1] if n.end > t - 0.02]

    def _notes(self, ctx, t):
        lay, st = self.lay, self.style
        W = lay.white_w
        vis = self._visible_notes(t)
        # a key is lit while its finger holds it down (the falling bar still shows the full length)
        sounding = [n for n in vis if n.start <= t < min(n.end, self.visual_end.get(n.id, n.end) + 0.035)]
        ctx.save()
        ctx.rectangle(0, lay.notes_top, lay.width, lay.key_top - lay.notes_top)
        ctx.clip()
        boxes = []
        for n in vis:
            y_bot = lay.key_top - (n.start - t) * self.speed
            y_top = lay.key_top - (n.end - t) * self.speed + 3.0
            if y_bot - y_top < 14:
                y_top = y_bot - 14
            if y_bot < lay.notes_top or y_top > lay.key_top:
                continue
            w = W * 0.82 if not is_black(n.pitch) else lay.black_w * 0.96
            x = lay.x_center(n.pitch) - w / 2
            boxes.append((n, x, y_top, w, y_bot - y_top))
        # white-key notes first so black-key notes sit on top
        boxes.sort(key=lambda b: is_black(b[0].pitch))
        r = st.radius
        for n, x, y, w, h in boxes:
            _rrect(ctx, x + 3, y + 4, w, h, r)
            ctx.set_source_rgba(0, 0, 0, 0.13)
            ctx.fill()
        for n, x, y, w, h in boxes:
            c = self.color(n)
            active = n.start <= t < n.end
            if active:
                c = _mix(c, (1, 1, 1), 0.28)
            if is_black(n.pitch):
                c = _mix(c, (0.1, 0.1, 0.15), 0.12)
            _rrect(ctx, x, y, w, h, r)
            g = cairo.LinearGradient(x, 0, x + w, 0)
            g.add_color_stop_rgb(0.0, *_mix(c, (1, 1, 1), 0.30))
            g.add_color_stop_rgb(0.28, *c)
            g.add_color_stop_rgb(1.0, *_mix(c, (0, 0, 0), 0.10))
            ctx.set_source(g)
            ctx.fill_preserve()
            ctx.set_source_rgb(0.07, 0.07, 0.09)
            ctx.set_line_width(st.outline)
            ctx.stroke()
            # glossy highlight strip
            _rrect(ctx, x + w * 0.16, y + 5, w * 0.16, max(0.0, h - 10), w * 0.08)
            ctx.set_source_rgba(1, 1, 1, 0.28)
            ctx.fill()
            if st.note_labels:
                vis_bot = min(y + h, lay.key_top)
                fs = min(W * 0.40, w * 0.62)
                if vis_bot - y > fs + 12:
                    name = self.pc_name[n.pitch % 12]
                    ctx.select_font_face(st.label_font, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
                    ctx.set_font_size(fs if len(name) == 1 else fs * 0.82)
                    ext = ctx.text_extents(name)
                    tx = x + w / 2 - ext.x_advance / 2
                    ty = vis_bot - fs * 0.45
                    ctx.move_to(tx, ty)
                    ctx.set_source_rgba(0.08, 0.08, 0.1, 0.85)
                    ctx.show_text(name)
        ctx.restore()
        return sounding

    def _body(self, ctx):
        lay, st = self.lay, self.style
        g = cairo.LinearGradient(0, lay.key_bottom, 0, lay.height)
        b = hex_rgb(st.body)
        g.add_color_stop_rgb(0, *_mix(b, (1, 1, 1), 0.10))
        g.add_color_stop_rgb(0.12, *b)
        g.add_color_stop_rgb(1, *_mix(b, (0, 0, 0), 0.4))
        ctx.set_source(g)
        ctx.rectangle(0, lay.key_bottom, lay.width, lay.height - lay.key_bottom)
        ctx.fill()

    def _keyboard(self, ctx, t, sounding):
        lay, st = self.lay, self.style
        W, kt, kh = lay.white_w, lay.key_top, lay.key_h
        pressed = {}
        for n in sounding:
            pressed[n.pitch] = self.color(n)
        # white keys
        for m in lay.keys():
            if is_black(m):
                continue
            x = lay.white_left(m)
            c = pressed.get(m)
            dip = 3 if c else 0
            _rrect_bottom(ctx, x + 1, kt, W - 2, kh + dip, 6)
            g = cairo.LinearGradient(0, kt, 0, kt + kh)
            if c:
                g.add_color_stop_rgb(0, *_mix(c, (0, 0, 0), 0.25))
                g.add_color_stop_rgb(0.25, *_mix(c, (1, 1, 1), 0.10))
                g.add_color_stop_rgb(1, *_mix(c, (1, 1, 1), 0.35))
            else:
                g.add_color_stop_rgb(0, 0.86, 0.86, 0.86)
                g.add_color_stop_rgb(0.08, 0.99, 0.99, 0.99)
                g.add_color_stop_rgb(0.85, 0.97, 0.97, 0.98)
                g.add_color_stop_rgb(1, 0.88, 0.88, 0.90)
            ctx.set_source(g)
            ctx.fill_preserve()
            ctx.set_source_rgb(0.55, 0.55, 0.58)
            ctx.set_line_width(1.2)
            ctx.stroke()
        # black keys
        bw, bh = lay.black_w, lay.black_h
        for m in lay.keys():
            if not is_black(m):
                continue
            x = lay.x_center(m) - bw / 2
            c = pressed.get(m)
            face = 0.07 * bh if c else 0.12 * bh
            _rrect_bottom(ctx, x, kt, bw, bh, 4)
            g = cairo.LinearGradient(x, 0, x + bw, 0)
            base = _mix(c, (0, 0, 0), 0.25) if c else (0.13, 0.13, 0.15)
            g.add_color_stop_rgb(0, *_mix(base, (0, 0, 0), 0.35))
            g.add_color_stop_rgb(0.5, *base)
            g.add_color_stop_rgb(1, *_mix(base, (0, 0, 0), 0.45))
            ctx.set_source(g)
            ctx.fill()
            top = _mix(c, (1, 1, 1), 0.05) if c else (0.22, 0.22, 0.25)
            _rrect_bottom(ctx, x + bw * 0.12, kt, bw * 0.76, bh - face, 3)
            g = cairo.LinearGradient(0, kt, 0, kt + bh - face)
            g.add_color_stop_rgb(0, *_mix(top, (0, 0, 0), 0.4))
            g.add_color_stop_rgb(0.85, *top)
            g.add_color_stop_rgb(1, *_mix(top, (1, 1, 1), 0.25))
            ctx.set_source(g)
            ctx.fill()
        # felt strip + shadow
        ctx.set_source_rgb(*hex_rgb(st.felt))
        ctx.rectangle(0, kt - 7, lay.width, 7)
        ctx.fill()
        g = cairo.LinearGradient(0, kt, 0, kt + 14)
        g.add_color_stop_rgba(0, 0, 0, 0, 0.28)
        g.add_color_stop_rgba(1, 0, 0, 0, 0)
        ctx.set_source(g)
        ctx.rectangle(0, kt, lay.width, 14)
        ctx.fill()
        # key labels
        if st.key_labels != "none":
            ctx.select_font_face(st.label_font, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
            for m in lay.keys():
                if is_black(m):
                    continue
                if st.key_labels == "c" and m % 12:
                    continue
                name = self.pc_name[m % 12] + (str(m // 12 - 1) if m % 12 == 0 else "")
                fs = W * (0.36 if len(name) == 1 else 0.31)
                ctx.set_font_size(fs)
                ext = ctx.text_extents(name)
                x = lay.white_left(m) + W / 2 - ext.x_advance / 2
                y = kt + kh - W * 0.30
                ctx.move_to(x, y)
                c = pressed.get(m)
                ctx.set_source_rgba(*((0.1, 0.1, 0.12) if c else (0.42, 0.43, 0.47)), 0.95)
                ctx.show_text(name)
        return pressed

    def _glow(self, ctx, t, sounding):
        lay = self.lay
        W = lay.white_w
        for n in sounding:
            c = self.color(n)
            x = lay.x_center(n.pitch)
            age = t - n.start
            a = 0.55 * math.exp(-age * 2.2) + 0.18
            rg = cairo.RadialGradient(x, lay.key_top, 0, x, lay.key_top, W * 1.1)
            rg.add_color_stop_rgba(0, *_mix(c, (1, 1, 1), 0.5), a)
            rg.add_color_stop_rgba(1, *c, 0)
            ctx.set_source(rg)
            ctx.rectangle(x - W * 1.2, lay.key_top - W * 1.2, W * 2.4, W * 1.4)
            ctx.fill()

    def _top_shadow(self, ctx):
        lay = self.lay
        g = cairo.LinearGradient(0, lay.notes_top, 0, lay.notes_top + 18)
        g.add_color_stop_rgba(0, 0, 0, 0, 0.22)
        g.add_color_stop_rgba(1, 0, 0, 0, 0)
        ctx.set_source(g)
        ctx.rectangle(0, lay.notes_top, lay.width, 18)
        ctx.fill()


def _rrect(ctx, x, y, w, h, r):
    r = max(0.0, min(r, w / 2, h / 2))
    ctx.new_sub_path()
    ctx.arc(x + w - r, y + r, r, -math.pi / 2, 0)
    ctx.arc(x + w - r, y + h - r, r, 0, math.pi / 2)
    ctx.arc(x + r, y + h - r, r, math.pi / 2, math.pi)
    ctx.arc(x + r, y + r, r, math.pi, 3 * math.pi / 2)
    ctx.close_path()


def _rrect_bottom(ctx, x, y, w, h, r):
    r = max(0.0, min(r, w / 2, h / 2))
    ctx.new_sub_path()
    ctx.move_to(x, y)
    ctx.line_to(x + w, y)
    ctx.arc(x + w - r, y + h - r, r, 0, math.pi / 2)
    ctx.arc(x + r, y + h - r, r, math.pi / 2, math.pi)
    ctx.close_path()
