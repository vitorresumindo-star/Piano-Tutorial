# Piano Studio

An automated generator for cinematic piano-tutorial videos. Give it a MIDI file. It produces a YouTube-ready MP4 with:

- an overhead 3D piano
- articulated hands driven by inverse kinematics, playing the assigned keys
- falling notes in separate colours for each hand
- sample-based grand-piano audio
- an animated title, an optional slow practice section and an outro
- a thumbnail and draft metadata

Every movement comes from structured musical data: MIDI → hand/finger assignment → kinematics. No generative video is involved.

```
MIDI ──► midi_io ──► fingering ──► animation (IK bake) ──► validation ──┬─► audio (sampler, reverb, LUFS)
                                                                        └─► scene data ──► Three.js renderer ──► FFmpeg MP4
```

---

## Quick start

```bash
./scripts/setup.sh                      # pip + npm deps, CC-BY piano samples (~400 MB), environment check
python3 -m pianostudio serve            # web app → http://127.0.0.1:8765
```

In the app, pick an example or upload a MIDI file, choose a theme, review the fingering, then click **Generate video**.

Command line:

```bash
python3 -m pianostudio analyze song.mid                       # validation report only
python3 -m pianostudio render song.mid --title "Clair de Lune" --composer "Debussy" \
        --preset presets/channel_default.json --video 1080p30 --practice
python3 -m pianostudio render song.mid --frames 300:390        # quick partial render
python3 -m pianostudio batch my_midis/ --preset presets/channel_default.json --out output/
python3 -m pianostudio doctor                                 # dependency / GPU check
python3 -m pytest                                             # automated QA (33 tests)
```

Each song's output goes to `output/<slug>/`:

| file | content |
|---|---|
| `<slug>_1920x1080_30fps.mp4` | H.264 High / yuv420p / AAC 320k, +faststart |
| `thumbnail.jpg` | 1280×720 rendered still plus title typography |
| `metadata.json`, `description.txt` | draft YouTube title, description (with chapters and credits), tags |
| `validation.md`, `validation.json` | machine-checked results and items that need human review |
| `fingering.json` | the final hand and finger for every note |
| `project.json` | the exact configuration used (reproducible) |
| `render.log` | renderer log |

---

## What is implemented

### 1. MIDI engine (`pianostudio/midi_io.py`, `model.py`)
- Reads notes, velocities, multi-track and multi-channel files, tempo maps, time signatures and sustain pedal (CC64).
- **No timing drift.** Each tick is converted to seconds from its own tempo segment, never by adding up deltas. Video frames are exact rationals (`frame_time(i) = i/fps`). Tested out to 50 minutes.
- Cleans up messy files: re-struck and overlapping same-pitch notes, unterminated notes, cross-track duplicates, out-of-range notes. Each fix is logged in the report.
- Hand detection, in order: track names (multi-language), two-track pitch ordering, two-channel ordering, then a heuristic split with running hand centres. A heuristic split is always flagged for review.
- Optional fingering sidecar: `song.fingering.json`.

### 2. Fingering (`fingering.py`)
- Beam search over chords. Fingers inside a chord always follow pitch order.
- Costs come from a hand-position model: natural finger offsets, comfortable and maximum spans per finger pair, movement scaled by the time available, thumb-under and finger-over crossings, repeated fingers, thumbs on black keys, and fingers still holding notes.
- Manual corrections (UI or `overrides` in the project file) become hard constraints.
- Reports `unplayable-stretch`, `too-many-notes`, `difficult-passage` and `fixed-finger-conflict`.
- **Automatic fingering is a plausible, physically checked suggestion. It is not guaranteed to match a professional pianist's fingering.** The report and the video description say so.

### 3. Hands and kinematics (`animation.py`, `web/js/hands.js`)
- The hand skeleton has 5 fingers × 3 phalanges, plus a two-axis thumb base, finger spread at the knuckle, splay for wide chords, and a 3-axis wrist. The proportions are adult-anatomical.
- **Choreography:**
  - The hand is placed as a weighted blend of every finger that is engaged (preparing, holding or releasing).
  - While keys are held, the wrist is clamped into the region from which every pressing finger can reach its key.
  - For leaps, keys are released visually a moment early, as a pianist would (the pedal keeps the sound). Audio keeps the exact MIDI durations.
- Fingers follow minimum-jerk approach, press, hold, release and lift curves. Key travel is synchronised so that **each key reaches the bottom exactly at the note onset**.
- A separation pass keeps neighbouring fingertips in order and apart.
- **IK:** vectorised analytic and bisection solver per finger (abduction, MCP, PIP, plus DIP coupled to PIP), with anatomical joint limits.
- **Python is the single source of truth.** Joint angles are baked at 120 Hz. The renderer only applies forward kinematics with identical conventions, so playback is deterministic.
- **Mesh:** the hand is a signed-distance model (palm, thenar and hypothenar eminences, tapered phalanges, knuckles, wrist). It is meshed once with surface nets, smoothed, and given distance-based skin weights on the same bone hierarchy. That gives a single smooth `SkinnedMesh`, plus nails, skin-tone variation, a pore normal map, and optional sleeves.

### 4. Falling notes (`web/js/studio.js`)
- One instanced quad per note, positioned in the shader from song time. There is no per-frame CPU work, and notes cannot drift away from the audio.
- The note plane is tilted to face the camera, so notes keep their true length without foreshortening, and they land at the back edge of the keys.
- Effects: rounded rectangles, a glow halo, three styles (flat, soft, neon), a brighter active note, a hit glow and spark particles at the landing line, and pressed keys tinted with the hand colour.
- Optional labels: finger numbers or note names (texture atlas). Practice mode always shows finger numbers.
- Adjustable travel time (note speed), glow, and difficulty presets (Beginner: slow with labels; Intermediate; Advanced: fast).

### 5. Audio (`audio.py`)
- Offline SFZ sampler (numpy/scipy) using **Salamander Grand Piano V3** (Yamaha C5, 16 velocity layers, CC BY 3.0). The credit is added automatically to the outro and to the description.
- Velocity layers and velocity tracking, sustain pedal, damper release, damper-less top keys, string-resonance release samples, and quick damping when a key is re-struck.
- Synthetic stereo convolution reverb (adjustable), optional metronome, and countdown clicks.
- ITU-R BS.1770 integrated-loudness normalisation (default −14 LUFS, YouTube's reference) with a true-peak limiter at −1 dBTP.
- If the samples are not installed, a fallback additive synth is used (clearly lower quality).

### 6. Video program (`program.py`)
- Segments: `intro` (animated title, difficulty chip, hand-colour legend), `performance` (progress bar), `countdown` (3-2-1; the notes are already falling), `practice` (default 60% speed, finger numbers, repeats), `outro` (branding).
- The same video-time ↔ song-time map drives the audio and the picture.

### 7. Themes (`themes/*.json`)
Four themes: **Midnight Studio**, **Ivory Minimal**, **Warm Concert Hall** (procedural wood) and **Modern Neon**. Each is plain JSON (colours, materials, lights, UI fonts, post-processing, camera tilt). You can edit or add themes without rebuilding anything.

### 8. Rendering and export (`renderer/render.mjs`, `export.py`)
- Headless Chromium (Playwright) renders the same Three.js scene used by the live preview. The renderer reads raw RGBA back and pipes it into FFmpeg (libx264 + AAC). Chunks can render in parallel with `--workers N` and are then concatenated.
- Presets: `1080p30`, `1080p60`, `4k30`, `4k60`, `shorts` (1080×1920 with portrait framing), `preview` (720p).
- **Integrity checks** after every export: codecs, resolution, fps, exact frame count, pixel format, and audio length equal to video length.
- **Hardware detection:** `doctor` and `produce()` check for an NVIDIA GPU and enable GPU flags if one is found. Otherwise rendering uses SwiftShader on the CPU (see *Performance*).
- `--cinematic` enables a full bloom post-processing chain. It is recommended only with a GPU.

### 9. Web app (`pianostudio/server.py`, `web/`)
- **Control panel:** song details, theme cards, hand colours, keyboard range (fit to song or all 88 keys), skin tone (6 presets), sleeves, falling-note settings, video structure, reverb, metronome, branding, export format, strict mode.
- **Large live 3D preview:** play, pause and scrub, synchronised to the rendered audio, with 16:9 and 9:16 previews.
- **Fingering editor:** a piano roll coloured by hand. Notes with problems are outlined. Click a note, then press `1`–`5` or `L`/`R`; overrides are saved and the preview updates. **Approve** marks the fingering as reviewed.
- **Validation tab:** click an issue to jump to it.
- **Renders tab:** progress, ETA, and links to the video, thumbnail, metadata and report.
- **Channel presets:** save and load.
- **Batch dialog.**

### 10. Batch production (`batch.py`)
Point it at a folder of MIDI files. Optional per-song settings go in `song.json` next to `song.mid`.

For each song it parses, fingers, animates and validates, then **renders only if validation passes**. Songs that need fingering review are **paused** with a `STATUS.txt` and their report, not exported. The channel preset applies to every song. The run writes `batch_summary.md/json`. One failed song does not stop the batch.

### 11. Quality assurance
`python3 -m pytest` runs 33 tests. They cover:
- MIDI notes, velocities, tempo changes without drift, time signatures, pedal, hand detection, chord grouping, overlap repair
- the 88-key layout and out-of-range rejection
- fingering: completeness, chord order, user overrides, impossible stretch, too many notes in one hand
- key-depression profile, and key bottom exactly at the onset
- fingertips on the intended key, with skin on the depressed surface
- forward kinematics equal to the IK targets, joint limits, no sinking into keys, smooth motion
- validation pass and block cases
- exact frame times and program mapping
- audio onsets within one video frame of the visual note time
- pedal sustain, velocity loudness, LUFS normalisation
- a real short render with export-integrity checks
- the JS renderer's fingertip positions matching the Python solver (< 0.001 cm)
- batch mode pausing a song that fails validation

The **debug overlay** (`--debug` or the checkbox) draws fingertip contact markers. `render.mjs --tips T` prints the renderer's own fingertip positions for comparison with Python.

**Machine-checked vs. human review.** The report lists both explicitly:
- **Machine-checked:** key mapping, finger conflicts, contact accuracy (≤ 0.3 cm), correct key, no floating fingertips, sustained fingertip collisions, hand overlap, timing.
- **Human review:** the musical and pedagogical quality of the fingering, heuristic left/right splits, and visual realism.

---

## Performance

With no GPU, frames are rasterised by SwiftShader (CPU WebGL). Measured in this repository's cloud container (4 vCPU, no GPU):

| setting | speed |
|---|---|
| 1080p, default quality, 1 worker | ~1.3 s/frame |
| 1080p, 2 workers | ~0.58 frames/s, so a 67 s video at 30 fps takes ~1 hour |
| bloom (`--cinematic`) on CPU | ~4–6 s/frame (not recommended without a GPU) |
| analysis + fingering + IK bake | < 0.3 s per song |
| audio render | ~10 s per minute of video |

A machine with a GPU (run with `--gpu`, detected automatically) is typically 20–100× faster, and 4K becomes practical. 4K on CPU works but costs about 4× the 1080p time.

## Limitations and unfinished work (honest list)

- **Hand realism.** The hands are procedurally modelled and skinned, with anatomically proportioned skeletons and smooth surfaces. They are clearly better than a jointed mannequin, but they are *not photoreal scanned hands*: no wrinkles, veins or subsurface scattering, and linear-blend skinning loses a little volume at strongly bent joints. The bone hierarchy is ready for a licensed, rigged GLTF hand model to be mapped onto it later. No such model is bundled, because no suitable openly licensed asset was available here.
- **Fingering.** The algorithm is plausible but not expert-level. Fast or virtuosic passages, voice-leading-aware fingering, finger substitution on held notes and redistribution between the hands are not modelled. Those passages get flagged.
- **Heuristic hand split.** This is the weakest part on single-track files. The test file `single_track_test.mid` is deliberately split imperfectly and is correctly **blocked** by validation.
- **Physical plausibility.** Collision checks look at fingertips (adjacent pairs, and left vs. right hand), not the full finger and palm geometry.
- **Rendering engine.** Three.js/WebGL rasterisation was chosen over Blender: Blender was not available here, and Cycles would cost minutes per frame on a CPU. Lighting is rasterised (shadow maps, a selective environment map), not path-traced.
- **Audio.** Una corda / sostenuto pedals, half-pedalling and sympathetic resonance are not modelled.
- **Copyright.** Only import MIDI files you have the rights to use. Arrangements of public-domain works can still be copyrighted.

## Repository layout

```
pianostudio/   Python package (midi_io, fingering, animation, validation, audio, program, project, export, batch, server, cli)
renderer/      render.mjs: headless Chromium → FFmpeg
web/           app UI (index.html, app.js, app.css), renderer scene (js/studio.js, hands.js, overlay.js), fonts (OFL)
themes/        visual presets (JSON)
presets/       channel presets (JSON)
examples/midi  demo MIDI files + generator script
tests/         pytest suite
scripts/       setup.sh
```

## Credits and licences
- Piano samples: Salamander Grand Piano V3 by Alexander Holm, CC BY 3.0. Downloaded by `setup.sh`, not committed.
- Fonts: Inter and Cormorant Garamond, SIL Open Font License.
- three.js: MIT.
- Demo music: *Minuet in G major*, BWV Anh. 114 (attributed to C. Petzold, public domain), new arrangement. *Morning Light* is an original étude written for this project.
