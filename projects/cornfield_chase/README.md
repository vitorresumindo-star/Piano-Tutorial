# Cornfield Chase (Interstellar): waiting for a licensed MIDI

This project is ready to render. The only missing piece is the music data.
*Cornfield Chase* (Hans Zimmer, 2014) is a copyrighted composition. No accurate
licensed arrangement was available in this workspace, and it was deliberately
**not** transcribed from memory, because that would produce an unverified,
possibly wrong arrangement.

## The file you need to add

`projects/cornfield_chase/cornfield_chase.mid`

* A **licensed** solo-piano arrangement of Cornfield Chase in MIDI format. For
  example, a licensed digital score from a sheet-music service that lets you
  export or download MIDI, or a MIDI recording made by a pianist from licensed
  sheet music.
* Best: two tracks named **Right Hand** and **Left Hand**. Otherwise the hands
  are split automatically, so check the result.
* Sustain pedal (CC64) and velocities are used if present.
* Any length. To use only part of a long arrangement, add an excerpt to
  `project.json`:
  `"excerpt": {"fromBar": 1, "toBar": 28}` (or `{"from": 0.0, "to": 72.0}` in seconds).

## Render

```bash
python3 -m pianostudio reel projects/cornfield_chase
```

Without the MIDI, the command stops with a message telling you which file is
missing. Nothing is substituted silently.

## Footage

The project uses the same **original** procedural scenes as the demo: a
golden-hour cornfield, a pickup in the dust, a dust storm, a farm under the
stars, a launch, orbit, a ringed giant and a black hole. If you have permission
to use real film clips, put them in `footage/` and replace any shot with:

```json
{"bar": 9, "clip": "footage/shot01.mp4", "in": 12.5, "focus": [0.5, 0.4], "zoom": [1.0, 1.06]}
```

`in` is the start time in the source clip. `focus` is the point kept in frame
when cropping to the top panel (x and y, as 0–1 fractions of the frame).
`zoom` is a slow push-in. Shots without `bar`/`time` are spread evenly and
snapped to bar lines.

## Posting

Platforms such as TikTok, Instagram and YouTube have their own rules for piano
covers of copyrighted music. Using a licensed arrangement covers the sheet
music. It does not automatically grant the right to publish a recording. Check
the platform's music policy before posting.
