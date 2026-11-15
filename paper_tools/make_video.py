"""Submission video from the rendered showcase episodes, the paper figures and the generated numbers.

    D:/HAIC-Claude/_testenv/python.exe paper/scripts/make_video.py              # 3-min and 1-min cuts
    D:/HAIC-Claude/_testenv/python.exe paper/scripts/make_video.py --cut 1min --preview 6

Inputs
    result/showcase-skateboard/..., result/video/showcase-*/...   paired renders (video.mp4 + paths.npz + episodes.csv)
    paper/figures/*.pdf, paper/generated/numbers.tex              slides and every number in the captions
    result/video/mainfigure_video/1..4.png (optional)             build-up stages of Fig. 2; the static figure otherwise
    result/video/mujoco/skateboard-<arm>-st2-ep0.mp4 (optional)   MuJoCo clips (research/sim2sim.py --video)
Voice-over: each segment's narration is spoken by the Windows speech synthesizer (offline; --voice none for a silent
video, --voice-dir for your own recordings) and the segment lasts at least as long as its narration.
Outputs: result/video/out/haic_<cut>.mp4 (1920x1080, 30 fps, H.264 + AAC), narration_<cut>.txt (timed script),
and with --preview N one PNG every N seconds.
"""
import argparse
import csv
import math
import re
from pathlib import Path

import pymupdf as fitz
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
PAPER, VID = ROOT / 'paper', ROOT / 'result' / 'video'
OUT = VID / 'out'
W, H, FPS = 1920, 1080, 30
BG, INK, MUTE = (247, 247, 245), (40, 40, 40), (120, 120, 120)
ARM = {'dyn': ((53, 96, 154), 'HAIC (baseline)'), 'dyn-hboth': ((232, 112, 42), 'OH-Both (ours)')}
REF = (140, 140, 140)
FONT = {k: f'C:/Windows/Fonts/{v}' for k, v in (('r', 'times.ttf'), ('b', 'timesbd.ttf'), ('i', 'timesi.ttf'))}
TASK_NAME = {'skateboard': 'Skateboarding', 'push_cart': 'Pushing a cart', 'box_stairslope': 'Carrying a box over a stair and a slope'}
TERM_TEXT = {'term/cum_body_pos_error': 'torso left the reference in the world frame (drift)',
             'term/cum_lost_contact_steps': 'lost hand-object contact',
             'term/cum_object_pos_error': 'object left its reference',
             'term/cum_body_ori_error': 'torso orientation left the reference',
             'term/cum_body_pos_error_local': 'body pose error', 'term/cum_body_ori_error_local': 'body orientation error'}


def font(size, kind='r'):
    return ImageFont.truetype(FONT[kind], size)


def numbers():
    txt = (PAPER / 'generated' / 'numbers.tex').read_text(encoding='utf-8')
    return {k: v for k, v in re.findall(r'\\newcommand\{\\(\w+)\}\{([^}]*)\}', txt)}


N = numbers()
MIN_DUR = [0.0]  # lower bound on the next segment's duration (its narration), set by main()
SUBS_ON = [False]  # True while building a narrated segment: its fixed captions are replaced by timed subtitles
SENT_T = [[]]  # start times [s] of the narration's sentences within the next segment, set by main()


def show_dir(task):
    for base in (VID / f'showcase-{task}' / task, ROOT / 'result' / f'showcase-{task}' / task):
        if (base / 'st0-ev0-n256').is_dir():
            yield base / 'st0-ev0-n256'


def render_dir(task, arm, env, cam):
    for d in show_dir(task):
        p = d / arm / f'render-env{env}-{cam}'
        if (p / 'video.mp4').is_file():
            return p
    raise FileNotFoundError(f'{task} {arm} env {env} {cam}')


def pdf_png(name, dpi=300):
    """Paper figure as an RGB image (cached under out/assets)."""
    out = OUT / 'assets' / f'{name}.png'
    if not out.is_file():
        out.parent.mkdir(parents=True, exist_ok=True)
        pix = fitz.open(PAPER / 'figures' / f'{name}.pdf')[0].get_pixmap(dpi=dpi, alpha=False)
        pix.save(out)
    return Image.open(out).convert('RGB')


# ------------------------------------------------------------------ drawing helpers
def text(d, xy, s, size=40, kind='r', fill=INK, anchor='la'):
    d.text(xy, s, font=font(size, kind), fill=fill, anchor=anchor)


def wrap(s, size, width, kind='r'):
    f, words, lines, cur = font(size, kind), s.split(), [], ''
    for w_ in words:
        t = (cur + ' ' + w_).strip()
        if f.getlength(t) > width and cur:
            lines.append(cur)
            cur = w_
        else:
            cur = t
    return lines + ([cur] if cur else [])


def caption(d, s, y=H - 42):
    if s and not SUBS_ON[0]:  # with a voice-over the subtitles are the spoken words, drawn by main()
        text(d, (W // 2, y), s, 34, 'i', MUTE, 'mm')


def header(d, title, sub=''):
    text(d, (60, 50), title, 50, 'b', INK, 'lm')
    if sub:
        text(d, (W - 60, 50), sub, 34, 'r', MUTE, 'rm')
    d.line([(60, 92), (W - 60, 92)], fill=(210, 210, 210), width=2)


def fade(img, t, dur, f=0.3):
    a = min(1.0, t / f, max(0.0, (dur - t) / f))
    if a >= 1.0:
        return img
    return Image.blend(Image.new('RGB', img.size, BG), img, a)


class Clip:
    """Sequential video reader: frame at a monotonically increasing index, holding the last one."""

    def __init__(self, path, size):
        self.r, self.size, self.i, self.last = imageio.get_reader(str(path)), size, -1, None
        self.n = self.r.count_frames()
        self.r = imageio.get_reader(str(path))

    def at(self, k):
        k = min(k, self.n - 1)
        while self.i < k:
            self.last = self.r.get_next_data()
            self.i += 1
        return Image.fromarray(self.last).resize(self.size, Image.BILINEAR)


# ------------------------------------------------------------------ segments: each is (duration [s], draw(t) -> Image)
def card(lines, dur, sub=None):
    """Centered text card; lines = [(text, size, kind, color)]."""
    dur = max(dur, MIN_DUR[0])
    def draw(t):
        img = Image.new('RGB', (W, H), BG)
        d = ImageDraw.Draw(img)
        total = sum(s * 1.45 for _, s, _, _ in lines)
        y = H / 2 - total / 2
        for s, size, kind, col in lines:
            for ln in wrap(s, size, W - 300, kind):
                text(d, (W / 2, y + size * 0.7), ln, size, kind, col, 'mm')
                y += size * 1.45
        caption(d, sub)
        return fade(img, t, dur)
    return dur, draw


def slide(fig, title, caps, dur, sub=''):
    """Paper figure with a slow zoom; caps = [(t_start, caption)]."""
    dur = max(dur, MIN_DUR[0])
    src = pdf_png(fig)
    box_w, box_h = W - 240, H - 290
    s = min(box_w / src.width, box_h / src.height)

    def draw(t):
        img = Image.new('RGB', (W, H), BG)
        d = ImageDraw.Draw(img)
        header(d, title, sub)
        z = s
        fw, fh = int(src.width * z), int(src.height * z)
        img.paste(src.resize((fw, fh), Image.LANCZOS), ((W - fw) // 2, 110 + (box_h - fh) // 2))
        cur = [c for t0, c in caps if t >= t0]
        caption(d, cur[-1] if cur else '')
        return fade(img, t, dur)
    return dur, draw


def episode_trace(rdir, env):
    p = np.load(rdir / 'paths.npz')
    row = next(r for r in csv.DictReader(open(rdir / 'episodes.csv', encoding='utf-8')) if int(r['env_id']) == env)
    n = int(row['episode_steps'])
    xy, rxy = p['root_xyz'][:n, env, :2], p['ref_root_xyz'][:n, env, :2]
    yaw, ryaw = p['root_yaw'][:n, env], p['ref_root_yaw'][:n, env]
    e = (yaw - ryaw + np.pi) % (2 * np.pi) - np.pi
    dx, dy = (xy - rxy).T
    lat = -dx * np.sin(ryaw) + dy * np.cos(ryaw)
    causes = [TERM_TEXT[k] for k in TERM_TEXT if row.get(k) == '1']
    return dict(n=n, xy=xy, rxy=rxy, e=np.degrees(e), lat=lat, ok=row['success'] == '1', causes=causes)


def compass(d, c, r, e_deg, col):
    d.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], outline=(190, 190, 190), width=3, fill=(255, 255, 255))
    for k in range(0, int(r * 0.85), 14):  # dashed reference heading
        d.line([(c[0], c[1] - k), (c[0], c[1] - min(k + 8, r * 0.85))], fill=REF, width=4)
    a = math.radians(e_deg)
    tip = (c[0] + r * 0.85 * math.sin(a), c[1] - r * 0.85 * math.cos(a))
    d.line([c, tip], fill=col, width=7)
    d.ellipse([tip[0] - 8, tip[1] - 8, tip[0] + 8, tip[1] + 8], fill=col)
    d.ellipse([c[0] - 6, c[1] - 6, c[0] + 6, c[1] + 6], fill=INK)


def paired(task, env, cam, title, caps, speed=1.0, arms=('dyn', 'dyn-hboth'), hold=1.0, note=''):
    """Two students on the same episode: videos side by side, heading compass and lateral error under each,
    a growing top view in the middle."""
    dirs = {a: render_dir(task, a, env, cam) for a in arms}
    tr = {a: episode_trace(dirs[a], env) for a in arms}
    vw, vh = 940, 536
    clips = {a: Clip(dirs[a] / 'video.mp4', (vw, vh)) for a in arms}
    T = max(max(tr[a]['n'] for a in arms) * 0.02 / speed + hold, MIN_DUR[0])
    # top view: reference heading at the start points up
    r0 = tr[arms[0]]['rxy']
    h0 = math.atan2(*(r0[min(25, len(r0) - 1)] - r0[0])[::-1])
    rot = np.array([[math.cos(math.pi / 2 - h0), -math.sin(math.pi / 2 - h0)],
                    [math.sin(math.pi / 2 - h0), math.cos(math.pi / 2 - h0)]])
    pts = {k: (v - r0[0]) @ rot.T for k, v in [('ref', r0)] + [(a, tr[a]['xy']) for a in arms]}
    allp = np.concatenate(list(pts.values()))
    lo, hi = allp.min(0) - 0.3, allp.max(0) + 0.3
    pw, ph, px0, py0 = 560, 290, (W - 560) // 2, 700
    VY = 140  # top of the videos
    sy = (ph - 40) / (hi[1] - lo[1])
    sx = min((pw - 40) / (hi[0] - lo[0]), 4 * sy)  # lateral axis exaggerated up to 4x
    stretch = sx / sy
    cx0 = px0 + pw / 2 - sx * (lo[0] + hi[0]) / 2
    cy0 = py0 + 20 + (ph - 40) / 2 + sy * (lo[1] + hi[1]) / 2

    def P(q):
        return [(cx0 + sx * x, cy0 - sy * y) for x, y in q]

    def draw(t):
        img = Image.new('RGB', (W, H), BG)
        d = ImageDraw.Draw(img)
        header(d, title, TASK_NAME[task] + (f'   ({speed:g}x)' if speed != 1 else ''))
        k = int(t * speed / 0.02)
        for j, a in enumerate(arms):
            col, lab = ARM[a]
            x0 = 13 + j * (vw + 14)
            img.paste(clips[a].at(k), (x0, VY))
            text(d, (x0, VY - 8), lab, 36, 'b', col, 'ls')
            kk = min(k, tr[a]['n'] - 1)
            text(d, (x0 + vw - 10, VY + 12), f't = {kk * 0.02:4.1f} s', 30, 'r', (255, 255, 255), 'ra')
            if k >= tr[a]['n'] and not tr[a]['ok']:  # terminated: dim the frozen frame
                ov = Image.new('RGB', (vw, vh), (30, 30, 30))
                img.paste(Image.blend(img.crop((x0, VY, x0 + vw, VY + vh)), ov, 0.55), (x0, VY))
                text(d, (x0 + vw / 2, VY + vh / 2 - 22), 'terminated', 54, 'b', (255, 255, 255), 'mm')
                text(d, (x0 + vw / 2, VY + vh / 2 + 30), tr[a]['causes'][0] if tr[a]['causes'] else '', 30, 'i',
                     (255, 255, 255), 'mm')
            elif k >= tr[a]['n']:
                text(d, (x0 + vw / 2, VY + vh - 40), 'episode completed', 40, 'b', (255, 255, 255), 'mm')
            # heading compass and lateral error
            bx = 60 if j == 0 else W - 60 - 560
            compass(d, (bx + 110, 845), 88, tr[a]['e'][kk], col)
            text(d, (bx + 230, 770), 'heading vs. reference', 30, 'r', MUTE)
            text(d, (bx + 230, 808), f'{tr[a]["e"][kk]:+5.1f} deg', 44, 'b', col)
            text(d, (bx + 230, 870), 'lateral error', 30, 'r', MUTE)
            text(d, (bx + 230, 908), f'{100 * tr[a]["lat"][kk]:+5.1f} cm', 44, 'b', col)
        # top view
        d.rectangle([px0, py0, px0 + pw, py0 + ph], outline=(215, 215, 215), width=2, fill=(255, 255, 255))
        text(d, (px0 + 10, py0 + 8), 'top view (dashed: reference)' + (f', lateral x{stretch:.1f}' if stretch > 1.2 else ''), 24, 'r', MUTE)
        ref = P(pts['ref'][:max(2, min(k, len(pts['ref'])) + 1)])
        for i in range(0, len(ref) - 1, 6):
            d.line(ref[i:i + 4], fill=REF, width=3)
        for a in arms:
            q = P(pts[a][:max(2, min(k, tr[a]['n']) + 1)])
            d.line(q, fill=ARM[a][0], width=5)
            if k >= tr[a]['n'] and not tr[a]['ok']:
                x, y = q[-1]
                d.line([(x - 12, y - 12), (x + 12, y + 12)], fill=ARM[a][0], width=6)
                d.line([(x - 12, y + 12), (x + 12, y - 12)], fill=ARM[a][0], width=6)
        cur = [c for t0, c in caps if t >= t0]
        caption(d, (cur[-1] if cur else '') + (f'  {note}' if note else ''))
        return fade(img, t, T)
    return T, draw


def bars(title, groups, dur, caps, sub=''):
    """Animated grouped bars: groups = [(label, {arm: value})]."""
    dur = max(dur, MIN_DUR[0])
    def draw(t):
        img = Image.new('RGB', (W, H), BG)
        d = ImageDraw.Draw(img)
        header(d, title, sub)
        g = min(1.0, t / 1.2)
        x0, y0, y1, lo, hi = 320, 930, 180, 60.0, 100.0
        d.line([(x0 - 20, y0), (W - 220, y0)], fill=INK, width=3)
        for v in range(60, 101, 10):
            y = y0 - (v - lo) / (hi - lo) * (y0 - y1)
            d.line([(x0 - 20, y), (W - 220, y)], fill=(225, 225, 225), width=1)
            text(d, (x0 - 35, y), f'{v}', 32, 'r', MUTE, 'rm')
        text(d, (150, (y0 + y1) / 2), 'success (%)', 34, 'r', MUTE, 'mm')
        gw = (W - 220 - x0) / len(groups)
        for i, (lab, vals) in enumerate(groups):
            for j, (a, v) in enumerate(vals.items()):
                bw = 150
                xa = x0 + i * gw + gw / 2 + (j - len(vals) / 2) * (bw + 16) + 8
                vv = lo + (v - lo) * g
                yt = y0 - (vv - lo) / (hi - lo) * (y0 - y1)
                d.rectangle([xa, yt, xa + bw, y0], fill=ARM[a][0])
                if g >= 1:
                    text(d, (xa + bw / 2, yt - 12), f'{v:.1f}', 36, 'b', INK, 'ms')
            text(d, (x0 + i * gw + gw / 2, y0 + 45), lab, 38, 'r', INK, 'mm')
        for j, a in enumerate(groups[0][1]):
            d.rectangle([W - 560 + j * 260, 125, W - 530 + j * 260, 155], fill=ARM[a][0])
            text(d, (W - 520 + j * 260, 140), ARM[a][1], 32, 'r', INK, 'lm')
        cur = [c for t0, c in caps if t >= t0]
        caption(d, cur[-1] if cur else '')
        return fade(img, t, dur)
    return dur, draw


def movie(path, title, dur_max, caps):
    """An external clip (PowerPoint animation, MuJoCo) fitted below the header."""
    r = imageio.get_reader(str(path))
    fps = r.get_meta_data().get('fps', 30)
    n = r.count_frames()
    r = imageio.get_reader(str(path))
    dur = min(dur_max, n / fps)
    state = {'i': -1, 'last': None}

    def draw(t):
        k = min(int(t * fps), n - 1)
        while state['i'] < k:
            state['last'] = r.get_next_data()
            state['i'] += 1
        img = Image.new('RGB', (W, H), BG)
        d = ImageDraw.Draw(img)
        header(d, title)
        src = Image.fromarray(state['last'])
        s = min((W - 120) / src.width, (H - 230) / src.height)
        src = src.resize((int(src.width * s), int(src.height * s)), Image.LANCZOS)
        img.paste(src, ((W - src.width) // 2, 105))
        cur = [c for t0, c in caps if t >= t0]
        caption(d, cur[-1] if cur else '')
        return fade(img, t, dur)
    return dur, draw


def side_by_side(paths, title, caps, hold=1.5, sub='', ended=None):
    """Two recordings without traces (MuJoCo), each labelled by arm; the shorter one is dimmed when it ends."""
    vw, vh = 940, 536
    clips = {a: Clip(p, (vw, vh)) for a, p in paths.items()}
    T = max(max(c.n for c in clips.values()) / 50 + hold, MIN_DUR[0])

    def draw(t):
        img = Image.new('RGB', (W, H), BG)
        d = ImageDraw.Draw(img)
        header(d, title, sub)
        k = int(t * 50)
        for j, (a, c) in enumerate(clips.items()):
            col, lab = ARM[a]
            x0 = 13 + j * (vw + 14)
            img.paste(c.at(k), (x0, 200))
            text(d, (x0, 192), lab, 36, 'b', col, 'ls')
            if k >= c.n:
                ov = Image.new('RGB', (vw, vh), (30, 30, 30))
                img.paste(Image.blend(img.crop((x0, 200, x0 + vw, 200 + vh)), ov, 0.55), (x0, 200))
                text(d, (x0 + vw / 2, 200 + vh / 2), (ended or {}).get(a, 'episode ended'), 50, 'b', (255, 255, 255), 'mm')
        cur = [c_ for t0, c_ in caps if t >= t0]
        for i, ln in enumerate(wrap(cur[-1] if cur and not subs else '', 34, W - 200, 'i')):
            text(d, (W // 2, 820 + 46 * i), ln, 34, 'i', MUTE, 'mm')
        return fade(img, t, T)
    subs = SUBS_ON[0]
    return T, draw


MUJOCO = {a: VID / 'mujoco' / f'skateboard-{a}-st2-ep0.mp4' for a in ('dyn', 'dyn-hboth')}


# ------------------------------------------------------------------ storyboards
# Each segment is (narration, factory); the factory is called after the narration is synthesized, with MIN_DUR set
# to the narration length, so that every segment lasts at least as long as its voice-over.
TITLE = [('Where Do Proprioceptive Humanoids Get Lost?', 66, 'b', INK),
         ('Observability-Aware Distillation for Underactuated Object Interaction', 44, 'r', INK),
         (' ', 30, 'r', INK), ('Supplementary video', 36, 'i', MUTE)]
FRAMES = VID / 'mainfigure_video'
INFO = {}  # authors, affiliations, venue, url, acknowledgment: paper/video_info.json (empty for double-blind review)


def title_lines(short=False):
    """Title card: paper title, then the authors and affiliations when INFO has them."""
    lines = TITLE[:2]
    if INFO.get('authors'):
        lines = lines + [(' ', 24, 'r', INK), (', '.join(INFO['authors']), 38, 'r', INK)]
        lines += [(a, 30, 'i', MUTE) for a in INFO.get('affiliations', [])]
    elif not short:
        lines = TITLE
    return lines


def credits():
    """Closing card (no narration): title, authors, affiliations, venue, link, acknowledgment."""
    lines = [('Where Do Proprioceptive Humanoids Get Lost?', 54, 'b', INK),
             ('Observability-Aware Distillation for Underactuated Object Interaction', 38, 'r', INK), (' ', 30, 'r', INK)]
    if INFO.get('authors'):
        lines.append((', '.join(INFO['authors']), 38, 'r', INK))
        lines += [(a, 30, 'i', MUTE) for a in INFO.get('affiliations', [])]
    else:
        lines.append(('Anonymous submission (double-blind review)', 34, 'i', MUTE))
    for k, size in (('venue', 32), ('url', 32), ('acknowledgment', 26)):
        if INFO.get(k):
            lines += [(' ', 16, 'r', INK), (INFO[k], size, 'r' if k != 'acknowledgment' else 'i', MUTE)]
    return card(lines, 5)
METHOD_CAPS = ['Simulation gives the teacher the privileged state and the student its onboard observations.',
               'The teacher (green) acts from the privileged state; it is used in training only.',
               'The student (blue) has a dynamics-aware world model and is distilled by matching latent and action.',
               'Ours (orange): the IMU-integrated heading error enters the world model and the actor, zero-initialized.']


def pct(k):
    return f'{float(N[k]):.0f}'


def method_frames(dur, stages=(0.0, 0.17, 0.38, 0.66), sentences=(0, 1, 2, 4)):
    """The four build-up stages of Fig. 2 (mainfigure_video/1..4.png), cross-faded when the narration reaches the
    given sentences (simulation, teacher, student, heading), else at fractions of the segment."""
    dur = max(dur, MIN_DUR[0])
    ims = [Image.open(FRAMES / f'{i}.png').convert('RGB') for i in (1, 2, 3, 4)]
    s = min((W - 80) / ims[0].width, (H - 250) / ims[0].height)
    ims = [im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS) for im in ims]
    st = SENT_T[0]
    t0 = [0.0] + [st[k] for k in sentences[1:]] if len(st) > max(sentences) else [f * dur for f in stages]

    def draw(t):
        img = Image.new('RGB', (W, H), BG)
        d = ImageDraw.Draw(img)
        header(d, 'Method: observability-complemented student')
        i = max(j for j in range(4) if t >= t0[j])
        a = min(1.0, (t - t0[i]) / 0.6) if i else 1.0
        fr = Image.blend(ims[i - 1], ims[i], a) if i and a < 1 else ims[i]
        img.paste(fr, ((W - fr.width) // 2, 110))
        caption(d, METHOD_CAPS[i])
        return fade(img, t, dur)
    return dur, draw


def method_segment(dur):
    if all((FRAMES / f'{i}.png').is_file() for i in (1, 2, 3, 4)):
        return method_frames(dur)
    caps = [(0, METHOD_CAPS[2]), (dur * 0.5, METHOD_CAPS[3])]
    return slide('fig_overview', 'Method: observability-complemented student', caps, dur)


BARS = [('Skateboard', {'dyn': float(N['DynPred']), 'dyn-hboth': float(N['DynHbothPred'])}),
        ('Push cart', {'dyn': float(N['pcDynPred']), 'dyn-hboth': float(N['pcDynHbothPred'])}),
        ('Box, stair & slope', {'dyn': float(N['bxDynPred']), 'dyn-hboth': float(N['bxDynHbothPred'])})]
# the speech synthesizer spells "HAIC" letter by letter only when written with dots
HAIC = 'H.A.I.C.'  # also read correctly by the neural voice


def three_min():
    s = [
        ('Where do proprioceptive humanoids get lost?', lambda: card(title_lines(), 5)),
        ('When a privileged teacher is distilled into a proprioceptive student, the student rarely falls. It drifts.',
         lambda: card([('A proprioceptive humanoid student rarely falls.', 60, 'b', INK),
                       ('It drifts.', 60, 'b', ARM['dyn'][0])], 5)),
        (f'The same skateboarding episode, run by the {HAIC} student on the left and by our student on the right. '
         f'The red spheres are the reference. {HAIC} slowly drifts sideways, and the episode ends when its torso '
         'leaves the reference in the world frame.',
         lambda: paired('skateboard', 4, 'chase', 'The problem: the student drifts',
                        [(0, 'Same episode, same random seed. The camera follows the reference; red spheres are reference keypoints.'),
                         (6, 'HAIC drifts sideways off the reference until the world-frame tracking check ends the episode.')])),
        (f'If this world-frame check is ignored, the {HAIC} student succeeds in {pct("DynPredDt")} percent of the episodes '
         f'instead of {pct("DynPred")}. With the teacher\'s privileged feature it reaches {pct("DynOraclePriv")} percent.',
         lambda: slide('fig_gap', 'Diagnosis: where the gap lives',
                       [(0, f'Ignoring world-frame drift, HAIC succeeds in {N["DynPredDt"]}% instead of {N["DynPred"]}% of the episodes.'),
                        (4.5, f'The teacher\'s privileged feature alone recovers {N["DynOraclePriv"]}%: the missing information is in the reference frame.')], 9)),
        (f'A counterfactual ablation of the teacher shows which input matters. Without its root orientation relative '
         f'to the reference, the teacher drops from {pct("ablTeacherBase")} to {pct("ablTeacherOri")} percent.',
         lambda: slide('fig_ablation', 'Diagnosis: which privileged input matters',
                       [(0, f'Replacing the teacher\'s root orientation w.r.t. the reference drops it from {N["ablTeacherBase"]}% to {N["ablTeacherOri"]}%.'),
                        (4.5, 'Object pose relative to the robot matters little: a better object estimate is not the remedy.')], 9)),
        ('Before a termination, the heading and lateral errors grow for seconds, while the local body pose stays far '
         'from its bound.',
         lambda: slide('fig_failure_aligned', 'Diagnosis: failures are slow drift',
                       [(0, 'Before a termination the heading and lateral errors grow for seconds, the local body error does not.')], 7)),
        (f'Our student keeps the {HAIC} design. The teacher sees the privileged state and is used in training only. '
         'The student, with a dynamics-aware world model, is distilled by matching its latent feature and action. '
         'What it cannot observe is its heading relative to the reference. An onboard I.M.U. integrates it, and the '
         'heading error enters the world model and the actor through zero-initialized weights.',
         lambda: method_segment(24)),
        ('The heading sensor model includes an alignment error, a drifting gyro bias and rate noise, about one point '
         'seven degrees at the end of an episode.',
         lambda: card([('Heading sensor model', 50, 'b', INK),
                       ('integrated IMU yaw rate since the reference anchor', 38, 'r', INK),
                       ('alignment error + gyro bias random walk + white rate noise', 38, 'r', INK),
                       ('about 1.7 deg (1 sigma) at the end of a 12.4 s episode', 38, 'i', MUTE)], 6)),
        (f'On skateboarding, success rises from {pct("DynPred")} to {pct("DynHbothPred")} percent.',
         lambda: paired('skateboard', 196, 'follow_ref', 'Result: skateboarding',
                        [(0, f'Skateboarding: HAIC {N["DynPred"]}%, OH-Both {N["DynHbothPred"]}% success (three students each).')], speed=1.25)),
        (f'Pushing a cart, from {pct("pcDynPred")} to {pct("pcDynHbothPred")} percent.',
         lambda: paired('push_cart', 199, 'chase', 'Result: pushing a cart',
                        [(0, f'Pushing a cart: HAIC {N["pcDynPred"]}%, OH-Both {N["pcDynHbothPred"]}%.')])),
        (f'And carrying a box over a stair and a slope, from {pct("bxDynPred")} to {pct("bxDynHbothPred")} percent.',
         lambda: paired('box_stairslope', 187, 'chase', 'Result: box over a stair and a slope',
                        [(0, f'Carrying a box over a stair and a slope: HAIC {N["bxDynPred"]}%, OH-Both {N["bxDynHbothPred"]}%.')], speed=1.25)),
        ('The gain replicates on all three tasks, while a control with the same network but a constant heading input '
         'does not improve.',
         lambda: bars('Success over three students and 3 x 1024 episodes per task', BARS, 7,
                      [(0, 'A capacity-matched control with a constant heading input does not improve on HAIC on any task.')])),
        (f'The benefit holds while the integrated heading error stays below a task-dependent level, about '
         f'{N["skRobBreakEven"]} degrees on the skateboard.',
         lambda: slide('fig_robustness_tasks', 'Robustness to heading-sensor error',
                       [(0, f'OH-Both stays above HAIC up to about {N["pcRobBreakEven"]} deg (cart), {N["skRobBreakEven"]} deg (skateboard) '
                            f'and {N["bxRobBreakEven"]} deg (box) of end-of-episode heading error.')], 7)),
        (f'Not every episode is saved. Here our student fails where {HAIC} succeeds: the heading removes lateral drift, '
         'but not drift along the track.',
         lambda: paired('push_cart', 7, 'chase', 'Failure case',
                        [(0, 'Not every episode is saved: here OH-Both fails where HAIC succeeds. The heading removes lateral drift, not along-track drift.')], speed=1.25)),
    ]
    if all(p.is_file() for p in MUJOCO.values()):
        s.append(('In a second simulator, MuJoCo, each push gives the board less speed. '
                  f'{HAIC} adapts its pushing, while our student keeps its stance and often falls, as here. '
                  'Where the board can drift, the heading still pays off. We report the falls as a limitation.',
                  lambda: side_by_side(MUJOCO, 'Second simulator: MuJoCo', [
                      (0, 'The same students in MuJoCo (no retraining). The push accelerates the board less than in Isaac Sim.'),
                      (5, 'Here HAIC completes the episode and OH-Both falls during fast pushing: the heading students push '
                          'less robustly under the different contact dynamics.')],
                      sub='Skateboarding, student 2, episode 0', ended={'dyn-hboth': 'fell'})))
    s += [
        ('All results are in simulation, and the sensor model still has to be validated on the robot.',
         lambda: card([('Limitations', 50, 'b', INK),
                       ('All results are in simulation; the sensor model must be validated on the robot.', 38, 'r', INK),
                       ('Along-track drift remains unobservable to a proprioceptive student.', 38, 'r', INK),
                       ('In MuJoCo, the heading students fall more often than HAIC during fast pushing.', 38, 'r', INK)], 6)),
        ('One onboard signal, the I.M.U.-integrated heading, recovers most of the teacher-student gap on three tasks.',
         lambda: card([('One onboard signal, the IMU-integrated heading,', 50, 'b', INK),
                       ('recovers most of the teacher-student gap on three tasks.', 50, 'b', INK),
                       (' ', 30, 'r', INK), ('Where Do Proprioceptive Humanoids Get Lost?', 36, 'i', MUTE)], 6)),
        ('', credits),
    ]
    return s


def one_min():
    return [
        ('Where do proprioceptive humanoids get lost?', lambda: card(title_lines(short=True), 4)),
        (f'A distilled proprioceptive humanoid rarely falls. It drifts away from the reference, as {HAIC} does on the left.',
         lambda: paired('skateboard', 4, 'chase', 'A proprioceptive student does not fall, it drifts',
                        [(0, 'HAIC (left) drifts off the reference and is terminated; OH-Both (right) completes the episode.')],
                        speed=1.5, hold=1.0)),
        ('We give the student one onboard signal: its I.M.U.-integrated heading relative to the reference, fed to the '
         'world model and the actor.', lambda: method_segment(12)),
        (f'Pushing a cart, success rises from {pct("pcDynPred")} to {pct("pcDynHbothPred")} percent,',
         lambda: paired('push_cart', 199, 'chase', 'Pushing a cart',
                        [(0, f'HAIC {N["pcDynPred"]}%, OH-Both {N["pcDynHbothPred"]}% success.')], speed=1.5, hold=0.8)),
        (f'and carrying a box over a stair and a slope, from {pct("bxDynPred")} to {pct("bxDynHbothPred")} percent.',
         lambda: paired('box_stairslope', 187, 'chase', 'Box over a stair and a slope',
                        [(0, f'HAIC {N["bxDynPred"]}%, OH-Both {N["bxDynHbothPred"]}% success.')], speed=1.5, hold=0.8)),
        ('On all three tasks, the heading recovers most of the teacher-student gap.',
         lambda: bars('Success over three students per task', BARS, 6,
                      [(0, 'One onboard signal, the IMU-integrated heading, recovers most of the gap.')])),
        ('', credits),
    ]


# ------------------------------------------------------------------ voice-over (Windows SAPI, offline)
SR, LEAD = 22050, 0.4  # sample rate; silence before each narration [s]


def synth(txt, wav, voice, rate):
    """Narration to a 22.05 kHz mono WAV with the Windows speech synthesizer (cached by text, voice and rate)."""
    import subprocess
    key, t = wav.with_suffix('.key'), wav.with_suffix('.txt')
    if wav.is_file() and key.is_file() and key.read_text(encoding='utf-8') == f'{voice}|{rate}|{txt}':
        return
    t.write_text(txt, encoding='utf-8')
    ps = ("Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
          f"$s.SelectVoice('{voice}'); $s.Rate = {rate}; "
          f"$f = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo({SR}, "
          "[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono); "
          f"$s.SetOutputToWaveFile('{wav}', $f); $s.Speak([IO.File]::ReadAllText('{t}')); $s.Dispose()")
    subprocess.run(['powershell', '-NoProfile', '-Command', ps], check=True)
    key.write_text(f'{voice}|{rate}|{txt}', encoding='utf-8')


def synth_edge(txt, wav, voice, rate):
    """Narration with a Microsoft neural voice (edge-tts, online: the text is sent to Microsoft's speech service),
    converted to 22.05 kHz mono WAV; cached by text, voice and rate."""
    import asyncio
    import subprocess
    import edge_tts
    import imageio_ffmpeg
    key, meta = wav.with_suffix('.key'), wav.with_suffix('.words.jsonl')
    if wav.is_file() and meta.is_file() and key.is_file() and key.read_text(encoding='utf-8') == f'{voice}|{rate}|{txt}':
        return
    import time
    mp3 = wav.with_suffix('.mp3')
    for attempt in range(4):  # the service occasionally resets the connection
        try:
            asyncio.run(edge_tts.Communicate(txt, voice, rate=f'{rate:+d}%', boundary='WordBoundary')
                        .save(str(mp3), metadata_fname=str(meta)))
            break
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                raise
            print(f'  edge-tts retry after {type(e).__name__}', flush=True)
            time.sleep(3 * (attempt + 1))
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error', '-i', str(mp3), '-ar', str(SR), '-ac', '1',
                    '-sample_fmt', 's16', str(wav)], check=True)
    key.write_text(f'{voice}|{rate}|{txt}', encoding='utf-8')


def _norm(s):
    return re.sub(r'[^0-9a-z]', '', s.lower())


def subtitles(narr, wav, dur_s):
    """Timed subtitle lines [(start, end, text)] (s from the start of the narration audio) and sentence start times.
    Word times come from the neural voice's word boundaries (<wav>.words.jsonl); without them (SAPI or a recording)
    the narration is spread over the audio in proportion to its characters."""
    import json
    toks = narr.split()
    t_on = [None] * len(toks)
    meta = wav.with_suffix('.words.jsonl')
    if meta.is_file():
        j = 0
        for line in meta.read_text(encoding='utf-8').splitlines():
            b = json.loads(line)
            if b.get('type') != 'WordBoundary':
                continue
            w_ = _norm(b['text'])
            for k in range(j, min(j + 4, len(toks))):
                tk = _norm(toks[k])
                if w_ and tk and (tk.startswith(w_) or w_.startswith(tk)):
                    if t_on[k] is None:
                        t_on[k] = b['offset'] / 1e7
                    j = k + 1
                    break
    if t_on[0] is None:
        t_on[0] = 0.0
    if any(t is None for t in t_on):  # spread the unmatched words by character count
        n_chars = np.cumsum([0] + [len(t) + 1 for t in toks])
        known = [(k, t) for k, t in enumerate(t_on) if t is not None] + [(len(toks), dur_s)]
        for (k0, ta), (k1, tb) in zip(known, known[1:]):
            for k in range(k0 + 1, k1):
                t_on[k] = ta + (tb - ta) * (n_chars[k] - n_chars[k0]) / max(1, n_chars[k1] - n_chars[k0])
    shown = [t.replace('H.A.I.C.', 'HAIC').replace('I.M.U.', 'IMU') for t in toks]
    ends = [tk[-1] in '.?!' and tk.count('.') <= 1 for tk in toks]  # sentence ends; "H.A.I.C." is not one
    f = font(34, 'i')

    def width(ix):
        return f.getlength(' '.join(shown[i] for i in ix))

    def split(ix):
        """One line if it fits, else split where the halves are most even (a comma is worth some imbalance)."""
        if width(ix) <= W - 260 or len(ix) < 2:
            return [ix]
        best = min(range(1, len(ix)), key=lambda c: abs(width(ix[:c]) - width(ix[c:]))
                   - (150 if shown[ix[c - 1]].endswith(',') else 0))
        return split(ix[:best]) + split(ix[best:])

    lines, cur = [], []
    for k in range(len(toks)):
        cur.append(k)
        if ends[k] or k == len(toks) - 1:
            lines += split(cur)
            cur = []
    subs = []
    for i, ln in enumerate(lines):
        end = t_on[lines[i + 1][0]] if i + 1 < len(lines) else dur_s + 0.5
        subs.append((t_on[ln[0]], end, ' '.join(shown[k] for k in ln)))
    sent = [t_on[0]] + [t_on[k + 1] for k in range(len(toks) - 1) if ends[k]]
    return subs, sent


def with_subtitles(draw, subs):
    def drawn(t):
        img = draw(t)
        cur = [s for s in subs if s[0] <= t < s[1]]
        if cur:
            text(ImageDraw.Draw(img), (W // 2, H - 42), cur[-1][2], 34, 'i', MUTE, 'mm')
        return img
    return drawn


def read_wav(wav):
    import wave
    with wave.open(str(wav)) as w:
        assert w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2, wav
        return np.frombuffer(w.readframes(w.getnframes()), np.int16)


def main():
    import subprocess
    import wave
    import imageio_ffmpeg
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cut', choices=['3min', '1min', 'both'], default='both')
    p.add_argument('--preview', type=float, default=0, help='also save one PNG every N seconds')
    p.add_argument('--tts', choices=['edge', 'sapi'], default='edge',
                   help='edge: Microsoft neural voice (online); sapi: Windows speech synthesizer (offline)')
    p.add_argument('--voice', default=None, help="voice name; 'none' for a silent video "
                   "(default en-US-AndrewNeural for edge, 'Microsoft Zira Desktop' for sapi)")
    p.add_argument('--rate', type=int, default=None, help='speaking rate: edge in percent (default +12), sapi -10..10 (default 2)')
    p.add_argument('--voice-dir', type=Path, default=None,
                   help='use <dir>/<cut>_<k>.wav (22.05 kHz mono 16 bit, e.g. your own recording) instead of SAPI')
    p.add_argument('--info', type=Path, default=PAPER / 'video_info.json',
                   help='authors, affiliations, venue, url and acknowledgment for the title card and the credits')
    a = p.parse_args()
    if a.info.is_file():
        import json
        INFO.update({k: v for k, v in json.loads(a.info.read_text(encoding='utf-8')).items() if not k.startswith('_')})
    if a.voice is None:
        a.voice = 'en-US-AndrewNeural' if a.tts == 'edge' else 'Microsoft Zira Desktop'
    if a.rate is None:
        a.rate = 12 if a.tts == 'edge' else 2
    OUT.mkdir(parents=True, exist_ok=True)
    for cut in (['3min', '1min'] if a.cut == 'both' else [a.cut]):
        story = three_min() if cut == '3min' else one_min()
        vdir = OUT / 'voice'
        vdir.mkdir(exist_ok=True)
        segs, audio, script = [], [], []
        for k, (narr, make) in enumerate(story):
            clip = np.zeros(0, np.int16)
            if narr and a.voice != 'none':
                wav = (a.voice_dir if a.voice_dir else vdir) / f'{cut}_{k}.wav'
                if not a.voice_dir:
                    (synth_edge if a.tts == 'edge' else synth)(narr, wav, a.voice, a.rate)
                clip = read_wav(wav)
            MIN_DUR[0] = LEAD + len(clip) / SR + 0.45 if len(clip) else 0.0
            subs, sent = subtitles(narr, wav, len(clip) / SR) if len(clip) else ([], [])
            SUBS_ON[0], SENT_T[0] = bool(len(clip)), [LEAD + t for t in sent]
            dur, draw = make()
            if subs:
                draw = with_subtitles(draw, [(LEAD + t0, LEAD + t1, s) for t0, t1, s in subs])
            n_frames = int(round(dur * FPS))
            track = np.zeros(int(round(n_frames / FPS * SR)), np.int16)
            i0 = int(LEAD * SR)
            track[i0:i0 + len(clip)] = clip[:max(0, len(track) - i0)]
            segs.append((n_frames, draw, SUBS_ON[0]))
            audio.append(track)
            script.append(f'{cut}_{k}  [{sum(n for n, _, _ in segs[:-1]) / FPS:6.1f} s, {n_frames / FPS:4.1f} s]  {narr}')
        (OUT / f'narration_{cut}.txt').write_text('\n'.join(script) + '\n', encoding='utf-8')
        total = sum(n for n, _, _ in segs) / FPS
        silent = OUT / f'haic_{cut}_silent.mp4'
        w = imageio.get_writer(str(silent), fps=FPS, codec='libx264', quality=8, macro_block_size=8)
        t_glob, next_prev = 0.0, 0.0
        for si, (n_frames, draw, subs_on) in enumerate(segs):
            SUBS_ON[0] = subs_on  # captions are drawn at render time: restore this segment's setting
            for f in range(n_frames):
                img = draw(f / FPS)
                w.append_data(np.asarray(img))
                if a.preview and t_glob >= next_prev:
                    img.save(OUT / f'preview_{cut}_{int(t_glob):03d}s.png')
                    next_prev += a.preview
                t_glob += 1 / FPS
            print(f'{cut}: segment {si + 1}/{len(segs)} done ({t_glob:.0f}/{total:.0f} s)', flush=True)
        w.close()
        path = OUT / f'haic_{cut}.mp4'
        if a.voice == 'none':
            silent.replace(path)
        else:
            aw = OUT / f'haic_{cut}_voice.wav'
            with wave.open(str(aw), 'wb') as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(SR)
                f.writeframes(np.concatenate(audio).tobytes())
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error', '-i', str(silent), '-i', str(aw),
                            '-c:v', 'copy', '-c:a', 'aac', '-b:a', '160k', '-shortest', str(path)], check=True)
            silent.unlink()
        print(f'wrote {path} ({total:.0f} s)')


if __name__ == '__main__':
    main()
