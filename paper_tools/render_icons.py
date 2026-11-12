"""Icons of the method overview (Fig. 2), layout in figures/drafts/fig_overview_layout.png.

    D:/HAIC-Claude/_testenv/python.exe scripts/render_icons.py
    D:/HAIC-Claude/_testenv/python.exe scripts/render_icons.py --ghost-t 0 2 4 6 --action-t 5.0 --azimuth 140

Scene icons (transparent PNG): poses of the golden Isaac rollout of the HAIC skateboard student
(golden.npz, written by HAIC/research/export_onnx.py) shown in the HOIC-baseline MuJoCo scene. Kinematics
only (mj_forward), no simulation. The reference ghost uses rollout poses, which track the reference closely.
Vector icons (PDF + PNG): latent bars, MLP, IMU chip, integrator and heading dial, in the colors of the layout.
Output: figures/icons/.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from matplotlib.patches import Arc, Circle, FancyBboxPatch, Rectangle
from PIL import Image

PAPER = Path(__file__).resolve().parents[1]
OUT = PAPER / 'figures' / 'icons'
BLUE, ORANGE, GRAY = (0.24, 0.44, 0.71), (0.88, 0.48, 0.12), (0.54, 0.56, 0.6)
DECK = (0.6, 0.47, 0.34, 1.0)
plt.rcParams.update({'font.family': 'Times New Roman', 'mathtext.fontset': 'stix'})


# ---------------------------------------------------------------- scene
def load(a):
    model = mujoco.MjModel.from_xml_path(str(a.hoic / 'assets/robots/g1/g1_29dof_rev_1_0_skateboard.xml'))
    model.vis.global_.offwidth = model.vis.global_.offheight = 2 * a.size
    model.vis.headlight.ambient[:] = 0.45
    model.vis.headlight.diffuse[:] = 0.55
    model.geom_rgba[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'board_deck')] = DECK
    return model, mujoco.MjData(model), np.load(a.golden)


def board_geoms(model):
    root = model.jnt_bodyid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'floating_board_base_joint')]
    bodies = [b for b in range(model.nbody) if b == root or model.body_rootid[b] == root]
    return np.isin(model.geom_bodyid, bodies)


def set_pose(model, data, g, t):
    """Robot and board at time t [s] of the golden rollout (50 Hz)."""
    k = min(int(round(t / 0.02)), len(g['joint_pos']) - 1)
    names = [str(n) for n in g['joint_names']]
    data.qpos[:3], data.qpos[3:7] = g['root_pos_w'][k], g['root_quat_w'][k]
    for j in range(model.njnt):
        n = model.joint(j).name
        if n in names:
            data.qpos[model.jnt_qposadr[j]] = g['joint_pos'][k, names.index(n)]
    qa = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'floating_board_base_joint')]
    data.qpos[qa:qa + 3], data.qpos[qa + 3:qa + 7] = g['obj_pos_w'][k], g['obj_quat_w'][k]
    mujoco.mj_forward(model, data)


def template_points(model, data, n=12):
    """Points on the surface of the board geoms (deck: top grid and rim; wheels: rims), world frame."""
    pts = []
    for gi in np.flatnonzero(board_geoms(model)):
        R, p, s = data.geom_xmat[gi].reshape(3, 3), data.geom_xpos[gi], model.geom_size[gi]
        if model.geom_type[gi] == mujoco.mjtGeom.mjGEOM_BOX and s[0] > 0.2:  # deck
            u, v = np.meshgrid(np.linspace(-1, 1, 2 * n), np.linspace(-1, 1, n // 2))
            local = np.stack([u.ravel() * s[0], v.ravel() * s[1], np.full(u.size, s[2])], 1)
        elif model.geom_type[gi] == mujoco.mjtGeom.mjGEOM_CYLINDER:  # wheel: axis is local z
            a = np.linspace(0, 2 * np.pi, 10, endpoint=False)
            local = np.stack([s[0] * np.cos(a), s[0] * np.sin(a), np.zeros_like(a)], 1)
        else:
            continue
        pts.append(local @ R.T + p)
    return np.concatenate(pts)


def add_sphere(scn, pos, r, rgba):
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([r, 0, 0]), np.asarray(pos, float),
                        np.eye(3).ravel(), np.asarray(rgba, np.float32))
    scn.ngeom += 1


def add_arrow(scn, a, b, w, rgba):
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                        np.asarray(rgba, np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, w, np.asarray(a, float), np.asarray(b, float))
    scn.ngeom += 1


def shot(model, data, a, lookat, dist, extra=None, alpha=None, azimuth=None, elevation=None):
    """RGBA image: geoms where alpha (per geom, 0 hides) says, background transparent (depth mask)."""
    rgba0 = model.geom_rgba.copy()
    if alpha is not None:
        model.geom_rgba[:, 3] = alpha
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
    if floor >= 0:
        model.geom_rgba[floor, 3] = 0
    cam = mujoco.MjvCamera()
    cam.lookat[:], cam.distance = lookat, dist
    cam.azimuth = a.azimuth if azimuth is None else azimuth
    cam.elevation = a.elevation if elevation is None else elevation
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[0] = opt.geomgroup[1] = 1
    s = 2 * a.size
    with mujoco.Renderer(model, s, s, max_geom=20000) as r:
        r.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
        r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        out = []
        for depth in (False, True):
            r.enable_depth_rendering() if depth else r.disable_depth_rendering()
            r.update_scene(data, cam, scene_option=opt)
            if extra:
                extra(r.scene)
            out.append(r.render().copy())
    model.geom_rgba[:] = rgba0
    rgb, d = out
    mask = d < 0.999 * d.max()
    # one-pixel erosion against the dark antialiasing fringe of the black clear color
    m = mask.copy()
    m[1:] &= mask[:-1]; m[:-1] &= mask[1:]; m[:, 1:] &= mask[:, :-1]; m[:, :-1] &= mask[:, 1:]
    img = Image.fromarray(np.dstack([rgb, (m * 255).astype(np.uint8)]), 'RGBA')
    return img.resize((a.size, a.size), Image.LANCZOS)


def crop(img, pad=8):
    box = img.getchannel('A').getbbox()
    if box is None:
        return img
    l, t, r, b = box
    return img.crop((max(l - pad, 0), max(t - pad, 0), min(r + pad, img.width), min(b + pad, img.height)))


def save(img, name):
    crop(img).save(OUT / f'{name}.png')
    print('wrote', OUT / f'{name}.png')


def scene_icons(a):
    model, data, g = load(a)
    board = board_geoms(model)
    robot_only = np.where(board, 0.0, model.geom_rgba[:, 3])
    board_only = np.where(board, model.geom_rgba[:, 3], 0.0)

    # proprioception: the robot alone, standing (t = 0)
    set_pose(model, data, g, 0.0)
    root = data.qpos[:3].copy()
    save(shot(model, data, a, root - [0, 0, 0.2], 2.6, alpha=robot_only), 'proprioception')

    # reference motion: ghost keyframes of the robot, older poses fainter, tinted towards gray
    frames = []
    for t in a.ghost_t:
        set_pose(model, data, g, t)
        frames.append((data.qpos[:3].copy(), t))
    center = np.mean([p for p, _ in frames], 0) - [0, 0, 0.2]
    span = max(np.ptp([p[:2] for p, _ in frames], 0).max(), 1.0)
    d = frames[-1][0] - frames[0][0]
    side = np.degrees(np.arctan2(d[1], d[0])) + 90 + a.ghost_view  # camera azimuth: side view, turned along the path
    canvas = Image.new('RGBA', (a.size, a.size), (0, 0, 0, 0))
    for i, (_, t) in enumerate(frames):
        set_pose(model, data, g, t)
        img = shot(model, data, a, center, 1.8 + 1.5 * span, alpha=robot_only, azimuth=side, elevation=-12)
        arr = np.asarray(img).astype(float)
        w = 0.25 + 0.75 * (i + 1) / len(frames)
        arr[..., :3] = arr[..., :3] * w + 200 * (1 - w)
        arr[..., 3] *= w
        canvas = Image.alpha_composite(canvas, Image.fromarray(arr.astype(np.uint8), 'RGBA'))
    save(canvas, 'reference_ghost')

    # board, template points, estimated pose and projection (t = 0)
    set_pose(model, data, g, 0.0)
    bid = model.jnt_bodyid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'floating_board_base_joint')]
    c, R = data.xpos[bid].copy(), data.xmat[bid].reshape(3, 3).copy()
    pts = template_points(model, data)

    def points(scn, rgba=(*BLUE, 1)):
        for p in pts:
            add_sphere(scn, p, 0.011, rgba)

    def axes(scn):
        for k, col in enumerate([(0.85, 0.2, 0.2, 1), (0.2, 0.7, 0.3, 1), (0.2, 0.35, 0.85, 1)]):
            add_arrow(scn, c + [0, 0, 0.03], c + [0, 0, 0.03] + 0.28 * R[:, k], 0.012, col)

    kw = dict(lookat=c, dist=1.5, elevation=-35)
    save(shot(model, data, a, alpha=board_only, **kw), 'board')
    save(shot(model, data, a, alpha=np.zeros(model.ngeom), extra=points, **kw), 'template_points')
    save(shot(model, data, a, alpha=board_only, extra=axes, **kw), 'board_pose')
    save(shot(model, data, a, alpha=np.where(board, 0.35, 0.0), extra=points, **kw), 'projection')

    # action: robot on the board during the push
    set_pose(model, data, g, a.action_t)
    save(shot(model, data, a, data.qpos[:3] - [0, 0, 0.3], 2.8), 'robot_on_board')


# ---------------------------------------------------------------- vector icons
def vec(name, draw, w=1.0, h=1.0):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis('off'); ax.set_aspect('equal')
    draw(ax)
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'{name}.{ext}', dpi=600, transparent=True)
    plt.close(fig)
    print('wrote', OUT / f'{name}.pdf')


def latent(color, seed):
    def draw(ax):
        v = np.random.default_rng(seed).uniform(0.2, 1.0, 8)
        for i, x in enumerate(v):
            ax.add_patch(Rectangle((0.08 + i * 0.105, 0.1), 0.075, 0.8 * x, fc=color, ec='none'))
    return draw


def mlp(ax):
    layers = [(0.15, 4), (0.5, 5), (0.85, 3)]
    pos = [[(x, 0.5 + (k - (n - 1) / 2) * 0.17) for k in range(n)] for x, n in layers]
    for l0, l1 in zip(pos, pos[1:]):
        for p in l0:
            for q in l1:
                ax.plot([p[0], q[0]], [p[1], q[1]], color=GRAY, lw=0.4, zorder=1)
    for layer in pos:
        for p in layer:
            ax.add_patch(Circle(p, 0.05, fc='white', ec=(0.2, 0.2, 0.2), lw=0.8, zorder=2))


def imu(ax):
    ax.add_patch(FancyBboxPatch((0.22, 0.22), 0.56, 0.56, boxstyle='round,pad=0,rounding_size=0.06',
                                fc=(0.25, 0.25, 0.27), ec='none'))
    for k in range(5):
        y = 0.3 + k * 0.1
        for x0, x1 in ((0.1, 0.22), (0.78, 0.9)):
            ax.plot([x0, x1], [y, y], color=(0.45, 0.45, 0.47), lw=1.2, solid_capstyle='butt')
            ax.plot([y, y], [x0, x1], color=(0.45, 0.45, 0.47), lw=1.2, solid_capstyle='butt')
    ax.add_patch(Arc((0.5, 0.5), 0.36, 0.2, theta1=200, theta2=520, color=ORANGE, lw=1.4))
    ax.annotate('', (0.67, 0.53), (0.66, 0.46), arrowprops=dict(arrowstyle='-|>', color=ORANGE, lw=1.2))


def integrator(ax):
    t = np.linspace(0.1, 0.9, 60)
    y = 0.35 + 0.12 * np.sin(9 * t) + 0.08 * t
    ax.fill_between(t, 0.2, y, color=(*ORANGE, 0.25), lw=0)
    ax.plot(t, y, color=ORANGE, lw=1.2)
    ax.plot([0.1, 0.92], [0.2, 0.2], color=(0.2, 0.2, 0.2), lw=0.8)
    ax.text(0.5, 0.78, r'$\int \dot{\psi}\,\mathrm{d}t$', ha='center', va='center', fontsize=9)


def dial(ax):
    ax.add_patch(Circle((0.5, 0.5), 0.4, fc='white', ec=(0.2, 0.2, 0.2), lw=0.9))
    ax.annotate('', (0.5, 0.88), (0.5, 0.5), arrowprops=dict(arrowstyle='-|>', color=GRAY, lw=1.1, ls='--'))
    e = np.deg2rad(28)
    ax.annotate('', (0.5 + 0.36 * np.sin(e), 0.5 + 0.36 * np.cos(e)), (0.5, 0.5),
                arrowprops=dict(arrowstyle='-|>', color=ORANGE, lw=1.4))
    ax.add_patch(Arc((0.5, 0.5), 0.4, 0.4, theta1=90 - 28, theta2=90, color=ORANGE, lw=1.0))
    ax.text(0.6, 0.78, r'$\tilde e$', fontsize=8, color=ORANGE)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--hoic', type=Path, default=Path('D:/HAIC-Claude/HOIC-baseline'))
    p.add_argument('--golden', type=Path, default=Path('D:/HAIC-Claude/result/sim2sim/skateboard/dyn/st0/golden.npz'))
    p.add_argument('--ghost-t', type=float, nargs='+', default=[3.0, 5.5, 7.0, 9.0], help='keyframe times [s]')
    p.add_argument('--ghost-view', type=float, default=55.0,
                   help='ghost camera: degrees from the side view towards looking along the path')
    p.add_argument('--action-t', type=float, default=5.0, help='time of the robot-on-board icon [s]')
    p.add_argument('--azimuth', type=float, default=140.0)
    p.add_argument('--elevation', type=float, default=-15.0)
    p.add_argument('--size', type=int, default=800, help='icon canvas [px] before cropping')
    p.add_argument('--vector-only', action='store_true')
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    vec('latent_teacher', latent(GRAY, 0), 1.0, 0.6)
    vec('latent_student', latent(BLUE, 0), 1.0, 0.6)
    vec('mlp', mlp)
    vec('imu', imu)
    vec('integrator', integrator)
    vec('heading_dial', dial)
    if not a.vector_only:
        scene_icons(a)


if __name__ == '__main__':
    main()
