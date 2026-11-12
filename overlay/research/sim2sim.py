"""MuJoCo sim-to-sim evaluation of exported students (research/export_students.py).

    python research/sim2sim.py --hoic D:/HAIC-Claude/HOIC-baseline --export D:/HAIC-Claude/result/sim2sim --check
    python research/sim2sim.py --hoic D:/HAIC-Claude/HOIC-baseline --export D:/HAIC-Claude/result/sim2sim --episodes 100
    python research/sim2sim.py ... --tasks skateboard --arms dyn --episodes 30 --calibrate 0.001 0.003 0.01 0.03 0.1
    python research/sim2sim.py ... --tasks skateboard --arms dyn --episodes 30 --calibrate 0.001 0.003 --calibrate-frictionloss 0.05 0.1 0.2
    python research/sim2sim.py ... --tasks skateboard --wheel-damping 0.003   # both arms at the calibrated value

The wheel damping is calibrated on the baseline (HAIC) students only: the value whose median signed
along-track error at 9 s is closest to zero; the improved arm is then evaluated at that value.

Runs on the CPU (mujoco + onnxruntime only, no Isaac). Scenes, gains, action mapping and the
observation layout of the student are those of the authors' MuJoCo deployment (HOIC-baseline,
utils/intermimic.py); this script adds the onboard heading sensor of the heading arms, starts every
episode from reference frame 0 like the Isaac evaluation, randomizes friction, object mass, initial
joints and observation noise per episode, and applies the Isaac termination criteria of the task
(world-frame and local torso / body errors, object errors for the box) to count successes.

--check compares the observation builder with the golden Isaac rollout stored by the export (every
input term, step by step) and the ONNX action with the exported one, and stops.
Outputs: <export>/<task>/<arm>/st<s>/mujoco/{episodes.csv,summary.json} and <export>/sim2sim.json.
"""
import argparse
import csv
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
TASKS = {  # scene, reference motion, object free joint, object body in the motion, object terminations
    'skateboard': ('g1_29dof_rev_1_0_skateboard.xml', 'board6', 'floating_board_base_joint', 'skateboard', False),
    'push_cart': ('g1_29dof_rev_1_0_push_cart.xml', 'push_cart', 'floating_cart_base_joint', 'cart', False),
    'box_stairslope': ('g1_29dof_rev_1_0_box_stairslope.xml', 'box_stairslope', 'floating_box_base_joint', 'mybox', True),
}
# Isaac termination criteria (cfg/task/base/hdmi-base.yaml): threshold, consecutive steps
TERM = {'torso_pos': (0.5, 25), 'torso_ori': (1.2, 25), 'local_pos': (0.5, 25), 'local_ori': (1.2, 25),
        'object_pos': (0.5, 25), 'object_ori': (0.8, 25)}
DRIFT = ('torso_pos', 'torso_ori', 'object_pos', 'object_ori')  # world-frame terms, off in drift-tolerant success
HEADING = {'init_align_std': 0.01, 'bias_std': 0.002, 'bias_rw_std': 0.0005, 'rate_noise_std': 0.005}
CAL_STEP = 450  # 9 s: along-track error used to calibrate the wheel damping (before most drift terminations)
OBS_NOISE = {'ang_vel': 0.05, 'gravity': 0.05, 'joint_pos': 0.015}  # policy group noise_std of the task config


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def yaw_wxyz(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def quat_angle(q1, q2):
    """Rotation angle between two wxyz quaternions."""
    d = abs(float(np.dot(q1, q2)) / (np.linalg.norm(q1) * np.linalg.norm(q2)))
    return 2 * math.acos(min(1.0, d))


class Student:
    """ONNX student with the HOIC observation builder plus the heading sensor."""

    def __init__(self, hoic, export_dir, motion_dir):
        sys.path.insert(0, str(hoic))
        from utils.intermimic import InterMimicRod  # noqa: E402  (HOIC-baseline)
        import onnxruntime
        meta = json.loads((export_dir / 'policy.json').read_text())
        self.in_keys = [k if isinstance(k, str) else '/'.join(k) for k in meta['in_keys']]
        self.out_keys = [k if isinstance(k, str) else '/'.join(k) for k in meta['out_keys']]
        self.rod = InterMimicRod(export_dir / 'policy.onnx', motion_dir)
        self.sess = onnxruntime.InferenceSession(str(export_dir / 'policy.onnx'), providers=['CPUExecutionProvider'])
        self.inputs = self.sess.get_inputs()
        assert len(self.inputs) == len(self.in_keys), (self.in_keys, [i.name for i in self.inputs])
        self.i_action = self.out_keys.index('action')
        self.heading = None  # set per episode by the caller for heading arms

    def feed(self, obs):
        out = {}
        for key, info in zip(self.in_keys, self.inputs):
            if key == 'heading_':
                v = self.heading[None].astype(np.float32)
            elif key in ('object_geo_', 'object_geo'):
                v = obs['object_geo']
            else:
                v = obs[key]
            width = info.shape[1] if len(info.shape) > 1 and isinstance(info.shape[1], int) else v.shape[1]
            assert v.shape[1] == width, f'input {key}: built {v.shape[1]} values, the student expects {width}'
            out[info.name] = v.astype(np.float32)
        return out

    def act(self, obs):
        return self.sess.run(None, self.feed(obs))[self.i_action].squeeze(0)


SUBTERMS = {'policy': (('ang_vel', 0, 3), ('gravity', 3, 6), ('joint_pos_hist', 6, 180), ('prev_actions', 180, 249)),
            'command': (('ref_body_pos', 0, 240), ('ref_joint_pos', 240, 355), ('phase', 355, 356))}


def check_golden(hoic, export_dir, motion_dir, first=10):
    """Rebuild every input term from the state stored in golden.npz and compare with Isaac (max abs
    difference over steps >= first, so that the joint and action histories are filled)."""
    g = np.load(export_dir / 'golden.npz')
    st = Student(hoic, export_dir, motion_dir)
    rod, ref = st.rod, st.rod.refmotion
    names = [str(n) for n in g['joint_names']]
    from utils.motion import UNITREE_JOINT_NAMES
    mj_from_isaac = [names.index(n) for n in UNITREE_JOINT_NAMES]  # MuJoCo (Unitree) order from Isaac order
    applied = g['rollout_action'] if 'rollout_action' in g.files else g['action']
    rep = {}

    def note(key, built, iso):
        rep[key] = max(rep.get(key, 0.0), float(np.abs(built - iso).max()))

    rod.dof_pos_buf[:] = g['joint_pos'][0][mj_from_isaac][rod.cor_obs_index]
    for step in range(len(g['action'])):
        ref.t = int(g['ref_t'][step]) - 2   # the observation uses frame ref_t - 1; update() advances by one
        q = g['root_quat_w'][step]
        state = {'qj': g['joint_pos'][step][mj_from_isaac], 'quat': np.array([q[1], q[2], q[3], q[0]]),
                 'ang_vel': g['root_ang_vel_b'][step]}
        ref.update()
        obs = rod._compute_obs(state)
        rod.action = applied[step]
        if step < first:
            continue
        for key in st.in_keys:
            iso = g[f'in/{key}'][step]
            if key == 'heading_':  # Isaac adds the integration error of the sensor; this is the true value
                e = wrap(yaw_wxyz(q) - yaw_wxyz(g['ref_root_quat_w'][step]))
                note(key, np.array([math.cos(e), math.sin(e)]), iso)
                continue
            built = obs['object_geo' if key.startswith('object_geo') else key][0]
            for name, lo, hi in SUBTERMS.get(key, ((key, 0, len(iso)),)):
                note(f'{key}/{name}', built[lo:hi], iso[lo:hi])
        a = st.sess.run(None, {i.name: g[f'in/{k}'][step][None].astype(np.float32)
                               for i, k in zip(st.inputs, st.in_keys)})[st.i_action].squeeze(0)
        note('onnx_vs_exported_action', a, g['action'][step])
        if 'rollout_action' in g.files:
            note('onnx_vs_rollout_action', a, g['rollout_action'][step])
    return {k: round(v, 5) for k, v in rep.items()}


def run_episode(job):
    """One MuJoCo episode; returns a row with success, terminations and end errors."""
    import mujoco
    hoic, export_dir, task, ep, seed, noise_scale, randomize, wheel_damping, wheel_frictionloss, solver = job[:10]
    video = job[10] if len(job) > 10 else None  # mp4 path: record this episode (chase camera, reference keypoints)
    xml, motion, obj_joint, obj_ref, obj_term = TASKS[task]
    motion_dir = REPO / 'data/motion/g1/mirobotA' / motion
    st = Student(hoic, Path(export_dir), motion_dir)
    rod, ref = st.rod, st.rod.refmotion
    rng = np.random.default_rng(seed * 100003 + ep)
    spec = mujoco.MjSpec.from_file(str(Path(hoic) / 'assets/robots/g1' / xml))
    if parse_solver(solver).get('isaac_assets') == '1':
        isaac_assets(spec, task)
    model = spec.compile()
    model.opt.timestep = 0.002
    obj_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, obj_joint)
    obj_body = model.jnt_bodyid[obj_jid]
    for k, v in parse_solver(solver).items():  # contact-solver settings (before the friction randomization)
        if k == 'impratio':
            model.opt.impratio = float(v)
        elif k == 'cone':
            model.opt.cone = {'pyramidal': mujoco.mjtCone.mjCONE_PYRAMIDAL, 'elliptic': mujoco.mjtCone.mjCONE_ELLIPTIC}[v]
        elif k == 'noslip':
            model.opt.noslip_iterations = int(v)
        elif k == 'floor_mu':  # sliding friction of the floor (contacts take the larger of the two geoms)
            model.geom_friction[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'floor'), 0] = float(v)
        elif k in ('isaac_assets', 'golden_init', 'heading_null'):
            pass  # applied to the spec before compiling
        elif k == 'deck_mu':  # sliding friction of the deck (Isaac: dynamic 1.5-3.0, static x1-2)
            model.geom_friction[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'board_deck'), 0] = float(v)
        elif k == 'board_scale':  # homogeneous object scale (Isaac: body_scale 0.8-1.1), geometry only
            s, sub = float(v), {obj_body}
            for b in range(model.nbody):
                if model.body_parentid[b] in sub and b != 0:
                    sub.add(b)
                    model.body_pos[b] *= s
            g = np.isin(model.geom_bodyid, list(sub))
            for arr in (model.geom_size, model.geom_pos, model.geom_rbound, model.geom_aabb):
                arr[g] *= s
        elif k == 'isaac_joints' and v == '1':  # Isaac armature per joint, no passive joint friction / damping
            import re
            acts = json.loads((Path(export_dir) / 'policy_cfg.json').read_text())['actuators'].values()
            for j in range(model.njnt):
                name, dof = model.joint(j).name, model.jnt_dofadr[j]
                if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE or 'wheel' in name:
                    continue
                for act in acts:
                    if any(re.fullmatch(e, name) for e in act['joint_names_expr']):
                        arm = act['armature']
                        arm = next((x for e, x in arm.items() if re.fullmatch(e, name)), None) \
                            if isinstance(arm, dict) else arm
                        if arm is not None:
                            model.dof_armature[dof] = float(arm)
                        model.dof_frictionloss[dof] = model.dof_damping[dof] = 0.0
        elif k == 'golden_mass' and v == '1':  # masses of the golden Isaac episode (env 0, object scale 1)
            gz = np.load(Path(export_dir) / 'golden.npz')
            board = [b for b in range(model.nbody) if b == obj_body or model.body_rootid[b] == obj_body]
            om = gz['obj_body_masses']  # Isaac: deck, then the wheels
            for b in board:
                n = model.body(b).name
                if b == obj_body or 'wheel' in n:
                    new = om[0] if b == obj_body else float(np.mean(om[1:]))
                    model.body_inertia[b] *= new / model.body_mass[b]
                    model.body_mass[b] = new
            rb = [b for b in range(1, model.nbody) if model.body_rootid[b] != obj_body]
            scale = float(gz['robot_body_masses'].sum()) / float(model.body_mass[rb].sum())
            model.body_mass[rb] *= scale  # Isaac randomizes link masses; match the total
            model.body_inertia[rb] *= scale
        else:
            raise ValueError(f'unknown solver setting {k}')
    if randomize:
        model.geom_friction[:, 0] *= rng.uniform(0.7, 1.3)
        m = rng.uniform(0.8, 1.2)
        model.body_mass[obj_body] *= m
        model.body_inertia[obj_body] *= m
    if wheel_damping is not None:  # rolling resistance of the skateboard / cart wheels (calibrated)
        for j in range(model.njnt):
            if 'wheel' in model.joint(j).name:
                model.dof_damping[model.jnt_dofadr[j]] = wheel_damping
    if wheel_frictionloss is not None:  # Coulomb (speed-independent) wheel friction
        for j in range(model.njnt):
            if 'wheel' in model.joint(j).name:
                model.dof_frictionloss[model.jnt_dofadr[j]] = wheel_frictionloss
    data = mujoco.MjData(model)
    nu = 29

    # start from reference frame 0 (root pose, joints, root velocity, object pose)
    bn = ref.body_names
    i_root, i_obj, i_torso = bn.index('pelvis'), bn.index(obj_ref), bn.index('torso_link')
    data.qpos[:3] = ref.body_pos_w[0, i_root]
    data.qpos[3:7] = ref.body_quat_w[0, i_root]
    data.qpos[7:7 + nu] = ref.joint_pos[0, :nu] + (rng.normal(0, 0.02, nu) if randomize else 0.0)
    qa = model.jnt_qposadr[obj_jid]
    data.qpos[qa:qa + 3] = ref.body_pos_w[0, i_obj]
    data.qpos[qa + 2] *= float(parse_solver(solver).get('board_scale', 1.0))  # resting height of a scaled object
    data.qpos[qa + 3:qa + 7] = ref.body_quat_w[0, i_obj]
    if parse_solver(solver).get('golden_init') == '1':  # exact initial state of the golden Isaac episode (reset noise)
        gz = np.load(Path(export_dir) / 'golden.npz')
        gnames = list(gz['joint_names'])
        data.qpos[:3], data.qpos[3:7] = gz['root_pos_w'][0], gz['root_quat_w'][0]
        for j in range(1, 1 + nu):
            i = gnames.index(model.joint(j).name)
            data.qpos[model.jnt_qposadr[j]] = gz['joint_pos'][0, i]
            data.qvel[model.jnt_dofadr[j]] = gz['joint_vel'][0, i]
        if 'obj_pos_w' in gz:
            data.qpos[qa:qa + 3], data.qpos[qa + 3:qa + 7] = gz['obj_pos_w'][0], gz['obj_quat_w'][0]
    mujoco.mj_forward(model, data)
    rod.reset_history()
    rod.dof_pos_buf[:] = data.qpos[7:7 + nu][rod.cor_obs_index]

    # heading sensor of the heading arms (same model as the Isaac observation, scaled by noise_scale)
    hs = {k: v * noise_scale for k, v in HEADING.items()}
    err, bias = rng.normal(0, hs['init_align_std']), rng.normal(0, hs['bias_std'])
    st.heading = np.array([1.0, 0.0])

    tb = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in bn if
          mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) >= 0}
    track = [n for n in tb if n not in (obj_ref, 'stairslope')]
    cnt = {k: 0 for k in TERM}
    flags = {k: False for k in TERM}
    dt, steps = 0.02, ref.motion_steps - 1
    n_sub = round(dt / model.opt.timestep)
    end_step, max_lat = steps, 0.0
    cal = {'along_cal_m': float('nan'), 'lat_cal_m': float('nan')}  # signed root errors at CAL_STEP
    drift = {'drift_step': -1, 'lat_at_drift_m': 0.0, 'along_at_drift_m': 0.0, 'heading_at_drift_deg': 0.0}
    # per-step trace in the layout of the Isaac paths.npz of research/showcase.py (NaN after the end)
    trace = {k: np.full((steps, 3), np.nan) for k in ('root_xyz', 'ref_root_xyz', 'obj_xyz', 'ref_obj_xyz')}
    trace.update({k: np.full(steps, np.nan) for k in ('root_yaw', 'ref_root_yaw')})
    trace['ankle_xyz'], trace['torque'] = np.full((steps, 2, 3), np.nan), np.full((steps, nu), np.nan)
    trace['obj_lin_vel'] = np.full((steps, 3), np.nan)
    from utils.sim import get_obs_from_mujoco, pd_control
    if video:
        import imageio.v2 as imageio
        model.vis.global_.offwidth, model.vis.global_.offheight = 1600, 912
        renderer = mujoco.Renderer(model, 912, 1600, max_geom=20000)
        cam = mujoco.MjvCamera()
        cam.distance, cam.elevation = 4.0, -22.0
        writer = imageio.get_writer(str(video), fps=round(1 / 0.02), codec='libx264', quality=8, macro_block_size=8)
    for step in range(steps):
        s = get_obs_from_mujoco(data, nu)
        if randomize:
            s['ang_vel'] = s['ang_vel'] + rng.normal(0, OBS_NOISE['ang_vel'], 3)
            s['qj'] = s['qj'] + rng.normal(0, OBS_NOISE['joint_pos'], nu)
        # Isaac: the observation at control step k uses reference frame k-1, the state is compared with frame k
        ref.t = max(step - 1, 0) - 1  # update() advances by one
        ref.update()
        yaw_r = yaw_wxyz(data.qpos[3:7])
        bias += rng.normal(0, hs['bias_rw_std'] * math.sqrt(dt))
        err += (bias + rng.normal(0, hs['rate_noise_std'])) * dt
        e = wrap(yaw_r + err - yaw_wxyz(ref.body_quat_w[ref.t, i_root]))
        st.heading = np.array([1.0, 0.0] if parse_solver(solver).get('heading_null') == '1' else [math.cos(e), math.sin(e)])
        obs = rod._compute_obs(s)
        if randomize:  # projected gravity is computed inside the builder
            obs['policy'][0, 3:6] += rng.normal(0, OBS_NOISE['gravity'], 3)
        a = st.act(obs)
        rod.action = a.copy()
        a = rod._add_wrist_6dof(a[rod.cor_hdmi_norm].copy())
        target = a * rod.action_scale + rod.default_pos
        for _ in range(n_sub):
            tau = pd_control(target, data.qpos[7:7 + nu], rod.dof_kps, 0.0, data.qvel[6:6 + nu], rod.dof_kds)
            data.ctrl[:nu] = np.clip(tau, -rod.torque_limits, rod.torque_limits)
            mujoco.mj_step(model, data)

        # termination errors at the new reference frame (Isaac: t advances, then terminations update)
        t = min(step + 1, ref.motion_steps - 1)
        rp, rq = ref.body_pos_w[t], ref.body_quat_w[t]
        pos = {n: data.xpos[b] for n, b in tb.items()}
        quat = {n: data.xquat[b] for n, b in tb.items()}
        errs = {'torso_pos': np.linalg.norm(rp[i_torso] - pos['torso_link']),
                'torso_ori': quat_angle(rq[i_torso], quat['torso_link'])}
        yr, yb = yaw_wxyz(rq[i_root]), yaw_wxyz(data.xquat[tb['pelvis']])
        Rr = np.array([[math.cos(yr), math.sin(yr), 0], [-math.sin(yr), math.cos(yr), 0], [0, 0, 1]])
        Rb = np.array([[math.cos(yb), math.sin(yb), 0], [-math.sin(yb), math.cos(yb), 0], [0, 0, 1]])
        r0, b0 = rp[i_root] * [1, 1, 0], pos['pelvis'] * [1, 1, 0]
        errs['local_pos'] = max(np.linalg.norm(Rr @ (rp[bn.index(n)] - r0) - Rb @ (pos[n] - b0)) for n in track)
        errs['local_ori'] = 0.0  # body-orientation terms rarely fire (<2% of Isaac failures); not reproduced
        errs['object_pos'] = np.linalg.norm(rp[i_obj] - data.xpos[obj_body]) if obj_term else 0.0
        errs['object_ori'] = quat_angle(rq[i_obj], data.xquat[obj_body]) if obj_term else 0.0
        d = data.xpos[tb['pelvis']] - rp[i_root]
        max_lat = max(max_lat, abs(-math.sin(yr) * d[0] + math.cos(yr) * d[1]))
        trace['root_xyz'][step], trace['ref_root_xyz'][step] = pos['pelvis'], rp[i_root]
        trace['obj_xyz'][step], trace['ref_obj_xyz'][step] = data.xpos[obj_body], rp[i_obj]
        trace['root_yaw'][step], trace['ref_root_yaw'][step] = yb, yr
        trace['ankle_xyz'][step] = [data.xpos[tb['left_ankle_roll_link']], data.xpos[tb['right_ankle_roll_link']]]
        trace['torque'][step] = data.ctrl[:nu]
        trace['obj_lin_vel'][step] = data.qvel[model.jnt_dofadr[obj_jid]:model.jnt_dofadr[obj_jid] + 3]
        if video:  # camera behind the robot along the reference heading, as the Isaac chase camera
            cam.lookat[:] = data.xpos[tb['pelvis']] * [1, 1, 0] + [0, 0, 0.8]
            cam.azimuth = math.degrees(yr)
            renderer.update_scene(data, cam)
            for n in track:  # red reference keypoints
                g = renderer.scene.geoms[renderer.scene.ngeom]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.03, 0, 0]), rp[bn.index(n)].astype(float),
                                    np.eye(3).ravel(), np.array([0.9, 0.05, 0.05, 1.0], np.float32))
                renderer.scene.ngeom += 1
            writer.append_data(renderer.render())
        if step + 1 == CAL_STEP:
            cal = {'along_cal_m': round(math.cos(yr) * d[0] + math.sin(yr) * d[1], 4),
                   'lat_cal_m': round(-math.sin(yr) * d[0] + math.cos(yr) * d[1], 4)}
        for k, (thr, n) in TERM.items():
            cnt[k] = cnt[k] + 1 if errs[k] >= thr else 0
            flags[k] |= cnt[k] >= n
        if drift['drift_step'] < 0 and any(flags[k] for k in DRIFT):  # first world-frame termination
            drift = {'drift_step': step + 1,
                     'lat_at_drift_m': round(-math.sin(yr) * d[0] + math.cos(yr) * d[1], 4),
                     'along_at_drift_m': round(math.cos(yr) * d[0] + math.sin(yr) * d[1], 4),
                     'heading_at_drift_deg': round(math.degrees(wrap(yb - yr)), 2)}
        # world-frame terms only mark the episode (drift-tolerant success continues past them, as pred_dt)
        if any(flags[k] for k in TERM if k not in DRIFT):
            end_step = step + 1
            break
    fell = bool(data.xpos[tb['pelvis']][2] < 0.4)
    local = any(flags[k] for k in TERM if k not in DRIFT)
    if video:
        writer.close()
        renderer.close()
    return {'episode': ep, 'success': int(not any(flags.values())),
            'success_drift_tolerant': int(not local and not fell),
            'end_step': end_step, 'fell': int(fell), 'max_lateral_m': round(max_lat, 4), **drift, **cal,
            **{f'term/{k}': int(v) for k, v in flags.items()}, '_trace': trace}


def box_inertia(m, x, y, z):
    return [m * (y * y + z * z) / 12, m * (x * x + z * z) / 12, m * (x * x + y * y) / 12]


def isaac_assets(spec, task):
    """Collision geometry and inertia of the Isaac training assets (research/dump_usd.py) in the HOIC scene.

    Robot: foot boxes 0.17 x 0.05 x 0.01 m, 0.03 m below the ankle-roll frame (HOIC: 0.17 x 0.06 x 0.02, 0.02 below).
    Skateboard: wheels 0.03 m wide (HOIC 0.06) with the centre of mass on the axle (HOIC: 0.1 m off along it),
    trucks 0.04 x 0.10 x 0.04, deck and wheel inertia of their shapes (HOIC deck: 0.01 on every axis).
    """
    for side in ('left', 'right'):
        g = spec.geom(f'{side}_ankle_roll_link')
        g.size, g.pos = [0.085, 0.025, 0.005], [0.035, 0.0, -0.03]
    if task != 'skateboard':
        return
    deck = spec.body('board_deck')
    deck.inertia = box_inertia(deck.mass, 0.8, 0.23, 0.02)
    for end in ('front', 'rear'):
        for g in spec.body(f'{end}_truck').geoms:
            g.size, g.quat = [0.02, 0.05, 0.02], [1, 0, 0, 0]
        for side in ('left', 'right'):
            w = spec.body(f'{end}_{side}_wheel')
            w.ipos = [0, 0, 0]
            r, h, m = 0.04, 0.03, w.mass
            w.inertia = [m * (3 * r * r + h * h) / 12, m * r * r / 2, m * (3 * r * r + h * h) / 12]
            spec.geom(f'{end}_{side}_wheel').size = [r, h / 2, 0]


def parse_solver(spec):
    """'impratio=10,cone=elliptic,noslip=10,floor_mu=1.5' -> dict ('' or None: the scene's settings)."""
    return dict(kv.split('=') for kv in spec.split(',')) if spec else {}


def evaluate(a, d, task, wheel_damping, wheel_frictionloss, solver=None):
    """Rows of all episodes and their stacked traces ([T, episodes, ...], like paths.npz)."""
    jobs = [(str(a.hoic), str(d), task, ep, a.seed, a.noise_scale, not a.no_randomize, wheel_damping,
             wheel_frictionloss, solver) for ep in range(a.episodes)]
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(run_episode, jobs))
    traces = [r.pop('_trace') for r in rows]
    return rows, {k: np.stack([t[k] for t in traces], axis=1) for k in traces[0]}


def along_track(tr):
    """Signed along-track root error [T, episodes] in the reference heading frame."""
    dx = tr['root_xyz'] - tr['ref_root_xyz']
    yr = tr['ref_root_yaw']
    return np.cos(yr) * dx[..., 0] + np.sin(yr) * dx[..., 1]


TRACE_T = (5.0, 5.5, 6.6, 7.5, 8.5, 9.0)  # s: push phase, glide, before the dismount


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--hoic', type=Path, required=True, help='HOIC-baseline checkout (scenes, gains)')
    p.add_argument('--export', type=Path, required=True, help='unzipped sim2sim-export.zip')
    p.add_argument('--tasks', nargs='+', default=list(TASKS))
    p.add_argument('--arms', nargs='+', default=['dyn', 'dyn-hboth'])
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--noise-scale', type=float, default=1.0, help='heading-sensor error scale')
    p.add_argument('--no-randomize', action='store_true')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--check', action='store_true', help='compare with the golden Isaac rollout and stop')
    p.add_argument('--wheel-damping', type=float, default=None,
                   help='wheel joint damping [N m s/rad]; default: the value of the scene')
    p.add_argument('--calibrate', type=float, nargs='+', default=None,
                   help='wheel dampings to sweep: report the median signed along-track error at 9 s per value '
                        '(run it with --arms dyn only, so that the improved arm never informs the choice)')
    p.add_argument('--wheel-frictionloss', type=float, default=None,
                   help='wheel joint Coulomb friction [N m]; default: the value of the scene')
    p.add_argument('--calibrate-frictionloss', type=float, nargs='+', default=None,
                   help='with --calibrate: wheel frictionloss values swept on a grid with the dampings')
    p.add_argument('--solver', default=None,
                   help="contact-solver settings, e.g. 'impratio=10,cone=elliptic,floor_mu=1.5'; default: the scene's")
    p.add_argument('--calibrate-solver', nargs='+', default=None,
                   help="with --calibrate: solver settings swept on the grid ('' for the scene's)")
    p.add_argument('--video', type=Path, default=None,
                   help='record --video-episodes of --students as <video>/<task>-<arm>-<student>-ep<k>.mp4 and stop '
                        '(no evaluation files are written)')
    p.add_argument('--video-episodes', type=int, nargs='+', default=[0])
    p.add_argument('--students', nargs='+', default=None, help='only these student folders, e.g. st2')
    a = p.parse_args()
    if a.video:
        a.video.mkdir(parents=True, exist_ok=True)
        for task in a.tasks:
            for arm in a.arms:
                for d in sorted((a.export / task / arm).glob('st*')):
                    if a.students and d.name not in a.students:
                        continue
                    for ep in a.video_episodes:
                        out = a.video / f'{task}-{arm}-{d.name}-ep{ep}.mp4'
                        r = run_episode((str(a.hoic), str(d), task, ep, a.seed, a.noise_scale, not a.no_randomize,
                                         a.wheel_damping, a.wheel_frictionloss, a.solver, out))
                        print(f'VIDEO {out}: success {r["success"]}, drift-tolerant {r["success_drift_tolerant"]}, '
                              f'fell {r["fell"]}, end step {r["end_step"]}', flush=True)
        return
    tag = (('' if a.wheel_damping is None else f'-wd{a.wheel_damping:g}')
           + ('' if a.wheel_frictionloss is None else f'-fl{a.wheel_frictionloss:g}')
           + ('' if not a.solver else '-' + a.solver.replace('=', '').replace(',', '-')))
    summary = {}
    for task in a.tasks:
        motion_dir = REPO / 'data/motion/g1/mirobotA' / TASKS[task][1]
        for arm in a.arms:
            for d in sorted((a.export / task / arm).glob('st*')):
                if a.check:
                    rep = check_golden(a.hoic, d, motion_dir)
                    print(f'{task} {arm} {d.name}: max |MuJoCo builder - Isaac| per input {json.dumps(rep)}', flush=True)
                    continue
                if a.calibrate:
                    for wd in a.calibrate:
                        for fl, sv in [(f, v) for f in a.calibrate_frictionloss or [a.wheel_frictionloss]
                                       for v in a.calibrate_solver or [a.solver]]:
                            rows, tr = evaluate(a, d, task, wd, fl, sv)
                            al = along_track(tr)
                            curve = ' '.join(f'{t:g}s {np.nanmedian(al[int(t * 50)]):+.2f}'
                                             f'[{np.nanpercentile(al[int(t * 50)], 25):+.2f},'
                                             f'{np.nanpercentile(al[int(t * 50)], 75):+.2f}]' for t in TRACE_T)
                            print(f'CALIBRATE {task} {arm} {d.name} wd={wd:g} fl={fl} solver={sv or "scene"}: along-track median[IQR] '
                                  f'{curve} | success {100 * np.mean([r["success"] for r in rows]):.0f}%', flush=True)
                    continue
                rows, tr = evaluate(a, d, task, a.wheel_damping, a.wheel_frictionloss, a.solver)
                out = d / ('mujoco' + tag)
                out.mkdir(exist_ok=True)
                np.savez_compressed(out / 'trace.npz', step_dt=0.02, **tr)
                with (out / 'episodes.csv').open('w', newline='', encoding='utf-8') as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0]))
                    w.writeheader()
                    w.writerows(rows)
                s = {'task': task, 'arm': arm, 'student': d.name, 'episodes': len(rows),
                     'success_rate': float(np.mean([r['success'] for r in rows])),
                     'drift_tolerant_success_rate': float(np.mean([r['success_drift_tolerant'] for r in rows])),
                     'fell_rate': float(np.mean([r['fell'] for r in rows])),
                     'median_max_lateral_m': float(np.median([r['max_lateral_m'] for r in rows])),
                     'median_along_cal_m': float(np.nanmedian([r['along_cal_m'] for r in rows])),
                     'terminations': {k: int(sum(r[f'term/{k}'] for r in rows)) for k in TERM},
                     'noise_scale': a.noise_scale, 'randomize': not a.no_randomize, 'seed': a.seed,
                     'wheel_damping': a.wheel_damping, 'wheel_frictionloss': a.wheel_frictionloss,
                     'solver': parse_solver(a.solver)}
                (out / 'summary.json').write_text(json.dumps(s, indent=2), encoding='utf-8')
                summary[f'{task}/{arm}/{d.name}'] = s
                print(f'{task} {arm} {d.name}: success {100 * s["success_rate"]:.1f}%, drift-tolerant '
                      f'{100 * s["drift_tolerant_success_rate"]:.1f}%, fell {100 * s["fell_rate"]:.1f}%, '
                      f'along-track at 9 s {s["median_along_cal_m"]:+.2f} m', flush=True)
    if summary:
        name = f'sim2sim{tag}.json'
        (a.export / name).write_text(json.dumps(summary, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
