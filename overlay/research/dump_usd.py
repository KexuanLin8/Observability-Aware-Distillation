"""Print the collision geometry of a USD asset (shapes, sizes, world transforms, mass, materials).

    python research/dump_usd.py active_adaptation/assets/objects/skateboard/skateboard.usd
    python research/dump_usd.py active_adaptation/assets/g1/g1_29dof_rubberhand-feet_box-eef_box-body_capsule.usd --filter ankle

Used to build the MuJoCo scene of the sim-to-sim check with Isaac's dimensions. Needs pxr
(the Python of Isaac Sim, research/setup/haic_env.sh); opens the file read-only, starts no simulation.
"""
import argparse
import sys

from pxr import Usd, UsdGeom, UsdPhysics


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('usd')
    p.add_argument('--filter', default='', help='only prims whose path contains this text')
    a = p.parse_args()
    stage = Usd.Stage.Open(a.usd)
    print(f'upAxis {UsdGeom.GetStageUpAxis(stage)}, metersPerUnit {UsdGeom.GetStageMetersPerUnit(stage)}')
    cache = UsdGeom.XformCache()
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if a.filter not in path:
            continue
        line = [f'{path} [{prim.GetTypeName()}]']
        if prim.IsA(UsdGeom.Xformable):
            m = cache.GetLocalToWorldTransform(prim)
            t = m.ExtractTranslation()
            line.append(f'pos ({t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f})')
            s = [m.GetRow3(i).GetLength() for i in range(3)]
            if any(abs(x - 1) > 1e-4 for x in s):
                line.append(f'scale ({s[0]:.4f}, {s[1]:.4f}, {s[2]:.4f})')
        for name in ('size', 'radius', 'height', 'axis', 'extent', 'points'):
            attr = prim.GetAttribute(name)
            if attr and attr.HasAuthoredValue():
                v = attr.Get()
                if name == 'points':
                    pts = list(v)
                    lo = [min(q[i] for q in pts) for i in range(3)]
                    hi = [max(q[i] for q in pts) for i in range(3)]
                    v = f'{len(pts)} pts, bbox {[round(x, 4) for x in lo]} .. {[round(x, 4) for x in hi]}'
                line.append(f'{name}={v}')
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            line.append('COLLISION')
            if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                line.append(f'approx={UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()}')
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            line.append('RIGID_BODY')
        if prim.HasAPI(UsdPhysics.MassAPI):
            mass = UsdPhysics.MassAPI(prim).GetMassAttr().Get()
            line.append(f'mass={mass}')
        if prim.HasAPI(UsdPhysics.MaterialAPI):
            mat = UsdPhysics.MaterialAPI(prim)
            line.append(f'friction s={mat.GetStaticFrictionAttr().Get()} d={mat.GetDynamicFrictionAttr().Get()}')
        if prim.IsA(UsdPhysics.Joint):
            line.append(f'joint {prim.GetTypeName()}')
            for name in ('physics:axis', 'physics:localPos0', 'physics:localPos1', 'physics:body0', 'physics:body1'):
                attr = prim.GetAttribute(name) or prim.GetRelationship(name)
                if attr:
                    v = attr.GetTargets() if isinstance(attr, Usd.Relationship) else attr.Get()
                    line.append(f'{name.split(":")[1]}={v}')
        print('  '.join(line))
    sys.stdout.flush()


if __name__ == '__main__':
    main()
