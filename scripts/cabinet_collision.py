"""
cabinet_collision.py
---------------------
Extracts per-link mesh collision geometry from the Sektion cabinet USD
and builds a cuRobo WorldConfig ready for MotionGen.

Usage (standalone helper):
    from cabinet_collision import build_cabinet_world_config
    world_cfg = build_cabinet_world_config(stage, cabinet_prim_path)

Usage (integrated in curobo_stretch.py):
    Replace `dummy_world` with the result of build_cabinet_world_config().
    Re-call motion_gen.update_world(world_cfg) any time a drawer moves.
"""

import numpy as np
import torch
from typing import Optional

# ── cuRobo ──────────────────────────────────────────────────────────────────
from curobo.geom.types import WorldConfig, Mesh, Cuboid
from curobo.types.base import TensorDeviceType

# ── Isaac Sim / USD ──────────────────────────────────────────────────────────
from pxr import Usd, UsdGeom, Gf


# ============================================================================
# Core helper: world-space transform from a UsdPrim
# ============================================================================


def _get_world_xform(prim: "Usd.Prim", time=Usd.TimeCode.Default()) -> np.ndarray:
    """Return a 4x4 world-space transform matrix for *prim*."""
    xformable = UsdGeom.Xformable(prim)
    mat: Gf.Matrix4d = xformable.ComputeLocalToWorldTransform(time)
    return np.array(mat).reshape(4, 4).T  # USD is row-major → transpose


def _mat4_to_curobo_pose(mat: np.ndarray):
    """
    Convert a 4×4 homogeneous matrix to cuRobo pose list
    [x, y, z, qw, qx, qy, qz].
    """
    from scipy.spatial.transform import Rotation

    trans = mat[:3, 3].tolist()
    rot_mat = mat[:3, :3]
    qxyz = Rotation.from_matrix(rot_mat).as_quat()  # [qx, qy, qz, qw]
    pose = trans + [float(qxyz[3]), float(qxyz[0]), float(qxyz[1]), float(qxyz[2])]
    return pose


# ============================================================================
# Mesh extraction
# ============================================================================


def _extract_mesh_from_prim(mesh_prim: "Usd.Prim"):
    """
    Return (vertices Nx3, faces Mx3) for a UsdGeomMesh prim,
    already in world space (vertices are transformed).

    Returns (None, None) if extraction fails.
    """
    usd_mesh = UsdGeom.Mesh(mesh_prim)
    if not usd_mesh:
        return None, None

    t = Usd.TimeCode.Default()
    pts_attr = usd_mesh.GetPointsAttr()
    idx_attr = usd_mesh.GetFaceVertexIndicesAttr()
    cnt_attr = usd_mesh.GetFaceVertexCountsAttr()

    if not (pts_attr.IsValid() and idx_attr.IsValid() and cnt_attr.IsValid()):
        return None, None

    raw_pts = np.array(pts_attr.Get(t), dtype=np.float32)  # (N, 3)
    raw_idx = np.array(idx_attr.Get(t), dtype=np.int32)
    raw_cnt = np.array(cnt_attr.Get(t), dtype=np.int32)

    # ── Apply world transform to vertices ───────────────────────────────────
    mat = _get_world_xform(mesh_prim, t)
    ones = np.ones((raw_pts.shape[0], 1), dtype=np.float32)
    pts_h = np.hstack([raw_pts, ones])  # (N, 4)
    verts_world = (mat @ pts_h.T).T[:, :3]  # (N, 3)

    # ── Triangulate (fan from first vertex of each polygon) ─────────────────
    triangles = []
    offset = 0
    for n in raw_cnt:
        fan_root = raw_idx[offset]
        for k in range(1, n - 1):
            triangles.append([fan_root, raw_idx[offset + k], raw_idx[offset + k + 1]])
        offset += n

    if len(triangles) == 0:
        return None, None

    faces = np.array(triangles, dtype=np.int32)
    return verts_world, faces


# ============================================================================
# Per-link mesh collector
# ============================================================================


def _collect_link_meshes(
    stage: "Usd.Stage",
    cabinet_root_path: str,
):
    """
    Walk the cabinet USD hierarchy.  For every rigid-body link (XForm or
    RigidBody scope) collect all descendant Mesh prims and merge them into
    a single (vertices, faces) pair so each link becomes ONE cuRobo Mesh.

    Returns list of (link_name, vertices Nx3, faces Mx3).
    """
    cabinet_prim = stage.GetPrimAtPath(cabinet_root_path)
    if not cabinet_prim.IsValid():
        raise ValueError(f"No prim found at '{cabinet_root_path}'")

    results = []

    # Traverse direct children as "links"
    for child in cabinet_prim.GetChildren():
        link_name = child.GetName()
        all_verts = []
        all_faces = []
        vert_offset = 0

        # Collect every Mesh under this link (recursive)
        for desc in Usd.PrimRange(child):
            if desc.GetTypeName() == "Mesh":
                verts, faces = _extract_mesh_from_prim(desc)
                if verts is None:
                    continue
                all_verts.append(verts)
                all_faces.append(faces + vert_offset)
                vert_offset += len(verts)

        if not all_verts:
            continue

        merged_verts = np.vstack(all_verts).astype(np.float32)
        merged_faces = np.vstack(all_faces).astype(np.int32)
        results.append((link_name, merged_verts, merged_faces))
        print(
            f"  [cabinet_collision] link '{link_name}': "
            f"{len(merged_verts)} verts, {len(merged_faces)} tris"
        )

    return results


# ============================================================================
# Public API
# ============================================================================


def build_cabinet_world_config(
    stage: "Usd.Stage",
    cabinet_prim_path: str = "/World/envs/env_0/Cabinet",
    add_floor: bool = True,
    floor_z: float = 0.0,
) -> WorldConfig:
    """
    Build a cuRobo WorldConfig containing one Mesh per cabinet link.

    Parameters
    ----------
    stage              : USD stage (from omni.usd.get_context().get_stage())
    cabinet_prim_path  : absolute USD path to the Cabinet root prim
    add_floor          : whether to include a thin floor Cuboid as ground plane
    floor_z            : Z height of the floor surface

    Returns
    -------
    WorldConfig ready to pass to MotionGenConfig.load_from_robot_config()
    """
    print(f"[cabinet_collision] Extracting meshes from '{cabinet_prim_path}' ...")
    link_data = _collect_link_meshes(stage, cabinet_prim_path)

    mesh_objects = []
    for link_name, verts, faces in link_data:
        # cuRobo Mesh expects pose as identity when vertices are already in
        # world space (which they are because we applied the world transform
        # during extraction).
        mesh_obj = Mesh(
            name=f"cabinet_{link_name}",
            pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # world-space identity
            vertices=verts.tolist(),
            faces=faces.tolist(),
        )
        mesh_objects.append(mesh_obj)

    # Optional ground-plane cuboid ────────────────────────────────────────────
    cuboid_objects = []
    if add_floor:
        cuboid_objects.append(
            Cuboid(
                name="floor",
                pose=[0.0, 0.0, floor_z - 0.025, 1.0, 0.0, 0.0, 0.0],
                dims=[10.0, 10.0, 0.05],
            )
        )

    world_cfg = WorldConfig(mesh=mesh_objects, cuboid=cuboid_objects)
    print(
        f"[cabinet_collision] WorldConfig built: "
        f"{len(mesh_objects)} mesh(es), {len(cuboid_objects)} cuboid(s)"
    )
    return world_cfg


# ============================================================================
# Dynamic update helpers (call when drawers/doors move)
# ============================================================================


def update_cabinet_world(
    motion_gen,
    stage: "Usd.Stage",
    cabinet_prim_path: str = "/World/envs/env_0/Cabinet",
    add_floor: bool = True,
    floor_z: float = 0.0,
) -> None:
    """
    Re-extract cabinet meshes from the current stage pose and push the
    updated WorldConfig into a live MotionGen instance.

    Call this whenever a drawer or door changes position (e.g. every N steps).
    """
    world_cfg = build_cabinet_world_config(
        stage, cabinet_prim_path, add_floor=add_floor, floor_z=floor_z
    )
    motion_gen.update_world(world_cfg)
    print("[cabinet_collision] MotionGen world updated.")


# ============================================================================
# Fallback: bounding-box approximation (no USD parsing needed)
# ============================================================================


def build_cabinet_bbox_world_config(
    cabinet_pos: list,
    cabinet_quat: list,
    cabinet_dims: Optional[list] = None,
    drawer_open_fraction: float = 0.0,
) -> WorldConfig:
    """
    Lightweight fallback that approximates the Sektion cabinet with a small
    set of Cuboids derived from measured dimensions.

    Parameters
    ----------
    cabinet_pos          : [x, y, z] world position of the cabinet base
    cabinet_quat         : [qw, qx, qy, qz] world orientation
    cabinet_dims         : override default [W, D, H] = [0.9, 0.6, 0.72] m
    drawer_open_fraction : 0 = closed, 1 = fully open (shifts drawer cuboids)

    Returns
    -------
    WorldConfig with Cuboid approximation of the cabinet.

    Sektion cabinet (IKEA-style) approximate geometry
    -------------------------------------------------
    Body shell  : 0.90 × 0.60 × 0.72  m  (W × D × H)
    Back panel  : full width/height, 0.02 m thick
    Side panels : 0.02 × D × H
    Top/bottom  : W × D × 0.02
    Drawer top  : 0.86 × 0.55 × 0.18  (half-height slot, slides out in +Y)
    Drawer bot  : 0.86 × 0.55 × 0.18
    """
    W, D, H = cabinet_dims or [0.90, 0.60, 0.72]
    t = 0.02  # panel thickness
    x, y, z = cabinet_pos
    qw, qx, qy, qz = cabinet_quat

    drawer_stroke = 0.35  # max travel in local +Y
    d_offset = drawer_open_fraction * drawer_stroke

    cuboids = [
        # Back panel
        Cuboid(
            "cab_back",
            pose=[x, y - D / 2 + t / 2, z + H / 2, qw, qx, qy, qz],
            dims=[W, t, H],
        ),
        # Left side
        Cuboid(
            "cab_left",
            pose=[x - W / 2 + t / 2, y, z + H / 2, qw, qx, qy, qz],
            dims=[t, D, H],
        ),
        # Right side
        Cuboid(
            "cab_right",
            pose=[x + W / 2 - t / 2, y, z + H / 2, qw, qx, qy, qz],
            dims=[t, D, H],
        ),
        # Bottom panel
        Cuboid(
            "cab_bottom", pose=[x, y, z + t / 2, qw, qx, qy, qz], dims=[W - 2 * t, D, t]
        ),
        # Top panel
        Cuboid(
            "cab_top",
            pose=[x, y, z + H - t / 2, qw, qx, qy, qz],
            dims=[W - 2 * t, D, t],
        ),
        # Middle shelf
        Cuboid(
            "cab_shelf", pose=[x, y, z + H / 2, qw, qx, qy, qz], dims=[W - 2 * t, D, t]
        ),
        # Upper drawer body
        Cuboid(
            "cab_drawer_top",
            pose=[x, y + D / 4 + d_offset, z + H * 0.75, qw, qx, qy, qz],
            dims=[W - 2 * t - 0.02, D / 2, H * 0.22],
        ),
        # Lower drawer body
        Cuboid(
            "cab_drawer_bot",
            pose=[x, y + D / 4 + d_offset, z + H * 0.28, qw, qx, qy, qz],
            dims=[W - 2 * t - 0.02, D / 2, H * 0.44],
        ),
    ]
    print(
        f"[cabinet_collision] Bbox WorldConfig: {len(cuboids)} cuboids "
        f"(drawer open {drawer_open_fraction*100:.0f}%)"
    )
    return WorldConfig(cuboid=cuboids)


# ============================================================================
# Integration snippet (copy into curobo_stretch.py)
# ============================================================================
INTEGRATION_EXAMPLE = """
# ── In your imports ──────────────────────────────────────────────────────────
from cabinet_collision import build_cabinet_world_config, update_cabinet_world
import omni.usd

# ── Replace dummy_world construction ─────────────────────────────────────────
stage = omni.usd.get_context().get_stage()
world_cfg = build_cabinet_world_config(
    stage,
    cabinet_prim_path="/World/envs/env_0/Cabinet",
    add_floor=True,
    floor_z=0.0,
)

# ── Pass world_cfg to MotionGenConfig (instead of dummy_world) ───────────────
motion_gen_config = MotionGenConfig.load_from_robot_config(
    robot_cfg,
    world_cfg,                         # <-- was dummy_world
    tensor_args,
    collision_checker_type=CollisionCheckerType.MESH,
    ...
)

# ── Update world when a drawer opens (inside the sim loop) ───────────────────
if step_count % 30 == 0 and not phase_three_done:
    update_cabinet_world(motion_gen, stage,
                         cabinet_prim_path="/World/envs/env_0/Cabinet")
"""

if __name__ == "__main__":
    print("Integration example:")
    print(INTEGRATION_EXAMPLE)
