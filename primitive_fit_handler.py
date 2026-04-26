"""
RunPod Serverless Handler — Primitive Fitting Pipeline
OBJ/GLB mesh → clean CAD geometry (STEP + IFC + Pascal nodes)

Advanced pipeline that:
  1. Loads mesh and samples dense point cloud
  2. Fits geometric primitives (planes, cylinders, boxes) via RANSAC
  3. Classifies primitives as architectural elements (wall, slab, door, window)
  4. Snaps walls to grid, aligns angles, cleans up geometry
  5. Exports as STEP (parametric CAD), IFC (BIM), and Pascal Editor JSON

This is the "long-term" pipeline — more accurate than Cloud2BIM,
specifically tuned for Pascal Editor's node schema.
"""
import os
import sys
import base64
import json
import math
import tempfile
import traceback
from io import BytesIO
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import runpod


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DetectedPlane:
    """A detected planar surface from the point cloud."""
    normal: np.ndarray          # unit normal
    offset: float               # distance from origin
    inlier_points: np.ndarray   # (N, 3) points
    centroid: np.ndarray        # (3,) center
    extent: tuple               # (width, height) of the bounding rect
    classification: str = "unknown"  # wall, slab, ceiling, etc.


@dataclass
class DetectedWall:
    """A wall extracted from vertical planes."""
    start: list    # [x, z]
    end: list      # [x, z]
    thickness: float
    height: float
    elevation: float
    normal_xz: list
    openings: list = field(default_factory=list)


@dataclass
class DetectedOpening:
    """A door or window opening in a wall."""
    type: str             # "door" or "window"
    position: list        # [x, y, z] world coordinates
    width: float
    height: float
    wall_u: float         # position along wall (0..1)
    elevation: float      # from floor


@dataclass
class DetectedSlab:
    """A floor/ceiling slab."""
    polygon: list         # [[x, z], ...] boundary
    elevation: float
    thickness: float = 0.2
    is_ceiling: bool = False


# ---------------------------------------------------------------------------
# Mesh loading & point cloud extraction
# ---------------------------------------------------------------------------

def load_mesh_to_pointcloud(data_bytes: bytes, fmt: str, num_samples=800_000):
    """Load mesh and extract dense point cloud with normals."""
    import open3d as o3d
    import trimesh as tm

    tmp = tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False)
    tmp.write(data_bytes)
    tmp.close()

    # Try trimesh first (better format support)
    try:
        scene_or_mesh = tm.load(tmp.name, force="mesh")
        if hasattr(scene_or_mesh, "dump"):
            mesh = tm.util.concatenate(scene_or_mesh.dump())
        else:
            mesh = scene_or_mesh

        # Sample points with normals
        points, face_indices = tm.sample.sample_surface(mesh, num_samples)
        normals = mesh.face_normals[face_indices]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.normals = o3d.utility.Vector3dVector(normals)

        if mesh.visual and hasattr(mesh.visual, "vertex_colors"):
            try:
                colors = mesh.visual.interpolate(points, face_indices)[:, :3] / 255.0
                pcd.colors = o3d.utility.Vector3dVector(colors)
            except Exception:
                pass
    except Exception:
        # Fallback to Open3D
        o3d_mesh = o3d.io.read_triangle_mesh(tmp.name)
        if len(o3d_mesh.vertices) == 0:
            raise ValueError(f"Could not load mesh from {fmt}")
        pcd = o3d_mesh.sample_points_uniformly(num_samples)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )

    os.unlink(tmp.name)
    return pcd


# ---------------------------------------------------------------------------
# Primitive fitting
# ---------------------------------------------------------------------------

def fit_planes_ransac(pcd, distance_threshold=0.03, min_points=300, max_planes=80):
    """Iterative RANSAC plane fitting — extract all significant planes."""
    import open3d as o3d

    points = np.asarray(pcd.points)
    remaining_idx = list(range(len(points)))
    planes = []

    for _ in range(max_planes):
        if len(remaining_idx) < min_points:
            break

        sub_pcd = pcd.select_by_index(remaining_idx)
        try:
            plane_model, inliers = sub_pcd.segment_plane(
                distance_threshold=distance_threshold,
                ransac_n=3,
                num_iterations=3000,
            )
        except Exception:
            break

        if len(inliers) < min_points:
            break

        a, b, c, d = plane_model
        normal = np.array([a, b, c])
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-6:
            break
        normal /= norm_len

        inlier_global = [remaining_idx[i] for i in inliers]
        inlier_pts = points[inlier_global]
        centroid = inlier_pts.mean(axis=0)

        # Compute oriented bounding rect on the plane
        extent = _compute_plane_extent(inlier_pts, normal)

        plane = DetectedPlane(
            normal=normal,
            offset=-d / norm_len,
            inlier_points=inlier_pts,
            centroid=centroid,
            extent=extent,
        )

        # Classify: horizontal (slab/ceiling) vs vertical (wall)
        verticality = abs(normal[1])
        if verticality > 0.85:
            if centroid[1] < np.median(points[:, 1]):
                plane.classification = "slab"
            else:
                plane.classification = "ceiling"
        elif verticality < 0.3:
            plane.classification = "wall"
        else:
            plane.classification = "unknown"

        planes.append(plane)

        # Remove used inliers
        inlier_set = set(inliers)
        remaining_idx = [
            remaining_idx[i] for i in range(len(remaining_idx)) if i not in inlier_set
        ]

    return planes


def _compute_plane_extent(points, normal):
    """Compute width/height of a planar point set."""
    # Create local 2D coordinate system on the plane
    if abs(normal[1]) > 0.9:
        # Horizontal plane — use X and Z
        u = np.array([1, 0, 0])
    else:
        up = np.array([0, 1, 0])
        u = np.cross(normal, up)
        u /= np.linalg.norm(u) + 1e-9

    v = np.cross(normal, u)
    v /= np.linalg.norm(v) + 1e-9

    centroid = points.mean(axis=0)
    centered = points - centroid

    proj_u = centered @ u
    proj_v = centered @ v

    width = proj_u.max() - proj_u.min()
    height = proj_v.max() - proj_v.min()

    return (float(width), float(height))


# ---------------------------------------------------------------------------
# Wall extraction & cleanup
# ---------------------------------------------------------------------------

def extract_walls_from_planes(wall_planes, snap_angle=5.0, snap_grid=0.05):
    """
    Convert vertical plane detections into clean wall segments.
    Snaps to common angles and optional grid.
    """
    walls = []

    for plane in wall_planes:
        pts = plane.inlier_points

        # Project to XZ plane
        xz = pts[:, [0, 2]]
        y_vals = pts[:, 1]

        if len(xz) < 10:
            continue

        # PCA to find wall direction
        mean = xz.mean(axis=0)
        centered = xz - mean
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Principal direction (along wall)
        principal = eigenvectors[:, -1]

        # Snap angle to nearest 0°, 45°, 90°, 135°
        angle = math.atan2(principal[1], principal[0])
        angle_deg = math.degrees(angle)
        snap_angles = [0, 45, 90, 135, 180, -45, -90, -135]
        closest = min(snap_angles, key=lambda a: abs(angle_deg - a))
        if abs(angle_deg - closest) < snap_angle:
            angle = math.radians(closest)
            principal = np.array([math.cos(angle), math.sin(angle)])

        # Project onto principal axis to find extents
        proj = centered @ principal
        t_min, t_max = proj.min(), proj.max()

        start_xz = mean + t_min * principal
        end_xz = mean + t_max * principal

        # Snap to grid
        if snap_grid > 0:
            start_xz = np.round(start_xz / snap_grid) * snap_grid
            end_xz = np.round(end_xz / snap_grid) * snap_grid

        # Thickness from perpendicular spread
        perp = eigenvectors[:, 0]
        perp_proj = centered @ perp
        thickness = float(perp_proj.max() - perp_proj.min())
        thickness = max(0.1, min(thickness, 0.6))
        # Snap thickness to common values
        common_thicknesses = [0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
        thickness = min(common_thicknesses, key=lambda t: abs(t - thickness))

        wall = DetectedWall(
            start=start_xz.tolist(),
            end=end_xz.tolist(),
            thickness=thickness,
            height=float(y_vals.max() - y_vals.min()),
            elevation=float(y_vals.min()),
            normal_xz=[float(plane.normal[0]), float(plane.normal[2])],
        )
        walls.append(wall)

    return walls


def merge_walls(walls, angle_threshold=5.0, distance_threshold=0.3, gap_threshold=0.5):
    """Merge collinear and overlapping wall segments."""
    if len(walls) < 2:
        return walls

    merged = []
    used = set()

    for i, w1 in enumerate(walls):
        if i in used:
            continue

        s1 = np.array(w1.start)
        e1 = np.array(w1.end)
        d1 = e1 - s1
        l1 = np.linalg.norm(d1)
        if l1 < 0.01:
            continue
        d1n = d1 / l1

        group = [w1]
        used.add(i)

        for j, w2 in enumerate(walls):
            if j in used:
                continue

            s2 = np.array(w2.start)
            e2 = np.array(w2.end)
            d2 = e2 - s2
            l2 = np.linalg.norm(d2)
            if l2 < 0.01:
                continue
            d2n = d2 / l2

            # Check alignment
            cos_a = abs(np.dot(d1n, d2n))
            if cos_a < math.cos(math.radians(angle_threshold)):
                continue

            # Check perpendicular distance
            mid2 = (s2 + e2) / 2
            v = mid2 - s1
            proj = np.dot(v, d1n) * d1n
            perp = np.linalg.norm(v - proj)

            if perp < distance_threshold:
                # Check gap distance
                projs = np.dot(np.array([s2, e2]) - s1, d1n)
                existing_projs = []
                for w in group:
                    for pt in [w.start, w.end]:
                        existing_projs.append(np.dot(np.array(pt) - s1, d1n))

                min_existing = min(existing_projs)
                max_existing = max(existing_projs)
                min_new = min(projs)
                max_new = max(projs)

                gap = max(min_new - max_existing, min_existing - max_new)
                if gap < gap_threshold:
                    group.append(w2)
                    used.add(j)

        # Merge group
        all_pts = []
        for w in group:
            all_pts.extend([np.array(w.start), np.array(w.end)])
        all_pts = np.array(all_pts)
        projs = all_pts @ d1n
        idx_min = projs.argmin()
        idx_max = projs.argmax()

        avg_thickness = np.mean([w.thickness for w in group])
        max_height = max(w.height for w in group)
        min_elev = min(w.elevation for w in group)

        merged.append(DetectedWall(
            start=all_pts[idx_min].tolist(),
            end=all_pts[idx_max].tolist(),
            thickness=float(avg_thickness),
            height=float(max_height),
            elevation=float(min_elev),
            normal_xz=group[0].normal_xz,
        ))

    return merged


# ---------------------------------------------------------------------------
# Opening detection (doors & windows)
# ---------------------------------------------------------------------------

def detect_openings(wall, all_points, floor_elevation):
    """
    Detect openings in a wall by analyzing point density in wall-local 2D space.
    Uses gap analysis with adaptive thresholding.
    """
    from scipy import ndimage

    start = np.array(wall.start)
    end = np.array(wall.end)
    wall_dir = end - start
    wall_length = np.linalg.norm(wall_dir)
    if wall_length < 0.5:
        return []

    wall_dir_n = wall_dir / wall_length
    wall_normal = np.array([-wall_dir_n[1], wall_dir_n[0]])

    # Filter points near this wall (within thickness + margin)
    margin = wall.thickness + 0.1
    xz = all_points[:, [0, 2]]
    y = all_points[:, 1]

    # Project onto wall coordinate system
    rel = xz - start
    u = rel @ wall_dir_n        # along wall
    v = rel @ wall_normal       # perpendicular to wall

    # Filter: within wall bounds
    mask = (
        (u >= -0.1) & (u <= wall_length + 0.1) &
        (abs(v) <= margin) &
        (y >= floor_elevation - 0.1) &
        (y <= floor_elevation + wall.height + 0.1)
    )

    wall_pts_u = u[mask]
    wall_pts_y = y[mask] - floor_elevation

    if len(wall_pts_u) < 50:
        return []

    # Create 2D density histogram
    n_u = max(15, int(wall_length / 0.1))
    n_v = max(12, int(wall.height / 0.1))

    hist, u_edges, v_edges = np.histogram2d(
        wall_pts_u, wall_pts_y, bins=[n_u, n_v],
        range=[[0, wall_length], [0, wall.height]]
    )

    if hist.max() == 0:
        return []

    # Adaptive threshold: cells with significantly less density than neighbors
    density = hist / hist.max()
    avg_density = np.mean(density[density > 0]) if np.any(density > 0) else 0
    threshold = min(0.2, avg_density * 0.3)

    empty = density < threshold
    labeled, num_features = ndimage.label(empty)

    openings = []
    for label_id in range(1, num_features + 1):
        region = np.where(labeled == label_id)
        u_min_idx, u_max_idx = region[0].min(), region[0].max()
        v_min_idx, v_max_idx = region[1].min(), region[1].max()

        u_min = u_edges[u_min_idx]
        u_max = u_edges[u_max_idx + 1]
        v_min = v_edges[v_min_idx]
        v_max = v_edges[v_max_idx + 1]

        width = u_max - u_min
        height = v_max - v_min
        area = width * height

        # Filter: reasonable opening sizes
        if width < 0.4 or width > 4.5:
            continue
        if height < 0.5 or height > 3.5:
            continue
        if area < 0.4:
            continue

        # Classify
        is_door = v_min < 0.25 and height > 1.5
        opening_type = "door" if is_door else "window"

        u_center = (u_min + u_max) / 2
        v_center = (v_min + v_max) / 2
        pos_xz = start + u_center * wall_dir_n

        # Snap common sizes
        if opening_type == "door":
            width = _snap_dimension(width, [0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 1.8, 2.0])
            height = _snap_dimension(height, [2.0, 2.1, 2.2, 2.4])
        else:
            width = _snap_dimension(width, [0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.4])
            height = _snap_dimension(height, [0.6, 0.8, 1.0, 1.2, 1.5, 1.8])

        openings.append(DetectedOpening(
            type=opening_type,
            position=[float(pos_xz[0]), float(floor_elevation + v_center), float(pos_xz[1])],
            width=float(width),
            height=float(height),
            wall_u=float(u_center / wall_length),
            elevation=float(v_min),
        ))

    return openings


def _snap_dimension(val, common_values, tolerance=0.15):
    """Snap a dimension to the nearest common value if close enough."""
    closest = min(common_values, key=lambda v: abs(v - val))
    if abs(closest - val) < tolerance:
        return closest
    return round(val, 2)


# ---------------------------------------------------------------------------
# Slab extraction
# ---------------------------------------------------------------------------

def extract_slabs(slab_planes, walls):
    """
    Create slab polygons from horizontal planes, trimmed by wall boundaries.
    """
    from scipy.spatial import ConvexHull

    slabs = []
    for plane in slab_planes:
        pts = plane.inlier_points[:, [0, 2]]  # XZ projection

        if len(pts) < 4:
            continue

        try:
            hull = ConvexHull(pts)
            polygon = pts[hull.vertices].tolist()
            # Simplify polygon if too many points
            if len(polygon) > 20:
                polygon = _simplify_polygon(polygon, tolerance=0.1)

            slabs.append(DetectedSlab(
                polygon=polygon,
                elevation=float(plane.centroid[1]),
                is_ceiling=plane.classification == "ceiling",
            ))
        except Exception:
            continue

    return slabs


def _simplify_polygon(polygon, tolerance=0.1):
    """Simplify polygon using Ramer-Douglas-Peucker algorithm."""
    if len(polygon) < 3:
        return polygon

    pts = np.array(polygon)
    n = len(pts)
    keep = [True] * n

    def rdp(start, end):
        if end - start < 2:
            return
        line_start = pts[start]
        line_end = pts[end]
        line_dir = line_end - line_start
        line_len = np.linalg.norm(line_dir)
        if line_len < 1e-9:
            return
        line_dir /= line_len

        max_dist = 0
        max_idx = start
        for i in range(start + 1, end):
            v = pts[i] - line_start
            proj = np.dot(v, line_dir) * line_dir
            dist = np.linalg.norm(v - proj)
            if dist > max_dist:
                max_dist = dist
                max_idx = i

        if max_dist < tolerance:
            for i in range(start + 1, end):
                keep[i] = False
        else:
            rdp(start, max_idx)
            rdp(max_idx, end)

    rdp(0, n - 1)
    return [polygon[i] for i in range(n) if keep[i]]


# ---------------------------------------------------------------------------
# STEP CAD export via CadQuery
# ---------------------------------------------------------------------------

def export_step(walls, slabs, openings_map, output_path):
    """
    Export clean parametric CAD geometry as STEP file using CadQuery.
    Falls back to OCC direct if CadQuery unavailable.
    """
    try:
        import cadquery as cq

        assembly = cq.Assembly()

        for i, wall in enumerate(walls):
            sx, sz = wall.start
            ex, ez = wall.end
            dx = ex - sx
            dz = ez - sz
            length = math.sqrt(dx * dx + dz * dz)
            angle = math.degrees(math.atan2(dz, dx))

            # Create wall as extruded rectangle
            wall_solid = (
                cq.Workplane("XY")
                .rect(length, wall.thickness)
                .extrude(wall.height)
            )

            # Cut openings
            for opening in wall.openings:
                u = opening.wall_u * length
                wall_solid = (
                    wall_solid
                    .faces(">Z" if opening.elevation > 0 else "<Z")
                    .workplane(offset=opening.elevation)
                    .center(u - length / 2, 0)
                    .rect(opening.width, wall.thickness + 0.02)
                    .cutBlind(opening.height)
                )

            # Position and rotate
            assembly.add(
                wall_solid,
                name=f"Wall_{i}",
                loc=cq.Location(
                    cq.Vector(sx, sz, wall.elevation),
                    cq.Vector(0, 0, 1),
                    angle,
                ),
            )

        for i, slab in enumerate(slabs):
            if len(slab.polygon) < 3:
                continue
            # Create slab as extruded polygon
            pts = [(p[0], p[1]) for p in slab.polygon]
            slab_solid = (
                cq.Workplane("XY")
                .polyline(pts).close()
                .extrude(slab.thickness)
            )
            assembly.add(
                slab_solid,
                name=f"Slab_{i}",
                loc=cq.Location(cq.Vector(0, 0, slab.elevation)),
            )

        assembly.save(output_path)
        return True

    except ImportError:
        print("[primitive-fit] CadQuery not available, skipping STEP export", flush=True)
        return False


# ---------------------------------------------------------------------------
# IFC export
# ---------------------------------------------------------------------------

def export_ifc(walls, slabs, output_path):
    """Export as IFC BIM model."""
    import ifcopenshell
    import ifcopenshell.api as api

    ifc = api.run("project.create_file")
    project = api.run("root.create_entity", ifc, ifc_class="IfcProject", name="Primitive Fit Import")
    api.run("unit.assign_unit", ifc)

    ctx = api.run("context.add_context", ifc, context_type="Model")
    body = api.run(
        "context.add_context", ifc,
        context_type="Model", context_identifier="Body",
        target_view="MODEL_VIEW", parent=ctx,
    )

    site = api.run("root.create_entity", ifc, ifc_class="IfcSite", name="Site")
    api.run("aggregate.assign_object", ifc, products=[site], relating_object=project)
    building = api.run("root.create_entity", ifc, ifc_class="IfcBuilding", name="Building")
    api.run("aggregate.assign_object", ifc, products=[building], relating_object=site)

    # Group by elevation into storeys
    wall_elevations = [w.elevation for w in walls]
    slab_elevations = [s.elevation for s in slabs if not s.is_ceiling]

    storey_elevations = sorted(set(
        [round(e, 1) for e in slab_elevations] or
        [round(e, 1) for e in wall_elevations] or
        [0.0]
    ))

    for si, elev in enumerate(storey_elevations):
        storey = api.run(
            "root.create_entity", ifc,
            ifc_class="IfcBuildingStorey", name=f"Level {si}",
        )
        storey.Elevation = elev
        api.run("aggregate.assign_object", ifc, products=[storey], relating_object=building)

        # Find walls for this storey
        next_elev = storey_elevations[si + 1] if si + 1 < len(storey_elevations) else elev + 10
        storey_walls = [
            w for w in walls
            if elev - 0.5 <= w.elevation <= next_elev - 0.5
        ]

        for wi, wall in enumerate(storey_walls):
            ifc_wall = api.run(
                "root.create_entity", ifc,
                ifc_class="IfcWall", name=f"Wall L{si}-{wi}",
            )
            api.run("spatial.assign_container", ifc, products=[ifc_wall], relating_structure=storey)

            # Add wall geometry
            sx, sz = wall.start
            ex, ez = wall.end
            _add_wall_ifc_geometry(ifc, ifc_wall, body, sx, sz, ex, ez,
                                   wall.thickness, wall.height, elev)

            # Add openings
            for oi, opening in enumerate(wall.openings):
                ifc_opening = api.run(
                    "root.create_entity", ifc,
                    ifc_class="IfcOpeningElement",
                    name=f"Opening L{si}-W{wi}-{oi}",
                )
                api.run("void.add_opening", ifc, opening_element=ifc_opening, element=ifc_wall)

                if opening.type == "door":
                    el = api.run("root.create_entity", ifc, ifc_class="IfcDoor",
                                 name=f"Door L{si}-W{wi}-{oi}")
                else:
                    el = api.run("root.create_entity", ifc, ifc_class="IfcWindow",
                                 name=f"Window L{si}-W{wi}-{oi}")
                api.run("void.add_filling", ifc, opening=ifc_opening, element=el)

        # Add slabs
        storey_slabs = [
            s for s in slabs
            if not s.is_ceiling and abs(s.elevation - elev) < 1.0
        ]
        for sbi, slab in enumerate(storey_slabs):
            ifc_slab = api.run(
                "root.create_entity", ifc,
                ifc_class="IfcSlab", name=f"Slab L{si}-{sbi}",
            )
            api.run("spatial.assign_container", ifc, products=[ifc_slab], relating_structure=storey)

    ifc.write(output_path)


def _add_wall_ifc_geometry(ifc, wall_element, context, sx, sz, ex, ez, thickness, height, elevation):
    """Add extruded area solid geometry to an IFC wall element."""
    dx = ex - sx
    dz = ez - sz
    length = math.sqrt(dx * dx + dz * dz)
    angle = math.atan2(dz, dx)
    half_t = thickness / 2

    points = [
        ifc.createIfcCartesianPoint([0.0, -half_t]),
        ifc.createIfcCartesianPoint([length, -half_t]),
        ifc.createIfcCartesianPoint([length, half_t]),
        ifc.createIfcCartesianPoint([0.0, half_t]),
    ]
    polyline = ifc.createIfcPolyline(points + [points[0]])
    profile = ifc.createIfcArbitraryClosedProfileDef("AREA", None, polyline)
    direction = ifc.createIfcDirection([0.0, 0.0, 1.0])
    solid = ifc.createIfcExtrudedAreaSolid(profile, None, direction, height)

    origin = ifc.createIfcCartesianPoint([sx, sz, elevation])
    z_dir = ifc.createIfcDirection([0.0, 0.0, 1.0])
    x_dir = ifc.createIfcDirection([math.cos(angle), math.sin(angle), 0.0])
    axis2 = ifc.createIfcAxis2Placement3D(origin, z_dir, x_dir)
    local_placement = ifc.createIfcLocalPlacement(None, axis2)

    shape_rep = ifc.createIfcShapeRepresentation(context, "Body", "SweptSolid", [solid])
    product_shape = ifc.createIfcProductDefinitionShape(None, None, [shape_rep])

    wall_element.Representation = product_shape
    wall_element.ObjectPlacement = local_placement


# ---------------------------------------------------------------------------
# Pascal Editor node generation
# ---------------------------------------------------------------------------

def generate_pascal_nodes(walls, slabs, storey_height=2.8):
    """Convert detected geometry to Pascal Editor scene nodes."""
    import string
    import random

    def make_id(prefix):
        chars = string.ascii_lowercase + string.digits
        suffix = "".join(random.choices(chars, k=16))
        return f"{prefix}_{suffix}"

    nodes = {}

    # Group walls/slabs by elevation into levels
    all_elevations = sorted(set(
        [round(w.elevation, 1) for w in walls] +
        [round(s.elevation, 1) for s in slabs if not s.is_ceiling]
    )) or [0.0]

    site_id = make_id("site")
    building_id = make_id("building")
    level_ids = []

    for li, elev in enumerate(all_elevations):
        level_id = make_id("level")
        level_ids.append(level_id)

        next_elev = all_elevations[li + 1] if li + 1 < len(all_elevations) else elev + 10

        # Walls for this level
        level_walls = [w for w in walls if elev - 0.5 <= w.elevation <= next_elev - 0.5]
        wall_ids = []
        for wi, wall in enumerate(level_walls):
            wall_id = make_id("wall")
            wall_ids.append(wall_id)

            children = []
            wall_node = {
                "object": "node",
                "id": wall_id,
                "type": "wall",
                "name": f"Wall {wi}",
                "parentId": level_id,
                "visible": True,
                "metadata": {},
                "children": children,
                "start": wall.start,
                "end": wall.end,
                "thickness": wall.thickness,
                "height": wall.height,
                "frontSide": "unknown",
                "backSide": "unknown",
            }
            nodes[wall_id] = wall_node

            # Openings
            for oi, opening in enumerate(wall.openings):
                if opening.type == "door":
                    oid = make_id("door")
                    nodes[oid] = {
                        "object": "node",
                        "id": oid,
                        "type": "door",
                        "name": f"Door {oi}",
                        "parentId": wall_id,
                        "visible": True,
                        "metadata": {},
                        "wallId": wall_id,
                        "position": opening.position,
                        "rotation": [0, 0, 0],
                        "width": opening.width,
                        "height": opening.height,
                        "frameThickness": 0.05,
                        "frameDepth": 0.07,
                        "threshold": True,
                        "thresholdHeight": 0.02,
                        "hingesSide": "left",
                        "swingDirection": "inward",
                        "segments": [
                            {"type": "panel", "heightRatio": 0.4, "columnRatios": [1],
                             "dividerThickness": 0.03, "panelDepth": 0.01, "panelInset": 0.04},
                            {"type": "panel", "heightRatio": 0.6, "columnRatios": [1],
                             "dividerThickness": 0.03, "panelDepth": 0.01, "panelInset": 0.04},
                        ],
                        "handle": True,
                        "handleHeight": 1.05,
                        "handleSide": "right",
                        "contentPadding": [0.04, 0.04],
                        "doorCloser": False,
                        "panicBar": False,
                        "panicBarHeight": 1.0,
                    }
                else:
                    oid = make_id("window")
                    nodes[oid] = {
                        "object": "node",
                        "id": oid,
                        "type": "window",
                        "name": f"Window {oi}",
                        "parentId": wall_id,
                        "visible": True,
                        "metadata": {},
                        "wallId": wall_id,
                        "position": opening.position,
                        "rotation": [0, 0, 0],
                        "width": opening.width,
                        "height": opening.height,
                        "frameThickness": 0.05,
                        "frameDepth": 0.07,
                        "columnRatios": [1],
                        "rowRatios": [1],
                        "columnDividerThickness": 0.03,
                        "rowDividerThickness": 0.03,
                        "sill": True,
                        "sillDepth": 0.08,
                        "sillThickness": 0.03,
                    }

        # Slabs
        level_slabs = [s for s in slabs if not s.is_ceiling and abs(s.elevation - elev) < 1.0]
        slab_ids = []
        for sbi, slab in enumerate(level_slabs):
            slab_id = make_id("slab")
            slab_ids.append(slab_id)
            nodes[slab_id] = {
                "object": "node",
                "id": slab_id,
                "type": "slab",
                "name": f"Slab L{li}",
                "parentId": level_id,
                "visible": True,
                "metadata": {},
                "polygon": slab.polygon,
                "holes": [],
                "elevation": 0.05,
            }

        nodes[level_id] = {
            "object": "node",
            "id": level_id,
            "type": "level",
            "name": f"Level {li}",
            "parentId": building_id,
            "visible": True,
            "metadata": {},
            "children": wall_ids + slab_ids,
            "level": li,
        }

    nodes[building_id] = {
        "object": "node",
        "id": building_id,
        "type": "building",
        "name": "Imported Building",
        "parentId": site_id,
        "visible": True,
        "metadata": {},
        "children": level_ids,
        "position": [0, 0, 0],
        "rotation": [0, 0, 0],
    }

    nodes[site_id] = {
        "object": "node",
        "id": site_id,
        "type": "site",
        "name": "Imported Site",
        "parentId": None,
        "visible": True,
        "metadata": {},
        "children": [building_id],
    }

    return nodes


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(pcd, params=None):
    """Run the full primitive fitting pipeline."""
    params = params or {}

    distance_thresh = params.get("distance_threshold", 0.03)
    min_points = params.get("min_wall_points", 300)
    snap_angle = params.get("snap_angle", 5.0)
    snap_grid = params.get("snap_grid", 0.05)

    print("[primitive-fit] Fitting planes via RANSAC...", flush=True)
    planes = fit_planes_ransac(pcd, distance_thresh, min_points)
    print(f"[primitive-fit] Detected {len(planes)} planes", flush=True)

    wall_planes = [p for p in planes if p.classification == "wall"]
    slab_planes = [p for p in planes if p.classification in ("slab", "ceiling")]
    print(f"[primitive-fit] Walls: {len(wall_planes)}, Slabs/Ceilings: {len(slab_planes)}", flush=True)

    # Extract walls
    print("[primitive-fit] Extracting walls...", flush=True)
    walls = extract_walls_from_planes(wall_planes, snap_angle, snap_grid)
    print(f"[primitive-fit] Raw walls: {len(walls)}", flush=True)

    walls = merge_walls(walls)
    print(f"[primitive-fit] Merged walls: {len(walls)}", flush=True)

    # Extract slabs
    print("[primitive-fit] Extracting slabs...", flush=True)
    slabs = extract_slabs(slab_planes, walls)
    print(f"[primitive-fit] Slabs: {len(slabs)}", flush=True)

    # Detect openings
    all_points = np.asarray(pcd.points)
    print("[primitive-fit] Detecting openings...", flush=True)
    for wall in walls:
        floor_elev = wall.elevation
        wall.openings = detect_openings(wall, all_points, floor_elev)
        if wall.openings:
            doors = sum(1 for o in wall.openings if o.type == "door")
            windows = sum(1 for o in wall.openings if o.type == "window")
            print(f"[primitive-fit]   Wall: {doors} doors, {windows} windows", flush=True)

    return walls, slabs


# ---------------------------------------------------------------------------
# RunPod Handler
# ---------------------------------------------------------------------------

def handler(event):
    try:
        inp = event["input"]

        data_b64 = inp.get("data", "")
        if "," in data_b64:
            data_b64 = data_b64.split(",")[-1]
        data_bytes = base64.b64decode(data_b64)

        fmt = inp.get("format", "obj").lower()
        params = inp.get("params", {})
        output_format = inp.get("output", "all")  # "ifc", "step", "pascal", or "all"

        print(f"[primitive-fit] Input: {len(data_bytes)} bytes, format={fmt}", flush=True)

        # Load mesh → point cloud
        pcd = load_mesh_to_pointcloud(data_bytes, fmt)
        print(f"[primitive-fit] Sampled {len(pcd.points)} points from mesh", flush=True)

        # Run pipeline
        walls, slabs = run_pipeline(pcd, params)

        total_openings = sum(len(w.openings) for w in walls)
        total_doors = sum(1 for w in walls for o in w.openings if o.type == "door")
        total_windows = sum(1 for w in walls for o in w.openings if o.type == "window")

        result = {
            "summary": {
                "walls": len(walls),
                "slabs": len(slabs),
                "doors": total_doors,
                "windows": total_windows,
                "total_openings": total_openings,
            }
        }

        # IFC export
        if output_format in ("ifc", "all"):
            ifc_path = "/tmp/primitive_fit_output.ifc"
            export_ifc(walls, slabs, ifc_path)
            with open(ifc_path, "rb") as f:
                result["ifc_b64"] = base64.b64encode(f.read()).decode()
            print(f"[primitive-fit] IFC exported", flush=True)

        # STEP export
        if output_format in ("step", "all"):
            step_path = "/tmp/primitive_fit_output.step"
            if export_step(walls, slabs, {}, step_path):
                with open(step_path, "rb") as f:
                    result["step_b64"] = base64.b64encode(f.read()).decode()
                print(f"[primitive-fit] STEP exported", flush=True)

        # Pascal Editor nodes
        if output_format in ("pascal", "all"):
            pascal_nodes = generate_pascal_nodes(walls, slabs)
            result["pascal_nodes"] = pascal_nodes
            print(f"[primitive-fit] Generated {len(pascal_nodes)} Pascal nodes", flush=True)

        print("[primitive-fit] Done!", flush=True)
        return result

    except Exception:
        traceback.print_exc()
        return {"error": traceback.format_exc()}


runpod.serverless.start({"handler": handler})
