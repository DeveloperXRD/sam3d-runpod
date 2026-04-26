"""
RunPod Pod Handler — Primitive Fitting Pipeline
OBJ/GLB mesh → clean CAD geometry (STEP + IFC + Pascal nodes)

Runs as HTTP server on a RunPod Pod (not serverless).
POST /run  → run pipeline
GET  /health → health check
"""
import os
import sys
import base64
import json
import math
import tempfile
import traceback
from io import BytesIO
from http.server import HTTPServer, BaseHTTPRequestHandler
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

PORT = int(os.environ.get("PORT", "8000"))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DetectedWall:
    start: list
    end: list
    thickness: float
    height: float
    elevation: float
    normal_xz: list
    openings: list = field(default_factory=list)

@dataclass
class DetectedOpening:
    type: str
    position: list
    width: float
    height: float
    wall_u: float
    elevation: float

@dataclass
class DetectedSlab:
    polygon: list
    elevation: float
    thickness: float = 0.2
    is_ceiling: bool = False


# ---------------------------------------------------------------------------
# Mesh loading
# ---------------------------------------------------------------------------

def load_mesh_to_pointcloud(data_bytes: bytes, fmt: str, num_samples=800_000):
    import open3d as o3d
    import trimesh as tm

    tmp = tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False)
    tmp.write(data_bytes)
    tmp.close()

    try:
        scene_or_mesh = tm.load(tmp.name, force="mesh")
        if hasattr(scene_or_mesh, "dump"):
            mesh = tm.util.concatenate(scene_or_mesh.dump())
        else:
            mesh = scene_or_mesh
        points, face_indices = tm.sample.sample_surface(mesh, num_samples)
        normals = mesh.face_normals[face_indices]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.normals = o3d.utility.Vector3dVector(normals)
    except Exception:
        o3d_mesh = o3d.io.read_triangle_mesh(tmp.name)
        if len(o3d_mesh.vertices) == 0:
            raise ValueError(f"Could not load mesh from {fmt}")
        pcd = o3d_mesh.sample_points_uniformly(num_samples)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    os.unlink(tmp.name)
    return pcd


# ---------------------------------------------------------------------------
# RANSAC plane fitting
# ---------------------------------------------------------------------------

def fit_planes_ransac(pcd, distance_threshold=0.03, min_points=300, max_planes=80):
    points = np.asarray(pcd.points)
    remaining_idx = list(range(len(points)))
    planes = []

    for _ in range(max_planes):
        if len(remaining_idx) < min_points:
            break
        sub_pcd = pcd.select_by_index(remaining_idx)
        try:
            plane_model, inliers = sub_pcd.segment_plane(
                distance_threshold=distance_threshold, ransac_n=3, num_iterations=3000)
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
        verticality = abs(normal[1])

        if verticality > 0.85:
            classification = "slab" if centroid[1] < np.median(points[:, 1]) else "ceiling"
        elif verticality < 0.3:
            classification = "wall"
        else:
            classification = "unknown"

        planes.append({
            "normal": normal, "inlier_points": inlier_pts,
            "centroid": centroid, "classification": classification,
        })

        inlier_set = set(inliers)
        remaining_idx = [remaining_idx[i] for i in range(len(remaining_idx)) if i not in inlier_set]

    return planes


# ---------------------------------------------------------------------------
# Wall extraction with snap
# ---------------------------------------------------------------------------

def extract_walls(wall_planes, snap_angle=5.0, snap_grid=0.05):
    walls = []
    for plane in wall_planes:
        pts = plane["inlier_points"]
        xz = pts[:, [0, 2]]
        y_vals = pts[:, 1]
        if len(xz) < 10:
            continue

        mean = xz.mean(axis=0)
        centered = xz - mean
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        principal = eigenvectors[:, -1]

        # Snap angle
        angle = math.atan2(principal[1], principal[0])
        angle_deg = math.degrees(angle)
        snap_angles = [0, 45, 90, 135, 180, -45, -90, -135]
        closest = min(snap_angles, key=lambda a: abs(angle_deg - a))
        if abs(angle_deg - closest) < snap_angle:
            angle = math.radians(closest)
            principal = np.array([math.cos(angle), math.sin(angle)])

        proj = centered @ principal
        t_min, t_max = proj.min(), proj.max()
        start_xz = mean + t_min * principal
        end_xz = mean + t_max * principal

        if snap_grid > 0:
            start_xz = np.round(start_xz / snap_grid) * snap_grid
            end_xz = np.round(end_xz / snap_grid) * snap_grid

        perp = eigenvectors[:, 0]
        perp_proj = centered @ perp
        thickness = float(perp_proj.max() - perp_proj.min())
        thickness = max(0.1, min(thickness, 0.6))
        common = [0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
        thickness = min(common, key=lambda t: abs(t - thickness))

        walls.append(DetectedWall(
            start=start_xz.tolist(), end=end_xz.tolist(),
            thickness=thickness, height=float(y_vals.max() - y_vals.min()),
            elevation=float(y_vals.min()),
            normal_xz=[float(plane["normal"][0]), float(plane["normal"][2])],
        ))
    return walls


def merge_walls(walls, angle_threshold=5.0, distance_threshold=0.3, gap_threshold=0.5):
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
            if abs(np.dot(d1n, d2n)) < math.cos(math.radians(angle_threshold)):
                continue
            mid2 = (s2 + e2) / 2
            v = mid2 - s1
            proj = np.dot(v, d1n) * d1n
            if np.linalg.norm(v - proj) < distance_threshold:
                group.append(w2)
                used.add(j)

        all_pts = np.array([pt for w in group for pt in [np.array(w.start), np.array(w.end)]])
        projs = all_pts @ d1n
        merged.append(DetectedWall(
            start=all_pts[projs.argmin()].tolist(), end=all_pts[projs.argmax()].tolist(),
            thickness=float(np.mean([w.thickness for w in group])),
            height=float(max(w.height for w in group)),
            elevation=float(min(w.elevation for w in group)),
            normal_xz=group[0].normal_xz,
        ))
    return merged


# ---------------------------------------------------------------------------
# Opening detection
# ---------------------------------------------------------------------------

def detect_openings(wall, all_points, floor_elevation):
    from scipy import ndimage

    start = np.array(wall.start)
    end = np.array(wall.end)
    wall_dir = end - start
    wall_length = np.linalg.norm(wall_dir)
    if wall_length < 0.5:
        return []
    wall_dir_n = wall_dir / wall_length
    wall_normal = np.array([-wall_dir_n[1], wall_dir_n[0]])

    margin = wall.thickness + 0.1
    xz = all_points[:, [0, 2]]
    y = all_points[:, 1]
    rel = xz - start
    u = rel @ wall_dir_n
    v = rel @ wall_normal

    mask = ((u >= -0.1) & (u <= wall_length + 0.1) & (abs(v) <= margin) &
            (y >= floor_elevation - 0.1) & (y <= floor_elevation + wall.height + 0.1))

    wall_pts_u = u[mask]
    wall_pts_y = y[mask] - floor_elevation
    if len(wall_pts_u) < 50:
        return []

    n_u = max(15, int(wall_length / 0.1))
    n_v = max(12, int(wall.height / 0.1))
    hist, u_edges, v_edges = np.histogram2d(
        wall_pts_u, wall_pts_y, bins=[n_u, n_v],
        range=[[0, wall_length], [0, wall.height]])
    if hist.max() == 0:
        return []

    density = hist / hist.max()
    avg_density = np.mean(density[density > 0]) if np.any(density > 0) else 0
    threshold = min(0.2, avg_density * 0.3)
    labeled, num_features = ndimage.label(density < threshold)

    openings = []
    for label_id in range(1, num_features + 1):
        region = np.where(labeled == label_id)
        u_min = u_edges[region[0].min()]
        u_max = u_edges[region[0].max() + 1]
        v_min = v_edges[region[1].min()]
        v_max = v_edges[region[1].max() + 1]
        width = u_max - u_min
        height = v_max - v_min
        if width < 0.4 or width > 4.5 or height < 0.5 or height > 3.5 or width * height < 0.4:
            continue

        is_door = v_min < 0.25 and height > 1.5
        u_center = (u_min + u_max) / 2
        v_center = (v_min + v_max) / 2
        pos_xz = start + u_center * wall_dir_n

        # Snap dimensions
        if is_door:
            width = _snap(width, [0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 1.8, 2.0])
            height = _snap(height, [2.0, 2.1, 2.2, 2.4])
        else:
            width = _snap(width, [0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.4])
            height = _snap(height, [0.6, 0.8, 1.0, 1.2, 1.5, 1.8])

        openings.append(DetectedOpening(
            type="door" if is_door else "window",
            position=[float(pos_xz[0]), float(floor_elevation + v_center), float(pos_xz[1])],
            width=float(width), height=float(height),
            wall_u=float(u_center / wall_length), elevation=float(v_min),
        ))
    return openings


def _snap(val, common, tol=0.15):
    c = min(common, key=lambda v: abs(v - val))
    return c if abs(c - val) < tol else round(val, 2)


# ---------------------------------------------------------------------------
# Slab extraction
# ---------------------------------------------------------------------------

def extract_slabs(slab_planes):
    from scipy.spatial import ConvexHull
    slabs = []
    for plane in slab_planes:
        pts = plane["inlier_points"][:, [0, 2]]
        if len(pts) < 4:
            continue
        try:
            hull = ConvexHull(pts)
            polygon = pts[hull.vertices].tolist()
            slabs.append(DetectedSlab(
                polygon=polygon, elevation=float(plane["centroid"][1]),
                is_ceiling=plane["classification"] == "ceiling",
            ))
        except Exception:
            continue
    return slabs


# ---------------------------------------------------------------------------
# IFC export
# ---------------------------------------------------------------------------

def export_ifc(walls, slabs, output_path):
    import ifcopenshell
    import ifcopenshell.api as api

    ifc = api.run("project.create_file")
    project = api.run("root.create_entity", ifc, ifc_class="IfcProject", name="Primitive Fit Import")
    api.run("unit.assign_unit", ifc)
    ctx = api.run("context.add_context", ifc, context_type="Model")
    body = api.run("context.add_context", ifc, context_type="Model",
                    context_identifier="Body", target_view="MODEL_VIEW", parent=ctx)
    site = api.run("root.create_entity", ifc, ifc_class="IfcSite", name="Site")
    api.run("aggregate.assign_object", ifc, products=[site], relating_object=project)
    building = api.run("root.create_entity", ifc, ifc_class="IfcBuilding", name="Building")
    api.run("aggregate.assign_object", ifc, products=[building], relating_object=site)

    wall_elevations = sorted(set([round(w.elevation, 1) for w in walls])) or [0.0]
    for si, elev in enumerate(wall_elevations):
        storey = api.run("root.create_entity", ifc, ifc_class="IfcBuildingStorey", name=f"Level {si}")
        storey.Elevation = elev
        api.run("aggregate.assign_object", ifc, products=[storey], relating_object=building)
        next_elev = wall_elevations[si + 1] if si + 1 < len(wall_elevations) else elev + 10
        for wi, wall in enumerate(w for w in walls if elev - 0.5 <= w.elevation <= next_elev - 0.5):
            ifc_wall = api.run("root.create_entity", ifc, ifc_class="IfcWall", name=f"Wall L{si}-{wi}")
            api.run("spatial.assign_container", ifc, products=[ifc_wall], relating_structure=storey)

            sx, sz = wall.start
            ex, ez = wall.end
            dx, dz = ex - sx, ez - sz
            length = math.sqrt(dx*dx + dz*dz)
            angle = math.atan2(dz, dx)
            half_t = wall.thickness / 2
            pts = [ifc.createIfcCartesianPoint([0.0, -half_t]),
                   ifc.createIfcCartesianPoint([length, -half_t]),
                   ifc.createIfcCartesianPoint([length, half_t]),
                   ifc.createIfcCartesianPoint([0.0, half_t])]
            polyline = ifc.createIfcPolyline(pts + [pts[0]])
            profile = ifc.createIfcArbitraryClosedProfileDef("AREA", None, polyline)
            solid = ifc.createIfcExtrudedAreaSolid(profile, None,
                        ifc.createIfcDirection([0.0, 0.0, 1.0]), wall.height)
            origin = ifc.createIfcCartesianPoint([sx, sz, elev])
            axis2 = ifc.createIfcAxis2Placement3D(origin,
                        ifc.createIfcDirection([0.0, 0.0, 1.0]),
                        ifc.createIfcDirection([math.cos(angle), math.sin(angle), 0.0]))
            ifc_wall.ObjectPlacement = ifc.createIfcLocalPlacement(None, axis2)
            ifc_wall.Representation = ifc.createIfcProductDefinitionShape(None, None,
                        [ifc.createIfcShapeRepresentation(body, "Body", "SweptSolid", [solid])])

            for oi, op in enumerate(wall.openings):
                ifc_op = api.run("root.create_entity", ifc, ifc_class="IfcOpeningElement",
                                  name=f"Opening L{si}-W{wi}-{oi}")
                api.run("void.add_opening", ifc, opening_element=ifc_op, element=ifc_wall)
                cls = "IfcDoor" if op.type == "door" else "IfcWindow"
                el = api.run("root.create_entity", ifc, ifc_class=cls, name=f"{op.type.title()} L{si}-W{wi}-{oi}")
                api.run("void.add_filling", ifc, opening=ifc_op, element=el)

    ifc.write(output_path)


# ---------------------------------------------------------------------------
# STEP export
# ---------------------------------------------------------------------------

def export_step(walls, slabs, output_path):
    try:
        import cadquery as cq
        assembly = cq.Assembly()
        for i, wall in enumerate(walls):
            sx, sz = wall.start
            ex, ez = wall.end
            dx, dz = ex - sx, ez - sz
            length = math.sqrt(dx*dx + dz*dz)
            angle = math.degrees(math.atan2(dz, dx))
            solid = cq.Workplane("XY").rect(length, wall.thickness).extrude(wall.height)
            assembly.add(solid, name=f"Wall_{i}",
                         loc=cq.Location(cq.Vector(sx, sz, wall.elevation), cq.Vector(0, 0, 1), angle))
        for i, slab in enumerate(slabs):
            if len(slab.polygon) < 3:
                continue
            pts = [(p[0], p[1]) for p in slab.polygon]
            solid = cq.Workplane("XY").polyline(pts).close().extrude(slab.thickness)
            assembly.add(solid, name=f"Slab_{i}",
                         loc=cq.Location(cq.Vector(0, 0, slab.elevation)))
        assembly.save(output_path)
        return True
    except (ImportError, Exception) as e:
        print(f"[primitive-fit] CadQuery unavailable ({type(e).__name__}: {e}), skipping STEP", flush=True)
        return False


# ---------------------------------------------------------------------------
# Pascal nodes
# ---------------------------------------------------------------------------

def generate_pascal_nodes(walls, slabs):
    import string, random
    def make_id(prefix):
        return f"{prefix}_{''.join(random.choices(string.ascii_lowercase + string.digits, k=16))}"

    nodes = {}
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
        level_walls = [w for w in walls if elev - 0.5 <= w.elevation <= next_elev - 0.5]
        wall_ids = []

        for wi, wall in enumerate(level_walls):
            wall_id = make_id("wall")
            wall_ids.append(wall_id)
            nodes[wall_id] = {
                "object": "node", "id": wall_id, "type": "wall",
                "name": f"Wall {wi}", "parentId": level_id,
                "visible": True, "metadata": {}, "children": [],
                "start": wall.start, "end": wall.end,
                "thickness": wall.thickness, "height": wall.height,
                "frontSide": "unknown", "backSide": "unknown",
            }
            for oi, op in enumerate(wall.openings):
                if op.type == "door":
                    oid = make_id("door")
                    nodes[oid] = {
                        "object": "node", "id": oid, "type": "door",
                        "name": f"Door {oi}", "parentId": wall_id,
                        "visible": True, "metadata": {}, "wallId": wall_id,
                        "position": op.position, "rotation": [0, 0, 0],
                        "width": op.width, "height": op.height,
                        "frameThickness": 0.05, "frameDepth": 0.07,
                        "threshold": True, "thresholdHeight": 0.02,
                        "hingesSide": "left", "swingDirection": "inward",
                        "segments": [
                            {"type": "panel", "heightRatio": 0.4, "columnRatios": [1],
                             "dividerThickness": 0.03, "panelDepth": 0.01, "panelInset": 0.04},
                            {"type": "panel", "heightRatio": 0.6, "columnRatios": [1],
                             "dividerThickness": 0.03, "panelDepth": 0.01, "panelInset": 0.04},
                        ],
                        "handle": True, "handleHeight": 1.05, "handleSide": "right",
                        "contentPadding": [0.04, 0.04],
                        "doorCloser": False, "panicBar": False, "panicBarHeight": 1.0,
                    }
                else:
                    oid = make_id("window")
                    nodes[oid] = {
                        "object": "node", "id": oid, "type": "window",
                        "name": f"Window {oi}", "parentId": wall_id,
                        "visible": True, "metadata": {}, "wallId": wall_id,
                        "position": op.position, "rotation": [0, 0, 0],
                        "width": op.width, "height": op.height,
                        "frameThickness": 0.05, "frameDepth": 0.07,
                        "columnRatios": [1], "rowRatios": [1],
                        "columnDividerThickness": 0.03, "rowDividerThickness": 0.03,
                        "sill": True, "sillDepth": 0.08, "sillThickness": 0.03,
                    }

        level_slabs = [s for s in slabs if not s.is_ceiling and abs(s.elevation - elev) < 1.0]
        slab_ids = []
        for sbi, slab in enumerate(level_slabs):
            slab_id = make_id("slab")
            slab_ids.append(slab_id)
            nodes[slab_id] = {
                "object": "node", "id": slab_id, "type": "slab",
                "name": f"Slab L{li}", "parentId": level_id,
                "visible": True, "metadata": {},
                "polygon": slab.polygon, "holes": [], "elevation": 0.05,
            }
        nodes[level_id] = {
            "object": "node", "id": level_id, "type": "level",
            "name": f"Level {li}", "parentId": building_id,
            "visible": True, "metadata": {},
            "children": wall_ids + slab_ids, "level": li,
        }

    nodes[building_id] = {
        "object": "node", "id": building_id, "type": "building",
        "name": "Imported Building", "parentId": site_id,
        "visible": True, "metadata": {}, "children": level_ids,
        "position": [0, 0, 0], "rotation": [0, 0, 0],
    }
    nodes[site_id] = {
        "object": "node", "id": site_id, "type": "site",
        "name": "Imported Site", "parentId": None,
        "visible": True, "metadata": {}, "children": [building_id],
    }
    return nodes


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(pcd, params=None):
    params = params or {}
    distance_thresh = params.get("distance_threshold", 0.03)
    min_points = params.get("min_wall_points", 300)
    snap_angle = params.get("snap_angle", 5.0)
    snap_grid = params.get("snap_grid", 0.05)

    print("[primitive-fit] Fitting planes...", flush=True)
    planes = fit_planes_ransac(pcd, distance_thresh, min_points)
    wall_planes = [p for p in planes if p["classification"] == "wall"]
    slab_planes = [p for p in planes if p["classification"] in ("slab", "ceiling")]
    print(f"[primitive-fit] Walls: {len(wall_planes)}, Slabs: {len(slab_planes)}", flush=True)

    walls = extract_walls(wall_planes, snap_angle, snap_grid)
    walls = merge_walls(walls)
    print(f"[primitive-fit] Merged walls: {len(walls)}", flush=True)

    slabs = extract_slabs(slab_planes)

    all_points = np.asarray(pcd.points)
    for wall in walls:
        wall.openings = detect_openings(wall, all_points, wall.elevation)

    return walls, slabs


# ---------------------------------------------------------------------------
# HTTP Handler for RunPod Pod
# ---------------------------------------------------------------------------

class PodHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/health", "/"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "service": "primitive-fit"}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path != "/run":
            self.send_response(404)
            self.end_headers()
            return

        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length))
            inp = body.get("input", body)

            data_b64 = inp.get("data", "")
            if "," in data_b64:
                data_b64 = data_b64.split(",")[-1]
            data_bytes = base64.b64decode(data_b64)

            fmt = inp.get("format", "obj").lower()
            params = inp.get("params", {})
            output_format = inp.get("output", "all")

            print(f"[primitive-fit] Input: {len(data_bytes)} bytes, format={fmt}", flush=True)

            pcd = load_mesh_to_pointcloud(data_bytes, fmt)
            print(f"[primitive-fit] Sampled {len(pcd.points)} points", flush=True)

            walls, slabs = run_pipeline(pcd, params)

            result = {
                "summary": {
                    "walls": len(walls), "slabs": len(slabs),
                    "doors": sum(1 for w in walls for o in w.openings if o.type == "door"),
                    "windows": sum(1 for w in walls for o in w.openings if o.type == "window"),
                    "total_openings": sum(len(w.openings) for w in walls),
                }
            }

            if output_format in ("ifc", "all"):
                ifc_path = "/tmp/primitive_fit_output.ifc"
                export_ifc(walls, slabs, ifc_path)
                with open(ifc_path, "rb") as f:
                    result["ifc_b64"] = base64.b64encode(f.read()).decode()

            if output_format in ("step", "all"):
                step_path = "/tmp/primitive_fit_output.step"
                if export_step(walls, slabs, step_path):
                    with open(step_path, "rb") as f:
                        result["step_b64"] = base64.b64encode(f.read()).decode()

            if output_format in ("pascal", "all"):
                result["pascal_nodes"] = generate_pascal_nodes(walls, slabs)

            print("[primitive-fit] Done!", flush=True)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())

        except Exception:
            traceback.print_exc()
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": traceback.format_exc()}).encode())

    def log_message(self, format, *args):
        print(f"[http] {args[0]}", flush=True)


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", PORT), PodHandler)
    print(f"[primitive-fit] HTTP server listening on port {PORT}", flush=True)
    server.serve_forever()
