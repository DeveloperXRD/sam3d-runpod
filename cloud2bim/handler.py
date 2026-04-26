"""
RunPod Pod Handler — Cloud2BIM
Accepts point cloud (PLY/XYZ/LAS) or OBJ mesh → returns IFC BIM model + Pascal nodes.

Runs as HTTP server on a RunPod Pod (not serverless).
POST /run  → run pipeline
GET  /health → health check
"""
import os
import sys
import base64
import json
import tempfile
import traceback
from io import BytesIO
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

import numpy as np

PORT = int(os.environ.get("PORT", "8000"))

# ---------------------------------------------------------------------------
# Core geometry detection (same logic as cloud2bim_handler.py)
# ---------------------------------------------------------------------------

def load_point_cloud(data_bytes: bytes, fmt: str):
    """Load point cloud from various formats."""
    import open3d as o3d

    tmp = tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False)
    tmp.write(data_bytes)
    tmp.close()

    if fmt in ("ply", "pcd", "xyz", "xyzn", "pts"):
        pcd = o3d.io.read_point_cloud(tmp.name)
    elif fmt in ("obj", "stl", "glb", "gltf"):
        mesh = o3d.io.read_triangle_mesh(tmp.name)
        if len(mesh.vertices) == 0:
            raise ValueError(f"Empty mesh from {fmt} file")
        pcd = mesh.sample_points_uniformly(number_of_points=500_000)
    elif fmt == "las":
        import laspy
        las = laspy.read(tmp.name)
        points = np.vstack((las.x, las.y, las.z)).T
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if hasattr(las, "red"):
            colors = np.vstack((las.red, las.green, las.blue)).T / 65535.0
            pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    os.unlink(tmp.name)
    return pcd


def preprocess_point_cloud(pcd, voxel_size=0.05):
    import open3d as o3d
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 4, max_nn=30)
    )
    return pcd_down


def detect_horizontal_planes(pcd, distance_threshold=0.05, min_points=500):
    points = np.asarray(pcd.points)
    remaining_indices = list(range(len(points)))
    planes = []

    for _ in range(20):
        if len(remaining_indices) < min_points:
            break
        sub_pcd = pcd.select_by_index(remaining_indices)
        plane_model, inliers = sub_pcd.segment_plane(
            distance_threshold=distance_threshold, ransac_n=3, num_iterations=2000,
        )
        a, b, c, d = plane_model
        normal = np.array([a, b, c])
        normal /= np.linalg.norm(normal) + 1e-9
        verticality = abs(normal[1])

        if verticality > 0.85 and len(inliers) >= min_points:
            elevation = -d / (b + 1e-9)
            inlier_global = [remaining_indices[i] for i in inliers]
            inlier_points = points[inlier_global]
            planes.append({
                "model": plane_model, "elevation": float(elevation),
                "inlier_indices": inlier_global, "points": inlier_points,
                "normal": normal.tolist(),
            })
            inlier_set = set(inliers)
            remaining_indices = [remaining_indices[i] for i in range(len(remaining_indices)) if i not in inlier_set]
        else:
            inlier_set = set(inliers)
            remaining_indices = [remaining_indices[i] for i in range(len(remaining_indices)) if i not in inlier_set]

    planes.sort(key=lambda p: p["elevation"])
    return planes


def cluster_into_storeys(horizontal_planes, min_storey_height=2.0, max_storey_height=5.0):
    if not horizontal_planes:
        return []
    storeys = []
    elevations = [p["elevation"] for p in horizontal_planes]
    i = 0
    while i < len(elevations) - 1:
        floor_elev = elevations[i]
        for j in range(i + 1, len(elevations)):
            height = elevations[j] - floor_elev
            if min_storey_height <= height <= max_storey_height:
                storeys.append({
                    "floor_elevation": floor_elev, "ceiling_elevation": elevations[j],
                    "height": height, "floor_plane": horizontal_planes[i],
                    "ceiling_plane": horizontal_planes[j],
                })
                i = j
                break
        else:
            i += 1
    if not storeys and len(elevations) >= 2:
        storeys.append({
            "floor_elevation": elevations[0], "ceiling_elevation": elevations[-1],
            "height": elevations[-1] - elevations[0],
            "floor_plane": horizontal_planes[0], "ceiling_plane": horizontal_planes[-1],
        })
    return storeys


def detect_walls_in_storey(pcd, storey, distance_threshold=0.05, min_points=200):
    points = np.asarray(pcd.points)
    y_min = storey["floor_elevation"] + 0.1
    y_max = storey["ceiling_elevation"] - 0.1
    mask = (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
    storey_indices = np.where(mask)[0]
    if len(storey_indices) < min_points:
        return []
    storey_pcd = pcd.select_by_index(storey_indices.tolist())
    remaining = list(range(len(storey_indices)))
    walls = []

    for _ in range(50):
        if len(remaining) < min_points:
            break
        sub_pcd = storey_pcd.select_by_index(remaining)
        plane_model, inliers = sub_pcd.segment_plane(
            distance_threshold=distance_threshold, ransac_n=3, num_iterations=1500,
        )
        a, b, c, d = plane_model
        normal = np.array([a, b, c])
        normal /= np.linalg.norm(normal) + 1e-9
        verticality = abs(normal[1])

        if verticality < 0.3 and len(inliers) >= min_points:
            inlier_pts = np.asarray(sub_pcd.points)[inliers]
            xz = inlier_pts[:, [0, 2]]
            mean = xz.mean(axis=0)
            centered = xz - mean
            if len(centered) > 2:
                cov = np.cov(centered.T)
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                principal = eigenvectors[:, -1]
                projections = centered @ principal
                t_min, t_max = projections.min(), projections.max()
                start_xz = mean + t_min * principal
                end_xz = mean + t_max * principal
                perp = eigenvectors[:, 0]
                perp_proj = centered @ perp
                thickness = float(perp_proj.max() - perp_proj.min())
                thickness = max(0.1, min(thickness, 0.5))
                walls.append({
                    "start": [float(start_xz[0]), float(start_xz[1])],
                    "end": [float(end_xz[0]), float(end_xz[1])],
                    "thickness": thickness, "height": storey["height"],
                    "normal_xz": [float(normal[0]), float(normal[2])],
                    "plane_model": [float(x) for x in plane_model],
                    "num_points": len(inliers), "inlier_points": inlier_pts,
                })
            inlier_set = set(inliers)
            remaining = [remaining[i] for i in range(len(remaining)) if i not in inlier_set]
        else:
            inlier_set = set(inliers)
            remaining = [remaining[i] for i in range(len(remaining)) if i not in inlier_set]
    return walls


def merge_collinear_walls(walls, angle_threshold=5.0, distance_threshold=0.3):
    if len(walls) < 2:
        return walls
    merged = []
    used = set()
    for i, w1 in enumerate(walls):
        if i in used:
            continue
        s1 = np.array(w1["start"])
        e1 = np.array(w1["end"])
        dir1 = e1 - s1
        dir1 /= np.linalg.norm(dir1) + 1e-9
        group = [w1]
        used.add(i)
        for j, w2 in enumerate(walls):
            if j in used:
                continue
            s2 = np.array(w2["start"])
            e2 = np.array(w2["end"])
            dir2 = e2 - s2
            dir2 /= np.linalg.norm(dir2) + 1e-9
            cos_angle = abs(np.dot(dir1, dir2))
            if cos_angle < np.cos(np.radians(angle_threshold)):
                continue
            mid2 = (s2 + e2) / 2
            v = mid2 - s1
            proj = np.dot(v, dir1) * dir1
            perp_dist = np.linalg.norm(v - proj)
            if perp_dist < distance_threshold:
                group.append(w2)
                used.add(j)
        all_points = []
        for w in group:
            all_points.extend([w["start"], w["end"]])
        all_pts = np.array(all_points)
        projections = all_pts @ dir1
        idx_min = projections.argmin()
        idx_max = projections.argmax()
        avg_thickness = np.mean([w["thickness"] for w in group])
        merged.append({
            "start": all_pts[idx_min].tolist(), "end": all_pts[idx_max].tolist(),
            "thickness": float(avg_thickness), "height": group[0]["height"],
            "normal_xz": group[0]["normal_xz"],
            "num_points": sum(w["num_points"] for w in group),
        })
    return merged


def detect_openings_in_wall(wall, storey_height, floor_elevation):
    openings = []
    if "inlier_points" not in wall:
        return openings
    pts = wall["inlier_points"]
    if len(pts) < 50:
        return openings
    start = np.array(wall["start"])
    end = np.array(wall["end"])
    wall_dir = end - start
    wall_length = np.linalg.norm(wall_dir)
    if wall_length < 0.5:
        return openings
    wall_dir_norm = wall_dir / wall_length
    xz = pts[:, [0, 2]]
    u = (xz - start) @ wall_dir_norm
    v = pts[:, 1] - floor_elevation
    n_bins_u = max(10, int(wall_length / 0.15))
    n_bins_v = max(8, int(storey_height / 0.15))
    hist, u_edges, v_edges = np.histogram2d(u, v, bins=[n_bins_u, n_bins_v])
    if hist.max() == 0:
        return openings
    density = hist / hist.max()
    threshold = 0.15
    empty = density < threshold
    labeled, num_features = _label_connected_components(empty)

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
        if width < 0.4 or width > 4.0:
            continue
        if height < 0.5 or height > 3.5:
            continue
        is_door = v_min < 0.3
        opening_type = "door" if is_door else "window"
        u_center = (u_min + u_max) / 2
        v_center = (v_min + v_max) / 2
        pos_xz = start + u_center * wall_dir_norm
        pos_y = floor_elevation + v_center
        openings.append({
            "type": opening_type,
            "position": [float(pos_xz[0]), float(pos_y), float(pos_xz[1])],
            "width": float(width), "height": float(height),
            "elevation_from_floor": float(v_min),
            "u_center": float(u_center),
        })
    return openings


# ---------------------------------------------------------------------------
# IFC export
# ---------------------------------------------------------------------------

def build_ifc(storeys_data, output_path):
    import ifcopenshell
    import ifcopenshell.api as api
    import math

    ifc = api.run("project.create_file")
    project = api.run("root.create_entity", ifc, ifc_class="IfcProject", name="Cloud2BIM Import")
    api.run("unit.assign_unit", ifc)
    ctx = api.run("context.add_context", ifc, context_type="Model")
    body = api.run("context.add_context", ifc, context_type="Model",
                    context_identifier="Body", target_view="MODEL_VIEW", parent=ctx)
    site = api.run("root.create_entity", ifc, ifc_class="IfcSite", name="Site")
    api.run("aggregate.assign_object", ifc, products=[site], relating_object=project)
    building = api.run("root.create_entity", ifc, ifc_class="IfcBuilding", name="Building")
    api.run("aggregate.assign_object", ifc, products=[building], relating_object=site)

    for si, storey_data in enumerate(storeys_data):
        storey = api.run("root.create_entity", ifc, ifc_class="IfcBuildingStorey", name=f"Level {si}")
        api.run("aggregate.assign_object", ifc, products=[storey], relating_object=building)
        elev = storey_data["floor_elevation"]
        storey.Elevation = elev

        for wi, wall in enumerate(storey_data.get("walls", [])):
            ifc_wall = api.run("root.create_entity", ifc, ifc_class="IfcWall", name=f"Wall L{si}-{wi}")
            api.run("spatial.assign_container", ifc, products=[ifc_wall], relating_structure=storey)
            sx, sz = wall["start"]
            ex, ez = wall["end"]
            thickness = wall.get("thickness", 0.2)
            height = wall.get("height", 2.8)

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
            origin = ifc.createIfcCartesianPoint([sx, sz, elev])
            z_dir = ifc.createIfcDirection([0.0, 0.0, 1.0])
            x_dir = ifc.createIfcDirection([math.cos(angle), math.sin(angle), 0.0])
            axis2 = ifc.createIfcAxis2Placement3D(origin, z_dir, x_dir)
            local_placement = ifc.createIfcLocalPlacement(None, axis2)
            shape_rep = ifc.createIfcShapeRepresentation(body, "Body", "SweptSolid", [solid])
            product_shape = ifc.createIfcProductDefinitionShape(None, None, [shape_rep])
            ifc_wall.Representation = product_shape
            ifc_wall.ObjectPlacement = local_placement

            for oi, opening in enumerate(wall.get("openings", [])):
                ifc_opening = api.run("root.create_entity", ifc, ifc_class="IfcOpeningElement",
                                       name=f"Opening L{si}-W{wi}-{oi}")
                api.run("void.add_opening", ifc, opening_element=ifc_opening, element=ifc_wall)
                if opening["type"] == "door":
                    el = api.run("root.create_entity", ifc, ifc_class="IfcDoor", name=f"Door L{si}-W{wi}-{oi}")
                else:
                    el = api.run("root.create_entity", ifc, ifc_class="IfcWindow", name=f"Window L{si}-W{wi}-{oi}")
                api.run("void.add_filling", ifc, opening=ifc_opening, element=el)

    ifc.write(output_path)
    return output_path


# ---------------------------------------------------------------------------
# Pascal Editor node generation
# ---------------------------------------------------------------------------

def generate_pascal_nodes(storeys_data):
    import string, random
    def make_id(prefix):
        chars = string.ascii_lowercase + string.digits
        return f"{prefix}_{''.join(random.choices(chars, k=16))}"
    nodes = {}
    site_id = make_id("site")
    building_id = make_id("building")
    level_ids = []

    for si, storey_data in enumerate(storeys_data):
        level_id = make_id("level")
        level_ids.append(level_id)
        wall_ids = []
        for wi, wall in enumerate(storey_data.get("walls", [])):
            wall_id = make_id("wall")
            wall_ids.append(wall_id)
            nodes[wall_id] = {
                "object": "node", "id": wall_id, "type": "wall",
                "name": f"Wall {wi}", "parentId": level_id,
                "visible": True, "metadata": {}, "children": [],
                "start": wall["start"], "end": wall["end"],
                "thickness": wall.get("thickness", 0.2),
                "height": wall.get("height", 2.8),
                "frontSide": "unknown", "backSide": "unknown",
            }
            for oi, opening in enumerate(wall.get("openings", [])):
                if opening["type"] == "door":
                    oid = make_id("door")
                    nodes[oid] = {
                        "object": "node", "id": oid, "type": "door",
                        "name": f"Door {oi}", "parentId": wall_id,
                        "visible": True, "metadata": {}, "wallId": wall_id,
                        "position": opening["position"], "rotation": [0, 0, 0],
                        "width": opening["width"], "height": opening["height"],
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
                        "position": opening["position"], "rotation": [0, 0, 0],
                        "width": opening["width"], "height": opening["height"],
                        "frameThickness": 0.05, "frameDepth": 0.07,
                        "columnRatios": [1], "rowRatios": [1],
                        "columnDividerThickness": 0.03, "rowDividerThickness": 0.03,
                        "sill": True, "sillDepth": 0.08, "sillThickness": 0.03,
                    }
        slab_ids = []
        if "slab_polygon" in storey_data:
            slab_id = make_id("slab")
            slab_ids.append(slab_id)
            nodes[slab_id] = {
                "object": "node", "id": slab_id, "type": "slab",
                "name": f"Slab L{si}", "parentId": level_id,
                "visible": True, "metadata": {},
                "polygon": storey_data["slab_polygon"], "holes": [], "elevation": 0.05,
            }
        nodes[level_id] = {
            "object": "node", "id": level_id, "type": "level",
            "name": f"Level {si}", "parentId": building_id,
            "visible": True, "metadata": {},
            "children": wall_ids + slab_ids, "level": si,
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


def compute_slab_polygon(wall_segments):
    if len(wall_segments) < 3:
        return None
    points = []
    for w in wall_segments:
        points.append(w["start"])
        points.append(w["end"])
    pts = np.array(points)
    try:
        hull = _convex_hull_2d(pts)
        return hull.tolist() if len(hull) >= 3 else None
    except Exception:
        return None


# ── Pure-numpy helpers (replace scipy to avoid numpy ABI issues) ──

def _label_connected_components(binary_mask):
    """Pure-numpy 4-connectivity connected components (replaces scipy.ndimage.label)."""
    mask = np.asarray(binary_mask, dtype=bool)
    labeled = np.zeros(mask.shape, dtype=np.int32)
    next_label = 1
    H, W = mask.shape
    for i in range(H):
        for j in range(W):
            if mask[i, j] and labeled[i, j] == 0:
                stack = [(i, j)]
                while stack:
                    y, x = stack.pop()
                    if y < 0 or y >= H or x < 0 or x >= W:
                        continue
                    if not mask[y, x] or labeled[y, x] != 0:
                        continue
                    labeled[y, x] = next_label
                    stack.extend([(y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)])
                next_label += 1
    return labeled, next_label - 1


def _convex_hull_2d(points):
    """2D convex hull via Andrew's monotone chain (replaces scipy.spatial.ConvexHull)."""
    pts = np.asarray(points, dtype=float)
    pts = np.unique(pts, axis=0)
    if len(pts) < 3:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(pcd, params=None):
    params = params or {}
    voxel_size = params.get("voxel_size", 0.05)
    wall_distance = params.get("wall_distance_threshold", 0.05)
    min_wall_points = params.get("min_wall_points", 200)

    print("[cloud2bim] Preprocessing...", flush=True)
    pcd_down = preprocess_point_cloud(pcd, voxel_size)
    print(f"[cloud2bim] Downsampled to {len(pcd_down.points)} points", flush=True)

    print("[cloud2bim] Detecting horizontal planes...", flush=True)
    h_planes = detect_horizontal_planes(pcd_down)
    print(f"[cloud2bim] Found {len(h_planes)} horizontal planes", flush=True)

    print("[cloud2bim] Clustering storeys...", flush=True)
    storeys = cluster_into_storeys(h_planes)
    if not storeys:
        pts = np.asarray(pcd_down.points)
        y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
        storeys = [{"floor_elevation": float(y_min), "ceiling_elevation": float(y_max),
                     "height": float(y_max - y_min), "floor_plane": None, "ceiling_plane": None}]

    storeys_data = []
    for si, storey in enumerate(storeys):
        print(f"[cloud2bim] Storey {si} (elev {storey['floor_elevation']:.2f}m)...", flush=True)
        walls = detect_walls_in_storey(pcd_down, storey, wall_distance, min_wall_points)
        walls_merged = merge_collinear_walls(walls)
        for wall in walls_merged:
            matching_raw = [w for w in walls if "inlier_points" in w
                           and abs(w["start"][0] - wall["start"][0]) < 0.5]
            if matching_raw:
                wall["inlier_points"] = np.vstack([w["inlier_points"] for w in matching_raw])
            openings = detect_openings_in_wall(wall, storey["height"], storey["floor_elevation"])
            wall["openings"] = openings
            wall.pop("inlier_points", None)
        slab_polygon = compute_slab_polygon(walls_merged)
        storey_out = {
            "floor_elevation": storey["floor_elevation"],
            "ceiling_elevation": storey["ceiling_elevation"],
            "height": storey["height"], "walls": walls_merged,
        }
        if slab_polygon:
            storey_out["slab_polygon"] = slab_polygon
        storeys_data.append(storey_out)

    return storeys_data


# ---------------------------------------------------------------------------
# HTTP Handler for RunPod Pod
# ---------------------------------------------------------------------------

class PodHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health" or self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "service": "cloud2bim"}).encode())
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

            fmt = inp.get("format", "ply").lower()
            params = inp.get("params", {})
            output_format = inp.get("output", "both")

            print(f"[cloud2bim] Input: {len(data_bytes)} bytes, format={fmt}", flush=True)

            pcd = load_point_cloud(data_bytes, fmt)
            print(f"[cloud2bim] Loaded {len(pcd.points)} points", flush=True)

            storeys_data = run_pipeline(pcd, params)

            result = {"storeys": len(storeys_data), "summary": []}
            for sd in storeys_data:
                result["summary"].append({
                    "floor_elevation": sd["floor_elevation"],
                    "walls": len(sd.get("walls", [])),
                    "openings": sum(len(w.get("openings", [])) for w in sd.get("walls", [])),
                    "has_slab": "slab_polygon" in sd,
                })

            if output_format in ("ifc", "both"):
                ifc_path = "/tmp/cloud2bim_output.ifc"
                build_ifc(storeys_data, ifc_path)
                with open(ifc_path, "rb") as f:
                    result["ifc_b64"] = base64.b64encode(f.read()).decode()

            if output_format in ("pascal", "both"):
                result["pascal_nodes"] = generate_pascal_nodes(storeys_data)

            print("[cloud2bim] Done!", flush=True)
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
    print(f"[cloud2bim] HTTP server listening on port {PORT}", flush=True)
    server.serve_forever()
