import cv2
import numpy as np
from pupil_apriltags import Detector
from PIL import Image
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Optional
from dataclasses import dataclass
from IPython.display import display

# 6bd70e6d-3756-49f8-9e4d-e9d91b60ae56.jfif
# 609bf0b1-5a70-4ba4-aa0a-e832bd64c3a3.jfif
# 2346cbbd-2e48-40a1-ba4f-d47f83529737.jfif
# a097261e-6674-4e96-a9d9-705be2be628d.jfif
IMAGE_PATH = "6bd70e6d-3756-49f8-9e4d-e9d91b60ae56.jfif"   # test image path
SCALE = 1.0
TAG_FAMILY = "tag36h11"

# For each repeated-ID polygon that represents an action region,
# tell the system how many physical tags define it.
EXPECTED_TAGS = {
    7:6,   # region defined by 6 tags that all share id=7
    0:4,
}

# Thresholds & weights (tune as needed)
TAU_VIS      = 0.5    # min visibility considered good for region
TAU_CONF     = 0.5    # min confidence considered good for region
TAU_REGION_S = 0.7    # region-level score threshold for pass
TAU_FRAME_S  = 0.75   # frame-level overall threshold
TAU_FRAC     = 0.75   # fraction of regions that must pass

W_VIS        = 0.4    # >>> visibility weight in final score
W_CONF       = 0.6    # >>> confidence weight in final score

#Polygon Formation
Point = Tuple[float, float]
Quad = List[Point]
Polygon = List[Point]

def round_pt(p: Point, tol=1e-6): 
    return (int(round(p[0]/tol)), int(round(p[1]/tol)))

def edge_key(a: Point, b: Point, tol=1e-6):
    ka, kb = round_pt(a, tol), round_pt(b, tol)
    return (ka, kb) if ka <= kb else (kb, ka)

def polygon_area(poly: Polygon):
    return 0.5 * abs(sum(
        x0*y1 - x1*y0
        for (x0, y0), (x1, y1) in zip(poly, poly[1:] + poly[:1])
    ))

def is_clockwise(poly: Polygon):
    return sum(
        (x2-x1)*(y2+y1)
        for (x1,y1),(x2,y2) in zip(poly, poly[1:]+poly[:1])
    ) > 0

def ensure_clockwise(poly: Polygon):
    return poly if is_clockwise(poly) else list(reversed(poly))

def remove_collinear(poly: Polygon, tol=1e-8):
    if len(poly) <= 3:
        return poly
    out = []
    for i in range(len(poly)):
        p0, p1, p2 = np.array(poly[i-1]), np.array(poly[i]), np.array(poly[(i+1)%len(poly)])
        cross = abs(np.cross(p1 - p0, p2 - p1))
        norm = np.linalg.norm(p1 - p0) * np.linalg.norm(p2 - p1)
        if norm == 0 or cross / (norm + 1e-12) > tol:
            out.append(tuple(p1))
    return out

def point_in_polygon(pt: Point, poly: Polygon):
    x, y = pt
    inside = False
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        if ((y1 > y) != (y2 > y)) and \
           (x < (x2-x1)*(y-y1)/(y2-y1+1e-20) + x1):
            inside = not inside
    return inside

def convex_hull(points: List[Point]) -> Polygon:
    pts = sorted(set(points))
    if len(pts) <= 1:
        return pts
    def cross(o, a, b): 
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    hull = remove_collinear(hull)
    return hull

def stitch_outer_boundary(quads: List[Quad], tol=1e-6):
    edge_count, edge_to_points, vert_map = defaultdict(int), {}, {}
    for q in quads:
        m = len(q)
        for i in range(m):
            a, b = q[i], q[(i+1)%m]
            k = edge_key(a, b, tol)
            edge_count[k] += 1
            edge_to_points[k] = (a, b)
            vert_map[round_pt(a, tol)] = a
            vert_map[round_pt(b, tol)] = b

    boundary_edges = [edge_to_points[k] for k, c in edge_count.items() if c == 1]
    if not boundary_edges:
        return []

    adj = defaultdict(list)
    for a, b in boundary_edges:
        ka, kb = round_pt(a, tol), round_pt(b, tol)
        adj[ka].append(kb)
        adj[kb].append(ka)

    loops, visited = [], set()
    for start in adj:
        for nb in adj[start]:
            ekey = tuple(sorted((start, nb)))
            if ekey in visited:
                continue
            loop, cur, prev = [], start, None
            while True:
                loop.append(vert_map[cur])
                nxts = [n for n in adj[cur] if n != prev]
                if not nxts:
                    break
                nxt = nxts[0]
                visited.add(tuple(sorted((cur, nxt))))
                prev, cur = cur, nxt
                if cur == start:
                    break
            if len(loop) >= 3:
                loops.append(loop)
    return loops

def build_boundary(quads: List[Quad],
                   use_stitch=True,
                   use_hull=True,
                   tol=1e-8,
                   require_inside=True,
                   H: Optional[np.ndarray]=None):
    pts = [tuple(p) for q in quads for p in q]
    meta = {"used": None, "loops": [], "fallback": False}
    poly = None

    if use_stitch:
        loops = stitch_outer_boundary(quads, tol=tol)
        clean = []
        for lp in loops:
            area = polygon_area(lp)
            if area > 1e-9:
                p = ensure_clockwise(lp)
                p = remove_collinear(p, tol)
                clean.append((area, p))
        clean.sort(key=lambda x: x[0], reverse=True)
        meta["loops"] = [p for _, p in clean]
        if clean:
            poly = clean[0][1]
            meta["used"] = "stitch"

    if poly is None and use_hull:
        hull = convex_hull(pts)
        hull = ensure_clockwise(hull)
        hull = remove_collinear(hull, tol)
        poly = hull
        meta["used"] = "convex_hull"
        meta["fallback"] = True

    if poly is None:
        meta["contains_all"] = False
        meta["proj"] = None
        return None, meta

    inside = True
    for q in quads:
        for v in q:
            if not point_in_polygon(v, poly):
                inside = False
                break
        if not inside:
            break
    meta["contains_all"] = inside

    if require_inside and not inside and use_hull and meta["used"] != "convex_hull":
        hull = convex_hull(pts)
        hull = ensure_clockwise(hull)
        hull = remove_collinear(hull, tol)
        poly = hull
        meta["used"] = "convex_hull_fallback"
        meta["fallback"] = True
        meta["contains_all"] = all(point_in_polygon(v, poly) for q in quads for v in q)

    if H is not None:
        pts_arr = np.array(poly, float)
        ones = np.ones((len(pts_arr), 1))
        hp = np.hstack([pts_arr, ones]) @ H.T
        hp /= hp[:, 2:3]
        meta["proj"] = [tuple(map(float, p[:2])) for p in hp]
    else:
        meta["proj"] = None

    return poly, meta

def intersect_polygons(subject: Polygon, clip: Polygon) -> Polygon:
    if not subject or not clip:
        return []
    if is_clockwise(clip):
        clip = list(reversed(clip))

    output = subject
    for i in range(len(clip)):
        A = clip[i]
        B = clip[(i+1) % len(clip)]
        inp = output
        output = []
        if not inp:
            break

        def inside(P):
            return (B[0]-A[0])*(P[1]-A[1]) - (B[1]-A[1])*(P[0]-A[0]) >= 0

        def seg_inter(P, Q):
            x1,y1 = P; x2,y2 = Q
            x3,y3 = A; x4,y4 = B
            den = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
            if abs(den) < 1e-12:
                return Q
            t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / den
            return (x1 + t*(x2-x1), y1 + t*(y2-y1))

        S = inp[-1]
        for E in inp:
            if inside(E):
                if not inside(S):
                    output.append(seg_inter(S, E))
                output.append(E)
            elif inside(S):
                output.append(seg_inter(S, E))
            S = E

    return remove_collinear(output, tol=1e-8)

def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


# Scoring Logic

@dataclass
class RegionMetrics:
    id: int
    vis_frac: float
    conf_score: float
    score: float
    passed: bool
    reason: str

@dataclass
class FrameDecision:
    passed: bool
    overall_score: float
    num_passed: int
    num_regions: int
    reasons: list
    per_region: list

def soft_component(x: float, good: float) -> float:
    """
    Map raw metric x (0..1) to a softer score:
    - below good, curve is gentle, not instantly 0
    - above good, saturates towards 1
    """
    x = clamp(x)
    if x <= 0.0:
        return 0.0
    return x ** 0.7  # adjust sensitivity

def grade_from_polygons(groups, boundaries) -> FrameDecision:
    if not boundaries:
        return FrameDecision(False, 0.0, 0, 0, ["no polygons"], [])

    # 1. choose station polygon = largest area
    station_id = None
    station_poly = None
    max_area = 0.0
    for tid, (poly, _) in boundaries.items():
        if poly is None:
            continue
        a = polygon_area(poly)
        if a > max_area:
            max_area = a
            station_id = tid
            station_poly = poly

    if station_poly is None:
        return FrameDecision(False, 0.0, 0, 0, ["no valid station polygon"], [])

    per_region = []
    sum_scores = 0.0
    num_pass = 0

    # 2. Only grade regions configured in EXPECTED_TAGS
    for tid, expected in EXPECTED_TAGS.items():
        expected = max(1, expected)

        if tid not in boundaries:
            # region polygon missing entirely
            vis = 0.0
            conf = 0.0
            reason = "region not detected"
        else:
            poly, _ = boundaries[tid]
            if poly is None:
                vis = 0.0
                conf = 0.0
                reason = "invalid polygon"
            else:
                # visibility: overlap between region and station
                inter = intersect_polygons(poly, station_poly)
                if not inter:
                    vis = 0.0
                else:
                    vis = clamp(
                        polygon_area(inter) /
                        (polygon_area(poly) + 1e-9)
                    )

                # confidence: how many of expected tags are visible
                detected = len(groups.get(tid, []))
                conf_raw = detected / float(expected)
                conf = clamp(conf_raw)
                reason = ""

        vis_score  = soft_component(vis, TAU_VIS)
        conf_score = soft_component(conf, TAU_CONF)

        # combined continuous score
        if (W_VIS + W_CONF) > 0:
            score = (W_VIS * vis_score + W_CONF * conf_score) / (W_VIS + W_CONF)
        else:
            score = 0.0

        # region pass/fail based on final score
        passed = score >= TAU_REGION_S

        # interpretable reason
        if not passed:
            if tid not in boundaries:
                reason = "region not detected"
            elif vis < TAU_VIS and conf < TAU_CONF:
                reason = f"low visibility ({vis:.2f}) & low confidence ({conf:.2f})"
            elif vis < TAU_VIS:
                reason = f"low visibility ({vis:.2f})"
            elif conf < TAU_CONF:
                reason = f"low confidence ({conf:.2f}, {len(groups.get(tid,[]))}/{expected} tags)"
            elif not reason:
                reason = "score below threshold"

        rm = RegionMetrics(
            id=tid,
            vis_frac=vis,
            conf_score=conf,
            score=score,
            passed=passed,
            reason=reason
        )
        per_region.append(rm)
        sum_scores += score
        if passed:
            num_pass += 1

    N = len(per_region)
    if N == 0:
        return FrameDecision(False, 0.0, 0, 0,
                             ["no EXPECTED_TAGS configured"], [])

    overall = sum_scores / N
    frac = num_pass / N

    reasons = []
    if overall < TAU_FRAME_S:
        reasons.append(f"overall score below {TAU_FRAME_S:.2f}")
    if frac < TAU_FRAC:
        reasons.append(f"only {num_pass}/{N} regions passed (<{TAU_FRAC:.2f})")
    for r in per_region:
        if not r.passed and r.reason:
            reasons.append(f"region {r.id} failed: {r.reason}")
    reasons = reasons[:4]

    frame_pass = (overall >= TAU_FRAME_S) and (frac >= TAU_FRAC)

    return FrameDecision(
        passed=frame_pass,
        overall_score=overall,
        num_passed=num_pass,
        num_regions=N,
        reasons=reasons,
        per_region=per_region
    )


detector = Detector(
    families=TAG_FAMILY,
    nthreads=4,
    quad_decimate=0.8,
    quad_sigma=0.4,
    refine_edges=True,
    decode_sharpening=0.5
)

img = cv2.imread(IMAGE_PATH)
if img is None:
    raise IOError(f"Cannot open {IMAGE_PATH}")

resized = cv2.resize(img, (0, 0), fx=SCALE, fy=SCALE)
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
gray = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
gray = cv2.GaussianBlur(gray, (3, 3), 0.6)
gray = cv2.copyMakeBorder(gray, 10, 10, 10, 10, cv2.BORDER_REPLICATE)

tags = detector.detect(gray)
print(f"Detected {len(tags)} tags")

if not tags:
    raise RuntimeError("No tags detected")

# group detections by tag_id
groups = defaultdict(list)
for tag in tags:
    corners = (np.array(tag.corners) - [10, 10]) / SCALE
    groups[tag.tag_id].append(corners.tolist())

# build polygons per tag_id
palette = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (255, 128, 0), (128, 0, 255), (0, 128, 255),
    (128, 255, 0), (255, 0, 128)
]

out = resized.copy()
boundaries = {}

for tag_id, quads in groups.items():
    poly, meta = build_boundary(quads)
    boundaries[tag_id] = (poly, meta)
    if poly is not None:
        color = palette[tag_id % len(palette)]
        p = np.array(poly, np.int32)
        cv2.polylines(out, [p], True, color, 2)
        cx, cy = np.mean(p[:, 0]), np.mean(p[:, 1])
        cv2.putText(out, f"ID {tag_id}", (int(cx), int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        print(f"Tag {tag_id}: {meta['used']} | contains_all={meta['contains_all']}")

for tag in tags:
    c = ((np.array(tag.corners) - [10, 10]) / SCALE).astype(int)
    cx, cy = int((tag.center[0] - 10) / SCALE), int((tag.center[1] - 10) / SCALE)
    cv2.polylines(out, [c], True, (200, 200, 200), 1)
    cv2.circle(out, (cx, cy), 3, (0, 0, 0), -1)

decision = grade_from_polygons(groups, boundaries)

print("\n=== Grading Result ===")
print("Frame pass:", decision.passed)
print("Overall score:", f"{decision.overall_score:.2f}")
for r in decision.per_region:
    print(f"region_{r.id}: score={r.score:.2f}, "
          f"vis={r.vis_frac:.2f}, conf={r.conf_score:.2f}, reason={r.reason}")
print("Frame reasons:", decision.reasons)

for r in decision.per_region:
    poly, _ = boundaries.get(r.id, (None, None))
    if poly is None:
        continue
    p = np.array(poly, np.int32)
    color = (0, 255, 0) if r.passed else (0, 0, 255)
    cv2.polylines(out, [p], True, color, 3)
    cx, cy = np.mean(p[:, 0]), np.mean(p[:, 1])
    label = f"{'P' if r.passed else 'F'} {r.score:.2f}"
    label_y = int(cy) + 24
    cv2.putText(out, label,
                (int(cx), label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

out_path = Path(IMAGE_PATH).with_name(Path(IMAGE_PATH).stem + "_graded.png")
cv2.imwrite(str(out_path), out)
print("Saved to", out_path)
display(Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)))
