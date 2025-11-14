#doms version

import cv2
import numpy as np
from pupil_apriltags import Detector
from IPython.display import display, clear_output
from PIL import Image
import time, platform
from collections import defaultdict


# ========= Tunables =========
CAMERA_INDEX = 0

SCALE_FACTOR = 1.0

TAG_FAMILY = "tag36h11"

PAD = 10

PERSIST_FOR_FRAMES = 180
# this here is for visual testing. it will keep frames on screen so you can look and see to verify. theres not really a beeter way to test this. 

TAG_SIZE_INCH = 1.0
# Physical tag side length (inches). Used to convert pixel area to square inches. this is hard coded so later, when we get the right sie, we can put in square inches and get the size of an area

# How to group tags into “teams”
GROUP_MODE = "exact"
# "exact": identical IDs together; "last_digit": IDs share last digit;
# i added this feature to give us some leway in how we decide to group tags. origionaly ried a few diffrent things, but the crent exact implemetation is best (in my opionon)

ALLOWED_GROUPS = None
# Optionally restrict rendering to a subset, e.g., {0,1}. None = render all groups found.

EDGE_THICKNESS = 4

LABEL_SCALE = 0.8

LABEL_THICK = 2

LABEL_COLOR = (0,255,255)
# BGR color for labels (yellows what i chose, change if you want).

GROUP_COLORS = [(0,0,255),(0,165,255),(0,255,0),(255,0,0),(255,0,255),(0,255,255)]
# Distinct colors per group (BGR). Cycled if you have more than the list length.

# ========= Helpers =========
# this code opeans the camra and runs the program in frame. nothing tp much to say on it. 
def open_camera(index=0):
    """Open a webcam reliably across OSes by trying the default and OS-specific backends."""
    cap = cv2.VideoCapture(index)

    if cap.isOpened(): return cap

    sys = platform.system().lower()
    backs = [cv2.CAP_AVFOUNDATION] if sys=="darwin" else ([cv2.CAP_MSMF, cv2.CAP_DSHOW] if sys=="windows" else [cv2.CAP_V4L2])
    for be in backs:
        cap = cv2.VideoCapture(index, be)
      
        if cap.isOpened(): return cap
      
    raise IOError("Could not open webcam. Check permissions/availability.")

'''
group_key_from_id: this code works as a helper for upcoming alogrighums. it is what determines what grpup to make the polygones into. in this case, it is hard coded to return the exact id of the tags to make groups. 
origionaly had diffrent features and options, all have been depricated due to iefficencies
'''  
def group_key_from_id(tag_id:int):
    
    """Map a raw tag ID to a ‘team’ key according to GROUP_MODE."""
    if GROUP_MODE == "exact":      return tag_id
    # Same exact ID -> same group (e.g., all 0s together).

    # if GROUP_MODE == "last_digit": return tag_id % 10 <- dont use this one

    return tag_id
 
''' 
corners_disp: Purpose:
    Give the tag’s 4 corner points in the same coordinate space as the image you draw on.

    Requires:
        - tag.corners: a 4×2 array of corner points from the detector.
        - PAD (int): how many pixels of border were added before detection.
        - SCALE_FACTOR (float > 0): how much the image was resized before detection.

    Returns:
        - A 4×2 float32 array whose points line up with your displayed frame.

    Why:
        The detector looked at a slightly modified image (extra border + possible resizing).
        This function removes that border and undoes the resizing so your overlays land exactly
        on the tag in the live view.
'''
def corners_disp(tag):
    """Return the tag’s 4 outer corners in *display* coordinates (undo PAD & SCALE_FACTOR)."""
    return ((np.array(tag.corners, np.float32) - np.array([PAD,PAD], np.float32)) / float(SCALE_FACTOR))
    # Math: tag.corners are in the detector’s padded coords. Subtract [PAD, PAD] to remove the +PAD offset
    # applied by copyMakeBorder on both x and y. Then divide by SCALE_FACTOR to undo any pre-detection
    # resizing (scaled_px -> original_px). If you draw on the *same* scaled image used for detection,
    # keep SCALE_FACTOR=1.0 or skip the division to avoid misalignment.





'''
 Estimate_px_per_in: Purpose:
        Estimate how many image pixels equal one real-world inch in the current view.

    Requires:
        - tags: a list/iterable of detected tags (each has .corners).
        - TAG_SIZE_INCH (float > 0): the real edge length of a printed tag.
        - corners_disp(): helper that converts detector corners to the displayed frame.

    Returns:
        - float: pixels-per-inch estimate, or None if there isn’t enough good data.

    Why:
        We need a scale to convert pixel areas into square inches. For each tag, we
        measure its four edges in the image, compare that average edge length to the
        tag’s known real size, then take the median across tags for a stable result.
'''

def estimate_px_per_in(tags):
    vals = []

    for t in tags:
        c = corners_disp(t)
        sides = [np.linalg.norm(c[0]-c[1]), np.linalg.norm(c[1]-c[2]),
                 np.linalg.norm(c[2]-c[3]), np.linalg.norm(c[3]-c[0])]
        # Four side lengths in pixels.
        avg = float(np.mean(sides))
        if avg > 1e-3: vals.append(avg / TAG_SIZE_INCH)
        # Convert px side length to px-per-inch using the known physical side.

    return (float(np.median(vals)) if vals else None)

def group_hull_from_all_corners(tags):
    """Build ONE polygon per group: convex hull of *all* outer corners from that group."""
    if not tags: return None
    pts = np.vstack([corners_disp(t) for t in tags]).astype(np.float32)
    if len(pts) < 3:
        return pts.astype(np.int32)  # line/segment
    # With <3 points, the best we can do is a line.
    hull = cv2.convexHull(pts, returnPoints=True).reshape(-1,2)
    # Why convex hull:
    # - We need ONE clean outer border around all tag edges; the hull encloses all points with no interior lines.
    # - Guarantees a non-self-intersecting polygon (planar), so edges never cross.
    # - Fast and parameter-free in OpenCV -> stable under noise/occlusion and easy to persist frame-to-frame.
    # - Alternatives: center-linking (MST/Delaunay) adds internal diagonals; concave hull/alpha-shape hugs
    #   indentations but needs tuning and can be brittle. Use hull for reliable, simple “outer outline.”
    return hull.astype(np.int32)
    # Int coordinates for drawing.

"""draw_label; Place a label near the polygon’s top-left without going off-screen. DONE FOr in screen testing"""
def draw_label(img, poly_pts, text):
    x,y,w,h = cv2.boundingRect(poly_pts.astype(np.int32))
    org = (x, max(0,y-10))
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, LABEL_SCALE, LABEL_THICK)
    if org[1]-th < 0:
        org = (x, min(img.shape[0]-5, y+h+th+5))
    # If above would go off-screen, put the label below.
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, LABEL_SCALE, LABEL_COLOR, LABEL_THICK, cv2.LINE_AA)
    # Draw the final label with anti-aliased text.

# ========= Main =========
cap = open_camera(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

detector = Detector(
    families=TAG_FAMILY, nthreads=2,
    quad_decimate=0.8, quad_sigma=0.4,
    refine_edges=True, decode_sharpening=0.5
)

# persist per-group
persist = {}  # g -> {"poly": np.ndarray, "area_in2": float|None, "age": int}
# Stores last known polygon & area per group to mitigate detection flicker.

print("Per-group SINGLE polygon from outer edges + area. (Ctrl+C to stop)")

try:
    while True:
        ok, frame = cap.read()
        if not ok: print("Camera frame not received."); break

        vis  = cv2.resize(frame, (0,0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
        # Apply SCALE_FACTOR to the color frame we will display and draw on.

        gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
        # Work in grayscale for robust tag detection.

        gray = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(gray)

        gray = cv2.GaussianBlur(gray, (3,3), 0.6)
        # Light blur to smooth noise without erasing corners.

        gray_padded = cv2.copyMakeBorder(gray, PAD, PAD, PAD, PAD, cv2.BORDER_REPLICATE)
        # Replicate edges so tags at the image boundary aren’t clipped during detection.

        dets = detector.detect(gray_padded)
        # Run AprilTag detection on the preprocessed image.

        # age out persistence
        for g in list(persist.keys()):
            persist[g]["age"] += 1

            if persist[g]["age"] > PERSIST_FOR_FRAMES:
                del persist[g]

        # group by rule
        groups = defaultdict(list)

        for t in dets:
            g = group_key_from_id(int(t.tag_id))
            # Map the tag’s ID to a group/team according to GROUP_MODE.
            if (ALLOWED_GROUPS is None) or (g in ALLOWED_GROUPS):
                groups[g].append(t)
            # Optionally filter to a subset of groups.

        # recompute one polygon per group
        for g, tags in groups.items():
            poly = group_hull_from_all_corners(tags)
            # Build the group’s single outline as the convex hull of all its outer corners.

            px_per_in = estimate_px_per_in(tags)
            # Estimate scale from the group’s tags (median px-per-inch across tags).

            area_in2 = None
            # Default area if scale or polygon is unavailable.

            if poly is not None and len(poly) >= 3 and px_per_in and px_per_in > 0:
                area_px2 = float(cv2.contourArea(poly.astype(np.float32)))
              # Computes the polygon’s area in pixel^2.
        # - Cast to float32 to match OpenCV’s preferred dtype and avoid type quirks.
        # - Assumes a closed, non-self-intersecting contour; our convex hull guarantees this.
        # - Returns absolute (unsigned) area by default; pass oriented=True if you ever need signed area.
        # - This is in pixels; we convert to square inches later using (px_per_in)**2.

                area_in2 = area_px2 / (px_per_in**2)
                # Convert to square inches using the estimated scale.

            persist[g] = {"poly": poly, "area_in2": area_in2, "age": 0}
            # Reset this group’s cached polygon and mark it as fresh.

        # draw persisted polygons + labels
        for i, (g, data) in enumerate(sorted(persist.items(), key=lambda x: x[0])):
            poly, area_in2 = data["poly"], data["area_in2"]
            # Retrieve the polygon and computed area for this group.

            if poly is None or len(poly) < 2: continue
            # Skip groups that don’t yet have enough geometry to draw.

            color = GROUP_COLORS[i % len(GROUP_COLORS)]
            # Cycle a stable color per group for readability.

            if len(poly) >= 3:
                cv2.polylines(vis, [poly.astype(np.int32)], True, color, EDGE_THICKNESS)
                # Draw the closed outer outline for the group (no interior edges).
            else:  # just a segment
                p1, p2 = tuple(poly[0]), tuple(poly[-1])
                cv2.line(vis, p1, p2, color, EDGE_THICKNESS)
                # If only two points exist, draw a simple segment.

            label = f"G{g}  Area ≈ {area_in2:.2f} in^2" if area_in2 is not None else f"G{g}  Area: n/a"
            # Compose the label text, including area when available.

            draw_label(vis, poly.astype(np.int32), label)
            # Place the label near the polygon, avoiding off-screen placement.

        # show per-tag boxes/IDs
        for t in dets:
            box = corners_disp(t).astype(int)
            cv2.polylines(vis, [box], True, (0,255,0), 2)

            cv2.putText(vis, f"ID:{t.tag_id}", (box[0][0], box[0][1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

        rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(rgb)

        clear_output(wait=True)

        display(img)
        # Show the current processed frame.

        time.sleep(0.01)

except KeyboardInterrupt:
    print("Stopped.")
    # Graceful stop on Ctrl+C.

finally:
    cap.release()
    # Free the camera device.

    clear_output(wait=True)
    # Clear the last frame from the notebook.

    print("Done.")
    # Final message for the session.
