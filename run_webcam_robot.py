"""
Live webcam → Fast-SAM-3D-Body pose estimation → GMR → Unitree G1 robot visualization.

Usage:
    conda activate fast_sam_3d_body
    python run_webcam_robot.py \
        --smpl-model-path checkpoints/smpl/SMPL_NEUTRAL.pkl \
        --nn-model-dir mhr2smpl/experiments/multiview_n30000_e500 \
        --mhr2smpl-mapping-path checkpoints/mhr_smpl_assets/mhr2smpl_mapping.npz \
        --mhr-mesh-path checkpoints/mhr_smpl_assets/mhr_face_mask.ply

Press Ctrl+C to stop.
"""

import argparse
import os
import queue
import sys
import threading
import time

# ---------------------------------------------------------------------------
# Performance env vars — set before any model imports so module-level checks
# (USE_COMPILE_BACKBONE, etc.) see these values.  All can be overridden from
# the shell: e.g.  USE_COMPILE=0 python run_webcam_robot.py ...
# ---------------------------------------------------------------------------
_perf_defaults = {
    # Decoder / compile
    "USE_COMPILE":                 "1",
    "USE_COMPILE_BACKBONE":        "1",
    "COMPILE_MODE":                "reduce-overhead",
    "COMPILE_WARMUP_BATCH_SIZES":  "1",
    "LAYER_DTYPE":                 "fp32",
    # Reduce decoder work
    "BODY_INTERM_PRED_LAYERS":     "0,1,2",
    "HAND_INTERM_PRED_LAYERS":     "0,1",
    "SKIP_KEYPOINT_PROMPT":        "1",
    "KEYPOINT_PROMPT_INTERM_INTERVAL": "999",
    # MHR head
    "MHR_NO_CORRECTIVES":          "1",
    # GPU hand preprocessing
    "GPU_HAND_PREP":               "1",
    # Backbone TRT (enabled once the engine is rebuilt for this GPU)
    "USE_TRT_BACKBONE":            "1",
    "TRT_BACKBONE_PATH":           "./checkpoints/sam-3d-body-dinov3/backbone_trt/backbone_dinov3_384_fp16.engine",
    # FOV estimator (MoGe2) — set before build_fov_estimator is imported so
    # module-level constants pick up the correct values.
    "FOV_MODEL":                   "s",   # small model (35M), fast enough for warmup
    "FOV_LEVEL":                   "0",   # fewest ViT tokens (1200), fastest inference
    "FOV_SIZE":                    "512", # resize input before running MoGe2
    "FOV_FAST":                    "1",   # skip normal_head, only predict depth/intrinsics
    # Debug off
    "DEBUG_NAN":                   "0",
}
for _k, _v in _perf_defaults.items():
    os.environ.setdefault(_k, _v)

import cv2
import numpy as np
import torch

# Use TF32 for FP32 matrix multiplications (Ampere/Ada/Blackwell/Hopper GPUs).
# Nearly identical accuracy to full FP32 but ~2x faster on Tensor Cores.
torch.set_float32_matmul_precision("high")

from loguru import logger
from scipy.spatial.transform import Rotation

# Suppress per-frame torch.cuda.empty_cache() calls inside the estimator —
# they synchronize the GPU and add 100-300 ms per frame for no benefit here.
_real_empty_cache = torch.cuda.empty_cache
_empty_cache_counter = 0
_EMPTY_CACHE_EVERY_N = 30  # only actually flush every 30 frames

def _throttled_empty_cache():
    global _empty_cache_counter
    _empty_cache_counter += 1
    if _empty_cache_counter % _EMPTY_CACHE_EVERY_N == 0:
        _real_empty_cache()

torch.cuda.empty_cache = _throttled_empty_cache

# ---------------------------------------------------------------------------
# Add GMR to the Python path before importing it
# ---------------------------------------------------------------------------
_GMR_ROOT = os.path.join(os.path.dirname(__file__), "..", "GMR")
if os.path.isdir(_GMR_ROOT):
    sys.path.insert(0, os.path.abspath(_GMR_ROOT))
else:
    # Fallback: try /root/code/GMR
    sys.path.insert(0, "/root/code/GMR")

# Fast-SAM-3D-Body imports

from mocap.core.multiview_mhr2smpl import MultiViewFusionRunner
from mocap.core.setup_estimator import build_default_estimator
from mocap.utils.video_source import create_video_source
from sam_3d_body.visualization.skeleton_visualizer import SkeletonVisualizer
from sam_3d_body.metadata.mhr70 import pose_info as mhr70_pose_info

# GMR imports
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer


# ---------------------------------------------------------------------------
# SMPL constants
# ---------------------------------------------------------------------------

# Standard SMPL 24-joint names (identical to SMPLX body-joint names used by GMR)
SMPL_JOINT_NAMES = [
    "pelvis",        # 0
    "left_hip",      # 1
    "right_hip",     # 2
    "spine1",        # 3
    "left_knee",     # 4
    "right_knee",    # 5
    "spine2",        # 6
    "left_ankle",    # 7
    "right_ankle",   # 8
    "spine3",        # 9
    "left_foot",     # 10
    "right_foot",    # 11
    "neck",          # 12
    "left_collar",   # 13
    "right_collar",  # 14
    "head",          # 15
    "left_shoulder", # 16
    "right_shoulder",# 17
    "left_elbow",    # 18
    "right_elbow",   # 19
    "left_wrist",    # 20
    "right_wrist",   # 21
    "left_hand",     # 22
    "right_hand",    # 23
]

# SMPL kinematic tree: parent[i] = index of joint i's parent (-1 = root)
SMPL_PARENTS = [
    -1,  # 0  pelvis
     0,  # 1  left_hip
     0,  # 2  right_hip
     0,  # 3  spine1
     1,  # 4  left_knee
     2,  # 5  right_knee
     3,  # 6  spine2
     4,  # 7  left_ankle
     5,  # 8  right_ankle
     6,  # 9  spine3
     7,  # 10 left_foot
     8,  # 11 right_foot
     9,  # 12 neck
     9,  # 13 left_collar
     9,  # 14 right_collar
    12,  # 15 head
    13,  # 16 left_shoulder
    14,  # 17 right_shoulder
    16,  # 18 left_elbow
    17,  # 19 right_elbow
    18,  # 20 left_wrist
    19,  # 21 right_wrist
    20,  # 22 left_hand
    21,  # 23 right_hand
]

# Rotation applied inside run_publisher._compute_body_quat to fix SMPL/camera convention
_X180 = Rotation.from_euler("x", 180.0, degrees=True)

# Standing root height in world frame (Z-up). Monocular camera gives no depth
# to the root translation, so we fix it to a sensible value.
_ROOT_HEIGHT_Z = 0.9  # metres


# ---------------------------------------------------------------------------
# One Euro Filter — adaptive low-pass filter (Casiez et al., 2012)
# Reduces jitter when signal is slow; minimises lag when signal moves fast.
# ---------------------------------------------------------------------------

class OneEuroFilter:
    """Vectorised One Euro Filter for numpy arrays.

    Parameters
    ----------
    min_cutoff : float
        Minimum cutoff frequency in Hz.  Lower = smoother when still.
    beta : float
        Speed coefficient.  Higher = less lag during fast motion.
    d_cutoff : float
        Cutoff for the derivative low-pass (usually left at 1.0 Hz).
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.05,
                 d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta       = beta
        self.d_cutoff   = d_cutoff
        self._x_prev: np.ndarray | None = None
        self._dx_prev: np.ndarray | None = None
        self._t_prev: float | None = None

    def reset(self):
        self._x_prev = self._dx_prev = self._t_prev = None

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        if self._x_prev is None:
            self._x_prev  = x.copy()
            self._dx_prev = np.zeros_like(x)
            self._t_prev  = t
            return x.copy()

        dt = t - self._t_prev
        if dt <= 0.0:
            return self._x_prev.copy()

        # Derivative estimate
        dx = (x - self._x_prev) / dt

        # Low-pass the derivative
        a_d    = 1.0 / (1.0 + 1.0 / (2.0 * np.pi * self.d_cutoff * dt))
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        # Per-element adaptive cutoff
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)

        # Low-pass the signal
        a     = 1.0 / (1.0 + 1.0 / (2.0 * np.pi * cutoff * dt))
        x_hat = a * x + (1.0 - a) * self._x_prev

        self._x_prev  = x_hat
        self._dx_prev = dx_hat
        self._t_prev  = t
        return x_hat


# ---------------------------------------------------------------------------
# Skeleton preview drawing — simplified body-only view
# ---------------------------------------------------------------------------

# Keypoint indices to render as dots.  Excludes:
#   22-40  right-hand intermediate joints  (only wrist=41 kept)
#   43-61  left-hand intermediate joints   (only wrist=62 kept)
#   63-66  extra elbow markers (olecranon, cubital fossa)
#   67-68  acromion markers (extra shoulder points)
_POSE_DOT_KPTS: frozenset = frozenset(range(21)) | {41, 62, 69}

# Only draw the first 25 skeleton_info entries (body + feet).
# Entries 25-64 are the detailed finger chains — replaced by _POSE_FINGER_RAYS.
_POSE_BODY_LINK_COUNT = 25

# Direct wrist-to-fingertip rays (no intermediate joints drawn).
# Right: wrist=41 → thumb4=21, middle4=29, pinky4=37
# Left:  wrist=62 → thumb4=42, middle4=50, pinky4=58
_POSE_FINGER_RAYS = [
    (41, 21), (41, 29), (41, 37),
    (62, 42), (62, 50), (62, 58),
]


def _draw_pose(image: np.ndarray, kp2d_conf: np.ndarray,
               skeleton, link_color, kpt_thr: float = 0.3) -> np.ndarray:
    """Draw body skeleton + wrist-to-tip finger rays on *image* (BGR, in-place copy).

    Parameters
    ----------
    kp2d_conf : (N, 3) array  — columns [x, y, score]
    skeleton   : list of [i, j] int pairs (from SkeletonVisualizer.skeleton)
    link_color : list of (R, G, B) tuples matching skeleton
    """
    img = image.copy()
    img_h, img_w = img.shape[:2]
    kpts   = kp2d_conf[:, :2]
    scores = kp2d_conf[:, 2]

    # --- body + feet links (first _POSE_BODY_LINK_COUNT entries) ---
    for sk_id in range(min(_POSE_BODY_LINK_COUNT, len(skeleton))):
        i, j = skeleton[sk_id]
        if scores[i] < kpt_thr or scores[j] < kpt_thr:
            continue
        if np.isnan(kpts[i, 0]) or np.isnan(kpts[j, 0]):
            continue
        p1 = (int(kpts[i, 0]), int(kpts[i, 1]))
        p2 = (int(kpts[j, 0]), int(kpts[j, 1]))
        if (0 < p1[0] < img_w and 0 < p1[1] < img_h and
                0 < p2[0] < img_w and 0 < p2[1] < img_h):
            color = tuple(int(c) for c in link_color[sk_id])
            cv2.line(img, p1, p2, color, 2)

    # --- minimal finger rays ---
    for w, t in _POSE_FINGER_RAYS:
        if w >= len(kpts) or t >= len(kpts):
            continue
        if scores[w] < kpt_thr or scores[t] < kpt_thr:
            continue
        if np.isnan(kpts[w, 0]) or np.isnan(kpts[t, 0]):
            continue
        p1 = (int(kpts[w, 0]), int(kpts[w, 1]))
        p2 = (int(kpts[t, 0]), int(kpts[t, 1]))
        cv2.line(img, p1, p2, (0, 200, 255), 2)

    # --- body keypoint dots ---
    for i in _POSE_DOT_KPTS:
        if i >= len(kpts) or scores[i] < kpt_thr:
            continue
        if np.isnan(kpts[i, 0]) or np.isnan(kpts[i, 1]):
            continue
        cv2.circle(img, (int(kpts[i, 0]), int(kpts[i, 1])), 5, (0, 0, 255), -1)

    return img


# ---------------------------------------------------------------------------
# Core conversion: SMPL params → GMR per-frame joint dict
# ---------------------------------------------------------------------------

def smpl_to_gmr_frame(
    body_pose_aa: np.ndarray,       # (21, 3) local axis-angle, joints 1-21
    joints_canonical: np.ndarray,  # (24, 3) root-relative, actual pose, identity global-orient
    global_rot_euler: np.ndarray,  # (3,)   ZYX Euler from estimator (camera frame)
    R_world_cam: np.ndarray,        # (3, 3) camera-to-world rotation matrix
    root_height_z: float = _ROOT_HEIGHT_Z,
) -> dict:
    """Convert one frame of SMPL pose to GMR's {joint_name: (pos, quat_wxyz)} dict.

    The returned dict contains all 24 SMPL body joints, but GMR only uses the
    ~13 joints listed in its IK config (pelvis, spine3, hips, knees, feet,
    shoulders, elbows, wrists).
    """
    # ---- Root orientation in world frame ----------------------------------
    # The estimator's global_rot is ZYX Euler in the camera convention where
    # SMPL Y-axis points down.  run_publisher._compute_body_quat corrects this
    # with an X-axis 180° flip.  We apply the same flip here.
    body_rot_cam = Rotation.from_euler("ZYX", global_rot_euler)
    root_rot_world = Rotation.from_matrix(R_world_cam) * _X180 * body_rot_cam

    # ---- Build per-joint local rotations (24 total) -----------------------
    # Joint 0  → root_rot_world (already in world frame)
    # Joints 1-21 → from body_pose_aa
    # Joints 22-23 → identity (left_hand / right_hand; not predicted by model)
    local_rots = [root_rot_world]
    for i in range(21):
        local_rots.append(Rotation.from_rotvec(body_pose_aa[i]))
    local_rots.append(Rotation.identity())  # 22: left_hand
    local_rots.append(Rotation.identity())  # 23: right_hand

    # ---- Chain kinematic tree to get global orientations ------------------
    global_rots = [None] * 24
    global_rots[0] = local_rots[0]
    for i in range(1, 24):
        parent = SMPL_PARENTS[i]
        global_rots[i] = global_rots[parent] * local_rots[i]

    # ---- World-space joint positions --------------------------------------
    # joints_canonical are computed with global_orient=zeros and actual body_pose,
    # so they reflect the true pose geometry in the identity-orientation frame.
    # Rotating by root_rot_world maps them into the world frame.
    world_joints = root_rot_world.apply(joints_canonical)
    # Shift the pelvis to a standing height (root-relative → absolute world)
    world_joints += np.array([0.0, 0.0, root_height_z], dtype=np.float64)

    # ---- Assemble the frame dict ------------------------------------------
    frame_dict = {}
    for i, name in enumerate(SMPL_JOINT_NAMES):
        frame_dict[name] = (
            world_joints[i].astype(np.float64),
            global_rots[i].as_quat(scalar_first=True).astype(np.float64),
        )
    return frame_dict


# ---------------------------------------------------------------------------
# Model warmup helper (mirrors run_publisher._warmup)
# ---------------------------------------------------------------------------

def _warmup(estimator, width: int = 640, height: int = 480):
    dummy = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    warmup_bbox = np.array([[0.0, 0.0, float(width - 1), float(height - 1)]], dtype=np.float32)
    for _ in range(2):
        estimator.process_one_image(dummy, bboxes=warmup_bbox, inference_type="body")
        if torch.cuda.is_available():
            torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

YOLO_MODEL_PATH = "checkpoints/yolo/yolo11m-pose.engine"
FOV_MODEL_SIZE = "s"
FOV_RESOLUTION_LEVEL = 0
FOV_FIXED_SIZE = 512
FOV_FAST_MODE = True


def main():
    parser = argparse.ArgumentParser(
        description="Live webcam → Fast-SAM-3D-Body → GMR → Unitree G1 visualization"
    )

    # Video source
    parser.add_argument("--device-index", type=int, default=0, help="Webcam device index")
    parser.add_argument("--width", type=int, default=640, help="Webcam capture width")
    parser.add_argument("--height", type=int, default=480, help="Webcam capture height")
    parser.add_argument("--fps", type=int, default=30, help="Webcam capture FPS")

    # Model checkpoints
    parser.add_argument("--smpl-model-path", type=str, required=True,
                        help="Path to SMPL_NEUTRAL.pkl")
    parser.add_argument("--nn-model-dir", type=str, required=True,
                        help="MHR→SMPL fusion model directory")
    parser.add_argument("--mhr2smpl-mapping-path", type=str, required=True,
                        help="Path to mhr2smpl_mapping.npz")
    parser.add_argument("--mhr-mesh-path", type=str, default=None,
                        help="Path to mhr_face_mask.ply (required for triangle_ids mapping)")
    parser.add_argument("--image-size", type=int, default=512, choices=[256, 384, 512],
                        help="SAM 3D body image size (must match TRT engine)")
    parser.add_argument("--yolo-model", type=str, default=YOLO_MODEL_PATH,
                        help="YOLO pose model path (.engine or .pt)")

    # Robot
    parser.add_argument("--robot", type=str, default="unitree_g1",
                        choices=["unitree_g1", "unitree_g1_with_hands", "unitree_h1",
                                 "unitree_h1_2", "booster_t1", "booster_t1_29dof",
                                 "stanford_toddy", "fourier_n1", "engineai_pm01",
                                 "kuavo_s45", "hightorque_hi", "galaxea_r1pro",
                                 "berkeley_humanoid_lite", "booster_k1",
                                 "pnd_adam_lite", "openloong", "tienkung", "fourier_gr3"],
                        help="Target robot for retargeting")
    parser.add_argument("--human-height", type=float, default=None,
                        help="Person height in metres (auto-estimated from SMPL betas if omitted)")
    parser.add_argument("--root-height", type=float, default=_ROOT_HEIGHT_Z,
                        help="Fixed world-space Z height for the pelvis (metres)")

    # Misc
    parser.add_argument("--preview", action="store_true",
                        help="Show webcam preview window with 2D skeleton overlay")
    parser.add_argument("--mesh-preview", action="store_true",
                        help="Overlay the MHR 3D mesh on the preview window (implies --preview; slower)")
    parser.add_argument("--min-person-confidence", type=float, default=0.75)
    parser.add_argument("--coord-debug", action="store_true",
                        help="Print root quaternion each frame for coordinate-frame diagnosis")

    parser.add_argument("--skip-frames", type=int, default=0,
                        help="Skip this many frames between inferences (0 = process every frame)")
    parser.add_argument("--robot-fps", type=int, default=30,
                        help="Robot viewer target FPS (decoupled from inference rate)")

    # Smoothing
    parser.add_argument("--no-smooth", action="store_true",
                        help="Disable One Euro Filter on robot qpos")
    parser.add_argument("--smooth-min-cutoff", type=float, default=1.0,
                        help="One Euro min cutoff Hz (lower = smoother when still, default 1.0)")
    parser.add_argument("--smooth-beta", type=float, default=0.05,
                        help="One Euro beta (higher = less lag on fast moves, default 0.05)")



    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # 1. Video source
    # -----------------------------------------------------------------------
    logger.info(f"Opening webcam device={args.device_index} {args.width}×{args.height} @ {args.fps}fps")
    video_source = create_video_source(
        "webcam",
        device_index=args.device_index,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )

    cam_intrinsics_np = video_source.get_camera_intrinsics()
    cam_intrinsics = (
        torch.from_numpy(np.asarray(cam_intrinsics_np, dtype=np.float32))
        if cam_intrinsics_np is not None else None
    )
    if cam_intrinsics is not None:
        logger.info(f"Camera intrinsics: fx={cam_intrinsics_np[0,0,0]:.1f} fy={cam_intrinsics_np[0,1,1]:.1f}")
    else:
        logger.info("No camera intrinsics — will use FOV estimator")

    # For a horizontal forward-facing webcam (no IMU) we bypass the gravity
    # pipeline and directly set the correct camera-to-world rotation.
    # Convention: (Rotation.from_matrix(R_world_cam) * X180).apply(v_body)
    #   maps SMPL Y-up → world Z-up and preserves vertical yaw correctly.
    # Camera frame: X=right, Y=down, Z=forward
    # World frame (Z-up): X=right, Y=forward, Z=up
    R_world_cam = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
    logger.info("Webcam: using hardcoded horizontal-camera R_world_cam")

    # -----------------------------------------------------------------------
    # 2. SAM 3D Body estimator
    # -----------------------------------------------------------------------
    logger.info("Loading SAM 3D Body estimator…")
    estimator = build_default_estimator(
        image_size=args.image_size,
        yolo_model_path=args.yolo_model,
        fov_model_size=FOV_MODEL_SIZE,
        fov_resolution_level=FOV_RESOLUTION_LEVEL,
        fov_fixed_size=FOV_FIXED_SIZE,
        fov_fast_mode=FOV_FAST_MODE,
    )
    logger.info("Warming up estimator…")
    _warmup(estimator, args.width, args.height)
    logger.info("Estimator ready")

    # -----------------------------------------------------------------------
    # 3. MHR → SMPL fusion runner
    # -----------------------------------------------------------------------
    logger.info("Loading MHR→SMPL fusion runner…")
    fusion_runner = MultiViewFusionRunner(
        smpl_model_path=args.smpl_model_path,
        model_dir=args.nn_model_dir,
        mapping_path=args.mhr2smpl_mapping_path,
        mhr_mesh_path=args.mhr_mesh_path,
    )
    logger.info("Fusion runner ready")

    # -----------------------------------------------------------------------
    # 4. GMR retargeter
    # -----------------------------------------------------------------------
    # Human height: use argument if given, otherwise estimate later from betas
    human_height = args.human_height  # may be None; updated on first frame
    logger.info(f"Initializing GMR for robot={args.robot}…")
    retarget = GMR(
        src_human="smplx",
        tgt_robot=args.robot,
        actual_human_height=human_height or 1.70,  # placeholder until we update below
        solver="daqp",
        damping=5e-1,
        use_velocity_limit=False,
    )
    logger.info("GMR ready")

    # -----------------------------------------------------------------------
    # 5. Robot motion viewer (MuJoCo)
    # -----------------------------------------------------------------------
    logger.info("Opening robot motion viewer…")
    robot_viewer = RobotMotionViewer(
        robot_type=args.robot,
        motion_fps=args.robot_fps,
        transparent_robot=0,
        record_video=False,
    )
    logger.info("Robot viewer ready")

    # -----------------------------------------------------------------------
    # 6. Optional 2D preview
    # -----------------------------------------------------------------------
    skeleton_vis = None
    if args.mesh_preview:
        args.preview = True  # mesh preview requires the preview window
    if args.preview:
        skeleton_vis = SkeletonVisualizer(line_width=2, radius=5)
        skeleton_vis.set_pose_meta(mhr70_pose_info)
        cv2.namedWindow(" ", cv2.WINDOW_NORMAL)  # makes window freely resizable
    if args.mesh_preview:
        # Use OSMesa (software GL) — EGL is unavailable inside Docker without
        # a properly configured EGL display device.
        os.environ["PYOPENGL_PLATFORM"] = "osmesa"
        from sam_3d_body.visualization.renderer import Renderer as _Renderer
        _MESH_COLOR = (0.65098039, 0.74117647, 0.85882353)  # light blue

    # -----------------------------------------------------------------------
    # 7. Shared state between inference thread and main (viewer) thread
    # -----------------------------------------------------------------------
    # latest_result: (qpos, human_data, vis_frame_bgr_or_None)
    _result_lock = threading.Lock()
    _latest_result = {"qpos": None, "human_data": None, "vis_frame": None}
    _stop_event = threading.Event()

    height_estimated = (args.human_height is not None)

    # One Euro Filter applied to qpos to reduce robot jitter
    qpos_filter = (
        None if args.no_smooth
        else OneEuroFilter(min_cutoff=args.smooth_min_cutoff, beta=args.smooth_beta)
    )
    if qpos_filter is not None:
        logger.info(
            f"One Euro Filter: min_cutoff={args.smooth_min_cutoff} Hz, beta={args.smooth_beta}"
        )

    # -----------------------------------------------------------------------
    # 8. Inference thread
    # -----------------------------------------------------------------------
    def _inference_loop():
        nonlocal height_estimated

        # torch.compile reduce-overhead mode creates thread-affine CUDA graph trees
        # whose manager is stored in C-level TLS.  The warmup ran on the main thread;
        # this thread needs its own TLS entries before any deferred cudagraph capture
        # fires here, otherwise cudagraph_trees.get_obj() raises AssertionError.
        # We set the keys directly via torch._C (always present) rather than relying
        # on the higher-level helper which may not exist in all PyTorch builds.
        try:
            import itertools as _itertools
            if not torch._C._is_key_in_tls("tree_manager_containers"):
                torch._C._set_obj_in_tls("tree_manager_containers", {})
            if not torch._C._is_key_in_tls("tree_manager_ids"):
                torch._C._set_obj_in_tls("tree_manager_ids", _itertools.count(0))
        except Exception:
            pass  # non-CUDA env or unexpected API change — safe to ignore

        infer_fps_counter = 0
        infer_fps_start = time.time()
        FPS_INTERVAL = 2.0
        skip_count = 0

        while not _stop_event.is_set():
            # -- Get frame --------------------------------------------------
            try:
                frame_bgr, _ = video_source.get_frame()
            except Exception as exc:
                logger.warning(f"Webcam read failed: {exc}")
                _stop_event.set()
                break
            if frame_bgr is None:
                continue

            # -- Optional frame skip ----------------------------------------
            if args.skip_frames > 0:
                skip_count += 1
                if skip_count % (args.skip_frames + 1) != 0:
                    continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # -- Pre-detect: always run on exactly one person (batch=1) --------
            # process_one_image with N>1 people triggers torch.compile to
            # specialise for the new batch size — a multi-second hang the first
            # time each new size is seen.  By pre-filtering to a single bbox we
            # guarantee batch=1 every frame regardless of scene occupancy.
            # For demos the subject stands in the centre; side viewers are
            # discarded by picking the detection whose box-centre is closest to
            # the middle of the frame.
            det = estimator.detector.run_human_detection(
                frame_bgr,
                bbox_thr=args.min_person_confidence,
                nms_thr=0.3,
                default_to_full_image=False,
            )
            boxes = det["boxes"] if isinstance(det, dict) else det

            if len(boxes) == 0:
                if args.preview:
                    with _result_lock:
                        _latest_result["vis_frame"] = frame_bgr.copy()
                continue

            if len(boxes) > 1:
                cx = (boxes[:, 0] + boxes[:, 2]) * 0.5
                cy = (boxes[:, 1] + boxes[:, 3]) * 0.5
                frame_cx = frame_bgr.shape[1] * 0.5
                frame_cy = frame_bgr.shape[0] * 0.5
                dist2 = (cx - frame_cx) ** 2 + (cy - frame_cy) ** 2
                idx = int(np.argmin(dist2))
            else:
                idx = 0
            best_bbox = boxes[idx : idx + 1, :4]  # (1, 4) XYXY

            # -- Pose estimation (body decoder only — hand decoder unused for robot)
            try:
                outputs = estimator.process_one_image(
                    frame_rgb,
                    bboxes=best_bbox,
                    cam_int=cam_intrinsics,
                    inference_type="body",
                )
            except AssertionError:
                # Rare: a new torch.compile specialisation triggered deferred
                # cudagraph capture before TLS was ready.  Re-initialise and
                # retry; the graph will be captured on the next call.
                try:
                    import itertools as _itertools
                    if not torch._C._is_key_in_tls("tree_manager_containers"):
                        torch._C._set_obj_in_tls("tree_manager_containers", {})
                    if not torch._C._is_key_in_tls("tree_manager_ids"):
                        torch._C._set_obj_in_tls("tree_manager_ids", _itertools.count(0))
                except Exception:
                    pass
                logger.warning("CUDA graph TLS miss — skipping frame, will recover next call")
                continue

            # Build preview frame
            vis_frame = frame_bgr if not args.preview else None

            if len(outputs) != 1:
                if args.preview:
                    vis_frame = frame_bgr.copy()
                with _result_lock:
                    _latest_result["vis_frame"] = vis_frame
                continue

            out = outputs[0]

            # -- MHR → SMPL conversion -------------------------------------
            pred_vertices = np.asarray(out["pred_vertices"], dtype=np.float32)
            pred_cam_t   = np.asarray(out["pred_cam_t"],    dtype=np.float32)

            body_pose_aa, joints_canonical, betas, _, smpl_verts = fusion_runner.infer(
                [(pred_vertices, pred_cam_t)]
            )

            if not height_estimated:
                h = float(1.66 + 0.1 * betas[0])
                retarget.actual_human_height = h
                logger.info(f"Auto-estimated human height from betas: {h:.2f} m")
                height_estimated = True

            # -- Build GMR frame dict --------------------------------------
            # Pass only the yaw (Y-axis) component from the monocular estimator.
            # Pitch and roll are zeroed — monocular lean estimates are too noisy
            # and could cause the robot to topple.
            _global_rot_raw = np.asarray(out["global_rot"], dtype=np.float64).reshape(3)
            global_rot_euler = np.zeros(3, dtype=np.float64)
            global_rot_euler[1] = _global_rot_raw[1]  # Y = yaw (left/right turn only)
            frame_dict = smpl_to_gmr_frame(
                body_pose_aa=body_pose_aa,
                joints_canonical=joints_canonical.astype(np.float64),
                global_rot_euler=global_rot_euler,
                R_world_cam=R_world_cam,
                root_height_z=args.root_height,
            )

            if args.coord_debug:
                logger.debug(f"pelvis quat [w,x,y,z]: {frame_dict['pelvis'][1]}")

            # -- Retarget to robot -----------------------------------------
            try:
                qpos = retarget.retarget(frame_dict, offset_to_ground=False)
            except Exception as exc:
                logger.warning(f"Retargeting failed: {exc}")
                continue

            # -- One Euro Filter on qpos -----------------------------------
            if qpos_filter is not None:
                qpos = qpos_filter(qpos, time.time())
                # Renormalise root quaternion (indices 3:7) after filtering
                q_norm = np.linalg.norm(qpos[3:7])
                if q_norm > 0.0:
                    qpos[3:7] /= q_norm

            # -- Build preview overlay -------------------------------------
            if args.preview:
                kp2d = out.get("pred_keypoints_2d")
                if kp2d is not None:
                    kp2d_conf = np.concatenate([kp2d, np.ones((kp2d.shape[0], 1))], axis=-1)
                    vis_frame = _draw_pose(
                        cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), kp2d_conf,
                        skeleton_vis.skeleton, skeleton_vis.link_color,
                    )
                else:
                    vis_frame = frame_bgr.copy()

                if args.mesh_preview:
                    try:
                        _fh, _fw = vis_frame.shape[:2]
                        _focal = float(out.get("focal_length") or 500.0)

                        def _render_canonical(verts, faces, pw=None, ph=None):
                            """Render verts on a white background, auto-scaled."""
                            _rh = ph or _fh
                            _rw = pw or _fw
                            _bg = np.ones((_rh, _rw, 3), dtype=np.uint8) * 255
                            v_min, v_max = verts.min(0), verts.max(0)
                            center = (v_min + v_max) / 2
                            extent = float(v_max[1] - v_min[1])  # body height
                            cam_t_z = _focal * extent / (0.8 * _rh)
                            cam_t = np.array([0.0, 0.0, cam_t_z], dtype=np.float32)
                            r = _Renderer(focal_length=_focal, faces=faces)
                            return (r(verts - center, cam_t, _bg,
                                      mesh_base_color=_MESH_COLOR,
                                      scene_bg_color=(1, 1, 1)) * 255).astype(np.uint8)

                        def _label(img, text):
                            """Stamp a small label in the top-left corner."""
                            out = img.copy()
                            cv2.putText(out, text, (8, 22),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1,
                                        cv2.LINE_AA)
                            return out

                        # Half-width panels for SMPL multi-angle views
                        _pw = _fw // 2

                        # Rotation matrices in camera-space (Y-down, Z-forward):
                        # rotate around Y-down axis to orbit the SMPL mesh.
                        # Ry(+90°): side view — body's +X side faces cam
                        # Ry(180°): back view
                        _Ry90  = np.array([[ 0, 0, 1], [0, 1, 0], [-1, 0, 0]], np.float32)
                        _Ry180 = np.array([[-1, 0, 0], [0, 1, 0], [ 0, 0,-1]], np.float32)
                        _Ry270 = np.array([[ 0, 0,-1], [0, 1, 0], [ 1, 0, 0]], np.float32)

                        # -- MHR panel: single front view (camera-space vertices)
                        _mhr_panel = _label(
                            _render_canonical(pred_vertices, estimator.faces),
                            "MHR front")

                        # -- SMPL panels (front / left-side / back / right-side)
                        # SMPL body-space: Y-up, front faces +Z.
                        # Renderer applies Rx(180°) which flips both Y and Z.
                        # Flip Y (Y-up→Y-down) AND Z (front+Z→-Z) so that after
                        # Rx(180°) the front ends up at +Z in OpenGL (closer to camera).
                        _sv = smpl_verts * np.array([1., -1., -1.], np.float32)
                        _smpl_f = _label(_render_canonical(_sv,              fusion_runner._smpl.faces, _pw), "SMPL front")
                        _smpl_l = _label(_render_canonical(_sv @ _Ry90,      fusion_runner._smpl.faces, _pw), "SMPL left")
                        _smpl_b = _label(_render_canonical(_sv @ _Ry180,     fusion_runner._smpl.faces, _pw), "SMPL back")
                        _smpl_r = _label(_render_canonical(_sv @ _Ry270,     fusion_runner._smpl.faces, _pw), "SMPL right")
                        # Stack into 2×2 grid, scale to single-panel size
                        _smpl_grid = np.concatenate([
                            np.concatenate([_smpl_f, _smpl_l], axis=1),
                            np.concatenate([_smpl_b, _smpl_r], axis=1),
                        ], axis=0)
                        _smpl_panel = cv2.resize(_smpl_grid, (_fw, _fh),
                                                 interpolation=cv2.INTER_AREA)

                        # skeleton | MHR mesh | SMPL 2×2 grid
                        vis_frame = np.concatenate([vis_frame, _mhr_panel, _smpl_panel], axis=1)
                    except Exception as _e:
                        import traceback as _tb; _tb.print_exc()

            # -- Publish result to main thread -----------------------------
            with _result_lock:
                _latest_result["qpos"]       = qpos.copy()
                _latest_result["human_data"] = retarget.scaled_human_data
                _latest_result["vis_frame"]  = vis_frame

            # -- Inference FPS stats ---------------------------------------
            infer_fps_counter += 1
            now = time.time()
            if now - infer_fps_start >= FPS_INTERVAL:
                fps = infer_fps_counter / (now - infer_fps_start)
                logger.info(f"Inference FPS: {fps:.1f}")
                infer_fps_counter = 0
                infer_fps_start = now

    infer_thread = threading.Thread(target=_inference_loop, daemon=True, name="inference")

    # -----------------------------------------------------------------------
    # 9. Main loop — robot viewer + preview (must run on main thread)
    # -----------------------------------------------------------------------
    logger.info("Starting live retargeting — press Ctrl+C to stop")
    infer_thread.start()

    try:
        while not _stop_event.is_set():
            with _result_lock:
                qpos       = _latest_result["qpos"]
                human_data = _latest_result["human_data"]
                vis_frame  = _latest_result["vis_frame"]

            if qpos is not None:
                robot_viewer.step(
                    root_pos=qpos[:3],
                    root_rot=qpos[3:7],
                    dof_pos=qpos[7:],
                    human_motion_data=human_data,
                    rate_limit=True,
                    follow_camera=False,
                )
            else:
                time.sleep(0.01)  # wait for first inference result

            if args.preview and vis_frame is not None:
                cv2.imshow(" ", vis_frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        _stop_event.set()
        infer_thread.join(timeout=5.0)
        video_source.release()
        if args.preview:
            cv2.destroyAllWindows()
        try:
            robot_viewer.close()
        except Exception:
            pass
        logger.success("Stopped.")


if __name__ == "__main__":
    main()
