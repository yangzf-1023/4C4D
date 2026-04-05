# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import os
import copy
from PIL import Image
import mediapy as media
from matplotlib import cm
from tqdm import tqdm
import torch

from scipy.interpolate import splprep, splev

def normalize(x: np.ndarray) -> np.ndarray:
    """Normalization helper function."""
    return x / (np.linalg.norm(x) + 1e-8)

def pad_poses(p: np.ndarray) -> np.ndarray:
    """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
    bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)

def unpad_poses(p: np.ndarray) -> np.ndarray:
    """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
    return p[..., :3, :4]

def viewmatrix(lookdir: np.ndarray, up: np.ndarray, position: np.ndarray) -> np.ndarray:
    """Construct lookat view matrix."""
    vec2 = normalize(lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, position], axis=1)
    return m

def focus_point_fn(poses: np.ndarray) -> np.ndarray:
    """Calculate nearest point to all focal axes in poses."""
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
    return focus_pt

def estimate_up_vector(poses: np.ndarray, focus_point: np.ndarray) -> np.ndarray:
    """
    Estimate a consistent up vector from camera poses.
    Uses the average Y-axis (up direction) of all cameras.
    """
    # Method 1: Average of all camera Y-axes
    up_vectors = poses[:, :3, 1]  # Extract Y-axis from all poses
    avg_up = np.mean(up_vectors, axis=0)
    avg_up = normalize(avg_up)
    
    # Method 2: Ensure up vector is perpendicular to average view direction
    positions = poses[:, :3, 3]
    avg_view_dir = normalize(focus_point - np.mean(positions, axis=0))
    
    # Make up perpendicular to view direction
    avg_up = avg_up - np.dot(avg_up, avg_view_dir) * avg_view_dir
    avg_up = normalize(avg_up)
    
    return avg_up

def fit_ellipse_to_points(points: np.ndarray, center: np.ndarray) -> tuple:
    """
    Fit an ellipse to 3D points around a center.
    Returns: (plane_normal, major_axis, minor_axis, semi_major, semi_minor)
    """
    # Center the points
    centered_points = points - center
    
    # Use PCA to find the best-fit plane
    _, _, Vt = np.linalg.svd(centered_points)
    plane_normal = Vt[-1]  # Normal is the smallest singular vector
    
    # Project points onto the plane
    # Create orthonormal basis for the plane
    major_axis = Vt[0]  # Direction of maximum variance
    minor_axis = np.cross(plane_normal, major_axis)
    minor_axis = normalize(minor_axis)
    
    # Project points onto the two plane axes
    x_coords = np.dot(centered_points, major_axis)
    y_coords = np.dot(centered_points, minor_axis)
    
    # Estimate semi-axes lengths
    semi_major = np.max(np.abs(x_coords))
    semi_minor = np.max(np.abs(y_coords))
    
    return plane_normal, major_axis, minor_axis, semi_major, semi_minor

def transform_poses_pca(poses: np.ndarray):
    """Transforms poses so principal components lie on XYZ axes.
    Args:
        poses: a (N, 3, 4) array containing the cameras' camera to world transforms.
    Returns:
        A tuple (poses, transform), with the transformed poses and the applied
        camera_to_world transforms.
    """
    t = poses[:, :3, 3]
    t_mean = t.mean(axis=0)
    t = t - t_mean

    eigval, eigvec = np.linalg.eig(t.T @ t)
    # Sort eigenvectors in order of largest to smallest eigenvalue.
    inds = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, inds]
    rot = eigvec.T
    if np.linalg.det(rot) < 0:
        rot = np.diag(np.array([1, 1, -1])) @ rot

    transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
    poses_recentered = unpad_poses(transform @ pad_poses(poses))
    transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

    # Flip coordinate system if z component of y-axis is negative
    if poses_recentered.mean(axis=0)[2, 1] < 0:
        poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
        transform = np.diag(np.array([1, -1, -1, 1])) @ transform

    return poses_recentered, transform

def generate_ellipse_path(poses: np.ndarray,
                          n_frames: int = 120,
                          const_speed: bool = True,
                          z_variation: float = 0.,
                          z_phase: float = 0.,
                          scale_factor: float = 1.0) -> np.ndarray:
    """Generate an elliptical render path based on the given poses."""
    # Calculate the focal point for the path (cameras point toward this).
    center = focus_point_fn(poses)
    z_base = np.min(poses[:, :3, 3][:, 2]) # z_median or z_mean = 0, center[2] is not available

    # Path height sits at z=0 (in middle of zero-mean capture pattern).
    offset = np.array([center[0], center[1], z_base])

    # Calculate scaling for ellipse axes based on input camera positions.
    sc = np.percentile(np.abs(poses[:, :3, 3] - offset), 90, axis=0) * scale_factor
    # sc = np.min(np.abs(poses[:, :3, 3] - offset), axis=0) * scale_factor

    # Use ellipse that is symmetric about the focal point in xy.
    low = -sc + offset
    high = sc + offset
    # Optional height variation need not be symmetric
    z_low = np.percentile((poses[:, :3, 3]), 10, axis=0)
    z_high = np.percentile((poses[:, :3, 3]), 90, axis=0)

    def get_positions(theta):
        # Interpolate between bounds with trig functions to get ellipse in x-y.
        # Optionally also interpolate in z to change camera height along path.
        return np.stack([
            low[0] + (high - low)[0] * (np.cos(theta) * .5 + .5),
            low[1] + (high - low)[1] * (np.sin(theta) * .5 + .5),
            z_base + z_variation * (z_low[2] + (z_high - z_low)[2] *
                        (np.cos(theta + 2 * np.pi * z_phase) * .5 + .5)),
        ], -1)

    theta = np.linspace(0, 2. * np.pi, n_frames + 1, endpoint=True)
    positions = get_positions(theta)

    # Throw away duplicated last position.
    positions = positions[:-1]

    # Set path's up vector to axis closest to average of input pose up vectors.
    avg_up = poses[:, :3, 1].mean(0)
    avg_up = avg_up / np.linalg.norm(avg_up)
    ind_up = np.argmax(np.abs(avg_up))
    up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])

    return np.stack([viewmatrix(p - center, up, p) for p in positions])

def generate_smooth_interpolation_path(poses: np.ndarray,
                                       n_frames: int = 120,
                                       smoothness: float = 0.0) -> np.ndarray:
    """
    Generate a smooth interpolation path through all input camera positions using splines.
    
    Args:
        poses: Input camera poses
        n_frames: Number of frames to generate
        smoothness: Smoothing parameter for spline (0 = pass through all points exactly)
    """
    # Find the focus point
    center = focus_point_fn(poses)
    
    # Get camera positions
    positions = poses[:, :3, 3]
    n_input = len(positions)
    
    # Sort positions by angle around the focus point (for closed loop)
    centered = positions - center
    
    # Project onto best-fit plane to get 2D angles
    _, _, Vt = np.linalg.svd(centered)
    basis_x = Vt[0]
    basis_y = Vt[1]
    
    x_coords = np.dot(centered, basis_x)
    y_coords = np.dot(centered, basis_y)
    angles = np.arctan2(y_coords, x_coords)
    
    # Sort by angle
    sort_idx = np.argsort(angles)
    positions_sorted = positions[sort_idx]
    
    # Add first point at end to close the loop
    positions_loop = np.vstack([positions_sorted, positions_sorted[0:1]])
    
    # Fit a smooth spline through the positions
    tck, u = splprep([positions_loop[:, 0], positions_loop[:, 1], positions_loop[:, 2]], 
                     s=smoothness, per=True)
    
    # Evaluate spline at n_frames points
    u_new = np.linspace(0, 1, n_frames, endpoint=False)
    interpolated = splev(u_new, tck)
    interpolated_positions = np.column_stack(interpolated)
    
    # Estimate up vector
    up = estimate_up_vector(poses, center)
    
    # Generate camera poses
    camera_poses = np.stack([viewmatrix(p - center, up, p) for p in interpolated_positions])
    
    return camera_poses

def generate_arc_path(poses: np.ndarray,
                      n_frames: int = 120,
                      scale_factor: float = 1.0,
                      arc_extension: float = -0.1,
                      clockwise: bool = True) -> np.ndarray: 
    """
    Generate an elliptical arc path that covers only the visible portion.
    
    Args:
        poses: Input camera poses
        n_frames: Number of frames to generate
        scale_factor: Scale factor for the ellipse size
        arc_extension: Extra arc length beyond input cameras
        clockwise: If True, camera moves clockwise (从上往下看); if False, counter-clockwise
    """
    print("\n" + "="*70)
    print("  Generating Ellipse Arc Camera Path")
    print("="*70)
    
    # Find the focus point
    center = focus_point_fn(poses)
    print(f"\n[1] Ellipse Center (Focal Point):")
    print(f"    Position: [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]")
    
    # Get camera positions
    positions = poses[:, :3, 3]
    
    # Fit an ellipse to the camera positions
    plane_normal, major_axis, minor_axis, semi_major, semi_minor = fit_ellipse_to_points(positions, center)
    
    print(f"\n[2] Fitted Ellipse Parameters:")
    print(f"    Semi-major axis: {semi_major:.4f}")
    print(f"    Semi-minor axis: {semi_minor:.4f}")
    print(f"    Plane normal: [{plane_normal[0]:.4f}, {plane_normal[1]:.4f}, {plane_normal[2]:.4f}]")
    
    
    ellipse_normal = np.cross(major_axis, minor_axis)
    ellipse_normal = ellipse_normal / np.linalg.norm(ellipse_normal)
    
    up = estimate_up_vector(poses, center)
    reference_up = up / np.linalg.norm(up)
    
    if np.dot(ellipse_normal, reference_up) < 0:
        minor_axis = -minor_axis
        ellipse_normal = -ellipse_normal
        print(f"\n[*] Flipped minor_axis to align with reference up vector")
    
    print(f"    Ellipse normal (aligned): [{ellipse_normal[0]:.4f}, {ellipse_normal[1]:.4f}, {ellipse_normal[2]:.4f}]")
    
    # Scale the ellipse
    semi_major *= scale_factor
    semi_minor *= scale_factor
    
    # Calculate angles of input cameras
    centered_positions = positions - center
    input_angles = []
    
    for pos in centered_positions:
        x = np.dot(pos, major_axis)
        y = np.dot(pos, minor_axis)
        angle = np.arctan2(y / semi_minor, x / semi_major)
        input_angles.append(angle)
    
    input_angles = np.array(input_angles)
    
    angles_normalized = np.mod(input_angles, 2 * np.pi)
    span_raw = np.max(input_angles) - np.min(input_angles)
    span_normalized = np.max(angles_normalized) - np.min(angles_normalized)
    wraps_around = span_normalized < span_raw - np.pi
    
    if wraps_around:
        angles_unwrapped = np.where(input_angles < 0, input_angles + 2*np.pi, input_angles)
        min_angle = np.min(angles_unwrapped)
        max_angle = np.max(angles_unwrapped)
        print(f"\n[4] Detected wrap-around at 0°")
    else:
        min_angle = np.min(input_angles)
        max_angle = np.max(input_angles)
        print(f"\n[4] No wrap-around detected")
    
    angle_span = max_angle - min_angle
    
    print(f"    Min angle: {np.degrees(min_angle):7.2f}°")
    print(f"    Max angle: {np.degrees(max_angle):7.2f}°")
    print(f"    Span: {np.degrees(angle_span):7.2f}°")
    
    # Extend the arc
    arc_padding = angle_span * abs(arc_extension)
    arc_start = min_angle - arc_padding
    arc_end = max_angle + arc_padding
    
    print(f"\n[5] Arc Extension:")
    print(f"    Extension: ±{np.degrees(arc_padding):7.2f}°")
    print(f"    Arc start: {np.degrees(arc_start):7.2f}°")
    print(f"    Arc end: {np.degrees(arc_end):7.2f}°")
    print(f"    Total arc: {np.degrees(arc_end - arc_start):7.2f}°")
    
    
    if not clockwise:  
        theta = np.linspace(arc_start, arc_end, n_frames, endpoint=True)
        print(f"\n[*] Direction: COUNTER-CLOCKWISE (angle increasing)")
    else:  
        theta = np.linspace(arc_end, arc_start, n_frames, endpoint=True)
        print(f"\n[*] Direction: CLOCKWISE (angle decreasing)")
    
    # Generate points along the arc
    ellipse_positions = np.zeros((n_frames, 3))
    for i, t in enumerate(theta):
        offset = semi_major * np.cos(t) * major_axis + semi_minor * np.sin(t) * minor_axis
        ellipse_positions[i] = center + offset
    
    print(f"\n[6] Up Vector:")
    print(f"    Direction: [{up[0]:.4f}, {up[1]:.4f}, {up[2]:.4f}]")
    
    # Generate camera poses
    camera_poses = np.stack([viewmatrix(p - center, up, p) for p in ellipse_positions])
    
    print(f"\n[7] Path Direction Verification:")
    print(f"    First frame angle: {np.degrees(theta[0]):7.2f}°")
    print(f"    Last frame angle: {np.degrees(theta[-1]):7.2f}°")
    print(f"    Angle change: {np.degrees(theta[-1] - theta[0]):7.2f}°")
    
    mid_idx = len(theta) // 2
    pos_start = ellipse_positions[0] - center
    pos_mid = ellipse_positions[mid_idx] - center
    pos_end = ellipse_positions[-1] - center
    
    cross1 = np.cross(pos_start, pos_mid)
    cross2 = np.cross(pos_mid, pos_end)
    avg_cross = (cross1 + cross2) / 2
    
    rotation_direction = "COUNTER-CLOCKWISE" if np.dot(avg_cross, ellipse_normal) > 0 else "CLOCKWISE"
    print(f"    Physical rotation: {rotation_direction}")
    print(f"    Expected rotation: {'CLOCKWISE' if clockwise else 'COUNTER-CLOCKWISE'}")
    
    if (clockwise and rotation_direction == "CLOCKWISE") or \
       (not clockwise and rotation_direction == "COUNTER-CLOCKWISE"):
        print(f"    ✓ Direction matches expectation!")
    else:
        print(f"    ✗ WARNING: Direction mismatch!")
    
    print(f"\n✓ Successfully generated {n_frames} camera poses")
    print("="*70 + "\n")
    
    return camera_poses

def generate_path(viewpoint_cameras, n_frames=480, traj='ellipse', scale_factor=1.0, total_frames=300, fix_time=False, selected_frame=-1):

    # Sample cameras
    viewpoint_cameras = [viewpoint_cameras[i] for i in range(0, len(viewpoint_cameras), total_frames)]
    
    c2ws = np.array([np.linalg.inv(np.asarray((cam[1].world_view_transform.T).cpu().numpy())) 
                     for cam in viewpoint_cameras])
    
    poses = c2ws[:, :3, :] @ np.diag([1, -1, -1, 1])
    
    # Generate new poses based on trajectory type
    if traj == 'ellipse':
        pose_recenter, colmap_to_world_transform = transform_poses_pca(poses)
        new_poses = generate_ellipse_path(poses=pose_recenter, n_frames=n_frames, scale_factor=scale_factor)
        new_poses = np.linalg.inv(colmap_to_world_transform) @ pad_poses(new_poses)
    elif traj == 'interpolate':
        new_poses = generate_smooth_interpolation_path(poses=poses, n_frames=n_frames)
    elif traj == 'arc':
        new_poses = generate_arc_path(poses=poses, n_frames=n_frames, scale_factor=scale_factor, clockwise=False)
    else:
        raise ValueError(f'Trajectory type {traj} not supported.')

    new_poses = new_poses @ np.diag([1, -1, -1, 1])
    
    if selected_frame >= 0:
        new_poses = [new_poses[selected_frame]] * n_frames
        
    traj_archived = traj
    # Create camera objects
    traj = []
    for i, pose_3x4 in enumerate(new_poses):
        cam = copy.deepcopy(viewpoint_cameras[0][1])
        cam.image_height = int(cam.image_height / 2) * 2
        cam.image_width = int(cam.image_width / 2) * 2
        
        if traj_archived != 'ellipse':
            c2w_4x4 = np.eye(4)
            c2w_4x4[:3, :] = pose_3x4
            w2c_4x4 = np.linalg.inv(c2w_4x4)
            cam.world_view_transform = torch.from_numpy(w2c_4x4.T).float().cuda()
        else:
            cam.world_view_transform = torch.from_numpy(np.linalg.inv(pose_3x4).T).float().cuda()
            
        cam.full_proj_transform = (cam.world_view_transform.unsqueeze(0).bmm(
            cam.projection_matrix.cuda().unsqueeze(0))).squeeze(0)
        cam.camera_center = cam.world_view_transform.inverse()[3, :3]
        if not fix_time:
            cam.timestamp = 10.0 / n_frames * i
        else:
            cam.timestamp = 10.0 / n_frames * i * total_frames / 300.0
        traj.append(cam)
    
    return traj

def load_img(pth: str) -> np.ndarray:
    """Load an image and cast to float32."""
    with open(pth, 'rb') as f:
        image = np.array(Image.open(f), dtype=np.float32)
    return image

def create_videos(base_dir, input_dir, out_name, num_frames=480):
    """Creates videos out of the images saved to disk."""
    # Last two parts of checkpoint path are experiment name and scene name.
    video_prefix = f'{out_name}'
    zpad = max(5, len(str(num_frames - 1)))
    idx_to_str = lambda idx: str(idx).zfill(zpad)

    os.makedirs(base_dir, exist_ok=True)
    render_dist_curve_fn = np.log

    # Load one example frame to get image shape and depth range.
    color_file = os.path.join(input_dir, 'renders', f'{idx_to_str(0)}.png')
    color_frame = load_img(color_file)
    shape = color_frame.shape
    p = 3
    distance_limits = np.percentile(color_frame.flatten(), [p, 100 - p])
    lo, hi = [render_dist_curve_fn(x) for x in distance_limits]
    print(f'Video shape is {shape[:2]}')

    video_kwargs = {
        'shape': shape[:2],
        'codec': 'h264',
        'fps': 48,
        'crf': 18,
    }

    for k in ['depth', 'normal', 'color','flow']:
        video_file = os.path.join(base_dir, f'{video_prefix}_{k}.mp4')
        input_format = 'gray' if k == 'alpha' else 'rgb'

        file_ext = 'png' if k in ['color', 'normal','flow'] else 'tiff'
        idx = 0

        if k == 'color':
            file0 = os.path.join(input_dir, 'renders', f'{idx_to_str(0)}.{file_ext}')
        elif k=='flow':
            file0 = os.path.join(input_dir, 'flow', f'{k}_{idx_to_str(0)}.{file_ext}')
        else:
            file0 = os.path.join(input_dir, 'vis', f'{k}_{idx_to_str(0)}.{file_ext}')

        if not os.path.exists(file0):
            print(f'Images missing for tag {k}')
            continue
        print(f'Making video {video_file}...')
        with media.VideoWriter(video_file, **video_kwargs, input_format=input_format) as writer:
            for idx in tqdm(range(num_frames)):
                # img_file = os.path.join(input_dir, f'{k}_{idx_to_str(idx)}.{file_ext}')
                if k == 'color':
                    img_file = os.path.join(input_dir, 'renders', f'{idx_to_str(idx)}.{file_ext}')
                elif k=='flow':
                    img_file = os.path.join(input_dir, 'flow', f'{idx_to_str(idx)}.{file_ext}')
                else:
                    img_file = os.path.join(input_dir, 'vis', f'{k}_{idx_to_str(idx)}.{file_ext}')

                if not os.path.exists(img_file):
                    ValueError(f'Image file {img_file} does not exist.')
                img = load_img(img_file)

                if k in ['color', 'normal', 'flow']:
                    img = img / 255.
                elif k.startswith('depth'):
                    img = render_dist_curve_fn(img)
                    img = np.clip((img - np.minimum(lo, hi)) / np.abs(hi - lo), 0, 1)
                    img = cm.get_cmap('turbo')(img)[..., :3]

                frame = (np.clip(np.nan_to_num(img), 0., 1.) * 255.).astype(np.uint8)
                writer.add_image(frame)
                idx += 1

def save_img_u8(img, pth):
    """Save an image (probably RGB) in [0, 1] to disk as a uint8 PNG."""
    with open(pth, 'wb') as f:
        Image.fromarray(
            (np.clip(np.nan_to_num(img), 0., 1.) * 255.).astype(np.uint8)).save(
                f, 'PNG')

def save_img_f32(depthmap, pth):
    """Save an image (probably a depthmap) to disk as a float32 TIFF."""
    with open(pth, 'wb') as f:
        Image.fromarray(np.nan_to_num(depthmap).astype(np.float32)).save(f, 'TIFF')