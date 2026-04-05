import os
import argparse
import glob

import numpy as np
import json
import sys
import math
import shutil
import sqlite3

IS_PYTHON3 = sys.version_info[0] >= 3
MAX_IMAGE_ID = 2**31 - 1

CREATE_CAMERAS_TABLE = """CREATE TABLE IF NOT EXISTS cameras (
    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    params BLOB,
    prior_focal_length INTEGER NOT NULL)"""

CREATE_DESCRIPTORS_TABLE = """CREATE TABLE IF NOT EXISTS descriptors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""

CREATE_IMAGES_TABLE = """CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    camera_id INTEGER NOT NULL,
    prior_qw REAL,
    prior_qx REAL,
    prior_qy REAL,
    prior_qz REAL,
    prior_tx REAL,
    prior_ty REAL,
    prior_tz REAL,
    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {}),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))
""".format(MAX_IMAGE_ID)

CREATE_TWO_VIEW_GEOMETRIES_TABLE = """
CREATE TABLE IF NOT EXISTS two_view_geometries (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    config INTEGER NOT NULL,
    F BLOB,
    E BLOB,
    H BLOB,
    qvec BLOB,
    tvec BLOB)
"""

CREATE_KEYPOINTS_TABLE = """CREATE TABLE IF NOT EXISTS keypoints (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)
"""

CREATE_MATCHES_TABLE = """CREATE TABLE IF NOT EXISTS matches (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB)"""

CREATE_NAME_INDEX = \
    "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)"

CREATE_ALL = "; ".join([
    CREATE_CAMERAS_TABLE,
    CREATE_IMAGES_TABLE,
    CREATE_KEYPOINTS_TABLE,
    CREATE_DESCRIPTORS_TABLE,
    CREATE_MATCHES_TABLE,
    CREATE_TWO_VIEW_GEOMETRIES_TABLE,
    CREATE_NAME_INDEX
])

def array_to_blob(array):
    if IS_PYTHON3:
        return array.tobytes()
    else:
        return np.getbuffer(array)

def blob_to_array(blob, dtype, shape=(-1,)):
    if IS_PYTHON3:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)
    else:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)

class COLMAPDatabase(sqlite3.Connection):

    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)

    def __init__(self, *args, **kwargs):
        super(COLMAPDatabase, self).__init__(*args, **kwargs)

        self.create_tables = lambda: self.executescript(CREATE_ALL)
        self.create_cameras_table = \
            lambda: self.executescript(CREATE_CAMERAS_TABLE)
        self.create_descriptors_table = \
            lambda: self.executescript(CREATE_DESCRIPTORS_TABLE)
        self.create_images_table = \
            lambda: self.executescript(CREATE_IMAGES_TABLE)
        self.create_two_view_geometries_table = \
            lambda: self.executescript(CREATE_TWO_VIEW_GEOMETRIES_TABLE)
        self.create_keypoints_table = \
            lambda: self.executescript(CREATE_KEYPOINTS_TABLE)
        self.create_matches_table = \
            lambda: self.executescript(CREATE_MATCHES_TABLE)
        self.create_name_index = lambda: self.executescript(CREATE_NAME_INDEX)

    def update_camera(self, model, width, height, params, camera_id):
        params = np.asarray(params, np.float64)
        cursor = self.execute(
            "UPDATE cameras SET model=?, width=?, height=?, params=?, prior_focal_length=1 WHERE camera_id=?",
            (model, width, height, array_to_blob(params), camera_id))
        return cursor.lastrowid

def camTodatabase(txtfile, database_path):
    import os

    # Only support PINHOLE
    camModelDict = {'PINHOLE': 1}

    if not os.path.exists(database_path):
        print("ERROR: database path doesn't exist -- please check database.db.")
        return
    # Open the database.
    db = COLMAPDatabase.connect(database_path)

    idList, modelList, widthList, heightList, paramsList = [], [], [], [], []

    with open(txtfile, "r") as cam:
        lines = cam.readlines()
        for i in range(0, len(lines), 1):
            line = lines[i].strip()
            if not line or line.startswith('#'):
                continue
            strLists = line.split()
            cameraId = int(strLists[0])
            cameraModelStr = strLists[1]
            if cameraModelStr not in camModelDict:
                db.close()
                raise ValueError(f"Only PINHOLE is supported. Found: {cameraModelStr}")
            cameraModel = camModelDict[cameraModelStr]
            width = int(strLists[2])
            height = int(strLists[3])
            params_raw = strLists[4:]
            if len(params_raw) != 4:
                db.close()
                raise ValueError(f"PINHOLE requires 4 params (fx fy cx cy). Got {len(params_raw)} for camera {cameraId}")
            params = np.array(params_raw, dtype=np.float64)
            idList.append(cameraId)
            modelList.append(cameraModel)
            widthList.append(width)
            heightList.append(height)
            paramsList.append(params)
            _ = db.update_camera(cameraModel, width, height, params, cameraId)

    db.commit()
    rows = db.execute("SELECT * FROM cameras")
    for i in range(0, len(idList), 1):
        camera_id, model, width, height, params, prior = next(rows)
        params = blob_to_array(params, np.float64)
        assert camera_id == idList[i]
        assert model == modelList[i] and width == widthList[i] and height == heightList[i]
        assert np.allclose(params, paramsList[i])

    db.close()

def do_system(arg):
    print(f"==== running: {arg}")
    err = os.system(arg)
    if err:
        print("FATAL: command failed")
        sys.exit(err)

# returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
def closest_point_2_lines(oa, da, ob, db):
    da = da / np.linalg.norm(da)
    db = db / np.linalg.norm(db)
    c = np.cross(da, db)
    denom = np.linalg.norm(c)**2
    t = ob - oa
    ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
    tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
    if ta > 0:
        ta = 0
    if tb > 0:
        tb = 0
    return (oa+ta*da+ob+tb*db) * 0.5, denom

def rotmat(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))

# ========== 从COLMAP cameras.txt读取相机参数（仅PINHOLE） ==========
def read_colmap_cameras(cameras_txt_path):
    cameras = {}
    with open(cameras_txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            camera_id = int(parts[0])
            model = parts[1]
            if model != 'PINHOLE':
                raise ValueError(f"Only PINHOLE is supported. Found camera {camera_id} with model {model}")
            width = int(parts[2])
            height = int(parts[3])
            params = [float(p) for p in parts[4:]]
            if len(params) != 4:
                raise ValueError(f"PINHOLE requires 4 params (fx fy cx cy). Got {len(params)} for camera {camera_id}")
            cameras[camera_id] = {
                'model': model,
                'width': width,
                'height': height,
                'params': params
            }
    return cameras

# ========== 从COLMAP images.txt读取图像和位姿 ==========
def read_colmap_images(images_txt_path):
    images = {}
    with open(images_txt_path, 'r') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line or line.startswith('#'):
                i += 1
                continue
            parts = line.split()
            image_id = int(parts[0])
            qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
            camera_id = int(parts[8])
            name = parts[9]
            R = quat2rotmat([qw, qx, qy, qz])
            pose = np.eye(4)
            pose[:3, :3] = R.T
            pose[:3, 3] = -R.T @ np.array([tx, ty, tz])
            images[name] = {
                'image_id': image_id,
                'camera_id': camera_id,
                'pose': pose
            }
            i += 2
    return images

def quat2rotmat(q):
    qw, qx, qy, qz = q
    R = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])
    return R

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path", default="", help="input path to the video")
    parser.add_argument('--skip_extract', action='store_true', default=False, help='whether to extract images from videos')
    parser.add_argument('--training_view', type=str, default=None, help='the views used for training')
    args = parser.parse_args()

    if args.path[-1] != '/':
        args.path += '/'
        
    videos = [os.path.join(args.path, vname) for vname in os.listdir(args.path) if vname.endswith(".mp4")]
    images_path = os.path.join(args.path, "images/")
    os.makedirs(images_path, exist_ok=True)
    
    if not args.skip_extract:
        for video in videos:
            cam_name = video.split('/')[-1].split('.')[-2]
            do_system(f"ffmpeg -i {video} -start_number 0 {images_path}/{cam_name}_%04d.png")
        
    images = [f[len(args.path):] for f in sorted(glob.glob(os.path.join(args.path, "images/", "*"))) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    cams = sorted(set([im[7:12] for im in images]))
    
    # ===== 修改1: 提前确定训练视角 =====
    cam_indices = []
    for cam in cams:
        cam_num = int(cam.replace('cam', ''))
        cam_indices.append(cam_num)
    
    desired_train_cams = cam_indices if args.training_view is None else args.training_view.split(',')
    desired_train_cams = set([int(cam) for cam in desired_train_cams])
    actual_train_cams = desired_train_cams.intersection(set(cam_indices))
    if actual_train_cams != desired_train_cams:
        raise ValueError(f"Some desired training cameras {desired_train_cams} are not present in the dataset. Present cameras: {set(cam_indices)}")
    
    print(f"[INFO] Training cameras: {sorted(actual_train_cams)}")
    # ===== 修改1结束 =====
    
    colmap_workspace = os.path.join(args.path, 'colmap_sfm')
    os.makedirs(colmap_workspace, exist_ok=True)
    db_path = os.path.join(colmap_workspace, 'database.db')
    
    first_frames_path = os.path.join(colmap_workspace, 'images')
    os.makedirs(first_frames_path, exist_ok=True)
    
    # ===== 修改2: 只选择训练视角的第一帧 =====
    print("[INFO] Selecting first frame from training cameras only...")
    first_frame_images = []
    for i, cam in enumerate(cams):
        # 只处理训练视角的相机
        if cam_indices[i] not in actual_train_cams:
            print(f"  - Skipping {cam} (not in training views)")
            continue
            
        first_frame = f"images/{cam}_0000.png"
        full_path = os.path.join(args.path, first_frame)
        if os.path.exists(full_path):
            link_path = os.path.join(first_frames_path, f"{cam}_0000.png")
            if not os.path.exists(link_path):
                try:
                    os.symlink(os.path.abspath(full_path), link_path)
                except OSError:
                    shutil.copy2(os.path.abspath(full_path), link_path)
            first_frame_images.append(first_frame)
            print(f"  - Added {first_frame}")
        else:
            print(f"  [WARNING] First frame not found: {first_frame}")
    # ===== 修改2结束 =====
    
    print(f"[INFO] Selected {len(first_frame_images)} first frames from training cameras for SfM reconstruction")
    print("[INFO] Running COLMAP Structure-from-Motion on training camera first frames only...")
    
    # 1) 特征提取：PINHOLE
    do_system(f"colmap feature_extractor \
                --database_path {db_path} \
                --image_path {first_frames_path} \
                --ImageReader.single_camera 1 \
                --ImageReader.camera_model PINHOLE \
                --SiftExtraction.use_gpu 1 \
                --SiftExtraction.max_num_features 16384 \
                --SiftExtraction.edge_threshold 10 \
                --SiftExtraction.peak_threshold 0.01 \
                --SiftExtraction.first_octave -1")

    # 2) 穷举匹配（exhaustive）
    do_system(f"colmap exhaustive_matcher \
                --database_path {db_path} \
                --SiftMatching.use_gpu 1 \
                --SiftMatching.max_ratio 0.9 \
                --SiftMatching.max_num_matches 32768 \
                --SiftMatching.cross_check 1")

    sfm_output_path = os.path.join(colmap_workspace, 'sparse')
    os.makedirs(sfm_output_path, exist_ok=True)

    # 3) Mapper
    do_system(f"colmap mapper \
                --database_path {db_path} \
                --image_path {first_frames_path} \
                --output_path {sfm_output_path} \
                --Mapper.init_min_num_inliers 20 \
                --Mapper.abs_pose_min_num_inliers 15 \
                --Mapper.ignore_watermarks 1 \
                --Mapper.ba_global_max_num_iterations 50")
    
    model_path = os.path.join(sfm_output_path, '0')
    if not os.path.exists(model_path):
        print("[ERROR] COLMAP reconstruction failed! No model found in default location.")
        print("[INFO] Checking for alternative model directories...")
        possible_dirs = [d for d in os.listdir(sfm_output_path) if os.path.isdir(os.path.join(sfm_output_path, d))]
        if possible_dirs:
            model_path = os.path.join(sfm_output_path, possible_dirs[0])
            print(f"[INFO] Using model from: {model_path}")
        else:
            sys.exit("[FATAL] No COLMAP reconstruction model found!")
    
    do_system(f"colmap model_converter \
                --input_path {model_path} \
                --output_path {model_path} \
                --output_type TXT")
    
    cameras_txt = os.path.join(model_path, 'cameras.txt')
    images_txt = os.path.join(model_path, 'images.txt')
    
    colmap_cameras = read_colmap_cameras(cameras_txt)  # Only PINHOLE accepted
    colmap_images = read_colmap_images(images_txt)
    
    print(f'[INFO] COLMAP reconstructed {len(colmap_images)} images with {len(colmap_cameras)} cameras')
    
    # Intrinsics: only PINHOLE
    first_camera = list(colmap_cameras.values())[0]
    if first_camera['model'] != 'PINHOLE':
        raise ValueError(f"Only PINHOLE is supported. Found {first_camera['model']}")
    W = first_camera['width']
    H = first_camera['height']
    fx = first_camera['params'][0]
    fy = first_camera['params'][1]
    cx = first_camera['params'][2]
    cy = first_camera['params'][3]
    
    print(f'[INFO] H = {H}, W = {W}, fx = {fx}, fy = {fy}, cx = {cx}, cy = {cy}')
    
    # ===== 修改3: 只为所有相机分配位姿，但优先使用COLMAP结果 =====
    N = len(cams)
    poses = []
    for i, cam in enumerate(cams):
        cam_first_frame = f"{cam}_0000.png"
        if cam_first_frame in colmap_images:
            # 这个相机在COLMAP重建中（即训练视角）
            pose = colmap_images[cam_first_frame]['pose']
            print(f"[INFO] Camera {cam} using COLMAP pose")
        else:
            # 这个相机不在COLMAP重建中（即测试视角），使用单位矩阵
            print(f"[WARNING] Camera {cam} not in COLMAP reconstruction (test view), using identity pose")
            pose = np.eye(4)
        poses.append(pose)
    poses = np.array(poses)
    # ===== 修改3结束 =====
    
    print(f'[INFO] loaded {len(images)} images from {len(cams)} videos, {N} poses')

    # Coordinate transforms (same as before)
    poses[:, 0:3, 1] *= -1
    poses[:, 0:3, 2] *= -1
    poses = poses[:, [1, 0, 2, 3], :]
    poses[:, 2, :] *= -1

    up = poses[:, 0:3, 1].sum(0)
    up = up / np.linalg.norm(up)
    R = rotmat(up, [0, 0, 1])
    R = np.pad(R, [0, 1])
    R[-1, -1] = 1

    poses = R @ poses

    # ===== 修改4: 只使用训练视角相机计算场景中心 =====
    totw = 0.0
    totp = np.array([0.0, 0.0, 0.0])
    train_indices = [i for i in range(N) if cam_indices[i] in actual_train_cams]
    
    for i in train_indices:
        mf = poses[i, :3, :]
        for j in train_indices:
            if j <= i:
                continue
            mg = poses[j, :3, :]
            p, w = closest_point_2_lines(mf[:,3], mf[:,2], mg[:,3], mg[:,2])
            if w > 0.01:
                totp += p * w
                totw += w
    # ===== 修改4结束 =====
    
    if totw > 0:
        totp /= totw
    else:
        print("[WARNING] Could not compute scene center from training views, using origin")
        totp = np.array([0.0, 0.0, 0.0])
    
    print(f'[INFO] totp = {totp}')
    poses[:, :3, 3] -= totp

    # ===== 修改5: 只使用训练视角相机计算平均距离 =====
    train_poses = poses[train_indices]
    avglen = np.linalg.norm(train_poses[:, :3, 3], axis=-1).mean()
    # ===== 修改5结束 =====
    
    if avglen > 0:
        poses[:, :3, 3] *= 4.0 / avglen
    else:
        print("[WARNING] Average camera distance is zero")
        avglen = 1.0

    print(f'[INFO] average radius (from training views) = {avglen}')
    
    train_frames = []
    test_frames = []
    
    for i in range(N):
        cam_name = cams[i]
        base_pose_transformed = poses[i]
        cam_images_list = [im for im in images if cam_name in im]
        for img_path in cam_images_list:
            img_name = img_path.lstrip("/")
            frame_num = int(img_name.split('.')[0][-4:])
            time = frame_num / 30.0
            frame_data = {
                'file_path': img_name.split('.')[0],
                'transform_matrix': base_pose_transformed.tolist(),
                'time': time
            }
            if cam_indices[i] in actual_train_cams:
                train_frames.append(frame_data)
            else:
                test_frames.append(frame_data)
    
    train_transforms = {
        'w': int(W),
        'h': int(H),
        'fl_x': float(fx),
        'fl_y': float(fy),
        'cx': float(cx),
        'cy': float(cy),
        'frames': train_frames,
    }
    test_transforms = {
        'w': int(W),
        'h': int(H),
        'fl_x': float(fx),
        'fl_y': float(fy),
        'cx': float(cx),
        'cy': float(cy),
        'frames': test_frames,
    }

    train_output_path = os.path.join(args.path, 'transforms_train.json')
    test_output_path = os.path.join(args.path, 'transforms_test.json')
    print(f'[INFO] write to {train_output_path} and {test_output_path}')
    with open(train_output_path, 'w') as f:
        json.dump(train_transforms, f, indent=2)
    with open(test_output_path, 'w') as f:
        json.dump(test_transforms, f, indent=2)
    
    # Dense reconstruction workspace (still PINHOLE)
    colmap_workspace_dense = os.path.join(args.path, 'tmp')
    blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    W, H, cx, cy, fx, fy = int(W), int(H), float(cx), float(cy), float(fx), float(fy)
    os.makedirs(os.path.join(colmap_workspace_dense, 'created', 'sparse'), exist_ok=True)
    
    fname2pose = {}
    with open(os.path.join(colmap_workspace_dense, 'created/sparse/cameras.txt'), 'w') as f:
        # Create a single PINHOLE camera with fx fy cx cy
        f.write(f'1 PINHOLE {W} {H} {fx} {fy} {cx} {cy}')
        for frame in train_frames:
            if frame['time'] == 0:
                fname = frame['file_path'].split('/')[-1] + '.png'
                pose = np.array(frame['transform_matrix']) @ blender2opencv
                fname2pose.update({fname: pose})
                
    os.makedirs(os.path.join(colmap_workspace_dense, 'images'), exist_ok=True)
    for fname in fname2pose.keys():
        src_path = os.path.join(images_path, fname)
        dst_path = os.path.join(colmap_workspace_dense, 'images', fname)
        if os.path.exists(src_path):
            if not os.path.exists(dst_path):
                try:
                    os.symlink(os.path.abspath(src_path), dst_path)
                except OSError:
                    shutil.copy2(os.path.abspath(src_path), dst_path)
        else:
            print(f"[WARNING] Image not found: {src_path}")
                
    with open(os.path.join(colmap_workspace_dense, 'created/sparse/images.txt'), 'w') as f:
        idx = 1
        for fname in fname2pose.keys():
            pose = fname2pose[fname]
            Rm = np.linalg.inv(pose[:3, :3])
            T = -np.matmul(Rm, pose[:3, 3])
            q0 = 0.5 * math.sqrt(max(1 + Rm[0, 0] + Rm[1, 1] + Rm[2, 2], 1e-12))
            q1 = (Rm[2, 1] - Rm[1, 2]) / (4 * q0)
            q2 = (Rm[0, 2] - Rm[2, 0]) / (4 * q0)
            q3 = (Rm[1, 0] - Rm[0, 1]) / (4 * q0)
            f.write(f'{idx} {q0} {q1} {q2} {q3} {T[0]} {T[1]} {T[2]} 1 {fname}\n\n')
            idx += 1
    
    with open(os.path.join(colmap_workspace_dense, 'created/sparse/points3D.txt'), 'w') as f:
        f.write('')
    
    db_path_dense = os.path.join(colmap_workspace_dense, 'database.db')
    
    do_system(f"colmap feature_extractor \
                --database_path {db_path_dense} \
                --image_path {os.path.join(colmap_workspace_dense, 'images')} \
                --ImageReader.camera_model PINHOLE")
    
    camTodatabase(os.path.join(colmap_workspace_dense, 'created/sparse/cameras.txt'), db_path_dense)
    
    do_system(f"colmap exhaustive_matcher  \
                --database_path {db_path_dense}")
    
    os.makedirs(os.path.join(colmap_workspace_dense, 'triangulated', 'sparse'), exist_ok=True)
    
    do_system(f"colmap point_triangulator   \
                --database_path {db_path_dense} \
                --image_path {os.path.join(colmap_workspace_dense, 'images')} \
                --input_path  {os.path.join(colmap_workspace_dense, 'created/sparse')} \
                --output_path  {os.path.join(colmap_workspace_dense, 'triangulated/sparse')}")
    
    do_system(f"colmap model_converter \
                --input_path  {os.path.join(colmap_workspace_dense, 'triangulated/sparse')} \
                --output_path  {os.path.join(colmap_workspace_dense, 'created/sparse')} \
                --output_type TXT")
    
    os.makedirs(os.path.join(colmap_workspace_dense, 'dense'), exist_ok=True)
    
    do_system(f"colmap image_undistorter  \
                --image_path  {os.path.join(colmap_workspace_dense, 'images')} \
                --input_path  {os.path.join(colmap_workspace_dense, 'created/sparse')} \
                --output_path  {os.path.join(colmap_workspace_dense, 'dense')}")
    
    do_system(f"colmap patch_match_stereo   \
                --workspace_path   {os.path.join(colmap_workspace_dense, 'dense')}")
    
    do_system(f"colmap stereo_fusion    \
                --workspace_path {os.path.join(colmap_workspace_dense, 'dense')} \
                --output_path {os.path.join(args.path, 'points3d.ply')}")
    
    shutil.rmtree(colmap_workspace_dense)
    if os.path.exists(os.path.join(args.path, 'points3d.ply.vis')):
        os.remove(os.path.join(args.path, 'points3d.ply.vis'))
    
    print(f"[INFO] Initial point cloud is saved in {os.path.join(args.path, 'points3d.ply')}.")