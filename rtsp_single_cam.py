import numpy as np
import cv2
import open3d as o3d
import threading
from ultralytics import YOLO
from sklearn.cluster import DBSCAN
from queue import Queue
import time
import torch
import logging
import math

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f" Using device: {device}")

# CAMERA PARAMETER
fisheye_params = {
    "width": 2592,
    "height": 1944,
    "cx": 1296.0,
    "cy": 972.0,
    "radius": 952.56,
    "theta_max_rad": 1.5708
}

cam_settings = [
    {"cx": 1296.0, "cy": 972.0, "fov": 90.0, "pitch": 58.0, "yaw": -18.0},
    {"cx": 1296.0, "cy": 972.0, "fov": 90.0, "pitch": 54.0, "yaw": 48.0},
    #{"cx": 1296.0, "cy": 972.0, "fov": 30.0, "pitch": 80.0, "yaw": 64.0},
    #{"cx": 1296.0, "cy": 972.0, "fov": 20.0, "pitch": 80.0, "yaw": 64.0},
    {"cx": 1296.0, "cy": 972.0, "fov": 90.0, "pitch": -20.0, "yaw": 150.0},
    {"cx": 1296.0, "cy": 972.0, "fov": 90.0, "pitch": 58.0, "yaw": 274.0},
]

def extract_fisheye_maps(cam_settings, out_w, out_h, fisheye_params):
    maps = []
    cx = fisheye_params["cx"]
    cy = fisheye_params["cy"]
    f_fish = fisheye_params["radius"] / fisheye_params["theta_max_rad"]
    x = np.linspace(0, out_w - 1, out_w)
    y = np.linspace(0, out_h - 1, out_h)
    xv, yv = np.meshgrid(x, y)
    cx_o = out_w / 2
    cy_o = out_h / 2

    for cam in cam_settings:
        yaw_deg = cam["yaw"]
        pitch_deg = cam["pitch"]
        fov_deg = cam["fov"]

        yaw = np.deg2rad(yaw_deg)
        pitch = np.deg2rad(pitch_deg)
        fov = np.deg2rad(fov_deg)

        fx = (out_w / 2) / np.tan(fov / 2)
        fy = fx

        x_cam = (xv - cx_o) / fx
        y_cam = (yv - cy_o) / fy
        z_cam = np.ones_like(x_cam)

        norm = np.sqrt(x_cam**2 + y_cam**2 + z_cam**2)
        x_cam_n = x_cam / norm
        y_cam_n = y_cam / norm
        z_cam_n = z_cam / norm

        R_yaw = np.array([
            [np.cos(-yaw), -np.sin(-yaw), 0],
            [np.sin(-yaw),  np.cos(-yaw), 0],
            [0,            0,           1]
        ])

        R_pitch = np.array([
            [1, 0,             0],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch),  np.cos(pitch)]
        ])

        R_fix = np.array([
            [0, -1, 0],
            [1,  0, 0],
            [0,  0, 1]
        ])

        R = R_fix @ R_yaw @ R_pitch

        rays = np.stack((x_cam_n, y_cam_n, z_cam_n), axis=-1)
        rays_rot = rays @ R.T

        theta = np.arccos(rays_rot[..., 2])
        phi = np.arctan2(rays_rot[..., 1], rays_rot[..., 0])
        r = f_fish * theta
        u = cx + r * np.cos(phi)
        v = cy + r * np.sin(phi)

        map_x = u.astype(np.float32)
        map_y = v.astype(np.float32)
        maps.append((map_x, map_y))

    return maps


fisheye_path = "19-dec/merged_2_crop_20.mp4"

# MODEL
model = YOLO("cam_4/18_11_25_SEN_PTd_TL_v1.0_10s.pt")
sgie_model = YOLO("Forklift_fork_load_seg_v1.1_11s.pt")

model.to(device)
model.fuse()
model.half()

sgie_model.to(device)
sgie_model.fuse()
sgie_model.half()
logging.info(" YOLO model loaded and fused for inference.")

homography_BEV = np.load("Homography for 229_camera_.npy")
homography_BEV_to_3D = np.load("Homography_for_bev_to_3d_full.npy")

point_queue = Queue()
mesh = o3d.io.read_triangle_mesh("OHLF_obj/OHLF_v2.8.3 (1).obj", enable_post_processing=True)
mesh.compute_vertex_normals()
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="3D Model", width=1280, height=720)
vis.add_geometry(mesh)
logging.info(" Open3D visualizer initialized")
current_spheres = []

def compute_basis(yaw_deg, pitch_deg):
    yaw_deg_corrected = yaw_deg - 90.0
    yaw, pitch = np.deg2rad([yaw_deg_corrected, pitch_deg])
    Rz = np.array([[np.cos(-yaw), -np.sin(-yaw), 0],
                   [np.sin(-yaw), np.cos(-yaw), 0],
                   [0, 0, 1]])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(pitch), -np.sin(pitch)],
                   [0, np.sin(pitch), np.cos(pitch)]])
    return np.diag([1, -1, 1]) @ (Rx @ Rz)

def rect_to_fisheye_point(u, v, view_params, fisheye_params, rect_w, rect_h):
    Hr, Wr = rect_h, rect_w
    cx_r, cy_r = Wr / 2, Hr / 2
    fov = view_params["fov"]
    f = (Wr / 2) / np.tan(np.deg2rad(fov / 2))
    x = (u - cx_r) / f
    y = (v - cy_r) / f
    z = 1
    ray_rect = np.array([x, y, z])
    ray_rect /= np.linalg.norm(ray_rect)
    Rot = compute_basis(view_params["yaw"], view_params["pitch"])
    ray_fish = Rot.T @ ray_rect
    Xf, Yf, Zf = ray_fish
    theta = np.arccos(np.clip(Zf, -1, 1))
    phi = np.arctan2(Yf, Xf)
    radius, theta_max = fisheye_params["radius"], fisheye_params["theta_max_rad"]
    cx_f, cy_f = fisheye_params["cx"], fisheye_params["cy"]
    r = (theta / theta_max) * radius
    u_f = cx_f + r * np.cos(phi)
    v_f = cy_f - r * np.sin(phi)
    return (float(u_f), float(v_f))

def map_2d_to_3d(u, v, homography_bev_to_3d, y_value=0.0):
    src = np.array([u, v, 1.0], dtype=np.float64)
    dst = homography_bev_to_3d @ src
    dst /= dst[2]
    return np.array([dst[0], y_value, dst[1]])

def angle(consider, compare):
    angle = math.degrees(math.atan2(compare[1] - consider[1], compare[0] - consider[0]))
    return (angle + 360) % 360

def nearest_point(center_s, target):
    center_s = np.array(center_s, dtype=np.float64)
    target = np.array(target, dtype=np.float64)
    distances = np.linalg.norm(center_s - target, axis=1)
    return center_s[np.argmin(distances)]

def update_spheres(points_3d):
    global vis, current_spheres

    for g in current_spheres:
        try:
            vis.remove_geometry(g, reset_bounding_box=False)
        except:
            pass
    current_spheres = []

    for class_id, pos, ang in points_3d:
        pos = np.asarray(pos, float)
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.25)
        col = [1, 0, 0] if class_id == 0 else [0, 1, 0]
        sphere.paint_uniform_color(col)
        sphere.translate(pos)
        vis.add_geometry(sphere, reset_bounding_box=False)
        current_spheres.append(sphere)

        if ang is None or (isinstance(ang, float) and np.isnan(ang)):
            continue

        L = 1.0
        t = math.radians(float(ang))
        tip = pos + np.array([math.cos(t) * L, 0, math.sin(t) * L])
        d = tip - pos
        length = np.linalg.norm(d)
        if length < 1e-6:
            continue

        cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=0.06, height=length)
        cyl.paint_uniform_color(col)

        direction = d / length
        z = np.array([0, 0, 1])
        dot = np.clip(np.dot(z, direction), -1, 1)
        axis = np.cross(z, direction)
        n = np.linalg.norm(axis)

        if n < 1e-6:
            if dot < 0:
                R = o3d.geometry.get_rotation_matrix_from_axis_angle(
                    np.array([1, 0, 0]) * math.pi
                )
                cyl.rotate(R, center=[0, 0, 0])
        else:
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(
                axis / n * math.acos(dot)
            )
            cyl.rotate(R, center=[0, 0, 0])

        cyl.translate((pos + tip) / 2)
        vis.add_geometry(cyl, reset_bounding_box=False)
        current_spheres.append(cyl)

    vis.poll_events()
    vis.update_renderer()

def uv_undistort_to_3d(centroid, K_fish, D_fish, dis_K_rect, homography_BEV, homography_BEV_to_3D):
    pt_f_arr = np.array([(centroid[0], centroid[1])], dtype=np.float32).reshape(-1, 1, 2)
    rect_point = cv2.fisheye.undistortPoints(pt_f_arr, K_fish, D_fish, P=dis_K_rect)
    mapped_pt = cv2.perspectiveTransform(rect_point, homography_BEV)
    mx, my = int(mapped_pt[0][0][0]), int(mapped_pt[0][0][1])
    return map_2d_to_3d(mx, my, homography_BEV_to_3D)


# CAMERA MATRICES
dis_rect_w, dis_rect_h = 2592, 2592
fov = 160.0
fov_RAD = np.deg2rad(fov)
focal = (dis_rect_w / 2) / np.tan(fov_RAD / 2)
dis_K_rect = np.array([[focal, 0.0, dis_rect_w / 2],
                       [0.0, focal, dis_rect_h / 2],
                       [0.0, 0.0, 1.0]])

f_fish = fisheye_params["radius"] / fisheye_params["theta_max_rad"]
K_fish = np.array([[f_fish, 0.0, fisheye_params["cx"]],
                   [0.0, f_fish, fisheye_params["cy"]],
                   [0.0, 0.0, 1.0]])
D_fish = np.zeros((4, 1))

TARGET_HEIGHT = 720
TARGET_WIDTH_FISHEYE = 720
TARGET_WIDTH_RECT = 720

cap_fish = cv2.VideoCapture(fisheye_path)

# FISHEYE MAPPING
rect_out_w = 1280
rect_out_h = 720
rect_maps = extract_fisheye_maps(cam_settings, rect_out_w, rect_out_h, fisheye_params)


def make_rect_grid_disp(views_native):
    h, w = views_native[0].shape[:2]
    grid = np.ones((h * 2, w * 2, 3), dtype=np.uint8) * 0
    grid[0:h, 0:w] = views_native[0]
    grid[0:h, w:2*w] = views_native[1]
    grid[h:2*h, 0:w] = views_native[2]
    grid[h:2*h, w:2*w] = views_native[3]
    return grid

# MAIN LOOP
def run():
    global vis
    frame_index = 0

    while True:
        t0 = time.time()

        ret_f, frame_fish = cap_fish.read()
        if not ret_f:
            break

        views = []
        for (map_x, map_y) in rect_maps:
            rect_img = cv2.remap(frame_fish, map_x, map_y, cv2.INTER_LINEAR)
            if rect_img is None or rect_img.size == 0:
                views = []
                break
            views.append(rect_img)

        if len(views) != len(rect_maps):
            continue

        try:
            with torch.inference_mode():
                results_list = model(views, device=device, verbose=False)
        except Exception as e:
            logging.warning(f" Model error during batch inference: {e}")
            continue

        all_points_this_cycle = []
        all_fisheye_points = []

        for i, result in enumerate(results_list):
            frame = views[i]
            rect_h, rect_w = frame.shape[:2]

            for box in result.boxes:
                cls = int(box.cls.item())
                if cls not in [0, 1]:
                    continue
                center_s = None
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf.item())
                scale_x, scale_y = rect_w / rect_out_w, rect_h / rect_out_h
                x1, x2, y1, y2 = x1 * scale_x, x2 * scale_x, y1 * scale_y, y2 * scale_y
                u_full, v_full = int((x1 + x2) / 2.0), int(y2)
                aspect_ratio = (y2 - y1) / (x2 - x1)
                corner_size = 50
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 3)
                if ((u_full < corner_size and v_full < corner_size) or
                    (u_full > rect_w - corner_size and v_full < corner_size) or
                    (u_full < corner_size and v_full > rect_h - corner_size) or
                    (u_full > rect_w - corner_size and v_full > rect_h - corner_size)):
                    continue


                if cls == 0:
                    color = (0, 0, 255)
                    aspect_ratio_cls = 0.5
                    aspect_ratio_cls_low_conf = 1.0
                elif cls == 1:
                    color = (0, 255, 0)
                    aspect_ratio_cls = 0.1
                
                if cls == 1:
                    logging.info(f"{i},{cls}, {aspect_ratio}, {conf}")

                if cls == 2:
                    p = 100
                    x1_g = max(0, x1 - p)
                    y1_g = max(0, y1 - p)
                    x2_g = min(rect_w, x2 + p)
                    y2_g = min(rect_h, y2 + p)
                    u_full = int((x2 + x1) / 2)
                    v_full = y1 + int((y2 - y1) * 0.8)
                    crop = frame[int(y1_g):int(y2_g), int(x1_g):int(x2_g)]
                    if crop.size == 0:
                        continue
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
                    cv2.circle(frame, (int(u_full), int(v_full)), 10, (255, 0, 0), -1)
                    logging.info(crop.shape)
                    results_s = sgie_model(crop)[0]
                    center_s = []
                    s_x = s_y = None
                    boxes_s = results_s.boxes
                    if boxes_s is None or len(boxes_s) == 0:
                        logging.info("No segmentation detections")
                        continue

                    boxes_xyxy = boxes_s.xyxy.cpu().numpy()
                    classes_s = boxes_s.cls.cpu().numpy()
                    for (xs1, ys1, xs2, ys2), cls_s in zip(boxes_xyxy, classes_s):
                        if int(cls_s) != 0:
                            continue
                        sx1 = int(xs1) + int(x1_g)
                        sy1 = int(ys1) + int(y1_g)
                        sx2 = int(xs2) + int(x1_g)
                        sy2 = int(ys2) + int(y1_g)
                        s_x = int((sx2 + sx1) / 2)
                        s_y = sy1 + int((sy2 - sy1) * 0.8)
                        center_s.append((s_x, s_y))
                        cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (0, 0, 0), 3)
                        cv2.circle(frame, (s_x, s_y), 10, (255, 255, 255), -1)
                else:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                    cv2.circle(frame, (int(u_full), int(v_full)), 10, (255, 0, 0), -1)

                if ((conf > 0.40) and (aspect_ratio >= aspect_ratio_cls)) or \
                   ((conf < 0.90) and (aspect_ratio <= aspect_ratio_cls_low_conf)):
                    pt_f = rect_to_fisheye_point(u_full, v_full, cam_settings[i], fisheye_params, rect_w, rect_h)
                    fork_pt_f = None

                    if center_s and pt_f is not None:
                        nearest = nearest_point(center_s, (u_full, v_full))
                        logging.info(f"nearest:{nearest}")
                        fork_pt_f = rect_to_fisheye_point(nearest[0], nearest[1], cam_settings[i],fisheye_params, rect_w, rect_h)
                        cv2.circle(frame_fish, (int(fork_pt_f[0]), int(fork_pt_f[1])), 10, (0, 0, 255), -1)
                    logging.info(f"pri:{pt_f}, sec:{fork_pt_f}")
                    if pt_f is None:
                        continue
                    if fork_pt_f is not None:
                        all_fisheye_points.append((cls, pt_f[0], pt_f[1], fork_pt_f[0], fork_pt_f[1]))
                    else:
                        all_fisheye_points.append((cls, pt_f[0], pt_f[1], np.nan, np.nan))
                else:
                    continue

        if len(all_fisheye_points) == 0:
            rect_grid_native = make_rect_grid_disp(views)
            fisheye_resized = cv2.resize(frame_fish, (TARGET_WIDTH_FISHEYE, TARGET_HEIGHT))
            rect_resized = cv2.resize(rect_grid_native, (TARGET_WIDTH_RECT, TARGET_HEIGHT))
            canvas = np.ones((TARGET_HEIGHT, TARGET_WIDTH_FISHEYE + TARGET_WIDTH_RECT, 3), dtype=np.uint8)
            canvas[:, 0:TARGET_WIDTH_FISHEYE] = fisheye_resized
            canvas[:, TARGET_WIDTH_FISHEYE:] = rect_resized

            if frame_index % 3 == 0:
                cv2.imshow("Unified Mapping", canvas)

            frame_index += 1
            if cv2.waitKey(1) == 27:
                break
            continue

        all_fisheye_points = np.array(all_fisheye_points, dtype=float)
        class_ = all_fisheye_points[:, 0].astype(int)
        uv = all_fisheye_points[:, 1:3]
        fork_uv = all_fisheye_points[:, 3:5]
        unique_class = np.unique(class_)
        mean_fisheye_points = []

        for c in unique_class:
            cls_points = uv[class_ == c]
            fork_cls = fork_uv[class_ == c]
            if c == 0:
                eps = 20
                min_samples = 1
            elif c == 1:
                eps = 120
                min_samples = 1
            logging.info(f"c:{c},cls_")
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(cls_points)
            labels = db.labels_
            unique_label = np.unique(labels)
            for cluster_id in unique_label:
                if cluster_id == -1:
                    continue
                cluster_points = cls_points[labels == cluster_id]
                cluster_fork_cls = fork_cls[labels == cluster_id]
                clean_cluster_fork_cls = [(u, v) for (u, v) in cluster_fork_cls if not (np.isnan(u) or np.isnan(v))]
                if len(cluster_points) > 1:
                    for i_a in range(len(cluster_points)):
                        for j_a in range(i_a + 1, len(cluster_points)):
                            d = np.linalg.norm(cluster_points[i_a] - cluster_points[j_a])
                            #logging.info(f"cls: ({c}) cluster {cluster_id}: dist({i_a},{j_a}) = {d:.2f} px ,(eps={eps})")
                else:
                    logging.info(f"cls: ({c}) only 1 point")
                centroid = cluster_points.mean(axis=0)

                p3d = uv_undistort_to_3d(centroid, K_fish, D_fish, dis_K_rect,homography_BEV, homography_BEV_to_3D)
                fork_point_mean = None
                if len(clean_cluster_fork_cls) > 0:
                    fork_cluster = np.array(clean_cluster_fork_cls)
                    fork_point_mean = fork_cluster.mean(axis=0)
                    p3d_fork = uv_undistort_to_3d(fork_point_mean, K_fish, D_fish, dis_K_rect, homography_BEV, homography_BEV_to_3D)
                    angle_cluster = angle((p3d[0], p3d[2]), (p3d_fork[0], p3d_fork[2]))
                else:
                    angle_cluster = None
                if fork_point_mean is not None:
                    mean_fisheye_points.append((c, centroid[0], centroid[1], fork_point_mean[0], fork_point_mean[1]))
                else:
                    mean_fisheye_points.append((c, centroid[0], centroid[1], np.nan, np.nan))
                all_points_this_cycle.append((c, np.abs(p3d), angle_cluster))
                logging.info(
                    f"cls: ({c}), angle:({angle_cluster}), "
                    f"fisheye:({centroid[0]:.3f},{centroid[1]:.3f}), "
                    f"3d: ({p3d[0]:.3f},{p3d[1]:.3f},{p3d[2]:.3f})"
                )

        for (c, u_f, v_f, f_u_f, f_v_f) in mean_fisheye_points:
            if c == 0:
                color = (0, 0, 255)
            elif c == 1:
                color = (0, 255, 0)
            if np.isfinite(f_u_f) and np.isfinite(f_v_f):
                cv2.circle(frame_fish, (int(f_u_f), int(f_v_f)), 10, (255, 255, 255), -1)
            cv2.circle(frame_fish, (int(u_f), int(v_f)), 10, color, -1)

        if all_points_this_cycle:
            point_queue.put(("update_3d", all_points_this_cycle))

        # canvas
        rect_grid_native = make_rect_grid_disp(views)
        fisheye_resized = cv2.resize(frame_fish, (TARGET_WIDTH_FISHEYE, TARGET_HEIGHT))
        rect_resized = cv2.resize(rect_grid_native, (TARGET_WIDTH_RECT, TARGET_HEIGHT))
        canvas = np.ones((TARGET_HEIGHT, TARGET_WIDTH_FISHEYE + TARGET_WIDTH_RECT, 3), dtype=np.uint8)
        canvas[:, 0:TARGET_WIDTH_FISHEYE] = fisheye_resized
        canvas[:, TARGET_WIDTH_FISHEYE:] = rect_resized

        if frame_index % 3 == 0:
            cv2.imshow("Unified Mapping", canvas)

        frame_index += 1
        if cv2.waitKey(1) == 27:
            break

    cap_fish.release()
    cv2.destroyAllWindows()

# THREAD MANAGEMENT
cv_thread = threading.Thread(target=run, daemon=True)
cv_thread.start()

try:
    while cv_thread.is_alive():
        while not point_queue.empty():
            msg_type, payload = point_queue.get()
            if msg_type == "update_3d":
                update_spheres(payload)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.02)
finally:
    vis.destroy_window()
    cv2.destroyAllWindows()
