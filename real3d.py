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
    {"cx": 1296.0, "cy": 972.0, "fov": 90.0, "pitch": 28.0, "yaw": 150.0},
    {"cx": 1296.0, "cy": 972.0, "fov": 90.0, "pitch": 58.0, "yaw": 294.0},
]

views_path = [
    "recordings_5/camera_1.mp4",
    "recordings_5/camera_2.mp4",
    "recordings_5/camera_3.mp4",
    "recordings_5/camera_4.mp4"
]
fisheye_path = "recordings_5/fisheye.mp4"

# MODEL
model = YOLO("16_09_25_SENd_v2.5_10s_960.pt")
model.to(device)
model.fuse()
model.half()
logging.info(" YOLO model loaded and fused for inference.")


homography_BEV = np.load("Homography for 229_camera_.npy")
homography_BEV_to_3D = np.load("Homography for bev to 3d.npy")

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

def update_spheres(points_3d):
    global vis, current_spheres
    for s in current_spheres:
        vis.remove_geometry(s, reset_bounding_box=False)
    current_spheres = []

    for (c,p) in points_3d:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.25)
        if c == 0:
            sphere.paint_uniform_color([1, 0, 0])
        elif c == 1:
            sphere.paint_uniform_color([0, 1, 0])
        sphere.translate(p)
        vis.add_geometry(sphere, reset_bounding_box=False)
        current_spheres.append(sphere)

    vis.poll_events()
    vis.update_renderer()

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
caps_rect = [cv2.VideoCapture(v) for v in views_path]


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
        for c in caps_rect:
            ret, f = c.read()
            if not ret:
                break
            views.append(f)
        if len(views) != 4:
            continue

        valid_views = []
        for v in views:
            if v is None or v.size == 0:
                valid_views = []
                break
            valid_views.append(cv2.resize(v, (960, 540)))

        if len(valid_views) != 4:
            continue

        try:
            with torch.inference_mode():
                results_list = model(valid_views, device=device, verbose=False)
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

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf.item())
                scale_x, scale_y = rect_w / 960.0, rect_h / 540.0
                x1, x2, y1, y2 = x1 * scale_x, x2 * scale_x, y1 * scale_y, y2 * scale_y
                u_full, v_full = int((x1 + x2) / 2.0), int(y2)
               
                if cls == 0:
                    color = (0, 0, 255)
                elif cls == 1:
                    color = (0, 255, 0)
                cv2.circle(frame, (int(u_full), int(v_full)), 10, color, -1)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                aspect_ratio = (y2-y1)/(x2-x1)
                logging.info(f"{i},{cls}, {aspect_ratio}, {conf}")

                if conf > 0.7:
                    pt_f = rect_to_fisheye_point(u_full, v_full, cam_settings[i], fisheye_params, rect_w, rect_h)
                    if pt_f is None:
                        continue
                    all_fisheye_points.append((cls,pt_f[0],pt_f[1]))
                else:
                    continue
        logging.info(all_fisheye_points)
        all_fisheye_points = np.array(all_fisheye_points, dtype=float)
        class_ = all_fisheye_points[:, 0].astype(int)
        uv = all_fisheye_points[:, 1:]
        unique_class = np.unique(class_)
        mean_fisheye_points = []
        for c in unique_class:
            cls_points = uv[class_==c]
            if c == 0:
                eps = 60
                min_samples = 1
            elif c == 1:
                eps = 120
                min_samples = 1

            db = DBSCAN(eps=eps, min_samples=min_samples).fit(cls_points)
            labels = db.labels_
            unique_label = np.unique(labels)
            for cluster_id in unique_label:
                if cluster_id == -1:
                    continue
                
                cluster_points = cls_points[labels == cluster_id]
                if len(cluster_points) > 1:
                    for i in range(len(cluster_points)):
                        for j in range(i+1, len(cluster_points)):
                            d = np.linalg.norm(cluster_points[i] - cluster_points[j])
                            logging.info(f"cls: ({c}) cluster {cluster_id}: dist({i},{j}) = {d:.2f} px ,(eps={eps})")
                else:
                    logging.info(f"cls: ({c}) only 1 point")
                centroid = cluster_points.mean(axis=0)
                mean_fisheye_points.append((c,centroid[0], centroid[1]))
                pt_f_arr = np.array([(centroid[0], centroid[1])], dtype=np.float32).reshape(-1, 1, 2)
                rect_point = cv2.fisheye.undistortPoints(pt_f_arr, K_fish, D_fish, P=dis_K_rect)
                mapped_pt = cv2.perspectiveTransform(rect_point, homography_BEV)
                mx, my = int(mapped_pt[0][0][0]), int(mapped_pt[0][0][1])
                p3d = map_2d_to_3d(mx, my, homography_BEV_to_3D)
                all_points_this_cycle.append((c,np.abs(p3d)))
                logging.info(f"cls: ({c}), fisheye:({centroid[0]:.3f},{centroid[1]:.3f}), 3d: ({p3d[0]:.3f},{p3d[1]:.3f},{p3d[2]:.3f})")

        for (c,u_f, v_f) in mean_fisheye_points:
            if c == 0:
               color = (0, 0, 255)
            elif c == 1:
                color = (0, 255, 0)
            cv2.circle(frame_fish, (int(u_f), int(v_f)), 10, color, -1)

        if all_points_this_cycle:
            point_queue.put(("update_3d", all_points_this_cycle))

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
    for c in caps_rect:
        c.release()
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
