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

fisheye_path = [
    "outputs_changed_reduced_time_23FPS/229/fisheye.mp4",
    "outputs_changed_reduced_time_23FPS/231/fisheye.mp4",
    "outputs_changed_reduced_time_23FPS/233/fisheye.mp4",
    "outputs_changed_reduced_time_23FPS/234/fisheye.mp4",

]

views_path = [
    ["outputs_changed_reduced_time_23FPS/229/view1.mp4", "outputs_changed_reduced_time_23FPS/229/view2.mp4", "outputs_changed_reduced_time_23FPS/229/view3.mp4", "outputs_changed_reduced_time_23FPS/229/view4.mp4"],
    ["outputs_changed_reduced_time_23FPS/231/view1.mp4", "outputs_changed_reduced_time_23FPS/231/view2.mp4", "outputs_changed_reduced_time_23FPS/231/view3.mp4", "outputs_changed_reduced_time_23FPS/231/view4.mp4"],
    ["outputs_changed_reduced_time_23FPS/233/view1.mp4", "outputs_changed_reduced_time_23FPS/233/view2.mp4", "outputs_changed_reduced_time_23FPS/233/view3.mp4", "outputs_changed_reduced_time_23FPS/233/view4.mp4"],
    ["outputs_changed_reduced_time_23FPS/234/view1.mp4", "outputs_changed_reduced_time_23FPS/234/view2.mp4", "outputs_changed_reduced_time_23FPS/234/view3.mp4", "outputs_changed_reduced_time_23FPS/234/view4.mp4"],
    
]


bev_img_original = cv2.imread("ohlf.png")

# MODEL
model = YOLO("16_09_25_SENd_v2.5_10s_960.pt")
model.to(device)
model.fuse()
model.half()
logging.info(" YOLO model loaded and fused for inference.")

homography_BEV_path = [
    "Homography for 229_camera_.npy",
    "Homography for 231_camera_.npy",
    "Homography for 233_camera_.npy",
    "Homography for 234_camera_full_area.npy",

]
homography_BEV_ = [np.load(v) for v in homography_BEV_path]

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

def canva(rect_view, fish_view):
    fisheye_resized = cv2.resize(fish_view, (TARGET_WIDTH_FISHEYE, TARGET_HEIGHT))
    rect_resized = cv2.resize(rect_view, (TARGET_WIDTH_RECT, TARGET_HEIGHT))
    canvas = np.ones((TARGET_HEIGHT, TARGET_WIDTH_FISHEYE + TARGET_WIDTH_RECT, 3), dtype=np.uint8)
    canvas[:, 0:TARGET_WIDTH_FISHEYE] = fisheye_resized
    canvas[:, TARGET_WIDTH_FISHEYE:] = rect_resized
    return canvas

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

    for (c, p) in points_3d:
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

TARGET_HEIGHT = 500
TARGET_WIDTH_FISHEYE = 400
TARGET_WIDTH_RECT = 400
cap_fish = [cv2.VideoCapture(fp) for fp in fisheye_path]
caps_rect = []
for cam_id in views_path:
    for cam_crop in cam_id:
        caps_rect.append(cv2.VideoCapture(cam_crop))

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
        fish_view = []
        for ff_id in cap_fish:
            ret_f, frame_fish = ff_id.read()
            if not ret_f:
                break
            fish_view.append(frame_fish)
        if len(fish_view) != 4:
            continue

        views = []
        for c in caps_rect:
            ret, f = c.read()
            if not ret:
                break
            views.append(f)
        if len(views) != 16:
            continue

        valid_views = []
        for v in views:
            if v is None or v.size == 0:
                valid_views = []
                break
            valid_views.append(cv2.resize(v, (960, 540)))

        if len(valid_views) != 16:
            continue

        canvas_229 = None
        canvas_231 = None
        canvas_233 = None
        canvas_234 = None
        

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
            cam = i // 4
            cam_set = i % 4
            for box in result.boxes:
                cls = int(box.cls.item())
                if cls not in [0, 1]:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf.item())
                scale_x, scale_y = rect_w / 960.0, rect_h / 540.0
                x1, x2, y1, y2 = x1 * scale_x, x2 * scale_x, y1 * scale_y, y2 * scale_y
                u_full, v_full = int((x1 + x2) / 2.0), int(y2)

                aspect_ratio = (y2 - y1) / (x2 - x1 + 1e-6)
                area = (x2-x1)*(y2-y1)
                logging.info(f"id:{cam},cam_set:{cam_set},en:{i},cls:{cls},ar:{aspect_ratio},area: {area},conf:{conf}")

                corner_size = 50
                # bottom_band = (v_full > rect_h - 10)
                if ((u_full < corner_size and v_full < corner_size) or
                    (u_full > rect_w - corner_size and v_full < corner_size) or
                    (u_full < corner_size and v_full > rect_h - corner_size) or
                    (u_full > rect_w - corner_size and v_full > rect_h - corner_size) ):
                    logging.info(f" removed id:{cam},cam_set:{cam_set},en:{i},cls:{cls},ar:{aspect_ratio},area: {area},conf:{conf}")
                    continue

                if cls == 0:
                    color = (0, 0, 255)
                    aspect_ratio_cls = 1.0
                    aspect_ratio_cls_low_ar = 0.8
                    aspect_ratio_cls_reg_high_ar = 3
                elif cls == 1:
                    color = (0, 255, 0)
                    aspect_ratio_cls = 0.8
                    aspect_ratio_cls_low_ar = 0.4
                    aspect_ratio_cls_reg_high_ar = 3.0

                cv2.circle(frame, (int(u_full), int(v_full)), 10, color, -1)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                if area < 10000:
                    continue
                do_not_use_rect_to_fisheye = False
                condition_match = False
                if (aspect_ratio >= aspect_ratio_cls_reg_high_ar):
                    do_not_use_rect_to_fisheye = True
                    if cam_set == 0:
                        rect_grid_native_229 = make_rect_grid_disp(views[:4])
                        canvas_229 = canva(rect_grid_native_229, fish_view[0])
                    elif cam_set == 1:
                        rect_grid_native_231 = make_rect_grid_disp(views[4:8])
                        canvas_231 = canva(rect_grid_native_231, fish_view[1])
                    elif cam_set == 2:
                        rect_grid_native_233 = make_rect_grid_disp(views[8:12])
                        canvas_233 = canva(rect_grid_native_233, fish_view[2])
                    elif cam_set == 3:
                        rect_grid_native_234 = make_rect_grid_disp(views[12:16])
                        canvas_234 = canva(rect_grid_native_234, fish_view[3])

                elif ((conf > 0.90) and (aspect_ratio >= aspect_ratio_cls)):
                    condition_match = True
                elif (aspect_ratio_cls_low_ar >= aspect_ratio):
                    condition_match = True
                else:
                    continue

                if (not do_not_use_rect_to_fisheye) and condition_match:
                    logging.info(f"condition id:{cam},cam_set:{cam_set},en:{i},cls:{cls},ar:{aspect_ratio},area: {area},conf:{conf}")
                    pt_f = rect_to_fisheye_point(u_full, v_full, cam_settings[cam_set], fisheye_params, rect_w, rect_h)
                    if pt_f is None:
                        continue
                    all_fisheye_points.append((cam, cls, pt_f[0], pt_f[1]))
                else:
                    continue

        logging.info(all_fisheye_points)
        if len(all_fisheye_points) == 0:
            if (canvas_229 is not None) and (frame_index % 3 == 0):
                cv2.imshow("Mapping_229", canvas_229)
            if (canvas_231 is not None) and (frame_index % 3 == 0):
                cv2.imshow("Mapping_231", canvas_231)
            if (canvas_233 is not None) and (frame_index % 3 == 0):
                cv2.imshow("Mapping_233", canvas_233)
            if (canvas_234 is not None) and (frame_index % 3 == 0):
                cv2.imshow("Mapping_234", canvas_234)

            if cv2.waitKey(1) == 27:
                break
            frame_index += 1
            continue

        all_fisheye_points = np.array(all_fisheye_points, dtype=float)
        cam_id_col = all_fisheye_points[:, 0].astype(int)
        class_col = all_fisheye_points[:, 1].astype(int)
        uv_col = all_fisheye_points[:, 2:4].astype(float)
        unique_cam_ids = np.unique(cam_id_col)

        mean_fisheye_points = []
        bev_points = []
        for cam_id in unique_cam_ids:
            homography_BEV = homography_BEV_[cam_id]
            mask_cam = (cam_id_col == cam_id)
            if not np.any(mask_cam):
                continue
            cam_classes = class_col[mask_cam]
            cam_uv = uv_col[mask_cam]

            unique_classes = np.unique(cam_classes)
            for c in unique_classes:
                cls_mask = (cam_classes == c)
                cls_points = cam_uv[cls_mask]
                if cls_points.shape[0] == 0:
                    continue

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
                        for ii in range(len(cluster_points)):
                            for jj in range(ii + 1, len(cluster_points)):
                                d = np.linalg.norm(cluster_points[ii] - cluster_points[jj])
                                logging.info(f"cls: ({c}) cluster {cluster_id}: dist({ii},{jj}) = {d:.2f} px ,(eps={eps})")
                    else:
                        logging.info(f"cls: ({c}) only 1 point")
                    centroid = cluster_points.mean(axis=0)
                    mean_fisheye_points.append((cam_id, c, centroid[0], centroid[1]))
                    pt_f_arr = np.array([(centroid[0], centroid[1])], dtype=np.float32).reshape(-1, 1, 2)
                    rect_point = cv2.fisheye.undistortPoints(pt_f_arr, K_fish, D_fish, P=dis_K_rect)
                    mapped_pt = cv2.perspectiveTransform(rect_point, homography_BEV)
                    mx, my = int(mapped_pt[0][0][0]), int(mapped_pt[0][0][1])
                    bev_points.append((cam_id,c, mx, my))
                    p3d = map_2d_to_3d(mx, my, homography_BEV_to_3D)
                    all_points_this_cycle.append((c, np.abs(p3d)))
                    logging.info(f"id: {cam_id},cls: ({c}), fisheye:({centroid[0]:.3f},{centroid[1]:.3f}), 3d: ({p3d[0]:.3f},{p3d[1]:.3f},{p3d[2]:.3f})")


        for (cam_id, c, u_f, v_f) in mean_fisheye_points:
            if c == 0:
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)
 
            h_f, w_f = fish_view[cam_id].shape[:2]
            u_clamped = max(0, min(w_f - 1, int(u_f)))
            v_clamped = max(0, min(h_f - 1, int(v_f)))
            cv2.circle(fish_view[cam_id], (u_clamped, v_clamped), 10, color, -1)


        if all_points_this_cycle:
            all_points_this_cycle = np.array([[cls, *vals] for cls, vals in all_points_this_cycle], dtype=float)
            p3d_class = all_points_this_cycle[:, 0].astype(int)
            p3d_value = all_points_this_cycle[:, 1:]
            unique_p3d_class = np.unique(p3d_class)

            mean_3d_points = []
            for u_class in unique_p3d_class:
                p3d_class_mask = (p3d_class == u_class)
                cls_p3d = p3d_value[p3d_class_mask]
                if cls_p3d.shape[0] == 0:
                    continue
                if u_class == 0:
                    eps = 1.2
                    min_samples = 1
                elif u_class == 1:
                    eps = 2.5
                    min_samples = 1

                db = DBSCAN(eps=eps, min_samples=min_samples).fit(cls_p3d)
                labels = db.labels_
                unique_label = np.unique(labels)
                for cluster_id in unique_label:
                    if cluster_id == -1:
                        continue
                    cluster_points = cls_p3d[labels == cluster_id]
                    if len(cluster_points) > 1:
                        for ii in range(len(cluster_points)):
                            for jj in range(ii + 1, len(cluster_points)):
                                d = np.linalg.norm(cluster_points[ii] - cluster_points[jj])
                                logging.info(f"cls: ({u_class}) cluster {cluster_id}: dist({ii},{jj}) = {d:.2f} px ,(eps={eps})")
                    centroid = cluster_points.mean(axis=0)
                    p3d = (centroid[0], centroid[1], centroid[2])
                    mean_3d_points.append((u_class, p3d))

            if mean_3d_points:
                point_queue.put(("update_3d", mean_3d_points))


        bev_display = bev_img_original.copy()
        h_bev, w_bev = bev_display.shape[:2]
        for (cam_id, c, u, v) in bev_points:
            if c == 0:
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)


            u_clamped = max(0, min(w_bev - 1, int(u)))
            v_clamped = max(0, min(h_bev - 1, int(v)))
            cv2.circle(bev_display, (u_clamped, v_clamped), 6, color, -1)
            label_text = f"ID:{cam_id} "
            cv2.putText(bev_display, label_text, (u_clamped + 10, v_clamped - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        cv2.imshow("BEV Mapping", bev_display)


        rect_grid_native_229 = make_rect_grid_disp(views[0:4])
        rect_grid_native_231 = make_rect_grid_disp(views[4:8])
        rect_grid_native_233 = make_rect_grid_disp(views[8:12])
        rect_grid_native_234 = make_rect_grid_disp(views[12:16])
                        
        canvas_229 = canva(rect_grid_native_229, fish_view[0])
        canvas_231 = canva(rect_grid_native_231, fish_view[1])
        canvas_233 = canva(rect_grid_native_233, fish_view[2])
        canvas_234 = canva(rect_grid_native_234, fish_view[3])


        if frame_index % 3 == 0:
            cv2.imshow("Mapping_229", canvas_229)
            cv2.imshow("Mapping_231", canvas_231)
            cv2.imshow("Mapping_233", canvas_233)
            cv2.imshow("Mapping_234", canvas_234)


        frame_index += 1
        if cv2.waitKey(1) == 27:
            break

    for cf in cap_fish:
        cf.release()
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
