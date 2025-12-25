from __future__ import annotations
import math
import numpy as np
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray
import tf2_ros
from tf2_geometry_msgs import do_transform_pose
import torch

from .projection import CameraIntrinsics, HeightmapSpec, depth_to_xyz, build_heightmaps
from .grasp_wrapper import GraspConfig, GraspModel


def quat_from_axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis /= (np.linalg.norm(axis) + 1e-12)
    s = math.sin(angle / 2.0)
    return axis[0] * s, axis[1] * s, axis[2] * s, math.cos(angle / 2.0)


def rpy_from_quat(x, y, z, w):
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    t2 = 2.0 * (w * y - z * x)
    t2 = max(-1.0, min(1.0, t2))
    pitch = math.asin(t2)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return roll, pitch, yaw


class GraspNode(Node):
    def __init__(self):
        super().__init__("grasp_node")

        # topics
        self.declare_parameter("rgb_topic", "camera/camera/image_raw")
        self.declare_parameter("depth_topic", "camera/camera/image_depth")
        self.declare_parameter("output_6dof_topic", "/grasp/target_pose_6dof")

        # intrinsics (CameraInfo отключено)
        for p in ("fx", "fy", "cx", "cy"):
            self.declare_parameter(p, 0.0)

        # frames
        self.declare_parameter("camera_frame", "")   # empty -> rgb.header.frame_id
        self.declare_parameter("base_frame", "base_link")

        # model
        self.declare_parameter("num_rotations", 16)
        self.declare_parameter("weights_path", "")

        # heightmap
        self.declare_parameter("heightmap_size", 224)
        self.declare_parameter("heightmap_resolution", 0.002)
        self.declare_parameter("plane_min", [-0.2, -0.2])
        self.declare_parameter("plane_max", [0.2, 0.2])
        self.declare_parameter("height_axis", 0)
        self.declare_parameter("plane_axes", [1, 2])
        self.declare_parameter("grasp_depth_offset", 0.0)
        self.declare_parameter("score_threshold", 0.0)

        # cadence
        self.declare_parameter("rate_hz", 2.0)
        self.declare_parameter("single_shot", False)

        # rotation & output
        self.declare_parameter("yaw_axis", [1.0, 0.0, 0.0])
        self.declare_parameter("euler_degrees", False)

        # YOLO (всегда внутри ноды)
        self.declare_parameter("yolo_weights", "")
        self.declare_parameter("yolo_conf", 0.25)
        self.declare_parameter("yolo_iou", 0.45)
        self.declare_parameter("yolo_imgsz", 640)

        self.pub = self.create_publisher(Float32MultiArray, self.get_parameter("output_6dof_topic").value, 10)
        self.bridge = CvBridge()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        fx, fy, cx, cy = (float(self.get_parameter(k).value) for k in ("fx", "fy", "cx", "cy"))
        self.K = CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy) if (fx > 0 and fy > 0) else None

        cfg = GraspConfig(
            num_rotations=int(self.get_parameter("num_rotations").value),
            force_cpu=False,
            weights_path=str(self.get_parameter("weights_path").value),
        )
        self.model = GraspModel(cfg)

        self.yolo = None
        w = str(self.get_parameter("yolo_weights").value).strip()
        if w:
            try:
                from ultralytics import YOLO  # type: ignore
                self.yolo = YOLO(w)
            except Exception as e:
                self.get_logger().error(f"YOLO init failed: {e}")

        self.tf_buffer = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=5.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # prebuild spec (assumed static)
        S = int(self.get_parameter("heightmap_size").value)
        res = float(self.get_parameter("heightmap_resolution").value)
        pmin = np.array(self.get_parameter("plane_min").value, dtype=np.float32)
        pmax = np.array(self.get_parameter("plane_max").value, dtype=np.float32)
        hax = int(self.get_parameter("height_axis").value)
        pax = self.get_parameter("plane_axes").value
        self.spec = HeightmapSpec(size=S, resolution=res, plane_min=pmin, plane_max=pmax,
                                  height_axis=hax, plane_axes=(int(pax[0]), int(pax[1])))

        rgb_sub = Subscriber(self, Image, self.get_parameter("rgb_topic").value)
        depth_sub = Subscriber(self, Image, self.get_parameter("depth_topic").value)
        ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size=5, slop=0.05).registerCallback(self._on_sync)

        self.rgb = None
        self.depth = None
        self.stamp = None
        self.last = None
        self.done = False

        hz = float(self.get_parameter("rate_hz").value)
        self.timer = self.create_timer(1.0 / max(hz, 1e-3), self._tick)

    def _on_sync(self, rgb, depth):
        self.rgb, self.depth = rgb, depth
        self.stamp = (rgb.header.stamp.sec, rgb.header.stamp.nanosec)

    def _depth_to_m(self, depth_msg):
        d = np.asarray(self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough"))
        return (d.astype(np.float32) * 0.001) if d.dtype == np.uint16 else d.astype(np.float32)

    def _yolo_mask(self, rgb_u8):
        if self.yolo is None:
            return None
        try:
            bgr = rgb_u8[..., ::-1]
            res = self.yolo.predict(
                source=bgr,
                conf=float(self.get_parameter("yolo_conf").value),
                iou=float(self.get_parameter("yolo_iou").value),
                imgsz=int(self.get_parameter("yolo_imgsz").value),
                verbose=False,
            )
            if not res or res[0].masks is None:
                return None
            m = res[0].masks.data.detach().cpu().numpy().astype(np.float32)
            return (m > 0.5).any(axis=0).astype(np.uint8) * 255
        except Exception as e:
            self.get_logger().error(f"YOLO inference failed: {e}")
            return None

    def _tick(self):
        if bool(self.get_parameter("single_shot").value) and self.done:
            return
        if self.rgb is None or self.depth is None or self.stamp is None:
            return
        if (not bool(self.get_parameter("single_shot").value)) and (self.stamp == self.last):
            return
        self.last = self.stamp
        if self.K is None:
            return

        rgb = np.asarray(self.bridge.imgmsg_to_cv2(self.rgb, desired_encoding="rgb8"), dtype=np.uint8)
        depth_m = self._depth_to_m(self.depth)

        mask = self._yolo_mask(rgb)
        if self.yolo is not None and mask is None:
            return

        xyz = depth_to_xyz(depth_m, self.K)
        color_hm, height_hm, mask_hm = build_heightmaps(rgb, xyz, self.spec, mask_u8=mask)

        q = self.model.infer_q(torch.from_numpy(color_hm).to(self.device),
                               torch.from_numpy(height_hm).to(self.device))

        valid = torch.from_numpy((mask_hm > 0).astype(np.float32)).to(self.device) if mask is not None \
            else torch.ones((self.spec.size, self.spec.size), device=self.device)
        if torch.sum(valid) <= 0:
            return

        q_masked = (q - torch.min(q)) * valid[None, :, :]
        flat = int(torch.argmax(q_masked).detach().cpu().item())
        k = flat // (self.spec.size * self.spec.size)
        rem = flat % (self.spec.size * self.spec.size)
        y = rem // self.spec.size
        x = rem % self.spec.size
        if float(q[k, y, x].detach().cpu().item()) < float(self.get_parameter("score_threshold").value):
            return

        u = (self.spec.size - 1 - x) * self.spec.resolution + float(self.spec.plane_min[0])
        v = (self.spec.size - 1 - y) * self.spec.resolution + float(self.spec.plane_min[1])
        h = float(height_hm[y, x] + float(self.get_parameter("grasp_depth_offset").value))

        p = np.zeros((3,), dtype=np.float64)
        p[int(self.spec.height_axis)] = h
        p[int(self.spec.plane_axes[0])] = u
        p[int(self.spec.plane_axes[1])] = v

        angle = math.radians(k * (360.0 / float(self.model.cfg.num_rotations)))
        qx, qy, qz, qw = quat_from_axis_angle(self.get_parameter("yaw_axis").value, angle)

        src = str(self.get_parameter("camera_frame").value).strip() or (self.rgb.header.frame_id or "camera_color_optical_frame")
        base = str(self.get_parameter("base_frame").value)

        pose = PoseStamped()
        pose.header.stamp = self.rgb.header.stamp
        pose.header.frame_id = src
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = float(p[0]), float(p[1]), float(p[2])
        pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w = float(qx), float(qy), float(qz), float(qw)

        try:
            tf = self.tf_buffer.lookup_transform(base, src, rclpy.time.Time.from_msg(self.rgb.header.stamp))
            pose_b = do_transform_pose(pose, tf)
        except Exception:
            return

        pos = pose_b.pose.position
        ori = pose_b.pose.orientation
        rx, ry, rz = rpy_from_quat(ori.x, ori.y, ori.z, ori.w)
        if bool(self.get_parameter("euler_degrees").value):
            rx, ry, rz = map(math.degrees, (rx, ry, rz))

        msg = Float32MultiArray()
        msg.data = [float(pos.x), float(pos.y), float(pos.z), float(rx), float(ry), float(rz)]
        self.pub.publish(msg)

        if bool(self.get_parameter("single_shot").value):
            self.done = True
            try:
                self.timer.cancel()
            except Exception:
                pass


def main():
    rclpy.init()
    n = GraspNode()
    rclpy.spin(n)
    n.destroy_node()
    rclpy.shutdown()
