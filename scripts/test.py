#!/home/user/anaconda3/envs/foundationpose/bin/python3
# coding :utf-8
# ROS1
import os
import sys
# Set megapose environment variables
sys.path.append('/home/user/anaconda3/envs/foundationpose/lib/python3.8/site-packages')
FOUNDATIONPOSE_SRC = "/home/user/FoundationPose"
if FOUNDATIONPOSE_SRC not in sys.path:
    sys.path.append(FOUNDATIONPOSE_SRC)

os.environ.setdefault("ULTRALYTICS_NO_INSTALL", "1")

import rospy
import numpy as np
import cv2
import trimesh

from sensor_msgs.msg import Image as RosImage, CameraInfo as RosCameraInfo
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge

from estimater import FoundationPose, draw_posed_3d_box, draw_xyz_axis
from ultralytics import YOLO

class FoundationPoseTracker:
    def __init__(self):
        # 讀參數
        self.bridge = CvBridge()
        self.get_parameters()
        self.create_subscriber_publisher()

    def get_parameters(self):
        # Subscriber Topic setting
        self.image_topic = rospy.get_param(rospy.get_name() + "/image_topic", "/camera/color/image_raw")
        self.info_topic = rospy.get_param(rospy.get_name() + "/info_topic", "/camera/color/camera_info")
        self.depth_topic = rospy.get_param(rospy.get_name() + "/depth_topic", "/")
        self.depth_info_topic = rospy.get_param(rospy.get_name() + "/depth_info_topic", "/")
        self.camera_tf = rospy.get_param(rospy.get_name() + "/camera_tf", "/")
        # Mesh and Model setting
        self.mesh_file = rospy.get_param(rospy.get_name() + "/mesh_file", "")
        self.det_onnx = rospy.get_param(rospy.get_name() + "/det_onnx", "")
        self.init_mode = rospy.get_param(rospy.get_name() + "/init_mode", "yolo")
        self.image_H = rospy.get_param(rospy.get_name() + "/image_H", 480)
        self.image_W = rospy.get_param(rospy.get_name() + "/image_W", 640)
        self.det_conf = rospy.get_param(rospy.get_name() + "/det_conf", 0.25)
        self.det_class = rospy.get_param(rospy.get_name() + "/det_class", 0)
        self.est_refine_iter = rospy.get_param(rospy.get_name() + "/est_refine_iter", 5)
        self.track_refine_iter = rospy.get_param(rospy.get_name() + "/track_refine_iter", 2)
        # Tracking setting
        self.roi_expand = rospy.get_param(rospy.get_name() + "/roi_expand", 0.01)
        self.iou_stride = rospy.get_param(rospy.get_name() + "/iou_stride", 1)
        self.iou_log = rospy.get_param(rospy.get_name() + "/iou_log", True)
        self.iou_thresh = rospy.get_param(rospy.get_name() + "/iou_thresh", 0.2)
        self.iou_patience = rospy.get_param(rospy.get_name() + "/iou_patience", 3)

        rospy.loginfo("Get subscriber topic parameter")
        rospy.loginfo("image_topic: {}, type: {}".format(self.image_topic, type(self.image_topic)))
        rospy.loginfo("info_topic: {}, type: {}".format(self.info_topic, type(self.info_topic)))
        rospy.loginfo("depth_topic: {}, type: {}".format(self.depth_topic, type(self.depth_topic)))
        rospy.loginfo("depth_info_topic: {}, type: {}".format(self.depth_info_topic, type(self.depth_info_topic)))
        rospy.loginfo("camera_tf: {}, type: {}".format(self.camera_tf, type(self.camera_tf)))
        
        rospy.loginfo("Get mesh and model parameter")
        rospy.loginfo("mesh_file: {}, type: {}".format(self.mesh_file, type(self.mesh_file)))
        rospy.loginfo("det_onnx: {}, type: {}".format(self.det_onnx, type(self.det_onnx)))
        rospy.loginfo("init_mode: {}, type: {}".format(self.init_mode, type(self.init_mode)))
        rospy.loginfo("image_H: {}, type: {}".format(self.image_H, type(self.image_H)))
        rospy.loginfo("image_W: {}, type: {}".format(self.image_W, type(self.image_W)))
        rospy.loginfo("det_conf: {}, type: {}".format(self.det_conf, type(self.det_conf)))
        rospy.loginfo("det_class: {}, type: {}".format(self.det_class, type(self.det_class)))
        rospy.loginfo("est_refine_iter: {}, type: {}".format(self.est_refine_iter, type(self.est_refine_iter)))
        rospy.loginfo("track_refine_iter: {}, type: {}".format(self.track_refine_iter, type(self.track_refine_iter)))
        
        rospy.loginfo("Get tracking parameter")
        rospy.loginfo("roi_expand: {}, type: {}".format(self.roi_expand, type(self.roi_expand)))
        rospy.loginfo("iou_stride: {}, type: {}".format(self.iou_stride, type(self.iou_stride)))
        rospy.loginfo("iou_log: {}, type: {}".format(self.iou_log, type(self.iou_log)))
        rospy.loginfo("iou_thresh: {}, type: {}".format(self.iou_thresh, type(self.iou_thresh)))
        rospy.loginfo("iou_patience: {}, type: {}".format(self.iou_patience, type(self.iou_patience)))

    def create_subscriber_publisher(self):
        # Subscriber
        self.image_sub = rospy.Subscriber(self.image_topic,RosImage,self.imageCallback, queue_size=1)
        self.info_sub = rospy.Subscriber(self.info_topic,RosCameraInfo,self.infoCallback, queue_size=1)
        self.depth_sub = rospy.Subscriber(self.depth_topic,RosImage,self.depthCallback, queue_size=1)
        self.depth_info_sub = rospy.Subscriber(self.depth_info_topic,RosCameraInfo,self.depthInfoCallback, queue_size=1)

    def imageCallback(self, msg):
        rospy.loginfo('Image received...')
        self.image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def infoCallback(self, msg):
        rospy.loginfo('CameraInfo received...')
        self.K = np.array(msg.K).reshape(3, 3)
        self.D = np.array(msg.D)
        self.R = np.array(msg.R).reshape(3, 3)
        self.P = np.array(msg.P).reshape(3, 4)

    def depthCallback(self, msg):
        rospy.loginfo('Depth received...')
        self.depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def depthInfoCallback(self, msg):
        rospy.loginfo('Depth CameraInfo received...')
        self.depth_K = np.array(msg.K).reshape(3, 3)
        self.depth_D = np.array(msg.D)
        self.depth_R = np.array(msg.R).reshape(3, 3)
        self.depth_P = np.array(msg.P).reshape(3, 4)

if __name__ == '__main__':
    rospy.init_node('foundationpose_tracker', anonymous=False)
    FoundationPoseTracker()
    rospy.spin()

