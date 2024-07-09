#!/usr/bin/env python
"""
A ROS node to get 3D values of bounding boxes returned by face_recognizer node.

This node gets the face bounding boxes and gets the real world coordinates of
them by using depth values. It simply gets the x and y values of center point
and gets the median value of face depth values as z value of face.

Author:
    Cagatay Odabasi -- cagatay.odabasi@ipa.fraunhofer.de
"""

import rospy

import message_filters

import numpy as np

from cv_bridge import CvBridge

from sensor_msgs.msg import Image

from cob_perception_msgs.msg import DetectionArray, Detection

import cv2
import os


def segment_foreground(depth_image):
    """
    Segments the foreground from a depth image using Otsu's Method.

    Args:
        depth_image (numpy.ndarray): The input depth image with floating point values.

    Returns:
        numpy.ndarray: The depth image with foreground kept and background set to NaN.
    """
    # Normalize depth image to 8-bit grayscale for Otsu's method
    depth_image_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Apply Otsu's thresholding
    _, foreground_mask = cv2.threshold(depth_image_normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Convert mask to binary format
    foreground_mask = foreground_mask.astype(np.uint8)

    # Create an output image and set background to NaN
    output_depth = np.full_like(depth_image, np.nan, dtype=np.float32)
    output_depth[foreground_mask == 0] = depth_image[foreground_mask == 0]

    return output_depth

def segment_foreground_auto(depth_image):
    """
    Segments the foreground from a depth image using K-means clustering.

    Args:
        depth_image (numpy.ndarray): The input depth image with floating point values.

    Returns:
        numpy.ndarray: The depth image with foreground kept and background set to NaN.
    """
    # Flatten the depth image to a single column
    pixels = depth_image.flatten().astype(np.float32)

    # Apply K-means clustering to find two clusters (foreground and background)
    pixels = pixels.reshape(-1, 1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 2
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Determine which cluster is the foreground (closer to the camera)
    foreground_cluster = np.argmin(centers)

    # Create a mask for the foreground
    foreground_mask = (labels.flatten() == foreground_cluster).astype(np.uint8)

    # Reshape the mask back to the shape of the depth image
    foreground_mask = foreground_mask.reshape(depth_image.shape)

    # Create an output image and set background to NaN
    output_depth = np.full_like(depth_image, np.nan, dtype=np.float32)
    output_depth[foreground_mask == 1] = depth_image[foreground_mask == 1]

    return output_depth


class ProjectionNode(object):
    """Get 3D values of bounding boxes returned by face_recognizer node.

    _bridge (CvBridge): Bridge between ROS and CV image
    pub (Publisher): Publisher object for face depth results
    f (Float): Focal Length
    cx (Int): Principle Point Horizontal
    cy (Int): Principle Point Vertical

    """

    def __init__(self):
        super(ProjectionNode, self).__init__()

        # init the node
        rospy.init_node('cob_object_projection', anonymous=False)

        self._bridge = CvBridge()

        (depth_topic, face_topic, output_topic, f, cx, cy) = self.get_parameters()

        # Subscribe to the face positions
        sub_obj = message_filters.Subscriber(face_topic, DetectionArray)

        sub_depth = message_filters.Subscriber(depth_topic, Image)

        # Advertise the result of Face Depths
        self.pub = rospy.Publisher(output_topic, DetectionArray, queue_size=1)

        # Create the message filter
        ts = message_filters.ApproximateTimeSynchronizer([sub_obj, sub_depth], 2, 0.9)

        ts.registerCallback(self.detection_callback)

        self.f = f
        self.cx = cx
        self.cy = cy
        # self.depth_image_path = depth_image_path
        #
        # d = Detection()
        # d.mask.roi.x = 139
        # d.mask.roi.y = 201
        # d.mask.roi.width = 330
        # d.mask.roi.height = 208
        # self.detection_callback(DetectionArray(detections=[d]))


        # spin
        rospy.spin()

    def shutdown(self):
        """
        Shuts down the node
        """
        rospy.signal_shutdown("See ya!")

    def detection_callback(self, msg, depth):
        """
        Callback for RGB images: The main logic is applied here

        Args:
        msg (cob_perception_msgs/DetectionArray): detections array
        depth (sensor_msgs/PointCloud2): depth image from camera
        """

        cv_depth = self._bridge.imgmsg_to_cv2(depth, "passthrough")

        # get the number of detections
        no_of_detections = len(msg.detections)

        ROI_SHRINK_FRACTION = 0.5

        # Check if there is a detection
        if no_of_detections > 0:
            for i, detection in enumerate(msg.detections):
                x = detection.mask.roi.x
                y = detection.mask.roi.y
                width = detection.mask.roi.width
                height = detection.mask.roi.height

                clip_x = int(round((width * ROI_SHRINK_FRACTION) / 2.0))
                clip_y = int(round((height * ROI_SHRINK_FRACTION) / 2.0))
                x += clip_x
                y += clip_y
                width -= 2 * clip_x
                height -= 2 * clip_y

                cv_depth_bounding_box = cv_depth[y:y + height, x:x + width]
                fg_depth = segment_foreground_auto(cv_depth_bounding_box)
                # cv2.imshow('segment', fg_depth)
                # cv2.imshow('segment_norm', (fg_depth / 4 * 65535).astype(np.uint16))
                # cv2.imshow('bbox', cv_depth_bounding_box)
                #
                # cv2.imshow('Original Depth Image',
                #            cv2.normalize(cv_depth_bounding_box, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
                # cv2.imshow('Segmented Foreground Image',
                #            cv2.normalize(fg_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                try:
                    depth_mean = np.nanmedian(fg_depth[np.nonzero(fg_depth)])
  #                  depth_mean = np.nanmedian(cv_depth_bounding_box[np.nonzero(cv_depth_bounding_box)])

                    real_x = (x + width / 2 - self.cx) * depth_mean / self.f
                    real_y = (y + height / 2 - self.cy) * depth_mean / self.f

                    msg.detections[i].pose.header = detection.header
                    msg.detections[i].pose.pose.position.x = real_x
                    msg.detections[i].pose.pose.position.y = real_y
                    msg.detections[i].pose.pose.position.z = depth_mean
                    msg.detections[i].pose.pose.orientation.w = 1.0  # no information; just return a valid quaternion

                except Exception as e:
                    print(e)

        self.pub.publish(msg)

    def get_parameters(self):
        """
        Gets the necessary parameters from parameter server

        Returns:
        (tuple) :
            depth_topic (String): Incoming depth topic name
            face_topic (String): Incoming face bounding box topic name
            output_topic (String): Outgoing depth topic name
            f (Float): Focal Length
            cx (Int): Principle Point Horizontal
            cy (Int): Principle Point Vertical
        """

        depth_topic = rospy.get_param("/cob_object_projection/depth_topic")
        face_topic = rospy.get_param('/cob_object_projection/face_topic')
        output_topic = rospy.get_param('/cob_object_projection/output_topic')
        f = rospy.get_param('/cob_object_projection/focal_length')
        cx = rospy.get_param('/cob_object_projection/cx')
        cy = rospy.get_param('/cob_object_projection/cy')

        return depth_topic, face_topic, output_topic, f, cx, cy


def main():
    """ main function
    """
    node = ProjectionNode()


if __name__ == '__main__':
    main()
