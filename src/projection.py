#!/usr/bin/env python
"""
A ROS node to get 3D coordinates of bounding boxes returned by object recognizer node.

This node gets the detected objects bounding boxes and gets the real world coordinates
by using depth values. We segment the foreground in a binary image and calculate its
centroid. The depth at the centroid gives the z coordinate.

Author:
    Cagatay Odabasi -- cagatay.odabasi@ipa.fraunhofer.de
"""

import rospy

import message_filters

import numpy as np

from scipy.signal import find_peaks

from cv_bridge import CvBridge

from sensor_msgs.msg import Image

from cob_perception_msgs.msg import DetectionArray

import cv2


def get_centroid(mask):
    # Calculate moments of the binary image
    M = cv2.moments(mask)

    # Calculate x,y coordinate of the centroid
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    else:
        cx, cy = 0, 0

    return cx, cy


def foreground_mask(depth_img):
    # Calculate the histogram
    histogram = cv2.calcHist([depth_img], [0], None, [256], [0, 256])

    # Smooth the histogram to find peaks with a stronger Gaussian filter
    smooth_histogram = cv2.GaussianBlur(histogram, (25, 25), 0)

    # Find the first peak in the smoothed histogram; pad with a 0 to also include peak at minimum depth
    smooth_histogram = np.concatenate([np.zeros(1), smooth_histogram.flatten()])
    min_height = 100
    peaks, _ = find_peaks(smooth_histogram, height=min_height)
    if len(peaks) == 0:
        highest_val = round(np.max(smooth_histogram))
        rospy.logdebug(f"No peaks found on depth image (highest value {highest_val} < {min_height})")
        return None
    first_peak_value = peaks[0].item()

    # Find the start of the first peak
    # The start of the peak is the point before the peak where the histogram value starts to rise significantly
    start_of_first_peak = 0
    for i in range(first_peak_value, 0, -1):
        if smooth_histogram[i] < 0.25 * smooth_histogram[first_peak_value]:
            start_of_first_peak = i
            break

    # Find the end of the first peak
    # The end of the peak is the point after the peak where the histogram value drops significantly
    end_of_first_peak = first_peak_value
    for i in range(first_peak_value, len(smooth_histogram)):
        if smooth_histogram[i] < 0.25 * smooth_histogram[first_peak_value]:
            end_of_first_peak = i
            break

    # Apply the threshold keeping the entire peak
    return cv2.inRange(depth_img, start_of_first_peak, end_of_first_peak)


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
        depth (sensor_msgs/Image): depth image from camera
        """

        depth_img = self._bridge.imgmsg_to_cv2(depth, "passthrough")

        # get the number of detections
        no_of_detections = len(msg.detections)

        FLOOR_SHRINK_FRACTION = 0.2

        # Check if there is a detection
        if no_of_detections > 0:
            for i, detection in enumerate(msg.detections):
                x = detection.mask.roi.x
                y = detection.mask.roi.y
                width = detection.mask.roi.width
                height = detection.mask.roi.height

                floor_rows_to_remove = int(FLOOR_SHRINK_FRACTION * height)

                # Crop the ROI image by removing the bottom 20% of the rows
                depth_img_roi = depth_img[y:y + height, x:x + width]
                depth_img_roi = depth_img_roi[:-floor_rows_to_remove, :]

                # Normalize the depth image to 0 .. 255 values and extract the foreground mask
                norm_depth_img = cv2.normalize(depth_img_roi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                fg_mask = foreground_mask(norm_depth_img)

                # Calculate the centroid and covert to real world coordinates
                if fg_mask is not None:
                    centroid_x, centroid_y = get_centroid(fg_mask)
                else:
                    # fallback to ROI center if we failed to extract the foreground mask
                    centroid_x, centroid_y = width // 2, height // 2

                centroid_depth = depth_img_roi[centroid_y, centroid_x]

                real_x = ((x + centroid_x) - self.cx) * centroid_depth / self.f
                real_y = ((y + centroid_y) - self.cy) * centroid_depth / self.f

                msg.detections[i].pose.header = detection.header
                msg.detections[i].pose.pose.position.x = real_x
                msg.detections[i].pose.pose.position.y = real_y
                msg.detections[i].pose.pose.position.z = centroid_depth
                msg.detections[i].pose.pose.orientation.w = 1.0  # no information; just return a valid quaternion

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
    ProjectionNode()


if __name__ == '__main__':
    main()
