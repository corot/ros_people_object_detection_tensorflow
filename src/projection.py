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

from scipy.signal import find_peaks

from cv_bridge import CvBridge

from sensor_msgs.msg import Image

from cob_perception_msgs.msg import DetectionArray, Detection

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


from matplotlib import pyplot as plt
def foreground_mask(depth_img):
    # Calculate the histogram
    histogram = cv2.calcHist([depth_img], [0], None, [256], [0, 256])

    # Smooth the histogram to find peaks with a stronger Gaussian filter
    smooth_histogram = cv2.GaussianBlur(histogram, (15, 15), 0)

    # Find the first peak in the smoothed histogram
#    first_peak_value = np.argmax(smooth_histogram).item()
    peaks, _ = find_peaks(smooth_histogram.flatten(), height=250)
    first_peak_value = peaks[0].item() if len(peaks) > 0 else 0

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

    # Apply the threshold using the first peak and its end
    try:
        fg_mask = cv2.inRange(depth_img, start_of_first_peak, end_of_first_peak)
    except TypeError as e:
        print(str(e))
        pass
    # fg_mask = cv2.threshold(depth_img, start_of_first_peak, 255, cv2.THRESH_TOZERO)[1]
    # fg_mask = cv2.threshold(fg_mask, end_of_first_peak, 255, cv2.THRESH_TOZERO_INV)[1]
    #
    # # Display the results
    # plt.figure(figsize=(10, 5))
    #
    # plt.subplot(1, 2, 1)
    # plt.imshow(depth_img, cmap='gray')
    # plt.title('Original Depth Image')
    # plt.axis('off')
    #
    # plt.subplot(1, 2, 2)
    # plt.plot(histogram, color='orange', label='Original Histogram')
    # plt.plot(smooth_histogram, color='blue', label='Smoothed Histogram')
    # plt.axvline(x=start_of_first_peak, color='r', linestyle='--', label=f'First Peak Start: {start_of_first_peak}')
    # plt.axvline(x=end_of_first_peak, color='g', linestyle='--', label=f'First Peak End: {end_of_first_peak}')
    # plt.title('Histogram with Threshold Range')
    # plt.xlabel('Pixel Value')
    # plt.ylabel('Frequency')
    # plt.legend()
    # plt.grid(True)
    #
    # plt.tight_layout()
    # plt.show()

    return fg_mask


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
        depth (sensor_msgs/PointCloud2): depth image from camera
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

                # Crop the image by removing the bottom 10% of the rows
                depth_img_roi = depth_img[y:y + height, x:x + width]
#                cv2.imshow('ROI', cv2.normalize(depth_img_roi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))

                depth_img_roi = depth_img_roi[:-floor_rows_to_remove, :]

                norm_depth_img = cv2.normalize(depth_img_roi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                fg_mask = foreground_mask(norm_depth_img)
                # cv2.imshow('norm', norm_depth_img)
                # cv2.imshow('mask', fg_mask)
                #
                # cv2.imshow('Original Depth Image',
                #            cv2.normalize(cv_depth_bounding_box, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
                # cv2.imshow('Segmented Foreground Image',
                #            cv2.normalize(fg_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                centroid_x, centroid_y = get_centroid(fg_mask)
                #print(f'Centroid of the mask is at: ({centroid_x}, {centroid_y})')

                # Visualize the centroid on the mask image
                # output = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
                # cv2.circle(output, (centroid_x, centroid_y), 5, (0, 0, 255), -1)
                # cv2.imshow('Centroid', output)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                #
                # cv2.imwrite('mask.png', norm_depth_img)

                depth_mean = depth_img_roi[centroid_y, centroid_x]
#                  depth_mean = np.nanmedian(cv_depth_bounding_box[np.nonzero(cv_depth_bounding_box)])

                real_x = ((x + centroid_x) - self.cx) * depth_mean / self.f
                real_y = ((y + centroid_y) - self.cy) * depth_mean / self.f

                msg.detections[i].pose.header = detection.header
                msg.detections[i].pose.pose.position.x = real_x
                msg.detections[i].pose.pose.position.y = real_y
                msg.detections[i].pose.pose.position.z = depth_mean
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
