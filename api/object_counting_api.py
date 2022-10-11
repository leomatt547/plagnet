import datetime
import time
import tensorflow as tf
import cv2
import numpy as np
from utils import visualization_utils as vis_util
import os

# Variables
total_passed_plastic = 0  # using it to count plastics
total_passed_small_plastic = 0
total_passed_medium_plastic = 0
total_passed_large_plastic = 0


def cumulative_object_counting_x_axis(input_video, detection_graph, category_index, roi, deviation, h, d, f, save_image=False):
    total_passed_small_plastic = 0
    total_passed_medium_plastic = 0
    total_passed_large_plastic = 0

    # input video
    cap = cv2.VideoCapture(input_video)

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    file_name = time.strftime("%Y%m%d-%H%M%S")
    folder_name = os.path.join("Output", file_name)
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
        os.mkdir(str(folder_name)+"/video")
    output_movie = cv2.VideoWriter(
        (str(folder_name)+"/video/"+str(file_name) +
         ".AVI"), fourcc, fps, (width, height))

    total_passed_small_plastic = 0
    total_passed_medium_plastic = 0
    total_passed_large_plastic = 0

    set_small_plastic = 0
    set_medium_plastic = 0
    set_large_plastic = 0

    small_cond = 15
    medium_cond = 10
    large_cond = 5

    reusable_bag = 0

    plastic_size_category = "..."
    with detection_graph.as_default():
        with tf.compat.v1.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name(
                'detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name(
                'detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name(
                'detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name(
                'num_detections:0')

            # for all the frames that are extracted from input video
            while (cap.isOpened()):
                ret, frame = cap.read()

                if not ret:
                    print("end of the video file...")
                    break

                input_frame = frame

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(input_frame, axis=0)

                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores,
                        detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                # insert information text to video frame
                font = cv2.FONT_HERSHEY_SIMPLEX

                # Visualization of the results of a detection.
                counter_small, \
                    counter_medium, \
                    counter_large = vis_util.visualize_boxes_and_labels_on_image_array_x_axis(cap.get(1),
                                                                                              input_frame,
                                                                                              1,
                                                                                              np.squeeze(
                        boxes),
                        np.squeeze(classes).astype(
                        np.int32),
                        np.squeeze(
                        scores),
                        category_index,
                        x_reference=roi,
                        deviation=deviation,
                        use_normalized_coordinates=True,
                        line_thickness=4,
                        camera_height=h,
                        camera_diameter=d,
                        camera_focus=f,
                        folder_name=folder_name,
                        save_image=save_image)

                """
                # set a min thresh score, say 0.8
                min_score_thresh = 0.5
                bboxes = boxes[scores > min_score_thresh]

                # get image size
                im_width = frame.shape[1]
                im_height = frame.shape[0]
                final_box = []
                for box in bboxes:
                    ymin, xmin, ymax, xmax = box
                    final_box.append(
                        [xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height])
                    # final_box.append([xmin , xmax , ymin , ymax])
                """
                # when the plastic passed over line and counted, make the color of ROI line green
                if counter_large == 1 or counter_medium == 1 or counter_small == 1:
                    cv2.line(input_frame, (roi, 0),
                             (roi, height), (0, 0xFF, 0), 5)
                    if counter_large == 1:
                        plastic_size_category = "Large"
                    elif counter_medium == 1:
                        plastic_size_category = "Medium"
                    elif counter_small == 1:
                        plastic_size_category = "Small"
                else:
                    cv2.line(input_frame, (roi, 0),
                             (roi, height), (0, 0, 0xFF), 5)

                total_passed_small_plastic = total_passed_small_plastic + counter_small
                total_passed_medium_plastic = total_passed_medium_plastic + counter_medium
                total_passed_large_plastic = total_passed_large_plastic + counter_large

                if total_passed_large_plastic % 5 == 0 and total_passed_large_plastic > 0:
                    total_passed_large_plastic = 0
                    set_large_plastic += 1
                    reusable_bag += 1

                if (total_passed_medium_plastic % 10 == 0) and total_passed_medium_plastic > 0:
                    total_passed_medium_plastic = 0
                    set_medium_plastic += 1
                    reusable_bag += 1

                if (total_passed_small_plastic % 15 == 0) and total_passed_small_plastic > 0:
                    total_passed_small_plastic = 0
                    set_small_plastic += 1
                    reusable_bag += 1

                if reusable_bag > 0:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(
                        input_frame,
                        'Congratulations!',
                        (frame.shape[1]-350, 30),
                        font,
                        1.2,
                        (0, 0xFF, 0xFF),
                        2,
                        cv2.FONT_HERSHEY_SIMPLEX,
                    )

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(
                        input_frame,
                        'You will get ',
                        (frame.shape[1]-230, 70),
                        font,
                        0.6,
                        (0xFF, 0, 0xFF),
                        2,
                        cv2.FONT_HERSHEY_SIMPLEX,
                    )

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(
                        input_frame,
                        str(reusable_bag),
                        (frame.shape[1]-250, 180),
                        font,
                        4,
                        (0xFF, 0, 0xFF),
                        8,
                        cv2.FONT_HERSHEY_SIMPLEX,
                    )

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(
                        input_frame,
                        'Reusable Plastic Bag',
                        (frame.shape[1]-270, 220),
                        font,
                        0.6,
                        (0, 150, 0),
                        2,
                        cv2.FONT_HERSHEY_SIMPLEX,
                    )
                # insert information text to video frame
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(
                    input_frame,
                    'Plastic Bag Detected: ' + str((set_large_plastic*large_cond)+total_passed_large_plastic+(
                        set_medium_plastic*medium_cond)+total_passed_medium_plastic+(set_small_plastic*small_cond)+total_passed_small_plastic),
                    (frame.shape[1] - 530, frame.shape[0] - 200),
                    font,
                    0.8,
                    (150, 0, 0),
                    2,
                    cv2.FONT_HERSHEY_SIMPLEX,
                )

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(
                    input_frame,
                    'Large Plastic      Medium Plastic      Small Plastic',
                    (frame.shape[1] - 580, frame.shape[0] - 20),
                    font,
                    0.6,
                    (0, 0xFF, 0xFF),
                    2,
                    cv2.FONT_HERSHEY_SIMPLEX,
                )

                cv2.putText(
                    input_frame,
                    str((set_large_plastic*large_cond) +
                        total_passed_large_plastic),
                    (frame.shape[1] - 590, frame.shape[0] - 75),
                    font,
                    4,
                    (0, 0, 255),
                    12,
                    cv2.FONT_HERSHEY_SIMPLEX,
                )

                cv2.putText(
                    input_frame,
                    str((set_medium_plastic*medium_cond) +
                        total_passed_medium_plastic),
                    (frame.shape[1] - 400, frame.shape[0] - 75),
                    font,
                    4,
                    (0, 0xFF, 0xFF),
                    12,
                    cv2.FONT_HERSHEY_SIMPLEX,
                )

                cv2.putText(
                    input_frame,
                    str((set_small_plastic*small_cond) +
                        total_passed_small_plastic),
                    (frame.shape[1] - 210, frame.shape[0] - 75),
                    font,
                    4,
                    (0, 100, 0),
                    12,
                    cv2.FONT_HERSHEY_SIMPLEX,
                )

                cv2.putText(
                    input_frame,
                    'Detected Plastic Bag Category: ' +
                    str(plastic_size_category),
                    (10, frame.shape[0] - 20),
                    font,
                    0.8,
                    (0xFF, 0, 0xFF),
                    2,
                    cv2.FONT_HERSHEY_SIMPLEX,
                )

                cv2.putText(
                    input_frame,
                    'ROI Line',
                    (545, roi-10),
                    font,
                    0.6,
                    (0, 0, 0xFF),
                    2,
                    cv2.LINE_AA,
                )

                timestamp = datetime.datetime.now()
                cv2.putText(input_frame, timestamp.strftime(
                    "%A %d %B %Y %I:%M:%S%p"), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

                output_movie.write(input_frame)
                print("writing frame")
                cv2.imshow('object counting', input_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()
