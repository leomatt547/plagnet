from utils.image_utils import image_saver
import os

is_small_plastic_detected = [0]
is_medium_plastic_detected = [0]
is_large_plastic_detected = [0]
bottom_position_of_detected_plastic = [0]


def count_objects_x_axis(top, bottom, right, left, crop_img, roi_position, deviation, h, d, f, folder_name, save_image=False):
    # Insert camera position
    height = abs(bottom - top)
    width = abs(right - left)

    length = ((h/d) * f)*width
    lebar = ((h/d) * f)*height

    area = length*lebar

    if (save_image):
        if not os.path.exists(os.path.join(folder_name, "images")):
            os.mkdir(str(folder_name)+"/images")
            os.mkdir(str(folder_name)+"/images/small")
            os.mkdir(str(folder_name)+"/images/medium")
            os.mkdir(str(folder_name)+"/images/large")
    if (abs(((right+left)/2)-roi_position) < deviation) and (area > 3000):
        is_large_plastic_detected.insert(0, 1)
        if (save_image):
            image_saver.save_large_plastic_image(str(folder_name)+"/images/large",
                                                 crop_img)  # save plastic image

    elif (abs(((right+left)/2)-roi_position) < deviation-10) and (area <= 3000) and (area >= 1200):
        is_medium_plastic_detected.insert(0, 1)
        if (save_image):
            image_saver.save_medium_plastic_image(str(folder_name)+"/images/medium",
                                                  crop_img)  # save plastic image

    elif (abs(((right+left)/2)-roi_position) < deviation-2.5) and (area < 1200):
        is_small_plastic_detected.insert(0, 1)
        if (save_image):
            image_saver.save_small_plastic_image(str(folder_name)+"/images/small",
                                                 crop_img)  # save plastic image

    return is_small_plastic_detected, is_medium_plastic_detected, is_large_plastic_detected
