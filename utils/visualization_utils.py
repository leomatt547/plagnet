# ----------------------------------------------
# --- Author         : Ahmet Ozlu
# --- Mail           : ahmetozlu93@gmail.com
# --- Date           : 27th January 2018
# ----------------------------------------------

"""A set of functions that are used for visualization.

These functions often receive an image, perform some visualization on the image.
The functions do not return a value, instead they modify the image itself.

"""

# Imports
import collections
import numpy as np
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import numpy
import os

# string utils - import
from utils.string_utils import custom_string_util

# image utils - image saver import
from utils.image_utils import image_saver

#  predicted_speed predicted_color module - import
from utils.object_counting_module import object_counter_x_axis

# color recognition module - import
# from utils.color_recognition_module import color_recognition_api

# Variables
is_plastic_detected = [0]
ROI_POSITION = [0]
DEVIATION = [0]
is_color_recognition_enable = [0]
mode_number = [0]
x_axis = [0]

_TITLE_LEFT_MARGIN = 10
_TITLE_TOP_MARGIN = 10

STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]

current_path = os.getcwd()


def draw_bounding_box_on_image_array(current_frame_number, image,
                                     ymin,
                                     xmin,
                                     ymax,
                                     xmax,
                                     color='red',
                                     thickness=4,
                                     display_str_list=(),
                                     use_normalized_coordinates=True,
                                     camera_height=0.0745,
                                     camera_diameter=0.25745,
                                     camera_focus=0.26,
                                     folder_name="Output/",
                                     save_image=False):
    """Adds a bounding box to an image (numpy array).

    Args:
      image: a numpy array with shape [height, width, 3].
      ymin: ymin of bounding box in normalized coordinates (same below).
      xmin: xmin of bounding box.
      ymax: ymax of bounding box.
      xmax: xmax of bounding box.
      color: color to draw bounding box. Default is red.
      thickness: line thickness. Default value is 4.
      display_str_list: list of strings to display in box
                        (each to be shown on its own line).
      use_normalized_coordinates: If True (default), treat coordinates
        ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
        coordinates as absolute.
    """
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
    is_small_plastic_detected, \
        is_medium_plastic_detected, \
        is_large_plastic_detected = draw_bounding_box_on_image(current_frame_number, image_pil, ymin, xmin, ymax, xmax, color,
                                                               thickness, display_str_list,
                                                               use_normalized_coordinates, camera_height,
                                                               camera_diameter,
                                                               camera_focus,
                                                               folder_name,
                                                               save_image)
    np.copyto(image, np.array(image_pil))
    return is_small_plastic_detected, is_medium_plastic_detected, is_large_plastic_detected


def draw_bounding_box_on_image(current_frame_number, image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color='red',
                               thickness=4,
                               display_str_list=(),
                               use_normalized_coordinates=True,
                               camera_height=0.0745,
                               camera_diameter=0.25745,
                               camera_focus=0.26,
                               folder_name="Output/",
                               save_image=False):
    """Adds a bounding box to an image.

    Each string in display_str_list is displayed on a separate line above the
    bounding box in black text on a rectangle filled with the input 'color'.
    If the top of the bounding box extends to the edge of the image, the strings
    are displayed below the bounding box.

    Args:
      image: a PIL.Image object.
      ymin: ymin of bounding box.
      xmin: xmin of bounding box.
      ymax: ymax of bounding box.
      xmax: xmax of bounding box.
      color: color to draw bounding box. Default is red.
      thickness: line thickness. Default value is 4.
      display_str_list: list of strings to display in box
                        (each to be shown on its own line).
      use_normalized_coordinates: If True (default), treat coordinates
        ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
        coordinates as absolute.
    """
    is_small_plastic_detected = [0]
    is_medium_plastic_detected = [0]
    is_large_plastic_detected = [0]
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    if use_normalized_coordinates:
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                      ymin * im_height, ymax * im_height)
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    draw.line([(left, top), (left, bottom), (right, bottom),
               (right, top), (left, top)], width=thickness, fill=color)

    image_temp = numpy.array(image)
    detected_plastic_image = image_temp[int(
        top):int(bottom), int(left):int(right)]

    '''if(bottom > ROI_POSITION): # if the plastic get in ROI area, plastic predicted_speed predicted_color algorithms are called - 200 is an arbitrary value, for my case it looks very well to set position of ROI line at y pixel 200'''
    if (x_axis[0] == 1):
        is_small_plastic_detected, \
            is_medium_plastic_detected, \
            is_large_plastic_detected = object_counter_x_axis.count_objects_x_axis(top,
                                                                                   bottom,
                                                                                   right,
                                                                                   left,
                                                                                   detected_plastic_image,
                                                                                   ROI_POSITION[0],
                                                                                   DEVIATION[0],
                                                                                   camera_height,
                                                                                   camera_diameter,
                                                                                   camera_focus,
                                                                                   folder_name,
                                                                                   save_image
                                                                                   )
    try:
        font = ImageFont.truetype('arial.ttf', 16)
    except IOError:
        font = ImageFont.load_default()

    display_str_list[0] = display_str_list[0]

    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]

    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = bottom + total_display_str_height

    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle(
            [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                              text_bottom)],
            fill=color)
        draw.text(
            (left + margin, text_bottom - text_height - margin),
            display_str,
            fill='black',
            font=font)
        text_bottom -= text_height - 2 * margin
        return is_small_plastic_detected, is_medium_plastic_detected, is_large_plastic_detected


def visualize_boxes_and_labels_on_image_array_x_axis(current_frame_number,
                                                     image,
                                                     mode,
                                                     boxes,
                                                     classes,
                                                     scores,
                                                     category_index,
                                                     x_reference=None,
                                                     deviation=None,
                                                     use_normalized_coordinates=False,
                                                     max_boxes_to_draw=20,
                                                     min_score_thresh=.5,
                                                     agnostic_mode=False,
                                                     line_thickness=4,
                                                     camera_height=0.0745,
                                                     camera_diameter=0.25745,
                                                     camera_focus=0.26,
                                                     folder_name="Output/",
                                                     save_image=False):
    """Overlay labeled boxes on an image with formatted scores and label names.

    This function groups boxes that correspond to the same location
    and creates a display string for each detection and overlays these
    on the image. Note that this function modifies the image in place, and returns
    that same image.

    Args:
      image: uint8 numpy array with shape (img_height, img_width, 3)
      boxes: a numpy array of shape [N, 4]
      classes: a numpy array of shape [N]. Note that class indices are 1-based,
        and match the keys in the label map.
      scores: a numpy array of shape [N] or None.  If scores=None, then
        this function assumes that the boxes to be plotted are groundtruth
        boxes and plot all boxes as black with no classes or scores.
      category_index: a dict containing category dictionaries (each holding
        category index `id` and category name `name`) keyed by category indices.
      instance_masks: a numpy array of shape [N, image_height, image_width], can
        be None
      keypoints: a numpy array of shape [N, num_keypoints, 2], can
        be None
      use_normalized_coordinates: whether boxes is to be interpreted as
        normalized coordinates or not.
      max_boxes_to_draw: maximum number of boxes to visualize.  If None, draw
        all boxes.
      min_score_thresh: minimum score threshold for a box to be visualized
      agnostic_mode: boolean (default: False) controlling whether to evaluate in
        class-agnostic mode or not.  This mode will display scores but ignore
        classes.
      line_thickness: integer (default: 4) controlling line width of the boxes.

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3) with overlaid boxes.
    """
    # Create a display string (and color) for every box location, group any boxes
    # that correspond to the same location.
    counter_small = 0
    counter_medium = 0
    counter_large = 0
    ROI_POSITION.insert(0, x_reference)
    DEVIATION.insert(0, deviation)
    x_axis.insert(0, 1)
    is_small_plastic_detected = []
    is_medium_plastic_detected = []
    is_large_plastic_detected = []
    mode_number.insert(0, mode)
    # is_color_recognition_enable.insert(0,color_recognition_status)
    box_to_display_str_map = collections.defaultdict(list)
    box_to_color_map = collections.defaultdict(str)
    if not max_boxes_to_draw:
        max_boxes_to_draw = boxes.shape[0]
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores is None or scores[i] > min_score_thresh:
            box = tuple(boxes[i].tolist())
            if scores is None:
                box_to_color_map[box] = 'black'
            else:
                if not agnostic_mode:
                    if classes[i] in category_index.keys():
                        class_name = category_index[classes[i]]['name']
                    else:
                        class_name = 'N/A'
                    display_str = '{}: {}%'.format(
                        class_name, int(100*scores[i]))
                else:
                    display_str = 'score: {}%'.format(int(100 * scores[i]))

                box_to_display_str_map[box].append(display_str)
                if agnostic_mode:
                    box_to_color_map[box] = 'DarkOrange'
                else:
                    box_to_color_map[box] = STANDARD_COLORS[
                        classes[i] % len(STANDARD_COLORS)]

    if (mode == 1):
        counting_mode = ""
    # Draw all boxes onto image.
    for box, color in box_to_color_map.items():
        ymin, xmin, ymax, xmax = box
        is_small_plastic_detected, \
            is_medium_plastic_detected, \
            is_large_plastic_detected = draw_bounding_box_on_image_array(current_frame_number,
                                                                         image,
                                                                         ymin,
                                                                         xmin,
                                                                         ymax,
                                                                         xmax,
                                                                         color=color,
                                                                         thickness=line_thickness,
                                                                         display_str_list=box_to_display_str_map[
                                                                             box],
                                                                         use_normalized_coordinates=use_normalized_coordinates,
                                                                         camera_height=camera_height,
                                                                         camera_diameter=camera_diameter,
                                                                         camera_focus=camera_focus,
                                                                         folder_name=folder_name,
                                                                         save_image=save_image)
    if (1 in is_small_plastic_detected):
        counter_small = 1
        del is_small_plastic_detected[:]
        is_small_plastic_detected = []

    elif (1 in is_medium_plastic_detected):
        counter_medium = 1
        del is_medium_plastic_detected[:]
        is_medium_plastic_detected = []

    elif (1 in is_large_plastic_detected):
        counter_large = 1
        del is_large_plastic_detected[:]
        is_large_plastic_detected = []

    if (mode == 1):
        counting_mode = counting_mode.replace(
            "['", " ").replace("']", " ").replace("%", "")
        counting_mode = ''.join([i for i in counting_mode.replace(
            "['", " ").replace("']", " ").replace("%", "") if not i.isdigit()])
        counting_mode = str(custom_string_util.word_count(counting_mode))
        counting_mode = counting_mode.replace("{", "").replace("}", "")

    return counter_small, counter_medium, counter_large
