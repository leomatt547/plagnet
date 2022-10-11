import cv2
import os

small_plastic_count = [0]
medium_plastic_count = [0]
large_plastic_count = [0]

current_path = os.getcwd()


def save_small_plastic_image(folder_name, source_image):
    cv2.imwrite(folder_name+"/small_" +
                str(len(small_plastic_count)) + ".png", source_image)
    small_plastic_count.insert(0, 1)


def save_medium_plastic_image(folder_name, source_image):
    cv2.imwrite(folder_name+"/medium_" +
                str(len(medium_plastic_count)) + ".png", source_image)
    medium_plastic_count.insert(0, 1)


def save_large_plastic_image(folder_name, source_image):
    cv2.imwrite(folder_name+"/large_" +
                str(len(large_plastic_count)) + ".png", source_image)
    large_plastic_count.insert(0, 1)
