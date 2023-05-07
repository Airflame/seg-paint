import math
import fastwer
import numpy as np
from PIL import Image
import cv2.cv2 as cv2
import pytesseract
from scipy.signal import convolve2d
from difflib import SequenceMatcher


class Metrics:
    @staticmethod
    def estimate_text_readability(filename: str, reference_text: str, print_info: bool = True):
        img = np.array(Image.open(filename))
        text = pytesseract.image_to_string(img, config='--psm 6', lang='pol').strip()
        gestalt = SequenceMatcher(None, reference_text, text).ratio()
        cer = fastwer.score_sent(text, reference_text, char_level=True)
        wer = fastwer.score_sent(text, reference_text, char_level=False)
        if print_info:
            print("Text matching ratio " + '{:.2%}'.format(gestalt))
            print("Character error rate " + str(cer))
            print("Word error rate " + str(wer))
        return {"gestalt": gestalt, "cer": cer, "wer": wer}

    @staticmethod
    def estimate_noise(I):
        H, W = I.shape
        M = [[1, -2, 1],
             [-2, 4, -2],
             [1, -2, 1]]
        sigma = np.sum(np.sum(np.absolute(convolve2d(I, M))))
        sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (W - 2) * (H - 2))
        return sigma

    @staticmethod
    def calculate_iou(image, ground_truth):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_ground_truth = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2HSV)
        colors = (
            (0, 255, 255),
            (15, 255, 255),
            (30, 255, 255),
            (60, 255, 255),
            (120, 255, 255),
            (137, 255, 130),
            (141, 255, 211),
        )
        iou_results = []
        for color in colors:
            mask_image = cv2.inRange(hsv_image, color, color)
            mask_ground_truth = cv2.inRange(hsv_ground_truth, color, color)
            target_image = cv2.bitwise_and(image, image, mask=mask_image)
            target_ground_truth = cv2.bitwise_and(ground_truth, ground_truth, mask=mask_ground_truth)
            target_intersection = cv2.bitwise_and(target_image, target_ground_truth)
            target_union = cv2.bitwise_or(target_image, target_ground_truth)
            cv2.imwrite("data/target_intersection.png", target_intersection)
            cv2.imwrite("data/target_union.png", target_union)
            res_intersection = cv2.cvtColor(target_intersection, cv2.COLOR_BGR2GRAY)
            res_union = cv2.cvtColor(target_union, cv2.COLOR_BGR2GRAY)
            intersection = cv2.countNonZero(res_intersection)
            union = cv2.countNonZero(res_union)
            if union != 0:
                iou_results.append(intersection / union)
        return sum(iou_results)/len(iou_results)
