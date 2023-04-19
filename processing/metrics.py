import math
import fastwer
import numpy as np
from PIL import Image
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
