import numpy as NP
import pickle
import datetime, glob, os
import cv2


class ColorLUT:
    def __init__(self):
        self.lut = NP.zeros((256, 256, 256), dtype=NP.uint8)
        pass

    @staticmethod
    def apply_mask(img: NP.ndarray, mask: NP.ndarray) -> NP.ndarray:
        new_img = NP.zeros_like(img)
        bmask = mask != 0

        new_img[bmask] = img[bmask]

        if 0:
            cv2.imshow("mask", mask)
            cv2.imshow("img", img)
            cv2.imshow("new_image", new_img)
            cv2.waitKey(10000)
        return new_img

    def build_lut_from_image(self, image: NP.ndarray, kernel_radius: int = 1,
                             this_color_bit_value: int = 1) -> NP.ndarray:

        # Ensure image is uint8
        # image = NP.clip(image, 0, 255).astype(NP.uint8)

        # Get unique RGB values
        unique_colors = NP.unique(image.reshape(-1, 3), axis=0)

        for r, g, b in unique_colors:
            r0, r1 = max(0, r - kernel_radius), min(255, r + kernel_radius)
            g0, g1 = max(0, g - kernel_radius), min(255, g + kernel_radius)
            b0, b1 = max(0, b - kernel_radius), min(255, b + kernel_radius)
            self.lut[r0:r1 + 1, g0:g1 + 1, b0:b1 + 1] |= this_color_bit_value
        # print(r,g,b)

        # print(GC(), NP.count_nonzero(self.lut))
        # print(self.lut)
        return self.lut, NP.count_nonzero(self.lut)  # for debug

    def zero_lut_from_image(self, image: NP.ndarray, kernel_radius: int = 1,
                            this_color_bit_value: int = 1) -> NP.ndarray:

        # Ensure image is uint8
        # image = NP.clip(image, 0, 255).astype(NP.uint8)

        # Get unique RGB values
        unique_colors = NP.unique(image.reshape(-1, 3), axis=0)

        # inv_bit_value = ~this_color_bit_value
        # Force the inverted bitmask to remain a uint8
        inv_bit_value = NP.uint8(~this_color_bit_value)

        for r, g, b in unique_colors:
            r0, r1 = max(0, r - kernel_radius), min(255, r + kernel_radius)
            g0, g1 = max(0, g - kernel_radius), min(255, g + kernel_radius)
            b0, b1 = max(0, b - kernel_radius), min(255, b + kernel_radius)
            self.lut[r0:r1 + 1, g0:g1 + 1, b0:b1 + 1] &= inv_bit_value

    def mask_with_lut_bitmask(self, image: NP.ndarray, bitmask: int) -> NP.ndarray:
        # image = NP.clip(image, 0, 255).astype(NP.uint8)
        r, g, b = image[..., 0], image[..., 1], image[..., 2]
        wtf = (self.lut[r, g, b] & bitmask) != 0

        return wtf

    def test_lut(self):
        img1 = cv2.imread("orange.jpg")
        hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        self.build_lut_from_image(hsv1, 3, 0b00000001)

        img2 = cv2.imread("blue.jpg")
        hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
        self.build_lut_from_image(hsv2, 3, 0b00000010)

        test_img = cv2.imread("scene.jpg")
        hsv_image = cv2.cvtColor(test_img, cv2.COLOR_BGR2HSV)
        mask_orange = self.mask_with_lut_bitmask(hsv_image, 0b00000001)
        mask_blue = self.mask_with_lut_bitmask(hsv_image, 0b00000010)

    @staticmethod
    def apply_lut(image: NP.ndarray, lut: NP.ndarray) -> NP.ndarray:

        h, w, _ = image.shape
        flat = image.reshape(-1, 3)
        keys = (flat[:, 0].astype(NP.uint32) << 16) | (flat[:, 1].astype(NP.uint32) << 8) | flat[:, 2]
        # print(f"{keys.shape=}", GC())
        # mask_flat = lut[keys]
        mask_flat = lut.ravel()[keys]
        return mask_flat.reshape(image.shape[:2])

    @staticmethod
    def apply_lut_return_image(image: NP.ndarray, lut: NP.ndarray) -> NP.ndarray:
        # H, W, _ = image.shape
        # flat = image.reshape(-1, 3)
        # keys = (flat[ :, 0 ].astype(NP.uint32) << 16) | (flat[ :, 1 ].astype(NP.uint32) << 8) | flat[ :, 2 ]
        #
        # # LUT returns RGB values per key
        # pixels_mapped = lut[ keys ]  # shape (H*W, 3)
        # return pixels_mapped.reshape(H, W, 3)

        def apply_lut_return_image(image: NP.ndarray, lut: NP.ndarray) -> NP.ndarray:
            r, g, b = image[..., 0], image[..., 1], image[..., 2]
            mask = lut[r, g, b] != 0

            new_img = NP.zeros_like(image)
            new_img[mask] = image[mask]

            return new_img
    @staticmethod
    def apply_lut_return_image_fast(image: NP.ndarray, lut: NP.ndarray) -> NP.ndarray:
        r, g, b = image[..., 0], image[..., 1], image[..., 2]
        mask = lut[r, g, b] != 0

        new_img = NP.zeros_like(image)
        new_img[mask] = image[mask]

        return new_img


def help():
    print("\n\nKEYS:")
    print("h: this help")
    print("s: save the LTU as a pickle file named with the current time")
    print("0/1 : zero or build a LTU using camera image.  11")
    print("z/o: overwrite LUT with zeros or ones")
    print("a/l/r: use all, or left half, or right half of image ONLY/")
    print("q:  cycle size of train image (only for training postive pixels)")



def cv2_named_windows(win_list:list, value=cv2.WINDOW_FREERATIO) :
    for name in win_list :
        print(name)
        cv2.namedWindow(name, value)

def find_most_recent_file(fpattern:str="*.pkl") :
    """
    load the latest of the files for a given file pattern  (eg *.pkl)

    :param fpattern:  a path plus pattern of files to consider
    :return: the latest of the files
    """

    latest_file = None
    # 1. Get a list of files matching your pattern
    files = glob.glob(fpattern)

    # 2. Check if the list isn't empty
    if files:
        # 3. Find the file with the maximum (most recent) modification time
        latest_file = max(files, key=os.path.getmtime)
    return latest_file


def load_lut_obj(fname) :
    with open(fname, "rb") as fin :
        obj = pickle.load(fin)
    return obj

