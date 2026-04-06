import time
import scipy.ndimage as ndi
import numpy as NP

np = NP
import pickle
import datetime, glob, os
import cv2
import scipy


class ColorLUT:
    def __init__(self):

        # Ensure the LUT is 1D and C-contiguous for cache efficiency
        # self.lut = np.zeros(256**3, dtype=np.uint8)
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

    def grow_lut(self, r: int) -> None:
        # assume self.lut is 0/1 or bool
        lut_bool = self.lut.astype(NP.bool_)

        lut_grown = ndi.binary_dilation(
            lut_bool,
            iterations=r  # 3x3x3 structuring element repeated
        )

        self.lut = lut_grown.astype(self.lut.dtype)

    def erode_lut(self, r: int) -> None:
        # assume self.lut is 0/1 or bool
        lut_bool = self.lut.astype(NP.bool_)

        lut_grown = ndi.binary_erosion(
            lut_bool,
            iterations=r  # 3x3x3 structuring element repeated
        )

        self.lut = lut_grown.astype(self.lut.dtype)

    def zero_lut_from_image(self, image: NP.ndarray) -> None:
        flat_pixels = image.reshape(-1, 3)
        self.lut[flat_pixels[:, 0], flat_pixels[:, 1], flat_pixels[:, 2]] = 0

    def build_lut_from_image(self, image: NP.ndarray) -> None:
        flat_pixels = image.reshape(-1, 3)
        self.lut[flat_pixels[:, 0], flat_pixels[:, 1], flat_pixels[:, 2]] = 1

    def mask_with_lut_bitmask(self, image: NP.ndarray, bitmask: int) -> NP.ndarray:
        # image = NP.clip(image, 0, 255).astype(NP.uint8)
        r, g, b = image[..., 0], image[..., 1], image[..., 2]
        wtf = (self.lut[r, g, b] & bitmask) != 0

        return wtf

    def apply_lut_faster(self, image: np.ndarray) -> np.ndarray:
        # 1. Update the indices in-place
        # We reuse self.indices to avoid memory allocation
        # Logic: (R << 16) | (G << 8) | B
        np.left_shift(image[:, :, 0], 16, out=self.indices, casting='safe')
        self.indices |= (image[:, :, 1].astype(np.uint32) << 8)
        self.indices |= image[:, :, 2]

        # 2. Map indices to LUT values into the pre-allocated mask_buffer
        np.take(self.lut, self.indices, out=self.mask_buffer)

        return self.mask_buffer

    def apply_lut(self, image: NP.ndarray) -> NP.ndarray:

        h, w, _ = image.shape
        flat = image.reshape(-1, 3)
        keys = (flat[:, 0].astype(NP.uint32) << 16) | (flat[:, 1].astype(NP.uint32) << 8) | flat[:, 2]
        # print(f"{keys.shape=}", GC())
        # mask_flat = lut[keys]
        mask_flat = self.lut.ravel()[keys]
        return mask_flat.reshape(image.shape[:2])

    def apply_lut_return_image(self, image: NP.ndarray) -> NP.ndarray:
        # H, W, _ = image.shape
        # flat = image.reshape(-1, 3)
        # keys = (flat[ :, 0 ].astype(NP.uint32) << 16) | (flat[ :, 1 ].astype(NP.uint32) << 8) | flat[ :, 2 ]
        #
        # # LUT returns RGB values per key
        # pixels_mapped = lut[ keys ]  # shape (H*W, 3)
        # return pixels_mapped.reshape(H, W, 3)

        H, W, _ = image.shape
        flat = image.reshape(-1, 3)
        new_img = NP.zeros_like(flat)
        for ndx, rgb_pixel in enumerate(flat):
            r, g, b = rgb_pixel
            lut_val = self.lut[r, g, b]
            if lut_val:
                # print(lut_val, r, g, b)
                new_img[ndx] = flat[ndx]

        return new_img.reshape(H, W, 3)

    def apply_lut_return_image_fast(self, image: NP.ndarray) -> NP.ndarray:
        r, g, b = image[..., 0], image[..., 1], image[..., 2]
        mask = self.lut[r, g, b] != 0

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


def cv2_named_windows(win_list: list, value=cv2.WINDOW_FREERATIO):
    for name in win_list:
        print(name)
        cv2.namedWindow(name, value)


def find_most_recent_file(fpattern: str = "*.pkl"):
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


def load_lut_obj(fname):
    with open(fname, "rb") as fin:
        obj = pickle.load(fin)
    return obj


def print_difflist(lst, scale: float = 1000):
    print([f"{(v1 - v0) * scale:.0f}" for v1, v0 in zip(lst[1:], lst)])


if __name__ == '__main__':

    win_list = ["maskedimage", "testimg", "frame", "mask"]
    # Load image
    cl = ColorLUT()

    tlist = [time.perf_counter()]
    if 0:
        latest_fname = find_most_recent_file("lut*.pkl")
        print(f"{latest_fname=}")
        obj = load_lut_obj(latest_fname)
        print(obj.keys())
        cl.lut = obj['lutobj']

    image_mode = 'l'
    half_mode = 0

    erode_mode = 0

    cap = cv2.VideoCapture(0)

    tlist.append(time.perf_counter())
    help()
    cv2_named_windows(win_list, cv2.WINDOW_FREERATIO)
    tlist.append(time.perf_counter())

    while True:
        tlist = [time.perf_counter()]
        ret, frame = cap.read()
        tlist.append(time.perf_counter())
        if image_mode in 'rl':
            h, w = frame.shape[:2]
            if image_mode == 'l':
                frame = frame[:, 0:int(w / 2)]
            else:
                frame = frame[:, int(w / 2):]

        cv2.imshow("frame", frame)
        tlist.append(time.perf_counter())
        if half_mode == 0:
            train_image = frame
        else:
            h, w = frame.shape[:2]
            scale_map = {1: 4, 2: 3, 3: 2.7, 4: 2.5, 5: 2.3}
            s = scale_map[half_mode]
            h4 = int(h / s)
            w4 = int(w / s)
            # print(w,w4, h,h4)
            train_image = frame[h4:h - h4, w4:w - w4]

        cv2.imshow("trainimage", cv2.resize(train_image, (0, 0), fx=2, fy=2))
        test_img = frame.copy()
        cv2.imshow("testimg", test_img)

        if 1:
            tlist.append(time.perf_counter())
            t = cl.apply_lut_return_image_fast(test_img)
            cv2.imshow("maskedimage", t)

        tlist.append(time.perf_counter())
        t = cl.apply_lut(test_img)
        cv2.imshow("mask", t)
        tlist.append(time.perf_counter())
        key = cv2.waitKey(1)

        if key >= 0:
            ch = chr(key)
            tlist2 = [time.perf_counter()]
            if ch == 'h':                help()

            if ch in 'alr':                image_mode = ch

            if ch in "0":                cl.zero_lut_from_image(frame)
            if ch in "1":                cl.build_lut_from_image(train_image)
            if ch in 'ed':
                erode_mode = ch
                print(f"{erode_mode=}")

            if ch in "23456789":
                print(f"{erode_mode=}")
                if erode_mode == 'd':
                    cl.grow_lut(int(ch) - 1)
                if erode_mode == 'e':
                    cl.erode_lut(int(ch) - 1)

            if ch == 'q':                half_mode = (half_mode + 1) % 6

            if ch == 'o':                cl.lut = NP.ones((256, 256, 256), dtype=NP.uint8)
            if ch == 'z':                cl.lut = NP.zeros((256, 256, 256), dtype=NP.uint8)

            if ch == 's' or key == 27:
                now = datetime.datetime.now()
                dt_string = now.strftime("%y%m%d_%H%M%S")
                fname = f"lut_{dt_string}.pkl"
                print(f"saving lut with {fname}")
                obj = dict(lutobj=cl.lut)
                with open(fname, "wb") as fout:
                    pickle.dump(obj, fout)
                if key == 27:
                    break
            tlist2.append(time.perf_counter())
            print("runtime2:", end="  ")
            print_difflist(tlist2)
        print_difflist(tlist)

