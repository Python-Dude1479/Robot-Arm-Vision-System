import cv2
import numpy as np
import LUT_Lib
import time


def check_circularity(area, perimeter):
    return np.pi * 4 * area / perimeter ** 2

def in_range(x, range):
    if x > range[0] and x < range[1]:
        return True
    return False

if __name__ == '__main__':

    prev_time = time.time()

    cl = LUT_Lib.ColorLUT()

    obj = LUT_Lib.load_lut_obj("lut_260405_192711.pkl")
    print(obj.keys())
    cl.lut = obj['lutobj']

    cap = cv2.VideoCapture(0)
    cv2.namedWindow("contours", cv2.WINDOW_FREERATIO)

    kernel = np.ones((5, 5), np.uint8)

    windows = ["t filled","maskedimage","contours"]

    LUT_Lib.cv2_named_windows(win_list=windows,value=cv2.WINDOW_FREERATIO)

    while True:
        ret, frame = cap.read()

        # ret2,all_c_frame = cap.read()

        # LUT

        t = cl.apply_lut_return_image_fast(frame, cl.lut)

        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        # conturs

        # preprocessing of the returned mask
        t_filled = cv2.morphologyEx(t, cv2.MORPH_CLOSE, kernel)

        gframe = cv2.cvtColor(t_filled, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(gframe, 127, 255, 0)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # connected = cv2.connectedComponentsWithStats(gframe)
        # nl, labels, stats, centroids = connected
        # print(connected )

        if 0:
            # print("stats:", nl, stats.shape)
            for s in stats:
                x, y, w, h, area = s
                print(x, y, w, h, area)
            if 0:
                stats = connected[2]
                for stat in stats[1:]:
                    area = stat[4]

        # print("contours:" + str(contours))

        filtered_contours = []

        for cnt in contours:
            # print("cnt:" + str(cnt))
            area = cv2.contourArea(cnt)
            p = cv2.arcLength(cnt, True)
            # print("area:" + str(area), "p:" + str(p))

            if p == 0: continue

            circularity = check_circularity(area, p)
            # print(circularity)

            if area > 45:
                if in_range(circularity, [.6, 1]):
                    filtered_contours.append(cnt)

        # cv2.drawContours(all_c_frame, contours, -1, (0,255,0), 1)

        cv2.drawContours(frame, filtered_contours, -1, (0, 255, 0), 3)

        cv2.imshow("contours", frame)
        cv2.imshow("thresh", thresh)
        cv2.imshow("gray", gframe)
        cv2.imshow("maskedimage", t)
        cv2.imshow("t filled", t_filled)
        # cv2.imshow("all contours", all_c_frame)

        print(f"FPS: {fps:.2f}")

        cv2.waitKey(1)
