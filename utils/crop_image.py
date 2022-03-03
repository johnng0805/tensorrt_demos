import cv2
import os


def crop_image(img, boxes, idx, mBlur=3, gBlur=(5, 5)):
    xmin, ymin, xmax, ymax = boxes[idx]

    cropped = img[int(ymin):int(ymax), int(xmin):int(xmax)]

    resized_img = cv2.resize(cropped, None, fx=2, fy=2,
                             interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(resized_img, cv2.COLOR_RGB2GRAY)

    medBlur = cv2.medianBlur(gray, mBlur)

    gaussBlur = cv2.GaussianBlur(medBlur, gBlur, 0)

    ret, thresh = cv2.threshold(
        gaussBlur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilation = cv2.dilate(thresh, rect_kern, iterations=1)

    return gaussBlur, dilation, thresh
