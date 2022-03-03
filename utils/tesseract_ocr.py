import pytesseract
import cv2


def extract_contour(img_dilation):
    try:
        contours, hierarchy = cv2.findContours(
            img_dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except:
        ret_img, contours, hierarchy = cv2.findContours(
            img_dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    sorted_contours = sorted(
        contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    return sorted_contours


def extract_text(img, img_dilation, img_thresh):
    img2 = img.copy()
    plate_num = ''

    sorted_contours = extract_contour(img_dilation)

    for cnt in sorted_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        height, width = img2.shape

        if height / float(h) > 6:
            continue
        ratio = h / float(w)

        if ratio < 1.5:
            continue
        area = h * w

        if width / float(w) > 15:
            continue

        if area < 100:
            continue

        roi = img_thresh[y-1:y+h+1, x:x+w]
        roi = cv2.bitwise_not(roi)
        roi = cv2.medianBlur(roi, 3)

        text = pytesseract.image_to_string(
            roi, config='--psm 8 --oem 1 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        plate_num += text

    return plate_num
