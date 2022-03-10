import cv2
from sklearn import datasets
from sklearn import svm


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


def init_svm():
    digits = datasets.load_digits()
    clf = svm.SVC()

    X = digits.data[:-10]
    y = digits.target[:-10]

    clf.fit(X, y)
    return clf, digits


def extract_text(img, img_dilation, img_thresh):
    img2 = img.copy()
    clf, digits = init_svm()
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
        print(clf.predict(digits.data[-5].reshape(1, -1)))

    return plate_num
