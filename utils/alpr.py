from openalpr import Alpr
import sys


def init_alpr():
    alpr = Alpr("eu", "/usr/share/openalpr/config/eu.conf", "openalpr/runtime_data/")
    if not alpr.is_loaded():
        print("Error loading OpenALPR")
        sys.exit(1)
    alpr.set_top_n(2)
    return alpr


def detect(alpr, thresh_img):
    #alpr = init_alpr()
    results = alpr.recognize_ndarray(thresh_img)

    if results['results'] != []:
        max_c = max(results['result'], key=lambda ev: ev['confidence'])
        #print(max_c['plate'])
        return max_c['plate']
