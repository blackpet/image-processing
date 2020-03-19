import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract

def find_chars(contour_list):
    matched_result_idx = []

    for d1 in contour_list:
        matched_contours_idx = []

        for d2 in contour_list:

            # 같은 Contour은 검사할 필요가 없다!
            if d1['idx'] == d2['idx']:
                continue

            dx = abs(d1['cx'] - d2['cx'])
            dy = abs(d1['cy'] - d2['cy'])

            diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)

            # 대각선 거리는?
            distance = np.linalg.norm(
                np.array([d1['cx'], d1['cy']])
                - np.array([d2['cx'], d2['cy']]))

            if dx == 0:
                angle_diff = 90
            else:
                angle_diff = np.degrees(np.arctan(dy / dx))

            area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h'])
            width_diff = abs(d1['w'] - d2['w']) / d1['w']
            height_diff = abs(d1['h'] - d2['h']) / d1['h']


            if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER \
            and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF \
            and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
                matched_contours_idx.append(d2['idx'])

        # append this contour
        matched_contours_idx.append(d1['idx'])

        if len(matched_contours_idx) < MIN_N_MATCHED:
            continue

        matched_result_idx.append(matched_contours_idx)

        unmatched_contour_idx = []
        for d4 in contour_list:
            if d4['idx'] not in matched_contours_idx:
                unmatched_contour_idx.append(d4['idx'])

        unmatched_contour = np.take(possible_contours, unmatched_contour_idx)

        # recursive
        recursive_contour_list = find_chars(unmatched_contour)

        for idx in recursive_contour_list:
            matched_result_idx.append(idx)

        break

    return matched_result_idx


# plt.style.use('dark_background')

# img_origin = cv2.imread('bowling-sj3.jpg')
img_origin = cv2.imread('car-number.jpg')
img_gray = cv2.cvtColor(img_origin, cv2.COLOR_BGR2GRAY)

height, width, channel = img_origin.shape
print(height, width, channel)

# plt.figure(figsize=(12,10))
# plt.imshow(img_origin, cmap='gray')

### Maximize Contrast (Optional)
# structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# imgTopHat = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, structuringElement)
# imgBlackHat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, structuringElement)
#
# imgGrayscalePlusTopHat = cv2.add(img_gray, imgTopHat)
# img_gray2 = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)



### Adaptive Thresholding
img_blurred = cv2.GaussianBlur(img_gray, ksize=(3, 3), sigmaX=0)
img_thresh = cv2.adaptiveThreshold(
    img_blurred,
    maxValue=255.0,
    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    thresholdType=cv2.THRESH_BINARY_INV,
    blockSize=19,
    C=9
)


### Find Contours
contours, _ = cv2.findContours(
    img_thresh,
    mode=cv2.RETR_LIST,
    method=cv2.CHAIN_APPROX_SIMPLE
)

temp_result = np.zeros((height, width, channel), dtype=np.uint8)

# cv2.drawContours(temp_result, contours=contours, contourIdx=-1, color=(255, 255, 255))


### Prepare Data
contours_dict = []

for ct in contours:
    x, y, w, h = cv2.boundingRect(ct)
    cv2.rectangle(temp_result, pt1=(x, y), pt2=(x+w, y+h), color=(255, 255, 255), thickness=2)

    #insert to dict
    contours_dict.append({
        'contour': ct,
        'x': x,
        'y': y,
        'w': w,
        'h': h,
        'cx': x + (w / 2),
        'cy': y + (h / 2)
    })


### Select Candidates by Char Size
MIN_AREA = 80
MIN_WIDTH, MIN_HEIGHT = 10, 30
MAX_WIDTH, MAX_HEIGHT = 30, 90
MIN_RATIO, MAX_RATIO = 0.25, 1.0

possible_contours = []

cnt = 0
for d in contours_dict:
    area = d['w'] * d['h']
    ratio = d['w'] / d['h']

    if area > MIN_AREA \
        and MIN_WIDTH < d['w'] < MAX_WIDTH and MIN_HEIGHT < d['h'] < MAX_HEIGHT \
        and MIN_RATIO < ratio < MAX_RATIO:
        print(d)
        d['idx'] = cnt
        ++cnt
        possible_contours.append(d)

# visualize possible contours
temp_result = np.zeros((height, width, channel), dtype=np.uint8)

for d in possible_contours:
    cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)

cv2.imshow('Contours', temp_result)


### Select Candidates by Arrangement of Contours
MAX_DIAG_MULTIPLYER = 5
MAX_ANGLE_DIFF = 12.0
MAX_AREA_DIFF = 0.5
MAX_WIDTH_DIFF = 0.8
MAX_HEIGHT_DIFF = 0.2
MIN_N_MATCHED = 3

result_idx = find_chars(possible_contours)

matched_result = []
for idx_list in result_idx:
    matched_result.append(np.take(possible_contours, idx_list))

#visualize possible contours
temp_result = np.zeros((height, width, channel), dtype=np.uint8)

for r in matched_result:
    for d in r:
        cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)




cv2.imshow('Car Number', img_origin)
# cv2.imshow('Gray', img_gray)
# cv2.imshow('Gray + TopHat + BlackHat', img_gray2)
cv2.imshow('Blurred', img_blurred)
cv2.imshow('Threshold', img_thresh)
cv2.imshow('Contours2', temp_result)

cv2.waitKey(0)









