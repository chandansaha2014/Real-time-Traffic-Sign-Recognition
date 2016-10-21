import numpy
import numpy as np
import matplotlib.pyplot as plt

import cv2

import CNN
import CNN.utils
import CNN.nms
import CNN.enums

import skimage
import skimage.io
import skimage.exposure
from skimage import data, color
from skimage.transform import hough_circle
from skimage.feature import peak_local_max, canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte


def detection_proposal(img_color, min_dim, max_dim, superclass_type=CNN.enums.SuperclassType._01_Prohibitory):
    if superclass_type == CNN.enums.SuperclassType._01_Prohibitory or superclass_type == CNN.enums.SuperclassType._03_Mandatory:
        return __detection_proposal_circles(img_color, min_dim, max_dim)
    elif superclass_type == CNN.enums.SuperclassType._02_Warning:
        return __detection_proposal_triangles(img_color, min_dim, max_dim)
    else:
        raise Exception("Sorry, non-supported superclass type to find detecttion proposal to.")


def detection_proposal_and_save(img_path, min_dim=40, max_dim=160, superclass_type=CNN.enums.SuperclassType._01_Prohibitory):
    # extract detection proposals for circle-based traffic signs
    # suppress the extracted circles to weak and strong regions
    # draw the regions on the image and save it

    # load picture and detect edges
    img_color = cv2.imread(img_path)
    regions_weak, regions_strong, img_map, circles = detection_proposal(img_color, min_dim, max_dim, superclass_type)

    for c in circles:
        # draw the outer circle
        cv2.circle(img_color, (c[0], c[1]), c[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(img_color, (c[0], c[1]), 2, (0, 0, 255), 3)

    # draw and save the result, for testing purposes
    red_color = (0, 0, 255)
    yellow_color = (84, 212, 255)
    for loc in regions_weak:
        cv2.rectangle(img_color, (loc[0], loc[1]), (loc[2], loc[3]), yellow_color, 1)
    for loc in regions_strong:
        cv2.rectangle(img_color, (loc[0], loc[1]), (loc[2], loc[3]), red_color, 2)

    # save the result
    cv2.imwrite("D://_Dataset//GTSDB//Test_Regions//_img2.png", img_color)


def __detection_proposal_circles(img_color, min_dim, max_dim):
    """
    This algorithm depends on opencv, which is much much better than this of skiimage
    :param img_color:
    :param min_dim:
    :param max_dim:
    :return:
    """

    img_filtered = cv2.pyrMeanShiftFiltering(img_color, 10, 10)
    img_filtered = cv2.cvtColor(img_filtered, cv2.COLOR_BGRA2GRAY)
    img_filtered = img_filtered.astype(float)
    img_blurred = cv2.GaussianBlur(img_filtered, (7, 7), sigmaX=0)
    img_laplacian = cv2.Laplacian(img_blurred, ddepth=cv2.CV_64F)

    weight = 0.01 * 40
    scale = 0.01 * 20
    img_sharpened = (1.5 * img_filtered) - (0.5 * img_blurred) - (weight * cv2.multiply(img_filtered, scale * img_laplacian))
    img_sharpened = img_sharpened.astype("uint8")

    dims = []
    if (max_dim - min_dim) > 10:
        dims = numpy.arange(max_dim, min_dim, -10, dtype=int)
        if dims[dims.shape[0] - 1] > min_dim:
            dims = numpy.append(dims, min_dim)
    else:
        dims = [max_dim, min_dim]
    # the problem with HoughCircles is it gives many false positives if the range between
    # min and max increase more than 10 pixels specially in the bigger windows size (>80)
    # so, if the range is detected to be bigger than 10 pixels, then do detection on several steps

    # loop on the dimensions, max radius is the current dim/2, while min radius = next dim/2
    circles = []
    for i in range(0, len(dims) - 1):
        max_r = int(dims[i] / 2)
        min_r = int(dims[i + 1] / 2)
        current_circles = cv2.HoughCircles(img_sharpened, cv2.HOUGH_GRADIENT, 1, min_r, param1=50, param2=30, minRadius=min_r, maxRadius=max_r)
        if current_circles is not None:
            circles.append(current_circles[0])

    # if no circles, then return
    if len(circles) == 0:
        return [], [], [], circles

    # convert circles to regions
    circles = numpy.vstack(circles)
    circles = (np.around(circles)).astype(int)
    center_x = circles[:, 0]
    center_y = circles[:, 1]
    radius = circles[:, 2]
    regions = numpy.asarray([center_x - radius, center_y - radius, center_x + radius, center_y + radius])
    regions = numpy.transpose(regions)

    # suppress the regions to extract the strongest ones
    min_overlap = 0
    overlap_thresh = 0.75
    regions_weak, regions_strong = CNN.nms.suppression(boxes=regions, overlap_thresh=overlap_thresh, min_overlap=min_overlap)

    # create binary map using only the strong regions
    img_shape = img_color.shape
    img_map = np.zeros(shape=(img_shape[0], img_shape[1]), dtype=bool)
    for r in regions_strong:
        img_map[r[1]:r[3], r[0]:r[2]] = True
    return regions_weak, regions_strong, img_map, circles


def __detection_proposal_triangles(img_color, min_dim, max_dim):
    # pre-process the image for better detection
    img_filtered = cv2.pyrMeanShiftFiltering(img_color, 10, 10)
    img_filtered = cv2.cvtColor(img_filtered, cv2.COLOR_BGRA2GRAY)
    img_filtered = img_filtered.astype(float)
    img_blurred = cv2.GaussianBlur(img_filtered, (7, 7), sigmaX=0)

    # img_laplacian = cv2.Laplacian(img_blurred, ddepth=cv2.CV_64F)
    # img_laplacian_thresh = cv2.bitwise_not(img_laplacian.astype("uint8"))
    # img_thresh = cv2.adaptiveThreshold(img_blurred.astype("uint8"), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # #image, contours, hierarchy = cv2.findContours(img_edge.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    import math

    img_edge = cv2.Canny(img_blurred.astype("uint8"), 10, 200, apertureSize=3)
    lines = []
    for angle in [1]:
        min_theta = math.radians(angle - 10)
        max_theta = math.radians(angle + 10)
        theta = math.radians(angle)
        # new_lines = cv2.HoughLines(img_edge, 2, theta, 100)
        # new_lines = cv2.HoughLines(img_edge, 1, theta, 10, min_theta=min_theta, max_theta=max_theta)
        # new_lines = cv2.HoughLines(img_edge, 1, numpy.pi / 180, 10)
        new_lines = cv2.HoughLinesP(img_edge, 1, np.pi / 180, 50, minLineLength=40, maxLineGap=10)
        if new_lines is not None:
            new_lines = new_lines.reshape(new_lines.shape[0], new_lines.shape[2])
            lines.append(new_lines)

    img_map = img_color.copy()
    if len(lines) > 0:
        lines = numpy.vstack(lines)
    print("line counr %d " % (len(lines)))

    # for probabilistic Hough
    for x1, y1, x2, y2 in lines:
        cv2.line(img_map, (x1, y1), (x2, y2), (0, 255, 0), 1)

    # # # for typical Hough
    # for rho, theta in lines[0:50]:
    #     # print("rho %f, theta %f" % (rho, math.degrees(theta)))
    #     a = np.cos(theta)
    #     b = np.sin(theta)
    #     x0 = a * rho
    #     y0 = b * rho
    #     x1 = int(x0 + 1000 * (-b))
    #     y1 = int(y0 + 1000 * (a))
    #     x2 = int(x0 - 1000 * (-b))
    #     y2 = int(y0 - 1000 * (a))
    #     cv2.line(img_map, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # color_red = (0, 255, 0)
    # for contour in contours:
    #     approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
    #     cv2.drawContours(img_map, [contour], 0, color_red, 1)
    # # if len(approx) == 3:

    cv2.imwrite("D://_Dataset//GTSDB//Test_Regions//_img_01.png", img_map)
    cv2.imwrite("D://_Dataset//GTSDB//Test_Regions//_img_02.png", img_edge)
    return

    regions_weak = []
    regions_strong = []
    triangles = []

    return regions_weak, regions_strong, img_map, triangles


def __proposal_old(img_preprocessed, min_dim, max_dim):
    """
    This algorithm depends on Hough circle detection using skii-image
    Which turned out to be crap
    :param img_preprocessed:
    :param min_dim:
    :param max_dim:
    :return:
    """

    img = img_preprocessed
    img_edge = canny(img * 255, sigma=3, low_threshold=10, high_threshold=50)

    centers = []
    accums = []
    radii = []
    regions = []

    # detect all radii within the range
    min_radius = int(min_dim / 2)
    max_radius = int(max_dim / 2)
    hough_radii = np.arange(start=min_radius, stop=max_radius, step=1)
    hough_res = hough_circle(img_edge, hough_radii)

    for radius, h in zip(hough_radii, hough_res):
        # for each radius, extract 2 circles
        num_peaks = 2
        peaks = peak_local_max(h, num_peaks=num_peaks)
        centers.extend(peaks)
        accums.extend(h[peaks[:, 0], peaks[:, 1]])
        radii.extend([radius] * num_peaks)

    # don't consider circles with accum value less than the threshold
    idx_sorted = np.argsort(accums)[::-1]
    accum_threshold = 0.3
    for idx in idx_sorted:
        idx = int(idx)
        if accums[idx] < accum_threshold:
            continue
        center_y, center_x = centers[idx]
        radius = radii[idx]
        x1 = center_x - radius
        y1 = center_y - radius
        x2 = center_x + radius
        y2 = center_y + radius

        # we don't want to collect circles, but we want to collect regions
        # later on, we'll suppress these regions
        regions.append([x1, y1, x2, y2])

    # suppress the regions to extract the strongest ones
    if len(regions) > 0:
        min_overlap = 3
        overlap_thresh = 0.7
        regions_weak, regions_strong = CNN.nms.suppression(boxes=regions, overlap_thresh=overlap_thresh, min_overlap=min_overlap)

        # create binary map using only the strong regions
        img_shape = img_preprocessed.shape
        map = np.zeros(shape=(img_shape[0], img_shape[1]), dtype=bool)
        for r in regions_strong:
            map[r[1]:r[3], r[0]:r[2]] = True
    else:
        regions_weak = []
        regions_strong = []
        map = []

    return regions_weak, regions_strong, map


def __tutorial_hough_circle_detection_skiimage(img_path, min_dim=40, max_dim=60):
    """
    This algorithm is crap, the one using opencv is much much better
    :param img_path:
    :param min_dim:
    :param max_dim:
    :return:
    """

    # load picture and detect edges
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    img_edges = canny(img, sigma=3, low_threshold=10, high_threshold=50)

    centers = []
    accums = []
    radii = []

    # detect all radii within the range
    min_radius = int(min_dim / 2)
    max_radius = int(max_dim / 2)
    hough_radii = np.arange(start=min_radius, stop=max_radius, step=1)
    hough_res = hough_circle(img_edges, hough_radii)

    for radius, h in zip(hough_radii, hough_res):
        # for each radius, extract 2 circles
        num_peaks = 2
        peaks = peak_local_max(h, num_peaks=num_peaks)
        centers.extend(peaks)
        accums.extend(h[peaks[:, 0], peaks[:, 1]])
        radii.extend([radius] * num_peaks)

    img_width = img.shape[1]
    img_height = img.shape[0]

    # get the sorted accumulated values
    # accums_sorted = np.asarray(accums)
    # accums_sorted = accums_sorted[idx_sorted]

    # don't consider circles with accum value less than the threshold
    accum_threshold = 0.3
    idx_sorted = np.argsort(accums)[::-1]

    # draw the most prominent n circles, i.e those
    # with the highest n peaks
    img_color = color.gray2rgb(img)
    for idx in idx_sorted:
        if accums[idx] < accum_threshold:
            continue
        center_y, center_x = centers[idx]
        radius = radii[idx]
        cx, cy = circle_perimeter(center_x, center_y, radius)
        cx[cx > img_width - 1] = img_width - 1
        cy[cy > img_height - 1] = img_height - 1
        img_color[cy, cx] = (220, 20, 20)

    # save the result
    skimage.io.imsave("D://_Dataset//GTSDB//Test_Regions//_img2_.png", img_color)


def __tutorial_hough_circle_detection_cv(img_path, min_dim=40, max_dim=60):
    img_color = cv2.imread(img_path)
    img_filtered = cv2.pyrMeanShiftFiltering(img_color, 10, 10)
    img_filtered = cv2.cvtColor(img_filtered, cv2.COLOR_BGRA2GRAY)
    img_filtered = img_filtered.astype(float)
    img_blurred = cv2.GaussianBlur(img_filtered, (7, 7), sigmaX=0)
    img_laplacian = cv2.Laplacian(img_blurred, ddepth=cv2.CV_64F)

    weight = 0.01 * 40
    scale = 0.01 * 20
    img_sharpened = (1.5 * img_filtered) - (0.5 * img_blurred) - (weight * cv2.multiply(img_filtered, scale * img_laplacian))
    img_sharpened = img_sharpened.astype("uint8")

    min_r = int(min_dim / 2)
    max_r = int(max_dim / 2)
    circles = cv2.HoughCircles(img_sharpened, cv2.HOUGH_GRADIENT, 1, 1, param1=50, param2=30, minRadius=min_r, maxRadius=max_r)

    if circles is not None:
        circles = np.around(circles).astype(int)
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(img_color, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(img_color, (i[0], i[1]), 2, (0, 0, 255), 3)

    cv2.imwrite("D://_Dataset//GTSDB//Test_Regions//_img2_1.png", img_sharpened)
    cv2.imwrite("D://_Dataset//GTSDB//Test_Regions//_img2_2.png", img_color)


def __tutorial_hough_circle_detection_cv_old(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    frame_gray = cv2.GaussianBlur(img, (5, 5), 2)

    edges = frame_gray - cv2.erode(frame_gray, None)
    _, bin_edge = cv2.threshold(edges, 0, 255, cv2.THRESH_OTSU)
    height, width = bin_edge.shape
    mask = numpy.zeros((height + 2, width + 2), dtype=numpy.uint8)
    cv2.floodFill(bin_edge, mask, (0, 0), 255)

    components = segment_on_dt(bin_edge)

    circles, obj_center = [], []
    contours, _ = cv2.findContours(components,
                                   cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        c = c.astype(numpy.int64)  # XXX OpenCV bug.
        area = cv2.contourArea(c)
        if 100 < area < 3000:
            arclen = cv2.arcLength(c, True)
            circularity = (pi_4 * area) / (arclen * arclen)
            if circularity > 0.5:  # XXX Yes, pretty low threshold.
                circles.append(c)
                box = cv2.boundingRect(c)
                obj_center.append((box[0] + (box[2] / 2), box[1] + (box[3] / 2)))

    return circles, obj_center
