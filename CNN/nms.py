# import the necessary packages
import numpy as np

# Malisiewicz et al.
def __tutorial(boxes, overlap_thresh, min_overlap):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # cast as numpy if needed
    if isinstance(boxes, list):
        boxes = np.asarray(boxes)

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []
    strong_pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        ii = idxs[:last]

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[ii])
        yy1 = np.maximum(y1[i], y1[ii])
        xx2 = np.minimum(x2[i], x2[ii])
        yy2 = np.minimum(y2[i], y2[ii])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        deleted_idx = np.concatenate(([last], np.where(overlap > overlap_thresh)[0]))
        idxs = np.delete(idxs, deleted_idx)

        # for strong suppression, only pick regions with many neighbours
        if deleted_idx.shape[0] >= min_overlap:
            strong_pick.append(i)
        else:
            pick.append(i)

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int"), boxes[strong_pick].astype("int")


# Malisiewicz et al. with mean-suppression
def suppression(boxes, overlap_thresh, min_overlap):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return [], []

    # cast as numpy if needed
    if isinstance(boxes, list):
        boxes = np.asarray(boxes)

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i" or boxes.dtype.kind == "u":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []
    strong_pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes
    # sort according to the area, so that the bigger box eat the smaller ones
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(area)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        ii = idxs[:last]

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = x1[ii]
        yy1 = y1[ii]
        xx2 = x2[ii]
        yy2 = y2[ii]

        last_x1 = x1[i]
        last_y1 = y1[i]
        last_x2 = x2[i]
        last_y2 = y2[i]

        # compute the width and height of the bounding box
        w = np.maximum(0, 1 + np.minimum(last_x2, xx2) - np.maximum(last_x1, xx1))
        h = np.maximum(0, 1 + np.minimum(last_y2, yy2) - np.maximum(last_y1, yy1))

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # get the indexes that need to be deleted
        overlap_cond = np.where(overlap > overlap_thresh)[0]
        deleted_i = np.concatenate(([last], overlap_cond))

        # x1_cond = np.intersect1d(np.where(xx1 >= last_x1)[0], np.where(xx1 <= last_x2)[0])
        # x2_cond = np.intersect1d(np.where(xx2 >= last_x1)[0], np.where(xx2 <= last_x2)[0])
        # y1_cond = np.intersect1d(np.where(yy1 >= last_y1)[0], np.where(yy1 <= last_y2)[0])
        # y2_cond = np.intersect1d(np.where(yy2 >= last_y1)[0], np.where(yy2 <= last_y2)[0])
        # y_cond = np.union1d(y1_cond, y2_cond)
        # xy_cond1 = np.intersect1d(x1_cond, y_cond)
        # xy_cond2 = np.intersect1d(x2_cond, y_cond)
        # xy_cond = np.union1d(xy_cond1, xy_cond2)
        # cond = np.intersect1d(overlap_cond, xy_cond)

        # instead of picking up the base box of the suppressed boxes
        # we might want instead to pick up their means
        mean_box = np.mean(boxes[idxs[deleted_i]], axis=0)

        # for if we have many neighbours, then it's strong suppression
        # else, it's week one
        if deleted_i.shape[0] >= min_overlap:
            strong_pick.append(mean_box)
        else:
            pick.append(mean_box)

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, deleted_i)

    # return only the bounding boxes that were picked using the integer data type
    return np.asarray(pick, dtype=int), np.asarray(strong_pick, dtype=int)
