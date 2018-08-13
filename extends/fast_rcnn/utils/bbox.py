from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange
import numpy as np

def bbox_overlaps(bboxes, query_boxes):
    '''
        bboxes: (N, 4) 
        query_boxes: (N, 4)
    '''

    N = bboxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K))
    for k in range(K):
        box_area=(query_boxes[k, 2] - query_boxes[k, 0]+1) * (query_boxes[k, 3] - query_boxes[k, 1]+1)
        for n in range(N):
            iw = min(bboxes[n, 2], query_boxes[k, 2]) - max(bboxes[n, 0], query_boxes[k, 0]) + 1
            if iw > 0:
                ih = min(bboxes[n, 3], query_boxes[k, 3]) - max(bboxes[n, 1], query_boxes[k,1])+1
                if ih >0:
                    ua = (bboxes[n, 2] - bboxes[n, 0]+1)*(bboxes[n, 3]-bboxes[n, 1]+1)+box_area - iw * ih
                    overlaps[n,k] = iw * ih / ua
    return overlaps