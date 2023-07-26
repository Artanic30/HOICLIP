import os
import pickle
import collections
import json
import numpy as np
from .swig_v1_categories import SWIG_INTERACTIONS


class SWiGEvaluator(object):
    ''' Evaluator for SWIG-HOI dataset '''
    def __init__(self, anno_file, output_dir):
        eval_hois = [x["id"] for x in SWIG_INTERACTIONS if x["evaluation"] == 1]
        size = max(eval_hois) + 1
        self.eval_hois = eval_hois

        self.gts = self.load_anno(anno_file)
        self.scores = {i: [] for i in range(size)}
        self.boxes = {i: [] for i in range(size)}
        self.keys = {i: [] for i in range(size)}
        self.swig_ap  = np.zeros(size)
        self.swig_rec = np.zeros(size)
        self.output_dir = output_dir

    def update(self, predictions):
        # update predictions
        for img_id, preds in predictions.items():
            for pred in preds:
                hoi_id = pred[0]
                score = pred[1]
                boxes = pred[2:]
                self.scores[hoi_id].append(score)
                self.boxes[hoi_id].append(boxes)
                self.keys[hoi_id].append(img_id)

    def accumulate(self):
        for hoi_id in self.eval_hois:
            gts_per_hoi = self.gts[hoi_id]
            ap, rec = calc_ap(self.scores[hoi_id], self.boxes[hoi_id], self.keys[hoi_id], gts_per_hoi)
            self.swig_ap[hoi_id], self.swig_rec[hoi_id] = ap, rec

    def summarize(self):
        eval_hois = np.asarray([x["id"] for x in SWIG_INTERACTIONS if x["evaluation"] == 1])
        zero_hois = np.asarray([x["id"] for x in SWIG_INTERACTIONS if x["frequency"] == 0 and x["evaluation"] == 1])
        rare_hois = np.asarray([x["id"] for x in SWIG_INTERACTIONS if x["frequency"] == 1 and x["evaluation"] == 1])
        nonrare_hois = np.asarray([x["id"] for x in SWIG_INTERACTIONS if x["frequency"] == 2 and x["evaluation"] == 1])

        full_mAP = np.mean(self.swig_ap[eval_hois])
        zero_mAP = np.mean(self.swig_ap[zero_hois])
        rare_mAP = np.mean(self.swig_ap[rare_hois])
        nonrare_mAP = np.mean(self.swig_ap[nonrare_hois])
        print("zero-shot mAP: {:.2f}".format(zero_mAP * 100.))
        print("rare mAP: {:.2f}".format(rare_mAP * 100.))
        print("nonrare mAP: {:.2f}".format(nonrare_mAP * 100.))
        print("full mAP: {:.2f}".format(full_mAP * 100.))

    def save_preds(self):
        with open(os.path.join(self.output_dir, "preds.pkl"), "wb") as f:
            pickle.dump({"scores": self.scores, "boxes": self.boxes, "keys": self.keys}, f)

    def save(self, output_dir=None):
        if output_dir is None:
            output_dir = self.output_dir
        with open(os.path.join(output_dir, "dets.pkl"), "wb") as f:
            pickle.dump({"gts": self.gts, "scores": self.scores, "boxes": self.boxes, "keys": self.keys}, f)

    def load_anno(self, anno_file):
        with open(anno_file, "r") as f:
            dataset_dicts = json.load(f)

        hoi_mapper = {(x["action_id"], x["object_id"]): x["id"] for x in SWIG_INTERACTIONS}

        size = max(self.eval_hois) + 1
        gts = {i: collections.defaultdict(list) for i in range(size)}
        for anno_dict in dataset_dicts:
            image_id = anno_dict["img_id"]
            box_annos = anno_dict.get("box_annotations", [])
            hoi_annos = anno_dict.get("hoi_annotations", [])
            for hoi in hoi_annos:
                person_box = box_annos[hoi["subject_id"]]["bbox"]
                object_box = box_annos[hoi["object_id"]]["bbox"]
                action_id = hoi["action_id"]
                object_id = box_annos[hoi["object_id"]]["category_id"]
                hoi_id = hoi_mapper[(action_id, object_id)]
                gts[hoi_id][image_id].append(person_box + object_box)

        for hoi_id in gts:
            for img_id in gts[hoi_id]:
                gts[hoi_id][img_id] = np.array(gts[hoi_id][img_id])

        return gts


def calc_ap(scores, boxes, keys, gt_boxes):
    if len(keys) == 0:
        return 0, 0

    if isinstance(boxes, list):
        scores, boxes, key = np.array(scores), np.array(boxes), np.array(keys)

    hit = []
    idx = np.argsort(scores)[::-1]
    npos = 0
    used = {}

    for key in gt_boxes.keys():
        npos += gt_boxes[key].shape[0]
        used[key] = set()

    for i in range(min(len(idx), 19999)):
        pair_id = idx[i]
        box = boxes[pair_id, :]
        key = keys[pair_id]
        if key in gt_boxes:
            maxi = 0.0
            k    = -1
            for i in range(gt_boxes[key].shape[0]):
                tmp = calc_hit(box, gt_boxes[key][i, :])
                if maxi < tmp:
                    maxi = tmp
                    k    = i
            if k in used[key] or maxi < 0.5:
                hit.append(0)
            else:
                hit.append(1)
                used[key].add(k)
        else:
            hit.append(0)
    bottom = np.array(range(len(hit))) + 1
    hit    = np.cumsum(hit)
    rec    = hit / npos
    prec   = hit / bottom
    ap     = 0.0
    for i in range(11):
        mask = rec >= (i / 10.0)
        if np.sum(mask) > 0:
            ap += np.max(prec[mask]) / 11.0

    return ap, np.max(rec)


def calc_hit(det, gtbox):
    gtbox = gtbox.astype(np.float64)
    hiou = iou(det[:4], gtbox[:4])
    oiou = iou(det[4:], gtbox[4:])
    return min(hiou, oiou)


def iou(bb1, bb2, debug = False):
    x1 = bb1[2] - bb1[0]
    y1 = bb1[3] - bb1[1]
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0

    x2 = bb2[2] - bb2[0]
    y2 = bb2[3] - bb2[1]
    if x2 < 0:
        x2 = 0
    if y2 < 0:
        y2 = 0

    xiou = min(bb1[2], bb2[2]) - max(bb1[0], bb2[0])
    yiou = min(bb1[3], bb2[3]) - max(bb1[1], bb2[1])
    if xiou < 0:
        xiou = 0
    if yiou < 0:
        yiou = 0

    if debug:
        print(x1, y1, x2, y2, xiou, yiou)
        print(x1 * y1, x2 * y2, xiou * yiou)
    if xiou * yiou <= 0:
        return 0
    else:
        return xiou * yiou / (x1 * y1 + x2 * y2 - xiou * yiou)


''' deprecated, evaluator
eval_hois = [x["id"] for x in SWIG_INTERACTIONS if x["evaluation"] == 1]
def swig_evaluation(predictions, gts):
    images, results = [], []
    for img_key, ps in predictions.items():
        images.extend([img_key] * len(ps))
        results.extend(ps)

    size = max(eval_hois) + 1
    swig_ap, swig_rec = np.zeros(size), np.zeros(size)

    scores = [[] for _ in range(size)]
    boxes = [[] for _ in range(size)]
    keys = [[] for _ in range(size)]

    for img_id, det in zip(images, results):
        hoi_id, person_box, object_box, score = int(det[0]), det[1], det[2], det[-1]
        scores[hoi_id].append(score)
        boxes[hoi_id].append([float(x) for x in person_box] + [float(x) for x in object_box])
        keys[hoi_id].append(img_id)

    for hoi_id in eval_hois:
        gts_per_hoi = gts[hoi_id]
        ap, rec = calc_ap(scores[hoi_id], boxes[hoi_id], keys[hoi_id], gts_per_hoi)
        swig_ap[hoi_id], swig_rec[hoi_id] = ap, rec

    return swig_ap, swig_rec


def prepare_swig_gts(anno_file):
    """
    Convert dataset to the format required by evaluator.
    """
    with open(anno_file, "r") as f:
        dataset_dicts = json.load(f)

    filename_to_id_mapper = {x["file_name"]: i for i, x in enumerate(dataset_dicts)}
    hoi_mapper = {(x["action_id"], x["object_id"]): x["id"] for x in SWIG_INTERACTIONS}

    size = max(eval_hois) + 1
    gts = {i: collections.defaultdict(list) for i in range(size)}
    for anno_dict in dataset_dicts:
        image_id = filename_to_id_mapper[anno_dict["file_name"]]
        box_annos = anno_dict.get("box_annotations", [])
        hoi_annos = anno_dict.get("hoi_annotations", [])
        for hoi in hoi_annos:
            person_box = box_annos[hoi["subject_id"]]["bbox"]
            object_box = box_annos[hoi["object_id"]]["bbox"]
            action_id = hoi["action_id"]
            object_id = box_annos[hoi["object_id"]]["category_id"]
            hoi_id = hoi_mapper[(action_id, object_id)]
            gts[hoi_id][image_id].append(person_box + object_box)

    for hoi_id in gts:
        for img_id in gts[hoi_id]:
            gts[hoi_id][img_id] = np.array(gts[hoi_id][img_id])

    return gts, filename_to_id_mapper
'''