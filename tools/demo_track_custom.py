import argparse
import os
import os.path as osp
import time
import pickle
import cv2
import numpy as np
import torch
from collections import defaultdict
import torchvision.ops.boxes as bops

from loguru import logger

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer


IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


class Predictor(object):
    def __init__(self, device=torch.device("cpu"), fp16=False):
        self.test_size = (480, 640)
        self.device = device
        self.fp16 = fp16
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    @staticmethod
    def inference(img_path):
        img_info = {"id": 0, "file_name": osp.basename(img_path)}
        img = cv2.imread(img_path)
        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img
        return img_info


def find_best_iou_match(pred_boxes, tlwh):
    if isinstance(pred_boxes, np.ndarray):
        pred_boxes = torch.from_numpy(pred_boxes)  # (M, 4), XYXY
    pred_boxes = pred_boxes.cpu().detach()
    x1, y1, w, h = tlwh
    tracked_box = torch.tensor([[x1, y1, x1 + w, y1 + h]])  # (1, 4), XYXY
    iou = bops.box_iou(pred_boxes, tracked_box).squeeze().numpy()  # (M,)
    matched_id = np.argmax(iou)
    return matched_id, iou[matched_id]


def detic_demo(args):
    predictor = Predictor(args.device, args.fp16)
    # current_time = time.localtime()
    # timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)

    video_path = os.path.join(args.dataset, args.video)
    color_im_folder = os.path.join(video_path, 'color')
    detic_output_folder = os.path.join(video_path, 'detic_output', args.detic_exp, 'instances')
    color_im_files = sorted(os.listdir(color_im_folder))
    color_im_paths = [os.path.join(color_im_folder, im_file) for im_file in color_im_files]
    instance_paths = [os.path.join(detic_output_folder, im_file.replace('png', 'pkl')) for im_file in color_im_files]
    output_folder = os.path.join(video_path, 'detic_output', args.detic_exp, 'byte_output')

    timer = Timer()
    tracker = BYTETracker(args, frame_rate=args.fps)
    results = []

    cluster_id_to_instance = defaultdict(list)
    for frame_id, (color_im_path, instance_path) in enumerate(zip(color_im_paths, instance_paths), 1):
        img_info = predictor.inference(color_im_path)
        with open(instance_path, 'rb') as fp:
            instances = pickle.load(fp)
            pred_boxes = instances.pred_boxes.tensor  # (M, 4)
            pred_scores = instances.scores  # (M,)
            # pred_masks = instances.pred_masks.numpy()  # (M, H, W)
            # pred_classes = instances.pred_classes.numpy()  # (M,)
            output = torch.hstack([pred_boxes, pred_scores[:, None]])  # (M, 5)

        if output.shape[0] != 0:
            online_targets = tracker.update(output, [img_info['height'], img_info['width']], predictor.test_size)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for target in online_targets:
                tlwh = target.tlwh
                tid = target.track_id
                # vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                # if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_scores.append(target.score)
                # save results
                line = f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{target.score:.2f}\n"
                results.append(line)
                instance_id, iou = find_best_iou_match(pred_boxes, tlwh)
                if iou > 0.5:
                    cluster_id_to_instance[tid].append((instance_path, instance_id))
            timer.toc()
            online_im = plot_tracking(
                img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id, fps=1. / timer.average_time
            )
        else:
            timer.toc()
            online_im = img_info['raw_img']

        # result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        if args.save_result:
            save_folder = osp.join(output_folder, 'vis')
            os.makedirs(save_folder, exist_ok=True)
            cv2.imwrite(osp.join(save_folder, osp.basename(color_im_path.replace('png', 'jpg'))), online_im)

        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break

    if args.save_result:
        res_file = osp.join(output_folder, "cluster_id_to_instance.pkl")
        with open(res_file, 'wb') as fp:
            pickle.dump(cluster_id_to_instance, fp)
        # with open(res_file, 'w') as f:
        #     f.writelines(results)
        # logger.info(f"save results to {res_file}")


def main():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("--save_result", action="store_true")

    # exp file
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true")
    parser.add_argument("--fuse", dest="fuse", default=False, action="store_true", help="Fuse conv and bn for testing.")

    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.4, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=150, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.7, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")

    # dataset
    parser.add_argument("--dataset", type=str, default=os.path.expanduser("~/dataset/CoRL_real"))
    parser.add_argument("--video", type=str, default="0001")
    parser.add_argument("--detic_exp", type=str, default="icra23-0.3")

    args = parser.parse_args()
    logger.info("Args: {}".format(args))

    detic_demo(args)


if __name__ == "__main__":
    main()
