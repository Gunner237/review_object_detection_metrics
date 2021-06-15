# Created by Vijay Rajagopal
import os
from src.evaluators.tube_evaluator import TubeEvaluator
import sys
import argparse
import logging
from src.utils import converter
from src.utils.enumerators import (BBFormat, BBType, CoordinatesType,
                                   MethodAveragePrecision)
import src.evaluators.coco_evaluator as coco_evaluator
import src.evaluators.pascal_voc_evaluator as pascal_voc_evaluator
import src.utils.converter as converter
import src.utils.general_utils as general_utils
from src.bounding_box import BoundingBox
import matplotlib.pyplot as plt

def parseArgs():
    parser = argparse.ArgumentParser()
    # Folder path to annotations
    parser.add_argument('--anno_gt', type=str)
    parser.add_argument('--anno_det', type=str)

    # Folder path to corresponding images:
    parser.add_argument('--img', type=str, required=False)
    #parser.add_argument('--img_gt', type=str, required=False)
    #parser.add_argument('--img_det', type=str, required=False)

    # format_gt: as in the application screenshot:
    parser.add_argument('--format_gt', type=str)

    # format_det: shares input from coord (either xyrb, xywh, or coco)
    parser.add_argument('--format_det', type=str)

    # Absolute or relative (abs, rel)
    #parser.add_argument('--gtcoord',  type=str)
    parser.add_argument('--coord_det', type=str)

    # Actual computation type:
    parser.add_argument('--metric', type=str)
    
    # metadata for metrics (not always needed)
    parser.add_argument('--names', '-n', type=str, default='')
    parser.add_argument('-t', '--threshold', type=float, default=0.5)
    
    # extra data (graphs and etc.)
    parser.add_argument('--prgraph', '-pr', action='store_true')
    parser.add_argument('-sp', '--savepath', type=str, required=False, default="./results/")
    
    # TODO: Add the same dataset analysis seen in the GUI:
    #parser.add_argument('--info', action='store_true')
    
    return parser.parse_args()

def verifyArgs(args):
    if not os.path.exists(args.anno_gt):
        raise Exception('--anno_gt path does not exist!')

    if not os.path.exists(args.anno_det):
        raise Exception('--anno_det path does not exist!')

    if args.threshold > 1 or args.threshold < 0:
        raise Exception('Incorrect range for threshold (0-1)')

    if args.prgraph and args.savepath == '':
        raise Exception("Precision-Recall graph specified but no save path given!")

    if args.format_gt == 'voc' and args.names == '':
        raise Exception("VOC or ImageNet ground truth format specified, but name file not specified.")

    if 'tube' == args.format_gt != args.format_det:
        raise Exception("Spatio-Temporal Tube AP specified in one format parameter but not other!")

    
    if args.img == '':
        logging.warning("Image path not specified. Assuming path is same as ground truth annotations.")
        args.img = args.anno_gt

    if args.names == '':
        logging.warning("Names property empty so assuming detection format is class_id based.")

    if not os.path.exists(args.savepath):
        logging.warning("Savepath directory %s is not found. Attempting to create folder"%(args.savepath))
        try:
            os.mkdir(args.savepath)
        except:
            logging.error("Could not create directory! Exiting...")
            raise Exception()


# TODO: Check if the plotting function is redundant for COCO eval
def plot_coco_pr_graph(results, mAP=None, ap50=None, savePath=None, showGraphic=True):
    result = None
    plt.close()
    for classId, result in results.items():
        if result is None:
            raise IOError(f'Error: Class {classId} could not be found.')

        if result['AP'] != None:
            precision = result['interpolated precision']
            recall = result['interpolated recall']
            plt.plot(recall, precision, label=f'{classId}')
        else:
            logging.warning(f"Class {classId} does not have results")
    
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    
    if mAP and ap50:
        map_str = "{0:.2f}%".format(mAP * 100)
        ap_str = "{0:.2f}%".format(ap50 * 100)
        plt.title(f'Precision x Recall curve, AP={ap_str}, AP @ 0.5={map_str}')
    else:
        plt.title('Precision x Recall curve')

    plt.legend(shadow=True)
    plt.grid()
    
    if savePath is not None:
        plt.savefig(os.path.join(savePath, 'all_classes.png'))
    if showGraphic is True:
        plt.show()
        plt.pause(0.05)


def __cli__(args):

    # check if args are correct:
    verifyArgs(args)

    # collect ground truth labels:
    if args.format_gt == 'coco':
        gt_anno = converter.coco2bb(args.anno_gt)
    elif args.format_gt == 'voc':
        gt_anno = converter.vocpascal2bb(args.anno_gt)
    elif args.format_gt == 'imagenet':
        gt_anno = converter.imagenet2bb(args.anno_gt)
    elif args.format_gt == 'labelme':
        gt_anno = converter.labelme2bb(args.anno_gt)
    elif args.format_gt == 'openimg':
        gt_anno = converter.openimage2bb(args.anno_gt, args.img)
    elif args.format_gt == 'yolo':
        gt_anno = converter.yolo2bb(args.anno_gt, args.img, args.names)
    elif args.format_gt == 'absolute':
        gt_anno = converter.text2bb(args.anno_gt, img_dir=args.img)
    elif args.format_gt == 'cvat':
        gt_anno = converter.cvat2bb(args.anno_gt)
    elif args.format_gt == 'tube':
        logging.warning("Spatio-Temporal Tube AP specified. Loading ground truth and detection results at same time...")
        tube = TubeEvaluator(args.anno_gt, args.anno_det)
    else:
        raise Exception("%s is not a valid ground truth annotation format. Valid formats are: coco, voc, imagenet, labelme, openimg, yolo, absolute, cvat"%args.anno_gt)

    # collect detection truth labels:
    if args.format_det == 'coco':
        logging.warning("COCO detection format specified. Ignoring 'coord_det'...")
        # load in json:
        det_anno = converter.coco2bb(args.anno_det, bb_type=BBType.DETECTED)
    elif args.format_det == 'tube':
        # ignore the detection reading phase
        pass
    else:
        if args.format_det == 'xywh':
            # x,y,width, height
            BB_FORMAT = BBFormat.XYWH
        elif args.format_det == 'xyrb':
            # x,y,right,bottom
            BB_FORMAT = BBFormat.XYX2Y2
        else:
            raise Exception("%s is not a valid detection annotation format"%args.anno_det)
    
        if args.coord_det == 'abs':
            COORD_TYPE = CoordinatesType.ABSOLUTE
        elif args.coord_det == 'rel':
            COORD_TYPE = CoordinatesType.RELATIVE
        else:
            raise Exception("%s is not a valid detection coordinate format"%args.coord_det)
        det_anno = converter.text2bb(args.anno_det, bb_type=BBType.DETECTED, bb_format=BB_FORMAT, type_coordinates=COORD_TYPE, img_dir=args.img)

        # If VOC specified, then switch id based to string for detection bbox:
        #if args.format_gt == 'voc' or args.format_gt == 'imagenet':
        if args.names != '':
            # if names file not given, assume id-based detection output
            with open(args.names, 'r') as r:
                names = list(map(str.strip, r.readlines()))
                for det in det_anno:
                    _out = names[int(det._class_id)]
                    det._class_id = _out

    # print out results of annotations loaded:
    print("%d ground truth bounding boxes retrieved"%(len(gt_anno)))
    print("%d detection bounding boxes retrieved"%(len(det_anno)))

    # compute bboxes with given metric:

    # COCO (101-POINT INTERPOLATION)
    if args.metric == 'coco':
        logging.info("Running metric with COCO metric")

        # use coco_out for PR graphs and coco_sum for just the AP
        coco_sum = coco_evaluator.get_coco_summary(gt_anno, det_anno)
        coco_out = coco_evaluator.get_coco_metrics(gt_anno, det_anno, iou_threshold=args.threshold)
        
        value_only = tuple([float(_i[1]) for _i in coco_sum.items()])
        print( ('\nCOCO metric:\n'
                'AP [.5:.05:.95]: %f\n'
                'AP50: %f\n'
                'AP75: %f\n'
                'AP Small: %f\n'
                'AP Medium: %f\n'
                'AP Large: %f\n'
                'AR1: %f\n'
                'AR10: %f\n'
                'AR100: %f\n'
                'AR Small: %f\n'
                'AR Medium: %f\n'
                'AR Large: %f\n\n'%value_only) )

        print("Class APs:")
        for item in coco_out.items():
            if item[1]['AP'] != None:
                print("%s AP50: %f\n"%(item[0], item[1]['AP']))
            else:
                logging.warning('AP for %s is None'%(item[0]))

        if args.prgraph:
            logging.warning("Graphing precision-recall is not supported!")
            #plot_coco_pr_graph(coco_out, mAP=coco_sum['AP50'], ap50=coco_sum['AP'], savePath=args.savepath, showGraphic=False)
        
        return coco_sum

    # 11-POINT INTERPOLATION:
    elif args.metric == 'voc2007':
        logging.info("Running metric with VOC2012 metric, using the 11-point interpolation approach")
        
        voc_sum = pascal_voc_evaluator.get_pascalvoc_metrics(gt_anno, det_anno, iou_threshold=args.threshold, method=MethodAveragePrecision.ELEVEN_POINT_INTERPOLATION)
        print("mAP: %f"%(voc_sum['mAP']))
        print("Class APs:")
        for class_item in voc_sum['per_class'].items():
            if class_item[1]['AP'] != None:
                print("%s AP: %f"%(class_item[0], class_item[1]['AP']))
            else:
                logging.warning('AP for %s is None'%(class_item[0]))

        if args.prgraph:
            pascal_voc_evaluator.plot_precision_recall_curve(voc_sum['per_class'], mAP=voc_sum['mAP'], savePath=args.savepath, showGraphic=False)
        return voc_sum

    # EVERY POINT INTERPOLATION:
    elif args.metric == 'voc2012' or args.metric == 'auc':
        logging.info("Running metric with VOC2012 metric, using the every point interpolation approach")

        voc_sum = pascal_voc_evaluator.get_pascalvoc_metrics(gt_anno, det_anno, iou_threshold=args.threshold)
        print("mAP: %f"%(voc_sum['mAP']))
        print("Class APs:")
        for class_item in voc_sum['per_class'].items():
            if class_item[1]['AP'] != None:
                print("%s AP: %f"%(class_item[0], class_item[1]['AP']))
            else:
                logging.warning('AP for %s is None'%(class_item[0]))
        
        if args.prgraph:
            pascal_voc_evaluator.plot_precision_recall_curve(voc_sum['per_class'], mAP=voc_sum['mAP'], savePath=args.savepath, showGraphic=False)
        return voc_sum
    
    # SST METRIC:
    elif args.metric == 'tube':
        tube_out = tube.evaluate()
        per_class, mAP = tube_out
        print("mAP: %f"%(mAP))
        print("Class APs:")
        for name, class_obj in per_class.items():
            print("%s AP: %f"%(name, class_obj['AP'])) 
        return tube_out
    else:
        # Error out for incorrect metric format
        raise Exception("%s is not a valid metric (coco, voc2007, voc2012, auc)"%(args.format_gt))

if __name__ == '__main__':
    args = parseArgs()
    __cli__(args)