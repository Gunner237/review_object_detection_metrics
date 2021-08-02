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
    parser.add_argument('--det_names',type=str,default='')
    
    # extra data (graphs and etc.)
    parser.add_argument('--plot', '-p', action='store_true')
    parser.add_argument('--save_path', '-sp', type=str, required=False, default="./results/")
    
    parser.add_argument('--plot_title', '-pt', type=str, required=False, default = '')
    
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

    if args.plot and args.save_path == '':
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
        
    if args.det_names == '':
        logging.warning("Detector names property empty so assuming detector and ground truth use same names file.")
        args.det_names = args.names

    if not os.path.exists(args.save_path):
        logging.warning("Savepath directory %s is not found. Attempting to create folder"%(args.save_path))
        try:
            os.mkdir(args.save_path)
        except:
            logging.error("Could not create directory! Exiting...")
            raise Exception()

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
            raise Exception("%s is not a valid detection annotation format"%args.format_det)
    
        if args.coord_det == 'abs':
            COORD_TYPE = CoordinatesType.ABSOLUTE
        elif args.coord_det == 'rel':
            COORD_TYPE = CoordinatesType.RELATIVE
        else:
            raise Exception("%s is not a valid detection coordinate format"%args.coord_det)
        det_anno = converter.text2bb(args.anno_det, bb_type=BBType.DETECTED, bb_format=BB_FORMAT, type_coordinates=COORD_TYPE, img_dir=args.img)

        # If VOC specified, then switch id based to string for detection bbox:
        #if args.format_gt == 'voc' or args.format_gt == 'imagenet':
        if args.det_names != '':
            # if names file not given, assume id-based detection output
            with open(args.det_names, 'r') as r:
                det_names = list(map(str.strip, r.readlines()))
                for det in det_anno:
                    try:
                        _index = int(det._class_id)
                        _out = det_names[_index]
                        det._class_id = _out
                    except ValueError:
                        print("Detection files have class IDs as integers!")

    bb_per_class = BoundingBox.get_amount_bounding_box_all_classes(det_anno)
    amount_bb_per_class = 'No class found'
    if len(bb_per_class) > 0:
        amount_bb_per_class = ''
        longest_class_name = len(max(bb_per_class.keys(), key=len))
        for c, amount in bb_per_class.items():
            c = c.ljust(longest_class_name, ' ')
            amount_bb_per_class += f'{c} : {amount}\n'
            
    bb_per_class_gt = BoundingBox.get_amount_bounding_box_all_classes(gt_anno)
    amount_bb_per_class_gt = 'No class found'
    if len(bb_per_class_gt) > 0:
        amount_bb_per_class_gt = ''
        longest_class_name_gt = len(max(bb_per_class_gt.keys(), key=len))
        for c, amount in bb_per_class_gt.items():
            c = c.ljust(longest_class_name_gt, ' ')
            amount_bb_per_class_gt += f'{c} : {amount}\n'
            
    total_images = BoundingBox.get_total_images(det_anno)

    # print out results of annotations loaded:
    print("%d images found with annotations"%total_images)
    print("%d ground truth bounding boxes retrieved"%(len(gt_anno)))
    print("%d detection bounding boxes retrieved"%(len(det_anno)))
    print("=== GROUND TRUTH ===")
    print(amount_bb_per_class_gt)
    print("=== DETECTIONS ===")
    print(amount_bb_per_class)
    
    f = open(f"{args.save_path}/detections.txt", "w+")
    f.write("%d images found with annotations\n"%total_images)
    f.write("%d ground truth bounding boxes retrieved\n"%(len(gt_anno)))
    f.write("%d detection bounding boxes retrieved\n\n"%(len(det_anno)))
    f.write("=== GROUND TRUTH ===\n")
    f.write(amount_bb_per_class_gt)
    f.write("\n=== DETECTIONS ===\n")
    f.write(amount_bb_per_class)
    f.close()

    # compute bboxes with given metric:
    
        # 11-POINT INTERPOLATION:
    if 'voc2007' in args.metric:
        logging.info("Running metric with VOC2012 metric, using the 11-point interpolation approach")
        
        voc_sum = pascal_voc_evaluator.get_pascalvoc_metrics(gt_anno, det_anno, iou_threshold=args.threshold, method=MethodAveragePrecision.ELEVEN_POINT_INTERPOLATION)
        print("mAP: %f"%(voc_sum['mAP']))
        print("Class APs:")
        for class_item in voc_sum['per_class'].items():
            if class_item[1]['AP'] != None:
                print("%s AP: %f"%(class_item[0], class_item[1]['AP']))
            else:
                logging.warning('AP for %s is None'%(class_item[0]))

        if args.plot:
            pascal_voc_evaluator.plot_precision_recall_curve(voc_sum['per_class'], mAP=voc_sum['mAP'], savePath=args.save_path, showGraphic=False)
        # return voc_sum
    else:
        pass

    # EVERY POINT INTERPOLATION:
    # elif args.metric == 'voc2012' or args.metric == 'auc':
    if 'voc2012' in args.metric:
        logging.info("Running metric with VOC2012 metric, using the every point interpolation approach")

        voc_sum = pascal_voc_evaluator.get_pascalvoc_metrics(gt_anno, det_anno, iou_threshold=args.threshold)
        
        # Output to console
        print("mAP: %f"%(voc_sum['mAP']))
        print("Class APs:")
        for class_item in voc_sum['per_class'].items():
            if class_item[1]['AP'] != None:
                print("%s AP: %f"%(class_item[0], class_item[1]['AP']))
            else:
                logging.warning('AP for %s is None'%(class_item[0]))
        
        # Output to text file
        f = open(f"{args.save_path}/metrics.txt", "a+")
        f.write("PASCAL METRIC (mAP):\n")
        f.write("mAP: %g\n\n"%(voc_sum['mAP']))
        f.write("PASCAL METRIC (AP per class):\n")
        for class_item in voc_sum['per_class'].items():
            if class_item[1]['AP'] != None:
                f.write("%s: %g\n"%(class_item[0], class_item[1]['AP']))
        f.close()
        
        if args.plot:
            # All classes
            pascal_voc_evaluator.plot_precision_recall_curve(voc_sum['per_class'], mAP=voc_sum['mAP'], savePath=args.save_path, showGraphic=False, customTitle=args.plot_title)
            
            # For each class
            pascal_voc_evaluator.plot_precision_recall_curves(voc_sum['per_class'], showAP=True, savePath=args.save_path, showGraphic=False, customTitle=args.plot_title)
            
        # return voc_sum
    else:
        pass

    # COCO (101-POINT INTERPOLATION)
    if 'coco' in args.metric:
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
        
        f = open(f"{args.save_path}/metrics.txt", "a+")
        f.write(('\nCOCO METRICS:\n'
                'AP [.5:.05:.95]: %f\n'
                'AP50: %f\n'
                'AP75: %f\n'
                'APsmall: %f\n'
                'APmedium: %f\n'
                'APlarge: %f\n'
                'AR1: %f\n'
                'AR10: %f\n'
                'AR100: %f\n'
                'ARsmall: %f\n'
                'ARmedium: %f\n'
                'ARlarge: %f\n\n'%value_only))
        f.close()

        print("Class APs:")
        for item in coco_out.items():
            if item[1]['AP'] != None:
                print("%s AP50: %f"%(item[0], item[1]['AP']))
            else:
                logging.warning('AP for %s is None'%(item[0]))

        if args.plot:
            logging.warning("Graphing precision-recall is not supported!")
        
        # return coco_sum
    else:
        pass
    
    # SST METRIC:
    if args.metric == 'tube':
        tube_out = tube.evaluate()
        per_class, mAP = tube_out
        print("mAP: %f"%(mAP))
        print("Class APs:")
        for name, class_obj in per_class.items():
            print("%s AP: %f"%(name, class_obj['AP'])) 
        return tube_out
    else:
        # Error out for incorrect metric format
        # raise Exception("%s is not a valid metric (coco, voc2007, voc2012, auc)"%(args.format_gt))
        pass

if __name__ == '__main__':
    args = parseArgs()
    __cli__(args)
