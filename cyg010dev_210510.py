import os
import sys
import cv2
import numpy as np
import json
import logging
import threading
import argparse
import time
import torch
from werkzeug.utils import import_string

# ****************** Basic Config ****************** #
# Add Python Path
PATH_PREFIX = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(PATH_PREFIX) if PATH_PREFIX not in sys.path else None
sys.path.append(f"{PATH_PREFIX}/../") if f"{PATH_PREFIX}/../" not in sys.path else None
# sys.path.append(f"{PATH_PREFIX}/../../") if f"{PATH_PREFIX}/../../" not in sys.path else None

# Save Path
SRC_IMG_PREFIX = f"{PATH_PREFIX}/../static"
MASK_IMG_PREFIX = f"{PATH_PREFIX}/../static"
CIRCLE_IMG_PREFIX = f"{PATH_PREFIX}/../static"

# Meta Config
cell_file = os.path.basename(__file__).split(".")[0]
MODEL = cell_file.split("_")[0].lower()
VERSION = cell_file.split("_")[1].lower()

# ****************** Detector Config ****************** #
CONF_PACKAGE = f"Cell.conf.{MODEL}_{VERSION}"

# Import Meta Config
meta_conf = import_string(f"{CONF_PACKAGE}.meta.MetaConfig")()
# Init Meta Config
meta_conf.gen_network_config(MODEL, VERSION)
DEBUG = meta_conf.DEBUG
LOG_LEVEL = meta_conf.LOG_LEVEL
# Config Log Level
logging.basicConfig(level=LOG_LEVEL,
                    format=(
                        # '%(filename)s: '
                        # '%(levelname)s: '
                        # '%(funcName)s: '
                        '%(lineno)d:\t'
                        '%(message)s')
                    )

# ****************** Detector Config ****************** #
sub_config_dict = {}
for sub_config in meta_conf.SUB_CONFIG_LIST:
    sub_config_dict[sub_config.lower()] = import_string(f'{CONF_PACKAGE}.sub.{sub_config}')()
# # white sample
# white_us_conf = import_string(f"Cell.conf.{MODEL}_{VERSION}_white_us.UsSideConfig")()
# white_usb_conf = import_string(f"Cell.conf.{MODEL}_{VERSION}_white_us.UsbSideConfig")()
# # black sample
# black_us_conf = import_string(f"Cell.conf.{MODEL}_{VERSION}_black_us.UsSideConfig")()
# black_usb_conf = import_string(f"Cell.conf.{MODEL}_{VERSION}_black_us.UsbSideConfig")()
# # 系统面
# ss_ss_conf = import_string(f"Cell.conf.{MODEL}_{VERSION}_ss.SsSideConfig")()

# Import Common
from Common.MqManager.rabbitMq import Worker, Rpc

# Import Glasses
from Glasses import img_transformer, img_crop, img_calc

# Import Memory
from Memory.blob import local
from Memory.doc import mongo

# Import Deeplearning Package
try:
    from torch import multiprocessing as mp
except:
    import multiprocessing as mp

# Import Loader
from Cell.loader import mmdet24 as mmdet24_loader
from Cell.detector import mmdet24 as mmdet24_detector

# Import Utils
from Common.Utils.utils import RtvThread


class Engine(object):
    def inference(self, ch=None, method=None, properties=None, body=None):
        try:
            ts = time.time()
            task_id = body['producer_info']['task_id']  # TASK_ID
            payload = body.get("payload")  # PAYLOAD
            middle_output = {}  # store subprocess output
            logging.info(f"################### {task_id} START ################### ")
            collection_rtv = {
                "task_id": task_id,
                "status": "SUCCESS",
                "results": {
                    "is_NG": False,
                    "total": 0,
                    "details": []
                },
                "path": {
                    "src_img": None,
                    "mask_img": None,
                    "aggr_img": None,
                },
                "pre_payload": body['payload'],
                "inferencer": {
                    "engine": self.inf_group,
                    "pid": os.getpid(),
                    "date": time.time()
                }
            }
            output_show_result = []
            output_hidden_result = []
            if body["img_code"] == "base64":
                # Image Transform Base64 to CV
                img = img_transformer.base64toCv(img_base64=body["img_data"])
                mask_img = img.copy()
            else:
                raise Exception("DO NOT Accept this img_code now!")
            logging.debug(f"""
            =================== {task_id} Src PAYLOAD =====================
            {payload}
            =================== {task_id} Src PAYLOAD =====================
            """)
            payload = meta_conf.trans_payload(payload, img)
            middle_output["cam"] = payload.get("cam").lower()
            middle_output["side"] = payload.get("side").lower()

            logging.debug(f"""
            =================== {task_id} Trans PAYLOAD =====================
            {payload}
            =================== {task_id} Trans PAYLOAD =====================
            """)
            # CAM = payload.get("cam").lower()
            # SIDE = payload.get("side").lower()
            logging.info(f"=================== {SIDE} START ===================== ")
            filter_config_str = meta_conf.config_selector(middle_output, img)
            middle_output["filter_config_str"] = filter_config_str
            filter_config = sub_config_dict[filter_config_str]
            logging.info(f"{'*' * 9} LOADING CONFIG {filter_config_str} as Filter Configuration {'*' * 9}")
            # Show Request Information
            logging.info(f"================= Request Information START =================")
            logging.info(f"""
            REQUEST DST: \t{body["producer_info"]["q_values"]}
            PAYLOAD: \t{payload}
            """)
            logging.info(f"================= Request Information END   =================")
            src_img_path = f"{SRC_IMG_PREFIX}/src_{task_id}.jpg"
            src_img_saver = threading.Thread(target=local.img_save_to_disk,
                                             args=(src_img_path, img), name="savetodisk")
            src_img_saver.start()
            logging.debug("src_img_saver starting")
            # Position Process
            p_bbox_result, p_segm_result = mmdet24_detector.inference_detector(self.positioning_model, img)

            # Clean GPU Memeory
            torch.cuda.empty_cache()

            p_bboxes, p_inds, p_labels, p_segms = \
                mmdet24_detector.render_defects_detail(
                    p_bbox_result, p_segm_result,
                    meta_conf.POSITIONING_SCORE
                )

            if (p_segm_result is not None) and len(p_inds):
                tmp = [0, 0, 0]
                np.random.seed(42)
                for i in p_inds:
                    mask, area, defect_name, position = mmdet24_detector.render_single_defect_detail(
                        i=i, bboxes=p_bboxes,
                        bbox_result=p_bbox_result,
                        segms=p_segms, labels=p_labels,
                        defect_name_list=meta_conf.POSITIONING_DEFECT_CLASSES
                    )
                    # return just biggest circle
                    if area > tmp[1]:
                        tmp = [i, area, position]
                X, Y, r, detect_r = meta_conf.crop(middle_output, tmp)
                middle_output["crop"] = [X, Y, r, detect_r]
                logging.info(f"================= Circle Information START =================")
                logging.info(f"""
                POSITION:\t{X}, {Y}, {r}
                """)
                logging.info(f"================= Circle Information  END   =================")

                # Crop Circle
                mask_img = img_crop.crop_cicle(img, X, Y, detect_r)
            else:
                logging.warning(f"!!! No Circle Detected!")
                mask_img = mask_img
            # Source Predict Result ter
            bbox_result, segm_result = mmdet24_detector.inference_detector(self.detector_model, mask_img)

            # Clean GPU Memeory
            torch.cuda.empty_cache()

            bboxes, inds, labels, segms = \
                mmdet24_detector.render_defects_detail(
                    bbox_result, segm_result,
                    meta_conf.DETECTOR_SCORE
                )
            return_img = img.copy()
            if DEBUG:
                try:
                    cv2.circle(return_img, (X, Y), r, (255, 0, 0), 3)
                    cv2.circle(return_img, (X, Y),
                               detect_r,
                               (0, 255, 0), 3
                               )
                    circle_img = return_img.copy()
                except Exception as e:
                    logging.warning(e)
                    circle_img = return_img.copy()
                src_mask_img = img.copy()
            else:
                src_mask_img = img.copy()

            if (segm_result is not None) and len(inds):
                np.random.seed(42)
                # Init Defect
                all_src_defects = []  # all defect
                src_defect_by_class = {}  # defect group by class
                for defect_class in meta_conf.DETECTOR_DEFECT_CLASSES:
                    src_defect_by_class[defect_class] = []
                for defect_i, i in enumerate(inds):
                    mask, area, defect_name, position = mmdet24_detector.render_single_defect_detail(
                        i=i, bboxes=bboxes,
                        bbox_result=bbox_result,
                        segms=segms, labels=labels,
                        defect_name_list=meta_conf.DETECTOR_DEFECT_CLASSES
                    )
                    all_src_defects.append([i, [mask, area, defect_name, position]])
                    src_defect_by_class[defect_name].append([i, [mask, area, defect_name, position]])
                # Map Filter
                single_t_pool = []
                group_t_pool = []
                # Filter Defects by ALL
                all_detector = RtvThread(filter_config.all_filter, args=(img, all_src_defects, middle_output))
                all_detector.start()
                # Filter Defects by Group
                for group_class, defect_group in src_defect_by_class.items():
                    group_detector = RtvThread(filter_config.group_filter, args=(img, defect_group, middle_output))
                    group_detector.start()
                    group_t_pool.append(group_detector)
                # Filter Defects by Self
                for single_detect in all_src_defects:
                    single_detector = RtvThread(filter_config.single_filter, args=(img, single_detect, middle_output))
                    single_detector.start()
                    single_t_pool.append(single_detector)
                # Reduce Result
                tmp_result = []
                tmp_result_idx = []
                for t in group_t_pool:
                    t.join()
                    for defect_detail in t.get_result():
                        if defect_detail[0] not in tmp_result_idx:
                            tmp_result_idx.append(defect_detail[0])
                            tmp_result.append(defect_detail[1])
                for t in single_t_pool:
                    t.join()
                    if defect_detail[0] not in tmp_result_idx:
                        tmp_result_idx.append(defect_detail[0])
                        tmp_result.append(defect_detail[1])

                all_detector.join()
                for defect_detail in all_detector.get_result():
                    if defect_detail[0] not in tmp_result_idx:
                        tmp_result_idx.append(defect_detail[0])
                        tmp_result.append(defect_detail[1])

                output_show_result = tmp_result
                for defect_detail in all_src_defects:
                    if defect_detail[0] not in tmp_result_idx:
                        output_hidden_result.append(defect_detail[1])

                for defect_detail in output_show_result:
                    # [mask, area, defect_name, position]
                    return_img = mmdet24_detector.put_single_defect_mask(
                        return_img, defect_detail[0], defect_detail[2],
                        defect_detail[1], defect_detail[3]
                    )
                    collection_rtv["results"]["details"].append({
                        "defect_name": defect_detail[2],
                        "area": str(defect_detail[1]),
                        "position": defect_detail[3],
                    })
                for defect_detail in output_hidden_result:
                    # [mask, area, defect_name, position]
                    return_img = mmdet24_detector.put_single_defect_mask(
                        src_mask_img, defect_detail[0], defect_detail[2],
                        defect_detail[1], defect_detail[3]
                    )

                collection_rtv["results"]["total"] = len(collection_rtv["results"]["details"])

                if len(collection_rtv["results"]["details"]) > 0:
                    collection_rtv["results"]["is_NG"] = True
                else:
                    collection_rtv["results"]["is_NG"] = False

            # Save Mask Image Anyway
            logging.info(f"================= SAVE MASK IMG START =================")
            mask_img_path = f"{MASK_IMG_PREFIX}/mask_{task_id}.jpg"
            mask_img_saver = threading.Thread(target=local.img_save_to_disk,
                                              args=(mask_img_path, return_img), name="savemasktodisk")
            mask_img_saver.start()
            collection_rtv["_id"] = task_id
            logging.info(f"================= SAVE MASK IMG  END   =================")

            # Save Aggr Img
            collection_rtv["path"]["aggr_img"] = None
            if DEBUG:
                aggr_img_path = f"{MASK_IMG_PREFIX}/aggr_{task_id}.jpg"
                local.imgs_save_to_disk(aggr_img_path, [src_mask_img, return_img, img, circle_img])
                collection_rtv["path"]["aggr_img"] = os.path.basename(aggr_img_path)

            # Join All IO
            src_img_saver.join()
            collection_rtv["path"]["src_img"] = os.path.basename(src_img_path)
            try:
                mask_img_saver.join()
                if DEBUG:
                    src_img_saver.join()
                collection_rtv["path"]["mask_img"] = os.path.basename(mask_img_path)
            except Exception as e:
                # TODO: 处理 Exception
                logging.warning(e)
                pass

            # TODO: handle UUID 重复
            # pymongo.errors.BulkWriteError: batch op errors occurred, full error:
            # {'writeErrors': [{'index': 0, 'code': 11000,
            # 'errmsg': 'E11000 duplicate key error collection: udfc_inference.infResult
            # index: _id_ dup key:

            self.mongo_client.insert(data=[collection_rtv])
            logging.info(f"================= Result START =================")
            logging.info(collection_rtv)
            logging.info(f"================= Result  END   =================")
            # Clean all Memory in this inference
            torch.cuda.empty_cache()
            logging.info(f"======= Total CT: [{os.getpid()}]{time.time() - ts}")
            logging.info(f"################### END ################### ")

            return collection_rtv
        except Exception as e:
            raise e
            logging.info(f"================= ERROR START =================")
            logging.error(e)
            logging.info(f"================= ERROR END =================")
            collection_rtv["status"] = "ERROR"
            return collection_rtv


class AsyncConsumer(Engine, Worker):
    def __init__(self, host="localhost", port=5672, inf_group="", weight=1, device="cuda:0"):
        super().__init__(host=host, port=port, inf_group=inf_group, weight=weight)
        for process in meta_conf.DETECT_PROCESS:
            config = getattr(meta_conf, f"{process.upper()}_DETECTOR_CONFIG")
            checkpoint = getattr(meta_conf, f"{process.upper()}_CHECKPOINT")
            attr_identity = f"{process}_model"
            setattr(
                self, attr_identity,
                mmdet24_loader.load_model(
                    config=config,
                    checkpoint=checkpoint,
                    device=device
                )
            )
            logging.info(f'Load {checkpoint} in {attr_identity}')
        self.__weight = weight
        self.inf_group = inf_group
        self.mongo_client = mongo.Manager(collection="infResult")
        logging.info(f"AsyncConsumer Init Finish")

    def callback(self, ch, method, properties, body):
        body = json.loads(body)
        collection_rtv = self.inference(
            ch=ch, method=method,
            properties=properties, body=body
        )
        self.channel.basic_ack(
            delivery_tag=method.delivery_tag
        )


class SyncConsumer(Rpc, Engine):
    def __init__(self, host="localhost", port=5672, inf_group="", weight=1, device="cuda:0"):
        super().__init__(host=host, port=port, inf_group=inf_group, weight=weight)
        for process in meta_conf.DETECT_PROCESS:
            config = getattr(meta_conf, f"{process.upper()}_CONFIG")
            checkpoint = getattr(meta_conf, f"{process.upper()}_CHECKPOINT")
            attr_identity = f"{process.lower()}_model"
            setattr(
                self, attr_identity,
                mmdet24_loader.load_model(
                    config=config,
                    checkpoint=checkpoint,
                    device=device
                )
            )
            logging.info(f'Load {checkpoint} in {attr_identity}')
        self.__weight = weight
        self.inf_group = inf_group
        self.mongo_client = mongo.Manager(collection="infResult")
        logging.info(f"SyncConsumer Init Finish")

    def response_body(self, body):
        # body = json.loads(body)
        collection_rtv = self.inference(body=body)
        return json.dumps(collection_rtv)


def start_consumer(weight, device, mode):
    if mode == "sync":
        logging.info("Starting Sync Mode Instance!")
        consumer = SyncConsumer(
            host=meta_conf.RABBIT_MQ_HOST,
            port=meta_conf.RABBIT_MQ_PORT,
            inf_group=f'{MODEL}.{VERSION}.{mode}',
            weight=weight,
            device=device
        )
    elif mode == "async":
        logging.info("Starting Async Mode Instance!")
        consumer = AsyncConsumer(
            host=meta_conf.RABBIT_MQ_HOST,
            port=meta_conf.RABBIT_MQ_PORT,
            inf_group=f'{MODEL}.{VERSION}.{mode}',
            weight=weight,
            device=device
        )
    else:
        raise ValueError("Mode Value Error!")
    consumer.declare().consume()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Start Inference Instances")
    parser.add_argument("--device", "-d", default="cuda:0")
    parser.add_argument("--weight", "-w", default=1)
    parser.add_argument("--mode", "-m", default="async", )
    args = parser.parse_args()

    logging.info(
        f"Starting {MODEL}.{VERSION}.{args.mode} instance"
        f" in {args.device} with "
        f"weight {args.weight}")
    try:
        mp.set_start_method('spawn')
    except Exception as e:
        logging.debug(e)

    start_consumer(device=args.device, weight=int(args.weight), mode=args.mode)
