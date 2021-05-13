#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Joachim WANG
# @Contact : xiaodong.wang@cygia.com
# @File    : meta.py
# @Software: PyCharm
# @Site    : CYGIA/UDF
# @Desc    :
# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Joachim WANG
# @Contact : xiaodong.wang@cygia.com
# @File    : base.py
# @Software: PyCharm
# @Site    : CYGIA/UDF
# @Desc    :
import logging
import os


class PositionProcess(object):
    @classmethod
    def crop(self, middle_output, tmp):
        """
        Using payload to ajust x y r and enable detect area
        :param payload:
        :param tmp:
        :return:
        """
        X = tmp[2][0] // 2 + tmp[2][2] // 2
        Y = tmp[2][1] // 2 + tmp[2][3] // 2
        r = max(tmp[2][2] // 2, tmp[2][3] // 2)
        detect_r = max(tmp[2][2] // 2, tmp[2][3] // 2)
        return X, Y, r, detect_r


class MetaConfig(PositionProcess):
    # Queue Config
    RABBIT_MQ_HOST = "127.0.0.1"
    RABBIT_MQ_PORT = 5672

    # Code Config
    LOG_LEVEL = logging.INFO
    DEBUG = True

    # Networks Settings
    NETWORK_ARTIFACT_PATH = "/app/NETWORKS"
    DETECT_PROCESS = [
        "POSITIONING", "DETECTOR",
    ]
    POSITIONING_SCORE = 0.8
    DETECTOR_SCORE = 0.5

    POSITIONING_DEFECT_CLASSES = ['circle']
    DETECTOR_DEFECT_CLASSES = [
        'dent', 'scratch',
        'wavelet', 'contamination', 'cloud',
        "fiber"
    ]

    SUB_CONFIG_LIST = [
        "DefaultConfig",
        # "DefaultConfig", "UsSideConfig", "UsbSideConfig",
        # "UsSideConfig", "UsbSideConfig", "SsSideConfig",
        "uSa"
        "BlackUsSideConfig",
        "WhiteUsSideConfig",
        "SsSideConfig",
        "UsbSideConfig"
    ]

    @classmethod
    def network_config_validator(cls, model, version):
        if not cls.NETWORK_ARTIFACT_PATH:
            raise RuntimeError("Miss NETWORK_ARTIFACT_PATH")
        if not cls.DETECT_PROCESS:
            raise RuntimeError("Miss DETECT_PROCESS")
        for process in cls.DETECT_PROCESS:
            if not hasattr(cls, f"{process.upper()}_SCORE"):
                raise RuntimeError(f"Miss {process.upper()}_SCORE")
            if not hasattr(cls, f"{process.upper()}_DEFECT_CLASSES"):
                raise RuntimeError(f"Miss {process.upper()}_DEFECT_CLASSES")
            arg_file = os.path.join(cls.NETWORK_ARTIFACT_PATH, f"{model}_{version}_{process.lower()}_model.pth")
            if not os.path.exists(arg_file):
                raise RuntimeError(f"Miss {arg_file}!")
            config_file = os.path.join(cls.NETWORK_ARTIFACT_PATH, f"{model}_{version}_{process.lower()}_config.py")
            if not os.path.exists(config_file):
                raise RuntimeError(f"Miss {config_file}!")
        return True

    @classmethod
    def gen_network_config(cls, model, version):
        """
        Generate Networks Configuration automatic
        :GENERATE PROPERTIES:
            MODEL_VERSION_positioning_config.py
            MODEL_VERSION_positioning_model.pth
            MODEL_VERSION_detector_config.py
            MODEL_VERSION_detector_model.pth
        :return:
        """
        cls.network_config_validator(model, version)
        for model_type in cls.DETECT_PROCESS:
            setattr(cls, f"{model_type}_CHECKPOINT",
                    f"./NETWORKS/{model}_{version}_{model_type.lower()}_model.pth")
            setattr(cls, f"{model_type}_CONFIG",
                    f"./NETWORKS/{model}_{version}_{model_type.lower()}_config.py")

    @classmethod
    def trans_payload(cls, payload, img):
        return payload

    @classmethod
    def config_selector(cls, middle_output, img):
        """
        using payload to choice
        :param payload:
        :return:
        """
        return "defaultconfig"
