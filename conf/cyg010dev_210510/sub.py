#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Joachim WANG
# @Contact : xiaodong.wang@cygia.com
# @File    : sub.py
# @Software: PyCharm
# @Site    : CYGIA/UDF
# @Desc    :
import math

class BaseConfig(object):
    @classmethod
    def single_filter(cls, img, defect, middle_output):
        """
        Single Defect Filter
        :param img:
        :param defect:
        :return:
        """
        return cls.pass_filter(img, defect)

    @classmethod
    def group_filter(cls, img, defect_group, middle_output):
        return cls.pass_filter(img, defect_group)

    @classmethod
    def all_filter(cls, img, all_defect, middle_output):
        return cls.pass_filter(img, all_defect)

    @classmethod
    def pass_filter(cls, img, defect):
        return defect

    @classmethod
    def deny_filter(cls, img, defect):
        return False


class DefaultConfig(BaseConfig):
    pass


class PerformConfig(BaseConfig):
    DEFAULT_OFFSET = {
        "OFFSET_X": 0,
        "OFFSET_Y": 0,
        "OFFSET_R": 200,
    }

    CIRCLE_OFFSET = {
        "default": DEFAULT_OFFSET,
        "a": DEFAULT_OFFSET,
        "o": DEFAULT_OFFSET,
        "m": DEFAULT_OFFSET,
    }

    # INK AREA
    DEFAULT_LENS_AREA_N_PIXEL = 100
    LENS_AREA_N_PIXEL = {
        "default": DEFAULT_LENS_AREA_N_PIXEL,
        "a": DEFAULT_LENS_AREA_N_PIXEL,
        "m": DEFAULT_LENS_AREA_N_PIXEL,
        "o": DEFAULT_LENS_AREA_N_PIXEL,
    }

    DEFAULT_LIMIT_RANGE = [0, float('inf')]
    # 油墨区缺陷面积限定
    INK_LIMIT_N_PIXEL = {
        "default": DEFAULT_LIMIT_RANGE,
        "dent": DEFAULT_LIMIT_RANGE,
        "scratch": DEFAULT_LIMIT_RANGE,
        "contamination": DEFAULT_LIMIT_RANGE,
        'fiber': DEFAULT_LIMIT_RANGE,
        'wavelet': DEFAULT_LIMIT_RANGE,
        'cloud': DEFAULT_LIMIT_RANGE,
    }
    # 透明区缺陷面积限定
    LENS_LIMIT_N_PIXEL = {
        "default": DEFAULT_LIMIT_RANGE,
        "dent": DEFAULT_LIMIT_RANGE,
        "scratch": DEFAULT_LIMIT_RANGE,
        "contamination": DEFAULT_LIMIT_RANGE,
        'fiber': DEFAULT_LIMIT_RANGE,
        'wavelet': DEFAULT_LIMIT_RANGE,
        'cloud': DEFAULT_LIMIT_RANGE,
    }

    DEFAULT_LIMIT_GRAY = [0, 255]
    # 油墨区灰度限定
    INK_LIMIT_GRAY = {
        "default": DEFAULT_LIMIT_GRAY,
        "dent": DEFAULT_LIMIT_GRAY,
        "scratch": DEFAULT_LIMIT_GRAY,
        "contamination": DEFAULT_LIMIT_GRAY,
        'fiber': DEFAULT_LIMIT_GRAY,
        'wavelet': DEFAULT_LIMIT_GRAY,
        'cloud': DEFAULT_LIMIT_GRAY,
    }
    # 透明区灰度限定
    LENS_LIMIT_GRAY = {
        "default": DEFAULT_LIMIT_GRAY,
        "dent": DEFAULT_LIMIT_GRAY,
        "scratch": DEFAULT_LIMIT_GRAY,
        "contamination": DEFAULT_LIMIT_GRAY,
        'fiber': DEFAULT_LIMIT_GRAY,
        'wavelet': DEFAULT_LIMIT_GRAY,
        'cloud': DEFAULT_LIMIT_GRAY,
    }

    # 增加透明区缺陷的统计条件
    TOTAL_DEFECTS_CONF = {
        "defects": ["dent"],
        "item_area": [[0, float('inf')]],
        "gray": [[0, 255]],
        "count": 3,
        "total_area": [0, float('inf')],
        "distance": [0, float('inf')]
    }
    # method
    @classmethod
    def is_ink(cls, cam, cx, cy, x, y):
        distance = math.sqrt(((cx - x) ** 2) + ((cy - y) ** 2))
        if distance > cls.LENS_AREA_N_PIXEL.get(cam.lower()):
            return True
        return False


class BlackUsSideConfig(PerformConfig, BaseConfig):
    CIRCLE_OFFSET = {
        "default": {
            "OFFSET_X": 0,
            "OFFSET_Y": 0,
            "OFFSET_R": 200,
        },
        "a": {
            "OFFSET_X": 0,
            "OFFSET_Y": 0,
            "OFFSET_R": 200,
        },
        "o": {
            "OFFSET_X": 0,
            "OFFSET_Y": 0,
            "OFFSET_R": 90 + 10,  # 需要缩小，则需要增大数值
        },
        "m": {
            "OFFSET_X": 0,
            "OFFSET_Y": 0,
            "OFFSET_R": 78,  # 需要缩小，则需要增大数值
        },
    }

    LENS_AREA_N_PIXEL = {
        "a": 100,
        "o": 405 + 30,
        "m": 370 + 20,
    }

    INK_LIMIT_N_PIXEL = {
        "dent": [0, float('inf')],
        "scratch": [0, float('inf')],
        "contamination": [0, float('inf')],
        'fiber': [0, float('inf')],
        'wavelet': [0, float('inf')],
        'cloud': [0, float('inf')],
    }

    LENS_LIMIT_N_PIXEL = {
        "dent": [0, float('inf')],
        "scratch": [0, float('inf')],
        "contamination": [0, float('inf')],
        'fiber': [0, float('inf')],
        'wavelet': [0, float('inf')],
        'cloud': [0, float('inf')],
    }

    # 普遍明显的缺陷
    LENS_LIMIT_GRAY = {
        "default": [0, 256],
        "dent": [0, 256],
        "scratch": [0, 256],
        "contamination": [0, 256],
        'fiber': [0, 256],
        'wavelet': [0, 256],
        'cloud': [0, 256],
    }

    INK_LIMIT_GRAY = {
        "default": [0, 255],
        "dent": [0, 256],
        "scratch": [0, 256],
        "contamination": [0, 256],
        'fiber': [0, 256],
        'wavelet': [0, 256],
        'cloud': [0, 256],
    }

    # 统计缺陷数量
    TOTAL_DEFECTS_CONF_1 = {
        "defects": ["dent", "cloud"],
        "item_area": [[0, float('inf')], [0, float('inf')]],
        "gray": [[0, 255], [0, 255]],
        "count": 1,
        "total_area": [1, float('inf')],
        "distance": [0, 45000]
    }


class WhiteUsSideConfig(BaseConfig):
    CIRCLE_OFFSET = {
        "default": {
            "OFFSET_X": 0,
            "OFFSET_Y": 0,
            "OFFSET_R": 200,
        },
        "a": {
            "OFFSET_X": 0,
            "OFFSET_Y": 0,
            "OFFSET_R": 200,
        },
        "o": {
            "OFFSET_X": 0,
            "OFFSET_Y": 0,
            "OFFSET_R": 90 + 10,  # 需要缩小，则需要增大数值
        },
        "m": {
            "OFFSET_X": 0,
            "OFFSET_Y": 0,
            "OFFSET_R": 78,  # 需要缩小，则需要增大数值
        },
    }

    LENS_AREA_N_PIXEL = {
        "a": 100,
        "o": 405 + 30 - 10,  # 需要缩小
        "m": 370 + 20 - 10,  # 需要缩小
    }

    INK_LIMIT_N_PIXEL = {
        "dent": [0, float('inf')],
        "scratch": [0, float('inf')],
        "contamination": [0, float('inf')],
        'fiber': [0, float('inf')],
        'wavelet': [0, float('inf')],
        'cloud': [0, float('inf')],
    }

    LENS_LIMIT_N_PIXEL = {
        "dent": [0, float('inf')],
        "scratch": [0, float('inf')],
        "contamination": [0, float('inf')],
        'fiber': [0, float('inf')],
        'wavelet': [0, float('inf')],
        'cloud': [0, float('inf')],
    }

    # 灰度限定
    INK_LIMIT_GRAY = {
        "dent": [0, 256],
        "scratch": [0, 256],
        "contamination": [0, 256],
        'fiber': [0, 256],
        'wavelet': [0, 256],
        'cloud': [0, 256],
    }

    LENS_LIMIT_GRAY = {
        "dent": [0, 256],
        "scratch": [0, 256],
        "contamination": [0, 256],
        'fiber': [0, 256],
        'wavelet': [0, 256],
        'cloud': [0, 256],
    }

    TOTAL_DEFECTS_CONF_1 = {
        "defects": ["dent", "cloud"],
        "item_area": [[0, float('inf')], [0, float('inf')]],
        "gray": [[0, 255], [0, 255]],
        "count": 1,
        "total_area": [0, float('inf')],
        "distance": [0, 45000]
    }


class SsSideConfig(BaseConfig):
    CIRCLE_OFFSET = {
        "default": {
            "OFFSET_X": 0,
            "OFFSET_Y": 0,
            "OFFSET_R": 100,
        },
        "a": {
            "OFFSET_X": 0,
            "OFFSET_Y": 0,
            "OFFSET_R": 100,
        },
        "o": {
            "OFFSET_X": 0,
            "OFFSET_Y": 0,
            "OFFSET_R": 120 - 10,  # 需要放大，则需要减小数值
        },
        "m": {
            "OFFSET_X": 0,
            "OFFSET_Y": 0,
            "OFFSET_R": 120,
        },
    }
    LENS_AREA_N_PIXEL = {
        "a": 100,
        "o": 440,
        "m": 390,
    }

    # 系统面屏蔽油墨区
    INK_LIMIT_N_PIXEL = {
        "dent": [0, float('inf')],
        "scratch": [0, float('inf')],
        "contamination": [0, float('inf')],
        'through': [0, float('inf')],
        'wavelet': [0, float('inf')],
        'cloud': [0, float('inf')],
    }

    LENS_LIMIT_N_PIXEL = {
        "dent": [0, float('inf')],
        "scratch": [0, float('inf')],
        "contamination": [0, float('inf')],
        'fiber': [0, float('inf')],
        'wavelet': [0, float('inf')],
        'cloud': [0, float('inf')],
    }
    # 普遍明显的缺陷
    LENS_LIMIT_GRAY = {
        "dent": [0, 256],
        "scratch": [0, 256],
        "contamination": [0, 256],
        'fiber': [0, 256],
        'wavelet': [0, 256],
        'cloud': [0, 256],
    }

    INK_LIMIT_GRAY = {
        "dent": [0, 256],
        "scratch": [0, 256],
        "contamination": [0, 256],
        'through': [0, 256],
        'wavelet': [0, 256],
        'cloud': [0, 256],
    }

    # 统计缺陷数量
    TOTAL_DEFECTS_CONF_1 = {
        "defects": ["dent", "cloud"],
        "item_area": [[0, float('inf')], [0, float('inf')]],
        "gray": [[0, 255], [0, 255]],
        "count": 1,
        "total_area": [0, float('inf')],
        "distance": [0, 45000]
    }


class UsbSideConfig(BaseConfig):
    CIRCLE_OFFSET = {
        "default": {
            "OFFSET_X": 0,
            "OFFSET_Y": 0,
            "OFFSET_R": 100,
        },
        "a": {
            "OFFSET_X": 0,
            "OFFSET_Y": 0,
            "OFFSET_R": 100,
        },
        "o": {
            "OFFSET_X": 0,
            "OFFSET_Y": 0,
            "OFFSET_R": -40,
        },
        "m": {
            "OFFSET_X": 0,
            "OFFSET_Y": 0,
            "OFFSET_R": -40 - 20,  # 需要放大
        },
    }
    LENS_AREA_N_PIXEL = {
        "a": 100,
        "o": 405 + 15,
        "m": 370 + 5,
    }

    INK_LIMIT_N_PIXEL = {
        "dent": [0, float('inf')],  # 油墨只检测dent贯穿
        "scratch": None,
        "contamination": None,
        'fiber': None,
        'wavelet': None,
        'cloud': None,
    }

    LENS_LIMIT_N_PIXEL = {
        "dent": None,
        "scratch": None,  # 透明区只检测scratch划痕
        "contamination": None,
        'fiber': None,
        'wavelet': None,
        'cloud': None,
    }

    # 灰度限定
    INK_LIMIT_GRAY = {
        "dent": [0, 256],
        "scratch": None,
        "contamination": None,
        'fiber': None,
        'wavelet': None,
        'cloud': None,
    }

    LENS_LIMIT_GRAY = {
        "dent": None,
        "scratch": None,
        "contamination": None,
        'fiber': None,
        'wavelet': None,
        'cloud': None,
    }

# class PerformConfig(BaseConfig):
#     @classmethod
#     def hidden_single_filter(cls):
#         # hide some class defect
#         cls.HIDDEN_CLASS
#         pass
#
#     @classmethod
#     def single_filter(cls, img, defect):
#         cls.hidden_single_filter()
#         pass
#
#
# class UsConfig(PerformConfig, BaseConfig):
#     HIDDEN_CLASS = []
#
#
# class HideBaseConfig(BaseConfig)
#     HIDDEN_CLASS = ["dent",]
#     def single_filter(cls, img, defect):
#         if defect[1][3] in cls.HIDDEN_CLASS:
#             return None
#         else:
#             return defect
#
# class UsConfig(UsBaseConfig, BaseConfig):
#     pass
