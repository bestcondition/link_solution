from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple, Optional

import cv2
import numpy as np
from ultralytics import YOLO

from link_solution.draw_graph import get_draw_point_by_unit
from link_solution.link_solution import Graph, Value, LinkSolution


class ClassEnum(Enum):
    empty = 0
    point = 1
    region = 2


@dataclass
class Box:
    c: ClassEnum
    ch: float
    cw: float
    bh: float
    bw: float
    g_point: tuple[int, int] = (-1, -1)
    color: tuple[int, int, int] = (-1, -1, -1)

    def get_tuple(self):
        return self.c, self.ch, self.cw, self.bh, self.bw

    def __hash__(self):
        return hash(self.get_tuple())

    def __eq__(self, other):
        return self.get_tuple() == other.get_tuple()


def get_min_max(c, b):
    return c - b / 2, c + b / 2


def has_intersection(box1: Box, box2: Box, attr: str) -> bool:
    min_1, max_1 = get_min_max(getattr(box1, f'c{attr}'), getattr(box1, f'b{attr}'))
    min_2, max_2 = get_min_max(getattr(box2, f'c{attr}'), getattr(box2, f'b{attr}'))
    if min_1 <= min_2 <= max_1:
        return True
    if min_1 <= max_2 <= max_1:
        return True
    if min_2 <= min_1 <= max_2:
        return True
    if min_2 <= max_1 <= max_2:
        return True
    return False


def color_diff(color1, color2):
    sum_diff = 0
    for i in range(3):
        sum_diff += abs(color1[i] - color2[i])
    return sum_diff


class ImageToSolution:
    def __init__(self, yolo_model: YOLO, image: np.ndarray):
        self.yolo_model = yolo_model
        self.ori_img = image
        self.image_draw = self.ori_img.copy()
        h, w = self.ori_img.shape[:2]
        self.max_h = 0
        self.min_h = h
        self.max_w = 0
        self.min_w = 0
        self.crop_image = self.ori_img
        self.box_list = []
        self.crop_h = -1
        self.crop_w = -1
        self.crop_top = -1
        self.crop_left = -1
        self.top = -1
        self.left = -1
        self.border_size = -1
        self.data: list[list[Optional[Box]]] = []

    def get_solution(self):
        self.set_crop_image()
        self.crop_h, self.crop_w = self.crop_image.shape[:2]
        self.set_box_list()
        self.set_g_points()
        self.set_box_color(self.box_list)
        h = max(
            box.g_point[0]
            for box in self.box_list
        ) + 1
        self.data = [
            [None] * h
            for _ in range(h)
        ]
        for box in self.box_list:
            self.data[box.g_point[0]][box.g_point[1]] = box
        block_rectangle_list = []
        graph = Graph(h, h, block_rectangle_list)
        for a in range(h):
            for b in range(h):
                graph.set_value((a, b), Value.block)
        for box in self.box_list:
            graph.set_value(box.g_point, Value.empty)
        point_pair_list = self.find_point_pair(self.box_list)
        ls = LinkSolution(graph, point_pair_list)
        box1 = self.box_list[0]
        box2 = self.box_list[1]
        center_h_diff = box1.ch - box2.ch
        center_g_h_diff = box1.g_point[0] - box2.g_point[0]
        self.border_size = center_h_diff / center_g_h_diff
        self.crop_top = int(box1.ch - self.border_size * box1.g_point[0])
        self.crop_left = int(box1.cw - self.border_size * box1.g_point[1])
        self.top = self.crop_top + self.min_h
        self.left = self.crop_left + self.min_w
        return ls

    def draw_path(self, path, color):
        box1 = self.data[path[0][0]][path[0][1]]
        box2 = self.data[path[1][0]][path[1][1]]
        self.draw_lin(box1, box2, color)

    def get_draw_point_by_unit(self, point: tuple[int, int]) -> tuple[int, int]:
        return get_draw_point_by_unit(point, self.top, self.left, self.border_size)

    def draw_lin(self, box1, box2, color):
        cv2.line(
            self.image_draw,
            self.get_draw_point_by_unit(box1.g_point),
            self.get_draw_point_by_unit(box2.g_point),
            color,
            int(self.border_size / 8),
        )

    def set_box_color(self, box_list: list[Box]):
        for box in box_list:
            color = self.crop_image[int(box.ch), int(box.cw)]
            box.color = int(color[0]), int(color[1]), int(color[2])

    @staticmethod
    def find_point_pair(box_list: list[Box]):
        box_set = set(
            box
            for box in box_list
            if box.c == ClassEnum.point
        )
        point_pair_list = []
        while box_set:
            box = box_set.pop()
            bl = list(box_set)
            bl.sort(key=lambda b: color_diff(box.color, b.color))
            point_pair_list.append((box.g_point, bl[0].g_point))
            box_set.remove(bl[0])
        return point_pair_list

    def set_box_list(self):
        box_list = self.get_box(self.crop_image)
        box_list = [
            box
            for box in box_list
            if box.c is not ClassEnum.region
        ]
        self.box_list = box_list

    def set_g_points(self):
        box_list = self.box_list
        h_attr_index_map = self.get_attr_index_map(box_list, 'h')
        w_attr_index_map = self.get_attr_index_map(box_list, 'w')
        h_len = max(h_attr_index_map.values()) + 1
        w_len = max(w_attr_index_map.values()) + 1
        assert h_len == w_len, f'{h_len} != {w_len}'
        for box in box_list:
            box.g_point = (h_attr_index_map[box], w_attr_index_map[box])

    @staticmethod
    def get_attr_index_map(box_list: list[Box], attr: str) -> dict[Box, int]:
        box_set = set(box_list)
        head_box_to_box_list = defaultdict(list)
        while box_set:
            head_box = box_set.pop()
            head_box_to_box_list[head_box].append(head_box)
            bl = list(box_set)
            bl.sort(key=lambda b: abs(getattr(head_box, f'c{attr}') - getattr(b, f'c{attr}')))
            for box in bl:
                if has_intersection(head_box, box, attr):
                    head_box_to_box_list[head_box].append(box)
                    box_set.remove(box)
        head_box_list = list(head_box_to_box_list.keys())
        head_box_list.sort(key=lambda b: getattr(b, f'c{attr}'))
        attr_index_map = {}
        for i, head_box in enumerate(head_box_list):
            attr_index_map[head_box] = i
            for box in head_box_to_box_list[head_box]:
                attr_index_map[box] = i
        return attr_index_map

    def set_crop_image(self):
        box_list = self.get_box(self.ori_img)
        for box in box_list:
            self.max_h = max(self.max_h, box.ch + box.bh / 2)
            self.min_h = min(self.min_h, box.ch - box.bh / 2)
            self.max_w = max(self.max_w, box.cw + box.bw / 2)
            self.min_w = min(self.min_w, box.cw - box.bw / 2)
        self.crop_image = self.ori_img[int(self.min_h):int(self.max_h), int(self.min_w):int(self.max_w)]

    def get_box(self, img: np.ndarray):
        box_list = []
        res = self.yolo_model(img)[0]
        for box in res:
            this_cls = ClassEnum(int(box.boxes.cls))
            cw, ch, bw, bh = box.boxes.xywh.numpy()[0]
            box_list.append(Box(
                c=this_cls,
                ch=ch,
                cw=cw,
                bh=bh,
                bw=bw,
            ))
        return box_list
