import random

import cv2
import numpy as np
from typing import NamedTuple

from link_solution.link_solution import Value, s_list, LinkSolution, Graph


class DrawGraph:
    def __init__(
            self,
            link_solution: LinkSolution,
            base_size: int = 4,
            little_point_size_times: float = 2.5,
            block_size_times: float = 25.0,
            point_size_times: float = 10.0,
            top_times: float = 25.0,
            left_times: float = 25.0,
            right_times: float = 50.0,
            bottom_times: float = 25.0,
    ):
        self.link_solution = link_solution
        self.graph = self.link_solution.graph

        self.base_size = base_size
        self.line_width = self.base_size
        self.block_size = int(self.line_width * block_size_times)
        self.point_size = int(self.line_width * point_size_times)
        self.little_point_size = int(self.line_width * little_point_size_times)
        self.top = int(self.base_size * top_times)
        self.left = int(self.base_size * left_times)
        self.right = int(self.base_size * right_times)
        self.bottom = int(self.base_size * bottom_times)
        self.h = (self.graph.h - 1) * self.block_size + self.top + self.bottom
        self.w = (self.graph.w - 1) * self.block_size + self.left + self.right

        self.line_color = (255, 255, 255)
        self.image = np.zeros((self.h, self.w, 3), np.uint8)

    # 设置渐变背景
    def set_gradient_background(self):
        color1 = self.random_color()
        color2 = self.random_color()

        # 计算渐变比例数组，避免重复计算
        ratios = [i / (self.h - 1) for i in range(self.h)]

        for i in range(self.h):
            # 计算当前行的渐变颜色
            ratio = ratios[i]
            color = (
                int(color1[0] * (1 - ratio) + color2[0] * ratio),
                int(color1[1] * (1 - ratio) + color2[1] * ratio),
                int(color1[2] * (1 - ratio) + color2[2] * ratio),
            )

            # 更新整行的颜色
            self.image[i, :] = [color] * self.w

    @staticmethod
    def random_color():
        return (
            np.random.randint(0, 255),
            np.random.randint(0, 255),
            np.random.randint(0, 255),
        )

    @staticmethod
    def get_draw_point(h_w_point):
        return h_w_point[1], h_w_point[0]

    def get_draw_point_by_unit(self, h_w_unit_point):
        h_w_point = self.get_relative_point(h_w_unit_point)
        return self.get_draw_point(h_w_point)

    def draw_circle(
            self,
            h_w_unit_point,  # 就是graph数据结构里面的point
            point_size,
            color,
    ):
        cv2.circle(
            self.image,
            self.get_draw_point_by_unit(h_w_unit_point),
            point_size,
            color,
            -1,
        )

    def draw(self):
        # 画背景
        self.set_gradient_background()
        # 画框架
        for point in self.graph.iter_non_block_point():
            self.finish_point(point)
        # 画特殊点
        for pp in self.link_solution.point_pair_list:
            color = self.random_color()
            for point in pp:
                # 打底色
                self.draw_circle(
                    point,
                    self.point_size,
                    self.line_color,
                )
                # 画填充色
                self.draw_circle(
                    point,
                    self.point_size - self.line_width,
                    color,
                )
        return self.image

    def finish_point(self, point):
        self.draw_circle(
            point,
            self.little_point_size,
            self.line_color,
        )
        for n_point in self.graph.get_non_block_neighbor_point_list(point):
            self.draw_line(point, n_point)

    def get_relative_point(self, point):
        h = self.top + point[0] * self.block_size
        w = self.left + point[1] * self.block_size
        return h, w

    def draw_line(self, point, n_point):
        cv2.line(
            self.image,
            self.get_draw_point_by_unit(point),
            self.get_draw_point_by_unit(n_point),
            self.line_color,
            self.line_width,
        )


class YoloLabelBox(NamedTuple):
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


class GenLabel:
    point_class_id = 1
    empty_class_id = 0

    def __init__(self, dg: DrawGraph):
        self.dg = dg
        self.graph = self.dg.graph
        self.data: list[list[YoloLabelBox]] = [
            [None] * self.graph.w
            for _ in range(self.dg.graph.h)
        ]

    def fill_data(self):
        for point in self.graph.iter_non_block_point():
            ph, py = point
            value = self.graph.get_value(point)
            h, w = self.dg.get_relative_point(point)
            if value == Value.empty:
                class_id = self.empty_class_id
                width = self.dg.little_point_size * 2
                height = self.dg.little_point_size * 2
            else:
                class_id = self.point_class_id
                width = self.dg.point_size * 2
                height = self.dg.point_size * 2
            self.data[ph][py] = YoloLabelBox(
                class_id,
                w / self.dg.w,
                h / self.dg.h,
                width / self.dg.w,
                height / self.dg.h,
            )

    def get_str(self):
        self.fill_data()
        return '\n'.join(
            ' '.join(map(str, self.data[ph][pw]))
            for ph, pw in self.graph.iter_non_block_point()
        )


def get_random_sample():
    graph_has_block_rate = 0.2  # 图像出现block的概率
    block_rate = 0.1  # block图像中，块block的概率
    point_rate = 0.2

    h = random.randint(4, 13)  # 随机高
    w = random.randint(4, 13)  # 随机宽
    block_rectangle_list = []
    point_list = []
    if random.random() < graph_has_block_rate:
        for i in range(h):
            for j in range(w):
                if random.random() < block_rate:
                    block_rectangle_list.append(((i, j), (i, j)))
    for i in range(h):
        for j in range(w):
            if random.random() < point_rate:
                point_list.append((i, j))
    if len(point_list) % 2 == 1:
        point_list.pop()
    random.shuffle(point_list)
    point_pair_list = []
    while point_list:
        point_pair_list.append((
            point_list.pop(),
            point_list.pop()
        ))
    graph = Graph(h, w, block_rectangle_list)
    ls = LinkSolution(graph, point_pair_list)
    dg = DrawGraph(
        ls,
        base_size=random.randint(4, 6),
        little_point_size_times=random.uniform(2.0, 3.0),
        block_size_times=random.uniform(24.0, 28.0),
        point_size_times=random.uniform(10.0, 13.0),
        top_times=random.uniform(28.0, 50.0),
        left_times=random.uniform(28.0, 50.0),
        right_times=random.uniform(28.0, 50.0),
        bottom_times=random.uniform(28.0, 50.0),
    )
    gl = GenLabel(dg)
    img = dg.draw()
    txt = gl.get_str()
    return img, txt


def main():
    for i in range(10):
        print(i)
        img, txt = get_random_sample()
        cv2.imwrite(f"t/test_{i:0>3}.jpg", img)
        with open(f"t/test_{i:0>3}.txt", "w") as f:
            f.write(txt)
