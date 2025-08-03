import time

from prettytable import PrettyTable


def manhattan_distance(point1, point2):
    """计算曼哈顿距离"""
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])


class Value:
    """填充值枚举"""
    block = '#'
    empty = ' '


class UnionFind:
    """并查集，用于校验联通情况"""

    def __init__(self):
        self.parent = {}

    def find(self, u):
        """查找集合的根节点"""
        if u not in self.parent:
            self.parent[u] = u
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]

    def union(self, u, v):
        """合并两个集合"""
        root_u = self.find(u)
        root_v = self.find(v)
        if root_u != root_v:
            self.parent[root_u] = root_v


class PointPair:
    """一组首尾点对，和他的值"""

    def __init__(self, start_point: tuple[int, int], end_point: tuple[int, int], value: int):
        self.start_point = start_point
        self.end_point = end_point
        self.value = value

    def __iter__(self):
        return iter((self.start_point, self.end_point))

    def __str__(self):
        return f'{self.value}-{self.start_point}-{self.end_point}'


class Graph:
    """地图信息"""

    def __init__(self, h, w, block_rectangle_list: list[tuple[tuple[int, int], tuple[int, int]]]):
        self.h = h
        self.w = w
        self.block_rectangle_list = block_rectangle_list
        self.data = [
            [Value.empty] * self.w
            for _ in range(self.h)
        ]
        # 设置不可走的点
        for block_rectangle in self.block_rectangle_list:
            for i in range(block_rectangle[0][0], block_rectangle[1][0] + 1):
                for j in range(block_rectangle[0][1], block_rectangle[1][1] + 1):
                    self.set_value((i, j), Value.block)

    def __str__(self):
        """打印表格用"""
        tb = PrettyTable()
        tb.header = False
        tb.hrules = True
        tb.add_rows(self.data)
        return str(tb)

    def iter_non_block_point(self):
        """迭代非block点"""
        for i in range(self.h):
            for j in range(self.w):
                point = i, j
                if self.get_value(point) != Value.block:
                    yield point

    def get_empty_out_len(self, point_list):
        """查找这组连周围空点的数量"""
        empty_set = set()
        for point in point_list:
            for e in self.get_empty_neighbor_point_list(point):
                empty_set.add(e)
        return len(empty_set)

    def get_func_list(self):
        """获取邻居的方法"""
        return [
            self.get_up_point,
            self.get_down_point,
            self.get_left_point,
            self.get_right_point,
        ]

    def get_neighbor_point_list(self, point):
        """获取邻居点"""
        neighbor_point_list = []
        for func in self.get_func_list():
            next_point = func(point)
            if next_point is not None:
                neighbor_point_list.append(next_point)
        return neighbor_point_list

    def get_empty_neighbor_point_list(self, point):
        """获取邻居中的空点"""
        neighbor_point_list = self.get_neighbor_point_list(point)
        empty_neighbor_point_list = []
        for neighbor_point in neighbor_point_list:
            if self.get_value(neighbor_point) == Value.empty:
                empty_neighbor_point_list.append(neighbor_point)
        return empty_neighbor_point_list

    def get_non_block_neighbor_point_list(self, point):
        neighbor_point_list = self.get_neighbor_point_list(point)
        non_block_neighbor_point_list = []
        for neighbor_point in neighbor_point_list:
            if self.get_value(neighbor_point) != Value.block:
                non_block_neighbor_point_list.append(neighbor_point)
        return non_block_neighbor_point_list

    def get_union_find(self) -> UnionFind:
        """获取当前空点组成的并查集，用于联通性测试"""
        uf = UnionFind()
        for i in range(self.h):
            for j in range(self.w):
                point = i, j
                value = self.get_value(point)
                if value == Value.empty:
                    for next_point in self.get_empty_neighbor_point_list(point):
                        uf.union(point, next_point)
        return uf

    def set_value(self, point, value):
        """设置点值"""
        self.data[point[0]][point[1]] = value

    def get_value(self, point):
        """获取点值"""
        return self.data[point[0]][point[1]]

    def get_up_point(self, point):
        """获取上方的点，没有则返回空"""
        if point[0] - 1 >= 0:
            return point[0] - 1, point[1]

    def get_down_point(self, point):
        """获取下方的点，没有则返回空"""
        if point[0] + 1 < self.h:
            return point[0] + 1, point[1]

    def get_left_point(self, point):
        """获取左方的点，没有则返回空"""
        if point[1] - 1 >= 0:
            return point[0], point[1] - 1

    def get_right_point(self, point):
        """获取右方的点，没有则返回空"""
        if point[1] + 1 < self.w:
            return point[0], point[1] + 1

    def no_empty(self) -> bool:
        """校验现在是否有空值"""
        for row in self.data:
            for value in row:
                if value == Value.empty:
                    return False
        return True


class LinkSolution:
    """链接问题解决方案"""

    def __init__(self, graph: Graph, point_pair_list: list[tuple[tuple[int, int], tuple[int, int]]]):
        self.graph = graph  # 地图信息
        point_set = set()
        # 校验数据合法性
        for point_pair in point_pair_list:
            for point in point_pair:
                if point in point_set:
                    raise ValueError(f'点对列表中存在重复点，{point}')
                point_set.add(point)
        self.point_pair_list = [
            PointPair(point1, point2, i)
            for i, (point1, point2) in enumerate(point_pair_list)
        ]  # 点对列表
        self.n = len(self.point_pair_list)  # 点对数量
        # 初始化地图，将点对值填入
        for i in range(self.n):
            point_pair = self.point_pair_list[i]
            for point in point_pair:
                self.graph.set_value(point, i)
        # 周围空格子多的点优先，这是一条更快到达解的策略
        self.point_pair_list.sort(key=lambda p: self.graph.get_empty_out_len(list(p)))

        self.i_path = [
            []
            for _ in range(self.n)
        ]

    def __str__(self):
        return str(self.graph)

    def solution(self):
        """解决方案"""
        flag = self.dfs(0, self.point_pair_list[0].start_point)
        if flag:
            for i in range(self.n):
                self.i_path[i].append(self.point_pair_list[i].end_point)
        return flag

    def get_neighbor_parent_set(self, point, uf: UnionFind):
        neighbor_parent_set = set()
        for next_point in self.graph.get_empty_neighbor_point_list(point):
            neighbor_parent_set.add(uf.find(next_point))
        return neighbor_parent_set

    def connect_check(self, i):
        uf = self.graph.get_union_find()
        """连通性校验"""
        for j in range(i, self.n):
            start_point = self.point_pair_list[j].start_point
            end_point = self.point_pair_list[j].end_point
            start_neighbor_parent_set = self.get_neighbor_parent_set(start_point, uf)
            end_neighbor_parent_set = self.get_neighbor_parent_set(end_point, uf)
            # 如果空邻居有相同的，说明联通，如果没有相同的，说明不联通
            if not start_neighbor_parent_set.intersection(end_neighbor_parent_set):
                return False
        return True

    def dfs(self, i, point, ) -> bool:
        """深度优先搜索，i表示正在解决第几组点，point表示上次选的点"""
        self.i_path[i].append(point)

        def return_false():
            self.i_path[i].pop()
            return False

        point_pair = self.point_pair_list[i]
        # 每次进行联通性校验，这个操作必须每次做，因为他时间复杂度小，剪枝效果好
        if not self.connect_check(i + 1):
            return return_false()
        # 最后一个点
        end_point = point_pair.end_point
        neighbor_point_list = self.graph.get_neighbor_point_list(point)
        # 按照曼哈顿距离排序，比较近地优先访问，这是一条更快到达解的策略
        neighbor_point_list.sort(key=lambda x: manhattan_distance(x, end_point))
        for next_point in neighbor_point_list:
            # 如果是最后一个点
            if next_point == end_point:
                # 如果是最后一组点
                if i == self.n - 1:
                    # 校验是否完成
                    rt = self.graph.no_empty()
                    if rt:
                        return rt
                else:
                    # 如果不是最后一组点，则进行下一组
                    rt = self.dfs(i + 1, self.point_pair_list[i + 1].start_point)
                    if rt:
                        return rt
                # 最后一个点，不需要做下面的操作
                continue
            # 拿到上面点的值
            next_value = self.graph.get_value(next_point)
            # 如果上面的点为空，则进行赋值操作
            if next_value == Value.empty:
                self.graph.set_value(next_point, point_pair.value)
                # 返回这个选择是否可行
                rt = self.dfs(i, next_point)
                if rt:
                    # 可行则返回可行
                    return rt
                else:
                    # 不可行则将上面点还原
                    self.graph.set_value(next_point, Value.empty)
        return return_false()


s1 = LinkSolution(
    Graph(
        9,
        9,
        [],
    ),
    [
        ((0, 0), (7, 7)),
        ((0, 1), (1, 4)),
        ((0, 8), (1, 5)),
        ((1, 6), (3, 6)),
        ((1, 7), (4, 7)),
        ((1, 8), (5, 7)),
        ((2, 2), (8, 0)),
        ((2, 5), (8, 5)),
        ((4, 2), (7, 4)),
        ((6, 4), (8, 4)),
    ],
)
s2 = LinkSolution(
    Graph(
        8,
        8,
        [],
    ),
    [
        ((0, 0), (2, 2)),
        ((0, 1), (1, 3)),
        ((0, 7), (1, 4)),
        ((1, 2), (4, 1)),
        ((2, 4), (6, 0)),
        ((2, 6), (6, 6)),
        ((5, 6), (7, 0)),
        ((7, 5), (6, 7)),
    ],
)
s3 = LinkSolution(
    Graph(
        5,
        5,
        [],
    ),
    [
        ((0, 3), (1, 0)),
        ((0, 4), (4, 0)),
        ((2, 2), (4, 2)),
        ((3, 3), (4, 1)),
    ],
)

s_list = [s1, s2, s3]


def main():
    t0 = time.time()
    for s in s_list:
        print(s.graph)
        f = s.solution()
        print(f)
        assert f
        print(s.graph)
    print(time.time() - t0)


if __name__ == '__main__':
    main()
