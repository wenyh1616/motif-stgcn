import numpy as np
import net.utils.tools as tools
class Graph():
    """ The Graph to model the skeletons extracted by the openpose

    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - dastance: Distance Partitioning
        - spatial: Spatial Configuration
        For more information, please refer to the section 'Partition Strategies'
            in our paper (https://arxiv.org/abs/1801.07455).

        layout (string): must be one of the follow candidates
        - openpose: Is consists of 18 joints. For more information, please
            refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose#output
        - ntu-rgb+d: Is consists of 25 joints. For more information, please
            refer to https://github.com/shahroudy/NTURGB-D

        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points

    """

    def __init__(self,
                 layout='openpose',
                 strategy='spatial'):
        self.get_edge(layout)
        self.get_adjacency(strategy)


    def __str__(self):
        return self.A, self.A_adj

    def get_edge(self, layout):
        if layout == 'openpose':
            self.num_node = 18
            self.self_link = [(i, i) for i in range(self.num_node)]
            self.inward = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12,
                                                                        11),
                             (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                             (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]
            self.outward = [(j, i) for (i, j) in self.inward]
            self.neighbor_link = self.inward + self.outward
            self.edge = self.self_link + self.neighbor_link
        elif layout == 'ntu-rgb+d':
            self.num_node = 25
            self.self_link = [(i, i) for i in range(self.num_node)]
            inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                              (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                              (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                              (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                              (22, 23), (23, 8), (24, 25), (25, 12)]
            self.inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
            self.outward = [(j, i) for (i, j) in self.inward]
            self.neighbor_link = self.inward + self.outward
            self.edge = self.self_link + self.neighbor_link
        else:
            raise ValueError("Do Not Exist This Layout.")

    def get_adjacency(self, strategy):
        if strategy == 'spatial':
            A = tools.get_spatial_graph(self.num_node, self.self_link, self.inward, self.outward)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")
