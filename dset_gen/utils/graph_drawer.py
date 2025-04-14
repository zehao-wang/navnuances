import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import copy
# Function to normalize a value
def normalize(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)

class GraphDrawer(object):
    def __init__(self, graph, tar_height, tar_width, start_node=None) -> None:
        self.G = graph
        self.tar_height, self.tar_width = tar_height, tar_width

        poses = {n: (d["position"][0], d["position"][1])  for n, d in self.G.nodes.items()}

        # Separate the x and y coordinates
        x_values = [pos[0] for pos in poses.values()]
        y_values = [pos[1] for pos in poses.values()]

        # Find the min and max values for x and y
        min_x, max_x = min(x_values), max(x_values)
        min_y, max_y = min(y_values), max(y_values)
        self.min_x, self.max_x, self.min_y, self.max_y = min_x, max_x, min_y, max_y
        
        normalized_positions = {node: (normalize(pos[0], min_x, max_x), normalize(pos[1], min_y, max_y))
                        for node, pos in poses.items()}
        self.poses = poses
        self.scaled_positions = {node: (int(pos[0] * self.tar_width), int(pos[1] * self.tar_height)) for node, pos in normalized_positions.items()}

        DPI = 300
        plt.figure(figsize=(self.tar_width / DPI, self.tar_height / DPI), dpi=DPI)
        if start_node is not None:
            node_color = []
            for node in self.G:
                if node == start_node:
                    node_color.append("green")
                else:
                    node_color.append("gray")
            nx.draw(self.G, pos=self.scaled_positions, with_labels=False, edge_color='gray', node_color=node_color, node_size=1, width=0.5)
        else:
            nx.draw(self.G, pos=self.scaled_positions, with_labels=False, edge_color='gray', node_color='gray', node_size=1, width=0.5)
        x_min, x_max = plt.xlim()
        y_min, y_max = plt.ylim()

        self.x_min = x_min
        self.y_min = y_min
        self.x_scale = self.tar_width / (x_max-x_min)
        self.y_scale = self.tar_height / (y_max-y_min)

        plt.savefig('./tmp_obs.png')
        plt.close()

        self.img = Image.open('./tmp_obs.png').convert('RGB')
        self.drawer = ImageDraw.Draw(self.img)
    
    def dump_img(self, path):
        self.img.save(path)

    def add_edge(self, edge, color='red'):
        if len(edge) == 0:
            return
        assert len(edge) == 2

        p1 = self.scaled_positions[edge[0]]
        p2 = self.scaled_positions[edge[1]]
        p1 = ((p1[0]-self.x_min) * self.x_scale, ((self.tar_height - p1[1]) - self.y_min) * self.y_scale)
        p2 = ((p2[0]-self.x_min) * self.x_scale, ((self.tar_height - p2[1]) - self.y_min) * self.y_scale)

        self.drawer = ImageDraw.Draw(self.img)
        self.drawer.line([p1, p2], fill=color, width=2)
    
    def add_dot(self, dot_name, color = 'gray', r=2):
        p1 = self.scaled_positions[dot_name]
        p1 = ((p1[0]-self.x_min) * self.x_scale, ((self.tar_height - p1[1]) - self.y_min) * self.y_scale)
        self.drawer = ImageDraw.Draw(self.img)
        self.drawer.ellipse((p1[0]-r, p1[1]-r, p1[0]+r, p1[1]+r), fill=color)
 
    def add_square(self, square_corners, color='blue', ori_coord=True):
        """ if ori_coord is False, v0 and v1 should be scaled_positions """
        v0 = list(square_corners[0])
        v1 = list(square_corners[1])

        if ori_coord:
            v0[0] = int(normalize(v0[0], self.min_x, self.max_x) * self.tar_width)
            v0[1] = int(normalize(v0[1], self.min_y, self.max_y) * self.tar_height)
            v1[0] = int(normalize(v1[0], self.min_x, self.max_x) * self.tar_width)
            v1[1] = int(normalize(v1[1], self.min_y, self.max_y) * self.tar_height)
        
        v0 = ((v0[0]-self.x_min) * self.x_scale, ((self.tar_height - v0[1]) - self.y_min) * self.y_scale)
        v1 = ((v1[0]-self.x_min) * self.x_scale, ((self.tar_height - v1[1]) - self.y_min) * self.y_scale)

        self.drawer.rectangle([(v0[0], v0[1]), (v1[0], v1[1])], outline=color)