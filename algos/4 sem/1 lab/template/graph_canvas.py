import tkinter as tk
import math
from config import GraphConfig
from utils import point_to_segment_dist

class GraphCanvas(tk.Canvas):
    def __init__(self, master=None, config=None, **kwargs):
        super().__init__(master, **kwargs)
        self.config = config if config else GraphConfig()
        self.vertices = []
        self.edges = []
        self.add_edge_mode = False
        self.first_vertex_index = None
        self.var_directed = None
        self.var_weighted = None
        self.entry_weight = None
        self.bind("<Button-1>", self.on_left_click)
        self.bind("<Button-3>", self.on_right_click)

    def set_add_edge_mode(self, enabled):
        self.add_edge_mode = enabled
        self.first_vertex_index = None

    def clear_graph(self):
        self.vertices.clear()
        self.edges.clear()
        self.redraw()

    def on_left_click(self, event):
        x, y = event.x, event.y
        if self.add_edge_mode:
            idx = self.find_nearest_vertex(x, y)
            if idx is not None:
                if self.first_vertex_index is None:
                    self.first_vertex_index = idx
                else:
                    second_vertex_index = idx
                    if second_vertex_index != self.first_vertex_index:
                        self.add_edge(self.first_vertex_index, second_vertex_index)
                    self.first_vertex_index = None
            self.redraw()
        else:
            name = str(len(self.vertices))
            self.vertices.append({"x": x, "y": y, "name": name})
            self.redraw()

    def on_right_click(self, event):
        x, y = event.x, event.y
        if self.add_edge_mode:
            edge_idx = self.find_nearest_edge(x, y, threshold=5)
            if edge_idx is not None:
                self.edges.pop(edge_idx)
                self.redraw()
        else:
            vertex_idx = self.find_nearest_vertex(x, y, threshold=10)
            if vertex_idx is not None:
                self.remove_vertex(vertex_idx)
                self.redraw()
            else:
                edge_idx = self.find_nearest_edge(x, y, threshold=5)
                if edge_idx is not None:
                    self.edges.pop(edge_idx)
                    self.redraw()

    def find_nearest_vertex(self, x, y, threshold=10):
        for i, v in enumerate(self.vertices):
            dx = v["x"] - x
            dy = v["y"] - y
            dist = math.hypot(dx, dy)
            if dist <= threshold:
                return i
        return None

    def find_nearest_edge(self, x, y, threshold=5):
        min_dist = math.inf
        nearest_idx = None
        for idx, (i, j, w) in enumerate(self.edges):
            v1 = self.vertices[i]
            v2 = self.vertices[j]
            dist = point_to_segment_dist(x, y, v1["x"], v1["y"], v2["x"], v2["y"])
            if dist < min_dist:
                min_dist = dist
                nearest_idx = idx
        if min_dist <= threshold:
            return nearest_idx
        return None

    def remove_vertex(self, idx):
        new_edges = []
        for (i, j, w) in self.edges:
            if i != idx and j != idx:
                new_edges.append((i, j, w))
        self.vertices.pop(idx)
        updated_edges = []
        for (i, j, w) in new_edges:
            new_i = i if i < idx else i - 1
            new_j = j if j < idx else j - 1
            updated_edges.append((new_i, new_j, w))
        self.edges = updated_edges

    def add_edge(self, i, j):
        if not self.edge_exists(i, j):
            weight = self.get_edge_weight(i, j)
            self.edges.append((i, j, weight))
            if self.var_directed and not self.var_directed.get():
                if not self.edge_exists(j, i):
                    self.edges.append((j, i, weight))

    def edge_exists(self, i, j):
        return any(e[0] == i and e[1] == j for e in self.edges)

    def get_edge_weight(self, i, j):
        if self.var_weighted and self.var_weighted.get():
            if self.entry_weight is not None:
                txt = self.entry_weight.get().strip()
                try:
                    return float(txt)
                except:
                    return self.calculate_distance(i, j)
            else:
                return self.calculate_distance(i, j)
        else:
            return self.calculate_distance(i, j)

    def calculate_distance(self, i, j):
        v1 = self.vertices[i]
        v2 = self.vertices[j]
        return math.hypot(v1["x"] - v2["x"], v1["y"] - v2["y"])

    def redraw(self):
        self.delete("all")
        edges_dict = {}
        for (i, j, w) in self.edges:
            edges_dict[(i, j)] = w
        drawn = set()
        for (i, j), w in edges_dict.items():
            if (i, j) in drawn:
                continue
            if (j, i) in edges_dict:
                w2 = edges_dict[(j, i)]
                self.draw_double_arrow(i, j, w, w2)
                drawn.add((i, j))
                drawn.add((j, i))
            else:
                self.draw_single_arrow(i, j, w)
                drawn.add((i, j))
        radius = 10
        for idx, v in enumerate(self.vertices):
            x, y = v["x"], v["y"]
            self.create_oval(
                x - radius, y - radius, x + radius, y + radius,
                fill=self.config.color_vertex,
                outline=self.config.color_vertex_outline,
                width=2
            )
            self.create_text(x + radius + 5, y, text=v["name"], anchor="w")

    def draw_single_arrow(self, i, j, weight):
        v1 = self.vertices[i]
        v2 = self.vertices[j]
        x1, y1 = v1["x"], v1["y"]
        x2, y2 = v2["x"], v2["y"]
        is_directed = (self.var_directed and self.var_directed.get())
        self.create_line(
            x1, y1, x2, y2,
            arrow=(tk.LAST if is_directed else None),
            arrowshape=self.config.arrow_shape,
            width=self.config.edge_width,
            fill=self.config.color_forward
        )
        if self.var_weighted and self.var_weighted.get():
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            self.create_text(
                mid_x, mid_y - 10,
                text=f"{weight:.1f}",
                fill=self.config.color_text_forward
            )

    def draw_double_arrow(self, i, j, w1, w2):
        v1 = self.vertices[i]
        v2 = self.vertices[j]
        x1, y1 = v1["x"], v1["y"]
        x2, y2 = v2["x"], v2["y"]
        is_directed = (self.var_directed and self.var_directed.get())
        self.create_line(
            x1, y1, x2, y2,
            arrow=(tk.LAST if is_directed else None),
            arrowshape=self.config.arrow_shape,
            width=self.config.edge_width,
            fill=self.config.color_forward
        )
        if self.var_weighted and self.var_weighted.get():
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            self.create_text(
                mid_x, mid_y - 10,
                text=f"{w1:.1f}",
                fill=self.config.color_text_forward
            )
        self.create_line(
            x2, y2, x1, y1,
            arrow=(tk.LAST if is_directed else None),
            arrowshape=self.config.arrow_shape,
            width=self.config.edge_width,
            fill=self.config.color_backward
        )
        if self.var_weighted and self.var_weighted.get():
            mid_x2 = (x2 + x1) / 2
            mid_y2 = (y2 + y1) / 2
            self.create_text(
                mid_x2, mid_y2 + 10,
                text=f"{w2:.1f}",
                fill=self.config.color_text_backward
            )
