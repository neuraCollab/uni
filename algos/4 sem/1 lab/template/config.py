class GraphConfig:
    def __init__(
        self,
        arrow_shape=(16, 20, 6),
        edge_width=2,
        color_forward="black",
        color_backward="black",
        color_vertex="orange",
        color_vertex_outline="black",
        color_text_forward="blue",
        color_text_backward="red"
    ):
        self.arrow_shape = arrow_shape
        self.edge_width = edge_width
        self.color_forward = color_forward
        self.color_backward = color_backward
        self.color_vertex = color_vertex
        self.color_vertex_outline = color_vertex_outline
        self.color_text_forward = color_text_forward
        self.color_text_backward = color_text_backward
