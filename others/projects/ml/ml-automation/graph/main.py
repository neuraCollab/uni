import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from networkx.algorithms.flow import max_flow_min_cost
from scipy.sparse.csgraph import csgraph_from_dense


n, m, q = list(map(int, input().split()))

arr = np.array([])
sub_graph = []
G = nx.DiGraph()
for _ in range(m):
    u, v, c, w, a = list(map(int, input().split()))
    G.add_edge(u, v, capacity=c, w=w, a=a)
    # sub_graph.append([u, v, c]) 
    
# G.add_edges_from[sub_graph]
for _ in range(q):
    s, d, f = list(map(int, input().split()))



    print(results)
    

pos = nx.spring_layout(G, seed = 7)
nx.draw_networkx_edges(G, pos)
nx.draw(G=G)
print(G)
plt.savefig("./grapg.png")