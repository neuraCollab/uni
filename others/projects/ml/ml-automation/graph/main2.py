import copy
import math
edges = []
d_edges = {}

all_paths = []
def dfs(parent, destination, goods, path = [0, []]):   

    if(not parent in d_edges.keys() or parent == destination):
        if parent == destination:
            all_paths.append(path)
            return [True, goods]
        return [False, goods]
    else:
        cnt = 0
        for i in range(len(d_edges[parent])):
            n_path = list(copy.deepcopy(path))
            n_path[1].append(d_edges[parent][i][2])
            goods -= int(d_edges[parent][i][1])
            # goods = max(1, goods)
            n_path[0] += d_edges[parent][i][1]
            d_edges[parent][i][1] = 0
            b, goods = dfs(d_edges[parent][i][0], destination, goods, n_path)
            # goods = max(1, goods)
            if b:
                cnt += 1
        if(goods != 0):
            if cnt == 0:
                return [False, goods]
            # cost = math.floor(goods / cnt)
            # cost = max(1, cost)
            # for i in range(len(all_paths) - 1, len(all_paths) - cnt-1,-1):
            #     goods -= cost
            #     all_paths[i][0] += cost
            all_paths[0][0]+=goods
            goods-=goods
            if goods >= 0:
                all_paths[-1][0] += goods
            else:
                print("AHHHH!!!")
                
    return [False, goods]
    
    
n, m, q = list(map(int, input().split()))


inx = 1
for _ in range(m):
    u, v, c, w, a = list(map(int, input().split()))
    if(not u in d_edges.keys()):
        d_edges.update({u : [[v, c, inx]]})
    else: 
        d_edges[u].append([v, c, inx])

    inx += 1

for _ in range(q):
    s, d, f = list(map(int, input().split()))
    
    dfs(s, d, f)
    print(len(all_paths))
    for e in all_paths:
        print(f"{len(e[1])} {e[0]}", end = " ")
        print(*e[1])
    all_paths.clear()
    



