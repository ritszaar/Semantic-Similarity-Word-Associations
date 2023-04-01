import queue

class Node:
    def __init__(self, id, type, content_id):
        self.id = id
        self.type = type
        self.content_id = content_id

global n, m, g, vis, par, dist

def dijkstra(sc):
    vis  = [False for i in range(n + 1)]
    par = [-1 for i in range(n + 1)]
    dist = [1e9 for i in range(n + 1)]

    dist[sc] = 0
    pq = queue.PriorityQueue()
    pq.put((dist[sc], sc))

    while not pq.empty():
        u = pq.get()[1]
        vis[u] = True
        for (v, w) in g[u]:
            if not vis[v]:
                if dist[v] > dist[u] + w:
                    par[v] = u
                    dist[v] = dist[u] + w
                    pq.put((dist[v], v))

    for i in range(1, n + 1):
        print(i)
        path = []
        u = i
        while u != -1:
            path.append(u)
            u = par[u]
        path.reverse()

        print("Distance: {}".format(dist[i]))
        print("Path    : ", end='')
        for j in range(len(path) - 1):
            print("{} --> ".format(path[j]), end='')
        print(i, end="\n\n")

with open("input.txt", "r") as f:
    line = f.readline().split(" ")
    n, m = int(line[0]), int(line[1])
    g = [[] for i in range(n + 1)]
    for i in range(m):
        line = f.readline().split(" ")
        u, v, w = int(line[0]), int(line[1]), float(line[2])
        g[u].append((v, w))
        g[v].append((u, w))

dijkstra(1)
