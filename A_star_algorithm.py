from locale import currency
from queue import PriorityQueue

from networkx import reconstruct_path

class Graph:
    def __init__(self):
        self.adj = {}

    def add_edge(self, u, v, w):                                                        # add edge between 2 nodes in directed graph
        if u not in self.adj:
            self.adj[u] = {}
        self.adj[u][v] = w

        if v not in self.adj:
            self.adj[v] = {}

    def neighbors(self, u):
        if u not in self.adj:                                                           # not in adjacent list yet
            return []
        return list(self.adj[u].keys())

    def cost(self, u, v):
        if u in self.adj and v in self.adj[u]:
            return self.adj[u][v]
        else:
            raise KeyError(f"No edge from {u} to {v}")
        

def A_star(graph, source, destination, heuristic):
    frontier = PriorityQueue()                                                          # min-heap to represent the boarder edge nodes
    frontier.put((0, source))                                                           # we need to sort the node by its priority (total cost)            

    came_from = {}
    cost_so_far = {}                        
    came_from[source] = None
    cost_so_far[source] = 0

    while not frontier.empty():                

        current_priority, current_node = frontier.get()                                 # return the first tuple in PQ which has the lowest cost, only node is useful

        if current_node == destination:
            break

        for next in graph.neighbors(current_node):                                      # up / down / left / right, 4 nodes beside the current one
            new_costSF = cost_so_far[current_node] + graph.cost(current_node, next)

            if next not in cost_so_far or new_costSF < cost_so_far[next]:               # next hasn't been visited (equivalent to does not have a so-far cost)
                cost_so_far[next] = new_costSF

                # calculate the total cost of nexts = so-far cost + estimate remaining cost
                total_priority = new_costSF + heuristic(next, destination)
                frontier.put((total_priority, next))
                came_from[next] = current_node

    shortest_path = self_reconstruct_path(source, destination, came_from)
    return came_from, shortest_path


def self_reconstruct_path(source, destination, came_from):
    # case that no path to destination
    if destination not in came_from:                                        
        return []
    
    path = []
    current = destination                                                               # start from destination, initializing
    
    while current is not None:                                                          # predecessor of source is None
        path.append(current)
        current = came_from[current]                                                    # predecessor of current node
    path.reverse()
    return path


def main():
    g = Graph()

    g.add_edge((0, 0), (0, 1), 1)
    g.add_edge((0, 0), (1, 0), 1)
    g.add_edge((0, 1), (0, 2), 1)
    g.add_edge((0, 1), (1, 1), 1)
    g.add_edge((1, 0), (1, 1), 1)
    g.add_edge((1, 0), (2, 0), 1)
    g.add_edge((0, 2), (1, 2), 1)
    g.add_edge((1, 1), (1, 2), 1)
    g.add_edge((1, 1), (2, 1), 1)
    g.add_edge((2, 0), (2, 1), 1)
    g.add_edge((1, 2), (2, 2), 1)
    g.add_edge((2, 1), (2, 2), 1)

    source = (0, 0)
    destination = (2, 2)

    # manhattan distance can avoid square root from Pythagorean theorem 
    def manhattan_distance(current_node, destination):
        x1, y1 = current_node
        x2, y2 = destination
        return abs(x2 - x1) + abs(y2 - y1)

   
    came_from, shortest_path = A_star(g, source, destination, manhattan_distance)

    # predecessor
    print("Came from:")
    for node, parent in came_from.items():
        print(f"{node}: {parent}")

    # shortest path
    print("\nShortest path from", source, "to", destination, ":")
    print(shortest_path)


# Run main() if the file is executed directly, otherwise not run main() when imported
# call the main()
if __name__ == '__main__':
    main()