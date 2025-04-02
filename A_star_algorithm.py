import unittest
from queue import PriorityQueue
from networkx import reconstruct_path
from graph_utils import Graph

def A_star(graph, source, destination, heuristic):
    frontier = PriorityQueue()                                                              # min-heap to represent the boarder edge nodes
    frontier.put((0, source))                                                               # we need to sort the node by its priority (total cost)            

    came_from = {}
    cost_so_far = {}                        
    came_from[source] = None
    cost_so_far[source] = 0

    while not frontier.empty():                

        current_priority, current_node = frontier.get()                                     # return the first tuple in PQ which has the lowest cost, only node is useful

        if current_node == destination:
            break

        # (u, v, weight)
        for neighbors, weight in graph.get_neighbors(current_node):                         # up / down / left / right, 4 nodes beside the current one
            new_costSF = cost_so_far[current_node] + weight

            if neighbors not in cost_so_far or new_costSF < cost_so_far[neighbors]:         # neighbor hasn't been visited (equivalent to does not have a so-far cost)
                cost_so_far[neighbors] = new_costSF
                total_priority = new_costSF + heuristic(neighbors, destination)             # calculate the total cost of nexts = so-far cost + estimate remaining cost
                frontier.put((total_priority, neighbors))
                came_from[neighbors] = current_node
            
    shortest_path = reconstruct_path(source, destination, came_from)
    # the return element of A* is restricted to 
    return came_from, shortest_path


def reconstruct_path(source, destination, came_from):
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


#  A start algorithm correctness test
# test case that use unittest  实例

class TestAStar(unittest.TestCase):
    def setUp(self):
        self.graph = Graph()
        self.graph.add_edge((0, 0), (0, 1), 1)
        self.graph.add_edge((0, 0), (1, 0), 1)
        self.graph.add_edge((0, 1), (0, 2), 1)
        self.graph.add_edge((0, 1), (1, 1), 1)
        self.graph.add_edge((1, 0), (1, 1), 1)
        self.graph.add_edge((1, 0), (2, 0), 1)
        self.graph.add_edge((0, 2), (1, 2), 1)
        self.graph.add_edge((1, 1), (1, 2), 1)
        self.graph.add_edge((1, 1), (2, 1), 1)
        self.graph.add_edge((2, 0), (2, 1), 1)
        self.graph.add_edge((1, 2), (2, 2), 1)
        self.graph.add_edge((2, 1), (2, 2), 1)
       
        # initialize attributes of instance TestAStar for source and destination nodes
        self.source = (0, 0)
        self.destination = (2, 2)

    def manhattan_distance(self, current, destination):
        x1, y1 = current
        x2, y2 = destination
        return (abs(x2 - x1) + abs(y2 - y1))
    
    def test_A_star_shortest_path(self):
        came_from, shortest_path = A_star(self.graph, self.source, self.destination, self.manhattan_distance)   # heuristic is a reference to the function manhattan_distance()   
        expected_path = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]

        self.assertEqual(shortest_path, expected_path)

        # predecessor
        print("Came from:")
        for node, parent in came_from.items():
            print(f"{node}: {parent}")

        # shortest path
        print("\nShortest path from", self.source, "to", self.destination, ":")
        print(shortest_path)

if __name__ == "__main__":
    # run the function if execute this file directly
    unittest.main()