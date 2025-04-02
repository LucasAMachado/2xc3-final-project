import unittest
import csv
import time 
import random
import matplotlib.pyplot as plt
import os
from graph_utils import Graph
from math import radians, sin, cos, atan2, sqrt
from shortest_path_algorithms import dijkstra
from A_star_algorithm import A_star


class TestSetting(unittest.TestCase):

    # all attributes that subclasses will use
    def setUp(self):
        # read 2 files and transfer relevant string figures into integers
        # nodes
        self.stations = {}                                                                                  # key is id of stations, values are (latitude, Longitude)       
        with open("london_stations.csv", newline='', encoding="utf-8") as file1:
            reader1 = csv.DictReader(file1)
            for row1 in reader1:
                station_id = int(row1["id"])
                latitude = float(row1["latitude"])
                longitude = float(row1["longitude"])

                # integrate position info of each station into coordinate tuples
                self.stations[station_id] = (latitude, longitude)

        # edges
        self.graph = Graph()
        self.edge_to_line = {}                                                                              # record line info related by edge between source and destination (in both direction)
        with open("london_connections.csv", newline='', encoding="utf-8") as file2:
            reader2 = csv.DictReader(file2)
            for row2 in reader2:
                source_id = int(row2["station1"])
                destination_id = int(row2["station2"])
                weight = self.haversine_distance(self.stations[source_id], self.stations[destination_id])   # calculate the edge weight from a node to another beside
                line_info = int(row2["line"])

                self.graph.add_edge(source_id, destination_id, weight, undirected=True)
                self.edge_to_line[(source_id, destination_id)] = line_info                                  
                self.edge_to_line[(destination_id, source_id)] = line_info

        # part 5.2, line test in three scenarios                
        self.same_line_pair = (49, 87)                                               # only need to take 1 line
        self.adjacent_line_pair = (28, 197)                                          # from line A to line B, need to take 2 lines
        self.several_transfers_pair = (40, 50)                                       # need to take 3 lines above

        # get all names of stations and store them into a list
        self.station_info_list = list(self.stations.keys())

    # all functions that subclasses will use
    """Since the Earth is a sphere and cannot be represented as a simple 2D grid, 
    we need to use the Haversine formula to calculate the actual distance between two points on the sphere."""

    def haversine_distance(self, coord1, coord2):
            latitude1, longitude1 = coord1
            latitude2, longitude2 = coord2
            R = 6371
            delta_lat = radians(latitude2 - latitude1)
            delta_lon = radians(longitude2 - longitude1)
            r_lat1 = radians(latitude1)
            r_lat2 = radians(latitude2)
            a = sin(delta_lat / 2)**2 + cos(r_lat1) * cos(r_lat2) * sin(delta_lon / 2)**2
            c = 2 * atan2(sqrt(a), sqrt(1 - a))
            return R * c
        
    # heuristic function based on haversine_distance
    # calculate the estimate remaining cost between current node and destination node
    def heuristic(self, station_a, station_b, stations):
        coord_info_a = stations[station_a]
        coord_info_b = stations[station_b]
        return self.haversine_distance(coord_info_a, coord_info_b)
    
        # part 5.2
    def compute_lines_used(self, shortest_path, edge_to_line):
        lines_used = set()
        for node_idx in range(len(shortest_path) - 1):                                  # from the source node to the node before the destination
            edge = (shortest_path[node_idx], shortest_path[node_idx + 1])
            if edge in edge_to_line:                                                                     
                lines_used.add(edge_to_line[edge])                                      # edge_to_line[edge] get relevant line info
        return lines_used

        
# inheritance test p5.1
class CompareDijAndA(TestSetting):
    def test_performance_compare(self):                                              # whole instance of class CompareDijAndA
        station_pair_num = 10
        generated_random_pairs = []

        # generate random (source, destination) pairs for each trial
        # while-loop ensure the number of tuple pairs is equal to number of trials after eliminating duplicate pairs
        while len(generated_random_pairs) < station_pair_num:
            source = random.choice(self.station_info_list)
            destination = random.choice(self.station_info_list)

            # avoid duplicate info
            # case 1: picking same nodes
            while destination == source:                                            # if statement doesn't work if function choose the same node as the another again
                destination = random.choice(self.station_info_list)                 # use a while loop to keep selecting  until destination != source
            
            # case 2: picking same tuple pairs
            check_pair = (source, destination)
            if check_pair not in generated_random_pairs:
                generated_random_pairs.append(check_pair)

        dijkstra_times = []
        A_star_times = []

        for (source, destination) in generated_random_pairs:

            # Dijkstra, dist_dij stores the distance between source node and all reachable nodes 
            start_1 = time.perf_counter()
            dist_dij, shortest_path_dij = dijkstra(self.graph, source, (self.graph.get_num_nodes() - 2))
            end_1 = time.perf_counter()
            dijkstra_times.append(end_1 - start_1)

            # A*
            start_2 = time.perf_counter()
            came_from, shortest_path_A = A_star(self.graph, source, destination, lambda u, v: self.heuristic(u, v, self.stations))
            end_2 = time.perf_counter()
            A_star_times.append(end_2 - start_2)

            # compare whether shortest path generated by 2 algorithms is the same

            # case 1: destination node is not reachable from source node in Dijkstra
            if destination not in dist_dij:
                self.fail(f"Dijkstra cannot reach {destination} (source = {source})")
            # case 2: A* algorithm cannot find any path from source node to destination node
            if not shortest_path_A:
                self.fail(f"A* cannot reach {destination} (source = {source})")

            # compute A* distance from source to destination: sum haversine_distance(stations[u], stations[v]) for each edge in path_A
            dist_A = 0.0
            for i in range(len(shortest_path_A) - 1):                               # iterate to len(LIST) - 1 since we need to access the last element of the list by [i + 1] 
                u = shortest_path_A[i]                                              # current node u, coord(a, b)
                v = shortest_path_A[i + 1]                                          # next node v, coord(b, c)
                dist_A += self.haversine_distance(self.stations[u], self.stations[v])


            # check whether same shortest path
            self.assertAlmostEqual(dist_dij[destination], dist_A, places=5, msg=f"Paths differ for {source} -> {destination}")

            # calculate average time
            avg_dijkstra_time = sum(dijkstra_times) / len(dijkstra_times)
            avg_A_star_time = sum(A_star_times) / len(A_star_times)
            print(f"\nDijkstra average time over {station_pair_num} runs: {avg_dijkstra_time:.6f} s")
            print(f"A*       average time over {station_pair_num} runs: {avg_A_star_time:.6f} s")

        draw_plot(dijkstra_times, A_star_times, station_pair_num)


# plot drawing and saving function
def draw_plot(dijkstra_times, a_star_times, station_pair_num):
    results_dir = "experiment_results"
    os.makedirs(results_dir, exist_ok=True)

    x_values = range(1, station_pair_num + 1)
    plt.figure()
    plt.plot(x_values, dijkstra_times, label="Dijkstra")
    plt.plot(x_values, a_star_times, label="A*")
    plt.legend()
    plt.xlabel("Trial Index")
    plt.ylabel("Time (seconds)")
    plt.title("A_star_vs_Dijkstra")
    plt.savefig(f"{results_dir}/A_star_vs_Dijkstra.png")
    print(f"Plot saved to {results_dir}/A_star_vs_Dijkstra.png")
    plt.close()


# inheritance test p5.2
class PerformanceLineUsage(TestSetting):
    
    def test_same_line(self):
        source, destination = self.same_line_pair
        _, shortest_path = A_star(                                                  # only need shortest path, no need the predecessor dictionary (came_from)
            self.graph, source, destination, 
            lambda u, v: self.haversine_distance(self.stations[u], self.stations[v])
        )
        lines = self.compute_lines_used(shortest_path, self.edge_to_line)
        print(f"Path from {source} to {destination}: {shortest_path}")
        print(f"Lines used: {lines}")
        
        self.assertEqual(len(lines), 1, "Stations on the same line should use only one line.")

    def test_adjacent_lines(self):
        source, destination = self.adjacent_line_pair
        _, shortest_path = A_star(
            self.graph, source, destination, 
            lambda u, v: self.haversine_distance(self.stations[u], self.stations[v])
        )
        lines = self.compute_lines_used(shortest_path, self.edge_to_line)
        print(f"Path from {source} to {destination}: {shortest_path}")
        print(f"Lines used: {lines}")
        
        self.assertEqual(len(lines), 2, "Stations on adjacent lines should use two lines.")

    def test_several_transfers(self):
        source, destination = self.several_transfers_pair
        _, shortest_path = A_star(
            self.graph, source, destination, 
            lambda u, v: self.haversine_distance(self.stations[u], self.stations[v])
        )
        lines = self.compute_lines_used(shortest_path, self.edge_to_line)
        print(f"Path from {source} to {destination}: {shortest_path}")
        print(f"Lines used: {lines}")
        
        self.assertGreaterEqual(len(lines), 3, "Stations requiring several transfers should use at least three lines.")


if __name__ == "__main__":
    unittest.main()