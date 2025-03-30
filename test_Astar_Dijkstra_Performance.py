import unittest
import csv
from graph_utils import Graph
from math import radians, sin, cos, atan2, sqrt
from shortest_path_algorithms import dijkstra
from A_star_algorithm import A_star



class CompareDijAndA(unittest.TestCase):

    def setUp(self):

        # nodes
        self.stations = {}                                                           # key is id of stations, values are (latitude, Longitude)       
        with open("london_stations.csv", newline='', encoding="utf-8") as file1:
            reader1 = csv.DictReader(file1)
            for row1 in reader1:
                station_id = row1["id"]
                latitude = float(row1["latitude"])
                longitude = float(row1["longitude"])

                # integrate position info of each station into coordinate tuples
                self.stations[station_id] = (latitude, longitude)

        # edges
        self.graph = Graph()
        with open("london_connections.csv", newline='', encoding="utf-8") as file2:
            reader2 = csv.DictReader(file2)
            for row2 in reader2:
                source_id = row2["station1"]
                destination_id = row2["station2"]

                """Since the Earth is a sphere and cannot be represented as a simple 2D grid, 
                we need to use the Haversine formula to calculate the actual distance between two points on the sphere."""

                weight = haversine_distance(self.stations[source_id], self.stations[destination_id])
                self.graph.add_edge(source_id, destination_id, weight)
        

        def haversine_distance(coord1, coord2):
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
        
        def performance_compare():
            pass

