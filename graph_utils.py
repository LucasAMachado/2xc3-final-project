class Graph:
    def __init__(self):
        self.graph = {} 

    def add_node(self, node):
        if node not in self.graph:
            self.graph[node] = []

    def add_edge(self, source, destination, weight, undirected=False):      # default as directed graph
        # add nodes if they do not exist
        self.add_node(source)
        self.add_node(destination)

        # add the edge
        self.graph[source].append((destination, weight))

        # ensure covering double direction graph - London subway
        if undirected == True:
            self.graph[destination].append((source, weight))
        
    def get_neighbors(self, node):
        if node in self.graph:
            return self.graph[node]
        return []

    def get_nodes(self):
        return list(self.graph.keys())

    def get_num_nodes(self):
        return len(self.graph)

    def get_num_edges(self):
        count = 0
        for node in self.graph:
            count += len(self.graph[node])
        return count

    def get_edge_weight(self, source, destination):
        if source in self.graph:
            for neighbor, weight in self.graph[source]:
                if neighbor == destination:
                    return weight
        return None


def generate_random_graph(num_nodes, density, min_weight=1, max_weight=10):
    import random

    graph = Graph()

    # add all nodes
    for i in range(num_nodes):
        graph.add_node(i)

    # add random edges based on density
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and random.random() < density:
                weight = random.randint(min_weight, max_weight)
                graph.add_edge(i, j, weight)

    return graph
