import matplotlib.pyplot as plt
import networkx as nx
from MCTS.node import Node

class Drawer:
    
    def __init__ (self):
        pass

    def build_tree_graph(self, node: Node, graph=None, parent_id=None, depth=2, max_nodes=100):
        if graph is None:
            graph = nx.DiGraph()

        if depth == 0 or len(graph) > max_nodes:
            return graph

        node_id = id(node)
        label = f"M:{node.state.last_move[1] if node.state.last_move else '?'}\nV:{node.visits}\nR:{round(node.reward, 1)}"
        graph.add_node(node_id, label=label)

        if parent_id is not None:
            graph.add_edge(parent_id, node_id)

        for child in node.children:
            self.build_tree_graph(child, graph, node_id, depth - 1, max_nodes)

        return graph


    def draw_tree(self, graph):
        pos = nx.spring_layout(graph, seed=42)
        labels = nx.get_node_attributes(graph, 'label')
        plt.figure(figsize=(12, 8))
        nx.draw(graph, pos, with_labels=True, labels=labels,
                node_color='lightblue', node_size=2000, font_size=8, font_weight='bold')
        plt.title("Monte Carlo Tree Search Tree")
        plt.axis('off')
        plt.show()



