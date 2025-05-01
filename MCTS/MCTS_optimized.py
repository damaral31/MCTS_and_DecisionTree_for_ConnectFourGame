import math
import random
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple, Any, Dict

import utils.config as config

from Game.ConnectFour import ConnectFour
from MCTS.node import Node


def worker_mcts(state: ConnectFour, iterations: int, exploration: float) -> Dict[int, Tuple[float, int]]:
    """
    Each worker runs its own mini-MCTS rooted at the same state.
    Returns: {move: (total_reward, total_visits)}
    """
    root = Node(state.copy())

    def selection(node: Node, turn: int) -> Tuple[Node, int]:
        while not node.is_terminal():
            if not node.fully_explored():
                return expansion(node), -1 * turn
            node = best_child(node)
            turn *= -1
        return node, turn

    def expansion(node: Node) -> Node:
        for col in node.state.legal_moves():
            if col not in node.children_move:
                new_state = node.state.copy()
                new_state.play(col)
                node.add_child(new_state, col)
                break
        return node.children[-1]

    def simulation(state: ConnectFour, turn: int, max_depth: int = 20) -> float:
        state = state.copy()
        moves = 0
        while not state.is_over() and moves < max_depth:
            legal = state.legal_moves()
            if 3 in legal:
                state.play(3)
            else:
                state.play(random.choice(legal))
            turn *= -1
            moves += 1

        if state.is_over():
            return 1.0 if turn == -1 else -1.0
        return 0.0

    def backpropagation(node: Node, reward: float, turn: int) -> None:
        while node is not None:
            node.visits += 1
            node.reward -= turn * reward
            node = node.parent
            turn *= -1

    def best_child(node: Node) -> Node:
        best_score = -float("inf")
        best_node = None
        log_visits = math.log(node.visits + 1)
        for child in node.children:
            exploit = child.reward / (child.visits + 1e-8)
            explore = math.sqrt(log_visits / (child.visits + 1e-8))
            score = exploit + exploration * explore

            if score == best_score:
                if child.visits >= best_node.visits:
                    best_node = child
            elif score > best_score:
                best_score = score
                best_node = child
        return best_node

    for _ in range(iterations):
        node, turn = selection(root, -1)
        reward = simulation(node.state, turn)
        backpropagation(node, reward, turn)

    move_stats = {}
    for _, child in enumerate(root.children):
        move = child.state.last_move[1]
        move_stats[move] = (child.reward, child.visits)

    return move_stats


class MonteCarlo:

    def __init__(self, iteration: int = config.ITERATION, exploration: float = config.EXPLORATION, debug: bool = False):
        self.iteration = iteration
        self.exploration = exploration
        self.cpu_cores = max(1, os.cpu_count() or 1)

        
        if debug and not hasattr(self, '_debug_printed'):
            print(f"Using {self.cpu_cores} CPU cores for MCTS.")
            print(f"Iterations per worker: {self.iteration // self.cpu_cores}")
            print(f"Exploration factor: {self.exploration}")
            print(f"Total iterations: {self.iteration}")
            self._debug_printed = True

        

    def search(self, root: Node) -> tuple[Any, list[Any]]:
        iterations_per_worker = self.iteration // self.cpu_cores

        with ProcessPoolExecutor(max_workers=self.cpu_cores) as executor:
            futures = [executor.submit(worker_mcts, root.state, iterations_per_worker, self.exploration)
                       for _ in range(self.cpu_cores)]

            all_stats = [f.result() for f in futures]

        merged_stats: Dict[int, Tuple[float, int]] = {}

        for stat in all_stats:
            for move, (reward, visits) in stat.items():
                if move not in merged_stats:
                    merged_stats[move] = (reward, visits)
                else:
                    r, v = merged_stats[move]
                    merged_stats[move] = (r + reward, v + visits)

        for move, (reward, visits) in merged_stats.items():
            found = False
            for child in root.children:
                if child.state.last_move[1] == move:
                    child.reward += reward
                    child.visits += visits
                    found = True
                    break
            if not found:
                new_state = root.state.copy()
                new_state.play(move)
                new_child = Node(new_state, root)
                new_child.reward = reward
                new_child.visits = visits
                root.children.append(new_child)
                root.children_move.append(move)
            root.visits += visits

        prob = [child.visits / root.visits for child in root.children]

        ans = max(root.children, key=lambda c: c.visits)
        return ans.state.last_move[1], prob

    def best_child(self, node: Node) -> Node:
        best_score = -float("inf")
        best_children = None
        log_parent_visits = math.log(node.visits + 1)

        for child in node.children:
            exploitation = child.reward / (child.visits + 1e-8)
            exploration = math.sqrt(log_parent_visits / (child.visits + 1e-8))
            score = exploitation + self.exploration * exploration

            if score == best_score:
                if child.visits >= best_children.visits:
                    best_children = child
            elif score > best_score:
                best_score = score
                best_children = child

        return best_children
