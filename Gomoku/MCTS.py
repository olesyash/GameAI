from copy import deepcopy
import random
import math
import numpy as np

class MCTS:
    def __init__(self, exploration_weight=1.4):
        self.exploration_weight = exploration_weight

    def search(self, initial_state, iterations=1000):
        """Performs MCTS to find the best move."""
        root = Node(initial_state)
        
        # Always check for winning moves first
        moves = initial_state.legal_moves()
        current_player = initial_state.current_player
        
        # Check each move for an immediate win or a winning threat
        for move in moves:
            test_state = initial_state.clone()
            if not test_state.make_move(move):
                continue
                
            # Check for immediate win
            if test_state.status == current_player:
                return Node(test_state)
            
            # Check if this creates a winning threat
            for next_move in test_state.legal_moves():
                next_state = test_state.clone()
                if next_state.make_move(next_move):
                    if next_state.status == current_player:
                        return Node(test_state)  # Return the first move that leads to a forced win
        
        # Only check for blocks if we don't have a winning move
        opponent = -current_player
        for move in moves:
            row, col = move
            for dr, dc in [(0,1), (1,0), (1,1), (1,-1)]:
                count = 0
                # Count forward
                r, c = row + dr, col + dc
                while 0 <= r < initial_state.board_size and 0 <= c < initial_state.board_size and initial_state.board[r,c] == opponent:
                    count += 1
                    r += dr
                    c += dc
                
                # Count backward
                r, c = row - dr, col - dc
                while 0 <= r < initial_state.board_size and 0 <= c < initial_state.board_size and initial_state.board[r,c] == opponent:
                    count += 1
                    r -= dr
                    c -= dc
                
                if count >= 3:  # Only block if it's critical (4 in a row)
                    test_state = initial_state.clone()
                    test_state.make_move(move)
                    return Node(test_state)

        # If no immediate win or critical block, do regular MCTS
        for i in range(iterations):
            if i % 100 == 0:
                print(f"\nIteration {i}")
                if i > 0:
                    print("\nCurrent top moves:")
                    moves = [(child.state.last_move, child.visits, child.wins/child.visits if child.visits > 0 else 0)
                            for child in root.children]
                    moves.sort(key=lambda x: (x[1], x[2]), reverse=True)
                    for move, visits, win_rate in moves[:5]:
                        print(f"Move {move}: {visits} visits, win_rate={win_rate:.3f}")
            
            node = self._select(root)
            if not node.state.status:
                node = self._expand(node)
            reward = self._simulate(node)
            self._backpropagate(node, reward)
        
        print("\nFinal move statistics:")
        moves = [(child.state.last_move, child.visits, child.wins/child.visits if child.visits > 0 else 0)
                for child in root.children]
        moves.sort(key=lambda x: (x[1], x[2]), reverse=True)
        for move, visits, win_rate in moves[:5]:
            print(f"Move {move}: {visits} visits, win_rate={win_rate:.3f}")
        
        # Return best child based on win rate and visits
        return max(root.children, key=lambda c: (c.wins/c.visits if c.visits > 0 else 0) + c.visits/1000)

    def _select(self, node):
        """Select a leaf node using UCB1."""
        while node.children and not node.state.status:
            if not node.fully_expanded():
                return node
            node = self._ucb_select(node)
        return node

    def _expand(self, node):
        """Expand node by adding a new child."""
        moves = node.state.legal_moves()
        tried_moves = {child.state.last_move for child in node.children}
        untried_moves = [move for move in moves if move not in tried_moves]
        
        if not untried_moves:
            return node
            
        move = random.choice(untried_moves)
        new_state = node.state.clone()
        new_state.make_move(move)
        
        child = Node(new_state, parent=node)
        node.children.append(child)
        return child

    def _evaluate_sequence(self, state, row, col, dr, dc, player):
        """Evaluate a sequence in a given direction for a player."""
        count = 1  # Start with 1 for the current position
        space_before = 0
        space_after = 0
        blocked_before = False
        blocked_after = False
        
        # Check forward
        r, c = row + dr, col + dc
        while 0 <= r < state.board_size and 0 <= c < state.board_size:
            if state.board[r,c] == player:
                if space_after > 0:  # If we found a piece after a space
                    break
                count += 1
            elif state.board[r,c] == 0:
                if space_after == 0:  # Only count first empty space
                    space_after = 1
                else:
                    break
            else:
                blocked_after = True
                break
            r += dr
            c += dc
            
        # Check backward
        r, c = row - dr, col - dc
        while 0 <= r < state.board_size and 0 <= c < state.board_size:
            if state.board[r,c] == player:
                if space_before > 0:  # If we found a piece after a space
                    break
                count += 1
            elif state.board[r,c] == 0:
                if space_before == 0:  # Only count first empty space
                    space_before = 1
                else:
                    break
            else:
                blocked_before = True
                break
            r -= dr
            c -= dc
            
        # Calculate threat level
        open_ends = (not blocked_before) + (not blocked_after)
        if count >= 5:
            return 10000  # Immediate win
        elif count == 4:
            if open_ends >= 1:
                return 5000  # One move from win
            return 100
        elif count == 3:
            if open_ends == 2:
                return 1000  # Strong threat
            elif open_ends == 1:
                return 100
        elif count == 2:
            if open_ends == 2:
                return 50
            elif open_ends == 1:
                return 10
        return count * open_ends

    def _evaluate_position(self, state, move, player):
        """Evaluate how good a move is for a player."""
        row, col = move
        score = 0
        
        # Check all directions
        for dr, dc in [(0,1), (1,0), (1,1), (1,-1)]:
            # First check our potential
            test_state = state.clone()
            test_state.board[row,col] = player
            score += self._evaluate_sequence(test_state, row, col, dr, dc, player)
            
            # Then check if we're blocking opponent
            opponent = -player
            test_state = state.clone()
            test_state.board[row,col] = opponent
            block_score = self._evaluate_sequence(test_state, row, col, dr, dc, opponent)
            if block_score >= 1000:  # If blocking a critical threat
                score = max(score, block_score * 0.9)  # Slightly less than making our own threat
        
        # Add small bonus for center proximity
        center = state.board_size // 2
        distance = abs(row - center) + abs(col - center)
        score += (10 - distance) * 0.1
        
        return score

    def _select_move(self, state, moves):
        """Select the best move from available moves."""
        if not moves:
            return None
            
        # First check for immediate wins
        current_player = state.current_player
        for move in moves:
            test_state = state.clone()
            if test_state.make_move(move):
                if test_state.status == current_player:
                    return move
                    
        # Then evaluate all moves
        best_score = float('-inf')
        best_moves = []
        
        for move in moves:
            score = self._evaluate_position(state, move, current_player)
            
            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)
        
        return random.choice(best_moves)

    def _simulate(self, node):
        """Simulate a random game from this node."""
        state = node.state.clone()
        current_player = state.current_player
        
        while not state.status:
            moves = state.legal_moves()
            if not moves:
                break
                
            move = self._select_move(state, moves)
            if not move or not state.make_move(move):
                break
        
        if state.status == current_player:
            return 1
        elif state.status == 0:
            return 0.5
        else:
            return 0

    def _backpropagate(self, node, reward):
        """Update statistics of all nodes up to the root."""
        while node is not None:
            node.visits += 1
            node.wins += reward
            reward = 1 - reward
            node = node.parent

    def _ucb_select(self, node):
        """Select a child node using UCB1 formula."""
        log_n_visits = math.log(node.visits) if node.visits > 0 else 0
        
        def ucb_value(child):
            if child.visits == 0:
                return float('inf')
            return (child.wins / child.visits) + self.exploration_weight * math.sqrt(log_n_visits / child.visits)
        
        return max(node.children, key=ucb_value)


class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0

    def fully_expanded(self):
        """Check if all possible moves from this state have been tried."""
        return len(self.children) == len(self.state.legal_moves())