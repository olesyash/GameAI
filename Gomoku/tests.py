from main import *
import unittest
from unittest.mock import Mock, patch
import torch
import numpy as np
from gomoku import Gomoku, BLACK, WHITE, DRAW
from main import evaluate_model, GameNetwork, PUCTPlayer, MCTSPlayer, save_game_data, load_games, BOARD_SIZE
import os
import datetime
import json

def print_board(board):
    """Helper function to print the board state"""
    for row in board:
        print(' '.join(['.' if x == 0 else 'X' if x == 1 else 'O' for x in row]))
    print()

class TestEvaluateModel(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.current_model = Mock(spec=GameNetwork)
        self.previous_model = Mock(spec=GameNetwork)
        
        # Mock the network's forward pass
        self.current_model.forward.return_value = (
            torch.tensor([0.1, 0.2, 0.7]),  # policy
            torch.tensor([0.5])  # value
        )
        self.previous_model.forward.return_value = (
            torch.tensor([0.1, 0.2, 0.7]),  # policy
            torch.tensor([-0.3])  # value
        )
    
    def test_value_perspectives(self):
        """Test function to verify how values are interpreted in PUCT and MCTS."""
        print("\n=== Testing Value Perspectives ===")
        
        # Initialize game and players
        game = Gomoku(board_size=BOARD_SIZE)
        
        # Create MCTS player
        mcts = MCTSPlayer(exploration_weight=1.4)
        
        # Create PUCT player with network
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        network = GameNetwork(BOARD_SIZE, device)
        network.to(device)
        try:
            network.load_model(os.path.join("models", "model_best.pt"))
            print("Loaded model successfully")
        except:
            print("No model found, using fresh network")
        
        puct = PUCTPlayer(1.4, game)
        puct.model = network
        
        # Create a simple test case with a clear advantage for Black
        # Black has 3 in a row and is about to win
        print("\nCreating a test position with advantage for Black:")
        game.board[3, 1] = BLACK  # Black
        game.board[3, 2] = BLACK  # Black
        game.board[3, 3] = BLACK  # Black
        game.next_player = BLACK  # Black to move (should be advantageous)
        
        print("\nBoard state (Black to move, Black has advantage):")
        print_board(game.board)
        
        # Test MCTS value
        print("\n--- MCTS Value Test (Black to move) ---")
        _, mcts_root = mcts.search(game.clone(), iterations=1000)
        mcts_value = mcts_root.value / max(1, mcts_root.visits)
        print(f"MCTS value (from root): {mcts_value:.4f}")
        print(f"Current player: {'Black' if game.next_player == 1 else 'White'} ({game.next_player})")
        print(f"Interpretation: {'Favorable for current player' if mcts_value > 0 else 'Unfavorable for current player'}")
        
        # Test PUCT value
        print("\n--- PUCT Value Test (Black to move) ---")
        _, puct_root = puct.best_move(game.clone(), iterations=1000, is_training=True)
        puct_value = puct_root.Q / max(1, puct_root.N)
        print(f"PUCT value (from root): {puct_value:.4f}")
        print(f"Current player: {'Black' if game.next_player == 1 else 'White'} ({game.next_player})")
        print(f"Interpretation: {'Favorable for current player' if puct_value > 0 else 'Unfavorable for current player'}")
        
        # Now switch to White's perspective
        game.next_player = WHITE  # White to move (should be disadvantageous)
        
        print("\nSame board state (White to move, White has disadvantage):")
        print_board(game.board)
        
        # Test MCTS value
        print("\n--- MCTS Value Test (White to move) ---")
        _, mcts_root = mcts.search(game.clone(), iterations=1000)
        mcts_value = mcts_root.value / max(1, mcts_root.visits)
        print(f"MCTS value (from root): {mcts_value:.4f}")
        print(f"Current player: {'Black' if game.next_player == 1 else 'White'} ({game.next_player})")
        print(f"Interpretation: {'Favorable for current player' if mcts_value > 0 else 'Unfavorable for current player'}")
        
        # Test PUCT value
        print("\n--- PUCT Value Test (White to move) ---")
        _, puct_root = puct.best_move(game.clone(), iterations=1000, is_training=True)
        puct_value = puct_root.Q / max(1, puct_root.N)
        print(f"PUCT value (from root): {puct_value:.4f}")
        print(f"Current player: {'Black' if game.next_player == 1 else 'White'} ({game.next_player})")
        print(f"Interpretation: {'Favorable for current player' if puct_value > 0 else 'Unfavorable for current player'}")
        
        # Test with a different position where White has advantage
        game = Gomoku(board_size=BOARD_SIZE)
        game.board[2, 1] = WHITE  # White
        game.board[2, 2] = WHITE  # White
        game.board[2, 3] = WHITE  # White
        game.next_player = WHITE  # White to move (should be advantageous)
        
        print("\nNew board state (White to move, White has advantage):")
        print_board(game.board)
        
        # Test MCTS value
        print("\n--- MCTS Value Test (White to move with advantage) ---")
        _, mcts_root = mcts.search(game.clone(), iterations=1000)
        mcts_value = mcts_root.value / max(1, mcts_root.visits)
        print(f"MCTS value (from root): {mcts_value:.4f}")
        print(f"Current player: {'Black' if game.next_player == 1 else 'White'} ({game.next_player})")
        print(f"Interpretation: {'Favorable for current player' if mcts_value > 0 else 'Unfavorable for current player'}")
        
        # Test PUCT value
        print("\n--- PUCT Value Test (White to move with advantage) ---")
        _, puct_root = puct.best_move(game.clone(), iterations=1000, is_training=True)
        puct_value = puct_root.Q / max(1, puct_root.N)
        print(f"PUCT value (from root): {puct_value:.4f}")
        print(f"Current player: {'Black' if game.next_player == 1 else 'White'} ({game.next_player})")
        print(f"Interpretation: {'Favorable for current player' if puct_value > 0 else 'Unfavorable for current player'}")
        
        # Now switch to Black's perspective
        game.next_player = BLACK  # Black to move (should be disadvantageous)
        
        print("\nSame board state (Black to move, Black has disadvantage):")
        print_board(game.board)
        
        # Test MCTS value
        print("\n--- MCTS Value Test (Black to move with disadvantage) ---")
        _, mcts_root = mcts.search(game.clone(), iterations=1000)
        mcts_value = mcts_root.value / max(1, mcts_root.visits)
        print(f"MCTS value (from root): {mcts_value:.4f}")
        print(f"Current player: {'Black' if game.next_player == 1 else 'White'} ({game.next_player})")
        print(f"Interpretation: {'Favorable for current player' if mcts_value > 0 else 'Unfavorable for current player'}")
        
        # Test PUCT value
        print("\n--- PUCT Value Test (Black to move with disadvantage) ---")
        _, puct_root = puct.best_move(game.clone(), iterations=1000, is_training=True)
        puct_value = puct_root.Q / max(1, puct_root.N)
        print(f"PUCT value (from root): {puct_value:.4f}")
        print(f"Current player: {'Black' if game.next_player == 1 else 'White'} ({game.next_player})")
        print(f"Interpretation: {'Favorable for current player' if puct_value > 0 else 'Unfavorable for current player'}")
        
        print("\n=== Test Complete ===")


class TestEvaluateModelVsMCTS(unittest.TestCase):
    def setUp(self):
        # Create a simple mock network
        self.mock_network = Mock()
        self.mock_network.predict = Mock(return_value=(Mock(), Mock()))  # Just return two mocks

    def test_evaluate_model_puct_always_loses(self):
        # Define move sequences
        mcts_moves_as_first = [
            (0, 0),  # Start horizontal line
            (0, 1),  # Continue line
            (0, 2),  # Continue line
            (0, 3),  # Continue line
            (0, 4),  # Win with line
        ]
        
        mcts_moves_as_second = [
            (1, 0),  # Start different horizontal line
            (1, 1),  # Continue line
            (1, 2),  # Continue line
            (1, 3),  # Continue line
            (1, 4),  # Win with line
        ]

        puct_moves_as_first = [
            (5, 5),  # Far from MCTS line
            (4, 1),  # Still far
            (3, 2),  # Still far
            (5, 3),  # Still far
        ]
        
        puct_moves_as_second = [
            (5, 5),  # Far from MCTS line
            (5, 4),  # Still far
            (3, 4),  # Still far
            (3, 2),  # Still far
        ]

        # Create mock PUCT player
        mock_puct = Mock()
        def mock_puct_best_move(game, iterations, is_training=False):
            print("\nPUCT's turn:")
            print(f"Game state before PUCT move:")
            print_board(game.board)
            
            move_index = len(game.move_history) // 2
            is_first = len(game.move_history) % 2 == 0
            moves = puct_moves_as_first if is_first else puct_moves_as_second
            
            if move_index >= len(moves):
                print("PUCT: No more moves")
                return None, None
                
            move = moves[move_index]

            print(f"PUCT making move: {move}")
            state = game.clone()
            state.make_move(move)
            print("Game state after PUCT move:")
            print_board(state.board)
            return state, None
            
        mock_puct.best_move = mock_puct_best_move
        mock_puct.model = self.mock_network

        # Create mock MCTS player
        mock_mcts = Mock()
        def mock_mcts_search(game, iterations):
            print("\nMCTS's turn:")
            print(f"Game state before MCTS move:")
            print_board(game.board)
            
            move_index = len(game.move_history) // 2
            is_first = len(game.move_history) % 2 == 0
            moves = mcts_moves_as_first if is_first else mcts_moves_as_second
            
            if move_index >= len(moves):
                print("MCTS: No more moves")
                return None, None
                
            move = moves[move_index]

            print(f"MCTS making move: {move}")
            state = game.clone()
            state.make_move(move)
            print("Game state after MCTS move:")
            print_board(state.board)
            node = Mock()
            node.state = state
            return node, None
            
        mock_mcts.search = mock_mcts_search

        # Patch both player creations
        with patch('main.PUCTPlayer', return_value=mock_puct), \
             patch('main.MCTSPlayer', return_value=mock_mcts):
            
            # Run evaluation
            win_rate = evaluate_model(
                self.mock_network,
                num_games=4
            )

            # Verify win rate calculation - PUCT should lose all games
            self.assertEqual(win_rate, 0.0, "PUCT should lose all games with predefined losing moves")


    def test_evaluate_model_puct_always_winning(self):
        # Define move sequences
        puct_moves_as_first = [
            (0, 0),  # Start horizontal line
            (0, 1),  # Continue line
            (0, 2),  # Continue line
            (0, 3),  # Continue line
            (0, 4),  # Win with line
        ]

        puct_moves_as_second = [
            (1, 0),  # Start different horizontal line
            (1, 1),  # Continue line
            (1, 2),  # Continue line
            (1, 3),  # Continue line
            (1, 4),  # Win with line
        ]

        mcts_moves_as_first = [
            (5, 5),  # Far from MCTS line
            (4, 1),  # Still far
            (3, 2),  # Still far
            (5, 3),  # Still far
        ]

        mcts_moves_as_second = [
            (5, 5),  # Far from MCTS line
            (5, 4),  # Still far
            (3, 4),  # Still far
            (3, 2),  # Still far
        ]

        # Create mock PUCT player
        mock_puct = Mock()

        def mock_puct_best_move(game, iterations, is_training=False):
            print("\nPUCT's turn:")
            print(f"Game state before PUCT move:")
            print_board(game.board)

            move_index = len(game.move_history) // 2
            is_first = len(game.move_history) % 2 == 0
            moves = puct_moves_as_first if is_first else puct_moves_as_second

            if move_index >= len(moves):
                print("PUCT: No more moves")
                return None, None

            move = moves[move_index]

            print(f"PUCT making move: {move}")
            state = game.clone()
            state.make_move(move)
            print("Game state after PUCT move:")
            print_board(state.board)
            return state, None

        mock_puct.best_move = mock_puct_best_move
        mock_puct.model = self.mock_network

        # Create mock MCTS player
        mock_mcts = Mock()

        def mock_mcts_search(game, iterations):
            print("\nMCTS's turn:")
            print(f"Game state before MCTS move:")
            print_board(game.board)

            move_index = len(game.move_history) // 2
            is_first = len(game.move_history) % 2 == 0
            moves = mcts_moves_as_first if is_first else mcts_moves_as_second

            if move_index >= len(moves):
                print("MCTS: No more moves")
                return None, None

            move = moves[move_index]

            print(f"MCTS making move: {move}")
            state = game.clone()
            state.make_move(move)
            print("Game state after MCTS move:")
            print_board(state.board)
            node = Mock()
            node.state = state
            return node, None

        mock_mcts.search = mock_mcts_search

        # Patch both player creations
        with patch('main.PUCTPlayer', return_value=mock_puct), \
                patch('main.MCTSPlayer', return_value=mock_mcts):

            # Run evaluation
            win_rate = evaluate_model(
                self.mock_network,
                num_games=4
            )

            # Verify win rate calculation - PUCT should lose all games
            self.assertEqual(win_rate, 1.0, "PUCT should lose all games with predefined losing moves")


    def test_evaluate_draw(self):
        # Define move sequences
        mcts_moves_as_first = [
            (2, 3),  
            (3, 2),  
            (4, 4),  
            (0, 0),  
            (2, 1),  
            (1, 4),
            (1, 2),
            (2, 0),
            (4, 2),
            (0, 2),
            (2, 5),
            (5, 5),
            (1, 3),
            (5, 1),
            (2, 4),
            (4, 5),
            (5, 2),
            (3, 0)


        ]
        
        puct_moves_as_second = [
            (3, 3),  # Far from MCTS line
            (2, 2),  # Still far
            (1, 1),  # Still far
            (4, 1),
            (4, 3),
            (0, 5),
            (3, 4),
            (0, 3),
            (1, 0),
            (0, 4),
            (1, 5),
            (5, 3),
            (3, 1),
            (4, 0),
            (0, 1),
            (5, 0),
            (5, 4),
            (3, 5)
        ]

        puct_moves_as_first = mcts_moves_as_first
        mcts_moves_as_second = puct_moves_as_second

        # Create mock PUCT player
        mock_puct = Mock()
        def mock_puct_best_move(game, iterations, is_training=False):
            print("\nPUCT's turn:")
            print(f"Game state before PUCT move:")
            print_board(game.board)
            
            move_index = len(game.move_history) // 2
            is_first = len(game.move_history) % 2 == 0
            moves = puct_moves_as_first if is_first else puct_moves_as_second
            
            if move_index >= len(moves):
                print("PUCT: No more moves")
                return None, None
                
            move = moves[move_index]

            print(f"PUCT making move: {move}")
            state = game.clone()
            state.make_move(move)
            print("Game state after PUCT move:")
            print_board(state.board)
            return state, None
            
        mock_puct.best_move = mock_puct_best_move
        mock_puct.model = self.mock_network

        # Create mock MCTS player
        mock_mcts = Mock()
        def mock_mcts_search(game, iterations):
            print("\nMCTS's turn:")
            print(f"Game state before MCTS move:")
            print_board(game.board)
            
            move_index = len(game.move_history) // 2
            is_first = len(game.move_history) % 2 == 0
            moves = mcts_moves_as_first if is_first else mcts_moves_as_second
            
            if move_index >= len(moves):
                print("MCTS: No more moves")
                return None, None
                
            move = moves[move_index]

            print(f"MCTS making move: {move}")
            state = game.clone()
            state.make_move(move)
            print("Game state after MCTS move:")
            print_board(state.board)
            node = Mock()
            node.state = state
            return node, None
            
        mock_mcts.search = mock_mcts_search

        # Patch both player creations
        with patch('main.PUCTPlayer', return_value=mock_puct), \
             patch('main.MCTSPlayer', return_value=mock_mcts):
            
            # Run evaluation
            win_rate = evaluate_model(
                self.mock_network,
                num_games=4
            )

            # Verify win rate calculation - PUCT should lose all games
            self.assertEqual(win_rate, 0.5, "Draw")

class TestGameDataSerialization(unittest.TestCase):
    def setUp(self):
        self.test_file = "test_training_data.json"
        # Clean up test file if it exists
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
            
    # def tearDown(self):
    #     # Clean up test file
    #     if os.path.exists(self.test_file):
    #         os.remove(self.test_file)
    #
    def test_save_and_load_game(self):
        # Create a deterministic game sequence
        original_game = Gomoku(BOARD_SIZE)
        moves_sequence = [
            ((0, 0), 1),  # Black plays top-left
            ((1, 1), -1), # White plays diagonal
            ((0, 1), 1),  # Black plays top
            ((1, 2), -1), # White plays middle
            ((0, 2), 1),  # Black plays top
            ((1, 3), -1), # White plays middle
            ((0, 3), 1),  # Black plays top
        ]
        
        # Generate some fake policies and Q-values
        policies = []
        q_values = []
        games = []
        
        # Play through the game and collect data
        for (move, player) in moves_sequence:
            # Save game state before move
            game_copy = Gomoku(BOARD_SIZE)
            game_copy.board = original_game.board.copy()
            game_copy.move_history = original_game.move_history.copy()
            games.append(game_copy)
            
            # Create a deterministic policy vector
            policy = np.zeros(BOARD_SIZE * BOARD_SIZE)
            move_idx = move[0] * BOARD_SIZE + move[1]
            policy[move_idx] = 1.0
            policies.append(policy)
            
            # Create a deterministic Q-value
            q_values.append(float(player) * 0.5)  # Positive for black, negative for white
            
            # Make the move
            original_game.make_move(move)
            
        # Save game data
        game_data = {
            "moves": moves_sequence,
            "policies": [p.tolist() for p in policies],
            "mcts_q_values": q_values,
            "winner": original_game.get_winner(),
            "timestamp": datetime.datetime.now().isoformat(),
            "board_size": BOARD_SIZE
        }
        save_game_data(game_data, filename=self.test_file)
        
        # Load game data
        loaded_states, loaded_policies, loaded_values = load_games(filename=self.test_file)
        
        # Verify data matches
        self.assertEqual(len(games), len(loaded_states), "Number of states should match")
        self.assertEqual(len(policies), len(loaded_policies), "Number of policies should match")
        self.assertEqual(len(q_values), len(loaded_values), "Number of Q-values should match")
        
        for i in range(len(games)):
            # Check game state matches
            self.assertEqual(games[i], loaded_states[i], f"Game state at move {i} doesn't match")
            
            # Check policy matches
            np.testing.assert_array_almost_equal(
                policies[i], 
                loaded_policies[i], 
                decimal=6,
                err_msg=f"Policy at move {i} doesn't match"
            )
            
            # Check Q-value matches
            self.assertAlmostEqual(
                q_values[i], 
                loaded_values[i], 
                places=6, 
                msg=f"Q-value at move {i} doesn't match"
            )


if __name__ == '__main__':
    unittest.main()
