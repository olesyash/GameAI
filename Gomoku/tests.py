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


class TestEvaluateAgents(unittest.TestCase):
    """Test class for the evaluate_agents function with mocked PUCTPlayer instances."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create a small board size for faster tests
        self.board_size = 5
        self.game = Gomoku(board_size=self.board_size)
        
        # Create an ELO rating system for testing
        self.elo_system = EloRating(k_factor=32, initial_rating=1200)
        
        # Create agent names for testing
        self.agent1_name = "TestAgent1"
        self.agent2_name = "TestAgent2"

    @patch('evaluate.play_evaluation_game')
    def test_evaluate_agents_all_agent1_wins(self, mock_play_game):
        """Test evaluate_agents when agent1 wins all games."""
        # Configure the mock to make agent1 win all games regardless of color
        # When agent1 plays as BLACK (even games), BLACK wins
        # When agent1 plays as WHITE (odd games), WHITE wins
        def mock_game_result(first_agent, second_agent, board_size):
            # Check which game we're on by counting calls
            game_num = mock_play_game.call_count - 1  # 0-indexed
            
            if game_num % 2 == 0:  # Even games: agent1 is BLACK
                return BLACK, []  # BLACK (agent1) wins
            else:  # Odd games: agent1 is WHITE
                return WHITE, []  # WHITE (agent1) wins
                
        mock_play_game.side_effect = mock_game_result
        
        # Create mock agents
        mock_agent1 = Mock()
        mock_agent2 = Mock()
        
        # Run the evaluation
        win_rate = evaluate_agents(
            mock_agent1, mock_agent2,
            self.agent1_name, self.agent2_name,
            num_games=10, board_size=self.board_size,
            elo_system=self.elo_system
        )
        
        # Verify the results
        self.assertEqual(win_rate, 1.0, "Agent1 should have a 100% win rate")
        self.assertEqual(mock_play_game.call_count, 10, "Should have played 10 games")
        
        # Check that ELO ratings were updated correctly
        agent1_rating = self.elo_system.get_rating(self.agent1_name)
        agent2_rating = self.elo_system.get_rating(self.agent2_name)
        self.assertGreater(agent1_rating, 1200, "Agent1's rating should have increased")
        self.assertLess(agent2_rating, 1200, "Agent2's rating should have decreased")

    @patch('evaluate.play_evaluation_game')
    def test_evaluate_agents_all_agent2_wins(self, mock_play_game):
        """Test evaluate_agents when agent2 wins all games."""
        # Configure the mock to make agent2 win all games regardless of color
        # When agent2 plays as BLACK (odd games), BLACK wins
        # When agent2 plays as WHITE (even games), WHITE wins
        def mock_game_result(first_agent, second_agent, board_size):
            # Check which game we're on by counting calls
            game_num = mock_play_game.call_count - 1  # 0-indexed
            
            if game_num % 2 == 0:  # Even games: agent1 is BLACK, agent2 is WHITE
                return WHITE, []  # WHITE (agent2) wins
            else:  # Odd games: agent2 is BLACK, agent1 is WHITE
                return BLACK, []  # BLACK (agent2) wins
                
        mock_play_game.side_effect = mock_game_result
        
        # Create mock agents
        mock_agent1 = Mock()
        mock_agent2 = Mock()
        
        # Run the evaluation
        win_rate = evaluate_agents(
            mock_agent1, mock_agent2,
            self.agent1_name, self.agent2_name,
            num_games=10, board_size=self.board_size,
            elo_system=self.elo_system
        )
        
        # Verify the results
        self.assertEqual(win_rate, 0.0, "Agent1 should have a 0% win rate")
        self.assertEqual(mock_play_game.call_count, 10, "Should have played 10 games")
        
        # Check that ELO ratings were updated correctly
        agent1_rating = self.elo_system.get_rating(self.agent1_name)
        agent2_rating = self.elo_system.get_rating(self.agent2_name)
        self.assertLess(agent1_rating, 1200, "Agent1's rating should have decreased")
        self.assertGreater(agent2_rating, 1200, "Agent2's rating should have increased")

    @patch('evaluate.play_evaluation_game')
    def test_evaluate_agents_alternating_wins(self, mock_play_game):
        """Test evaluate_agents with alternating wins between agents."""
        # Configure the mock to make the first player (BLACK) always win
        # This means agent1 wins when it plays BLACK, and agent2 wins when it plays BLACK
        mock_play_game.return_value = (BLACK, [])  # BLACK always wins
        
        # Create mock agents
        mock_agent1 = Mock()
        mock_agent2 = Mock()
        
        # Run the evaluation
        win_rate = evaluate_agents(
            mock_agent1, mock_agent2,
            self.agent1_name, self.agent2_name,
            num_games=10, board_size=self.board_size,
            elo_system=self.elo_system
        )
        
        # Verify the results
        # Since they alternate who plays as BLACK, and BLACK always wins,
        # agent1 should win 5 games out of 10 (when it plays as BLACK)
        self.assertEqual(win_rate, 0.5, "Agent1 should have a 50% win rate")
        self.assertEqual(mock_play_game.call_count, 10, "Should have played 10 games")
        
        # Check that ELO ratings remain close to initial
        agent1_rating = self.elo_system.get_rating(self.agent1_name)
        agent2_rating = self.elo_system.get_rating(self.agent2_name)
        self.assertAlmostEqual(agent1_rating, 1200, delta=50, msg="Agent1's rating should be close to initial")
        self.assertAlmostEqual(agent2_rating, 1200, delta=50, msg="Agent2's rating should be close to initial")

    @patch('evaluate.play_evaluation_game')
    def test_evaluate_agents_with_draws(self, mock_play_game):
        """Test evaluate_agents with some games ending in draws."""
        # Configure the mock to include draws (0) and BLACK wins
        # Game 0: BLACK (agent1) wins
        # Game 1: Draw
        # Game 2: BLACK (agent2) wins
        # Game 3: Draw
        mock_play_game.side_effect = [(BLACK, []), (0, []), (BLACK, []), (0, [])]
        
        # Create mock agents
        mock_agent1 = Mock()
        mock_agent2 = Mock()
        
        # Run the evaluation
        win_rate = evaluate_agents(
            mock_agent1, mock_agent2,
            self.agent1_name, self.agent2_name,
            num_games=4, board_size=self.board_size,
            elo_system=self.elo_system
        )
        
        # Verify the results
        # Agent1 wins in game 0 (as BLACK), loses in game 2 (as WHITE)
        # Games 1 and 3 are draws (count as 0.5 each)
        # So agent1's score is 1 + 0 + 0.5 + 0.5 = 2 out of 4 games = 0.5 win rate
        self.assertEqual(win_rate, 0.5, "Agent1 should have a 50% win rate with draws")
        self.assertEqual(mock_play_game.call_count, 4, "Should have played 4 games")

    def test_evaluate_agents_integration(self):
        """Integration test for evaluate_agents with mocked PUCTPlayer behavior."""
        # Create mock PUCTPlayer instances
        mock_agent1 = Mock()
        mock_agent2 = Mock()
        
        # Mock the best_move method to simulate gameplay
        def agent1_best_move(game, iterations, is_training=False):
            # Agent1 always plays a winning move when possible
            state = game.clone()
            moves = state.legal_moves()
            if moves:
                state.make_move(moves[0])  # Just make the first legal move
            return state, None
            
        def agent2_best_move(game, iterations, is_training=False):
            # Agent2 always plays a suboptimal move
            state = game.clone()
            moves = state.legal_moves()
            if moves:
                state.make_move(moves[-1])  # Make the last legal move
            return state, None
        
        mock_agent1.best_move = agent1_best_move
        mock_agent2.best_move = agent2_best_move
        
        # Run the evaluation with real play_evaluation_game function
        win_rate = evaluate_agents(
            mock_agent1, mock_agent2,
            self.agent1_name, self.agent2_name,
            num_games=10, board_size=self.board_size,
            elo_system=self.elo_system
        )
        
        # We expect agent1 to perform better, but exact win rate depends on game dynamics
        self.assertGreaterEqual(win_rate, 0.0, "Win rate should be at least 0")
        self.assertLessEqual(win_rate, 1.0, "Win rate should be at most 1")

if __name__ == '__main__':
    unittest.main()
