from typing import Tuple, List, Optional
from gomoku import Gomoku, BLACK, WHITE, ONGOING
from elo import EloRating


def play_evaluation_game(agent1, agent2, board_size: int = 15) -> Tuple[int, List[Tuple[int, int]]]:
    """Play a single game between two agents and return the winner and move history.
    
    Args:
        agent1: First agent (plays as BLACK)
        agent2: Second agent (plays as WHITE)
        board_size: Size of the Gomoku board
    
    Returns:
        Tuple containing:
        - Winner (1 for BLACK/agent1, -1 for WHITE/agent2, 0 for DRAW)
        - List of moves made during the game
    """
    game = Gomoku(board_size)
    moves = []

    while game.status == ONGOING:  # While game is ongoing
        current_agent = agent1 if game.next_player == BLACK else agent2
        move, root = current_agent.best_move(game, iterations=1600, is_training=False)

        if move is None:
            break  # No valid moves available

        game.make_move(move.last_move)  # Pass move as a tuple
        moves.append(move.last_move)

    winner = game.get_winner()
    return winner, moves


def evaluate_agents(agent1, agent2, agent1_name: str, agent2_name: str,
                    num_games: int = 100, board_size: int = 15, elo_system: Optional[EloRating] = None) -> float:
    """Evaluate two agents by playing multiple games and updating their ELO ratings.
    
    Args:
        agent1: First agent
        agent2: Second agent
        agent1_name: Name/version identifier for agent1
        agent2_name: Name/version identifier for agent2
        num_games: Number of games to play
        board_size: Size of the Gomoku board
        elo_system: Optional EloRating system to use. If None, creates a new one.
        
    Returns:
        float: Win rate of agent1 (between 0 and 1)
    """
    if elo_system is None:
        elo_system = EloRating()
        
    wins_a = 0
    wins_b = 0
    draws = 0

    print(f"Starting evaluation: {agent1_name} vs {agent2_name}")
    print(f"Initial ratings - {agent1_name}: {elo_system.get_rating(agent1_name)}, "
          f"{agent2_name}: {elo_system.get_rating(agent2_name)}")

    for game_num in range(num_games):
        # Alternate which agent plays black
        if game_num % 2 == 0:
            first_agent, second_agent = agent1, agent2
        else:
            first_agent, second_agent = agent2, agent1
     

        winner, _ = play_evaluation_game(first_agent, second_agent, board_size)

        # Convert result to ELO score from agent1's perspective (not first player's perspective)
        if winner == BLACK:  # First player (BLACK) won
            if game_num % 2 == 0:  # agent1 was BLACK
                score = 1.0
                wins_a += 1
            else:  # agent2 was BLACK
                score = 0.0
                wins_b += 1
        elif winner == WHITE:  # Second player (WHITE) won
            if game_num % 2 == 0:  # agent1 was BLACK but white won
                score = 0.0
                wins_b += 1
            else:  # agent2 was BLACK but white won
                score = 1.0
                wins_a += 1
        else:  # DRAW
            score = 0.5
            draws += 1

        # Always update ELO from agent1's perspective
        elo_system.update_ratings(agent1_name, agent2_name, score)

        if (game_num + 1) % 10 == 0:
            print(f"\nAfter {game_num + 1} games:")
            print(f"{agent1_name} wins: {wins_a}")
            print(f"{agent2_name} wins: {wins_b}")
            print(f"Draws: {draws}")
            print(f"Current ratings - {agent1_name}: {elo_system.get_rating(agent1_name)}, "
                  f"{agent2_name}: {elo_system.get_rating(agent2_name)}")

    print("\nFinal Results:")
    print(f"Games played: {num_games}")
    print(f"{agent1_name} wins: {wins_a} ({wins_a / num_games * 100:.1f}%)")
    print(f"{agent2_name} wins: {wins_b} ({wins_b / num_games * 100:.1f}%)")
    print(f"Draws: {draws} ({draws / num_games * 100:.1f}%)")
    print(f"Final ratings - {agent1_name}: {elo_system.get_rating(agent1_name)}, "
          f"{agent2_name}: {elo_system.get_rating(agent2_name)}")

    return wins_a / num_games  # Return win rate of agent1
