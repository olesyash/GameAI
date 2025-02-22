import math
from typing import Dict, Tuple
import json
import os

class EloRating:
    def __init__(self, k_factor: int = 16, initial_rating: int = 1200):
        """Initialize the ELO rating system.
        
        Args:
            k_factor (int): The K-factor determines how much ratings can change after each game
            initial_rating (int): The starting rating for new agents
        """
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.ratings: Dict[str, int] = {}
        self.history: Dict[str, list] = {}  # Track rating history for each agent
        self.games_played: Dict[str, int] = {}  # Track number of games for each agent
        
        # Load existing ratings if available
        self._load_ratings()
    
    def get_expected_score(self, rating_a: int, rating_b: int) -> float:
        """Calculate expected score for player A when playing against player B.
        
        Args:
            rating_a (int): Rating of player A
            rating_b (int): Rating of player B
            
        Returns:
            float: Expected score (between 0 and 1)
        """
        return 1 / (1 + math.pow(10, (rating_b - rating_a) / 400))
    
    def _get_k_factor(self, agent: str) -> float:
        """Get dynamic K-factor based on number of games played.
        
        K-factor decreases as more games are played to stabilize ratings.
        """
        games = self.games_played.get(agent, 0)
        if games < 10:
            return self.k_factor
        elif games < 20:
            return self.k_factor * 0.75
        else:
            return self.k_factor * 0.5
            
    def update_ratings(self, agent_a: str, agent_b: str, score: float) -> Tuple[int, int]:
        """Update ratings after a game.
        
        Args:
            agent_a (str): Name/version of agent A
            agent_b (str): Name/version of agent B
            score (float): Actual score of the game for agent A
                        (1.0 for win, 0.5 for draw, 0.0 for loss)
        
        Returns:
            Tuple[int, int]: Updated ratings for agent A and agent B
        """
        # Initialize ratings if not present
        if agent_a not in self.ratings:
            self.ratings[agent_a] = self.initial_rating
            self.history[agent_a] = [self.initial_rating]
            self.games_played[agent_a] = 0
        if agent_b not in self.ratings:
            self.ratings[agent_b] = self.initial_rating
            self.history[agent_b] = [self.initial_rating]
            self.games_played[agent_b] = 0
        
        # Update games played
        self.games_played[agent_a] += 1
        self.games_played[agent_b] += 1
        
        rating_a = self.ratings[agent_a]
        rating_b = self.ratings[agent_b]
        
        expected_a = self.get_expected_score(rating_a, rating_b)
        
        # Use dynamic K-factor based on games played
        k_factor = min(self._get_k_factor(agent_a), self._get_k_factor(agent_b))
        
        # Update ratings
        rating_change = int(k_factor * (score - expected_a))
        self.ratings[agent_a] += rating_change
        self.ratings[agent_b] -= rating_change
        
        # Update history
        self.history[agent_a].append(self.ratings[agent_a])
        self.history[agent_b].append(self.ratings[agent_b])
        
        # Save updated ratings
        self._save_ratings()
        
        return self.ratings[agent_a], self.ratings[agent_b]
    
    def get_rating(self, agent: str) -> int:
        """Get the current rating for an agent.
        
        Args:
            agent (str): Name/version of the agent
            
        Returns:
            int: Current rating of the agent
        """
        return self.ratings.get(agent, self.initial_rating)
    
    def get_rating_history(self, agent: str) -> list:
        """Get the rating history for an agent.
        
        Args:
            agent (str): Name/version of the agent
            
        Returns:
            list: List of historical ratings for the agent
        """
        return self.history.get(agent, [self.initial_rating])
    
    def _save_ratings(self):
        """Save ratings and history to a JSON file."""
        data = {
            'ratings': self.ratings,
            'history': self.history,
            'games_played': self.games_played
        }
        with open('elo_ratings.json', 'w') as f:
            json.dump(data, f)
    
    def _load_ratings(self):
        """Load ratings and history from JSON file if it exists."""
        if os.path.exists('elo_ratings.json'):
            with open('elo_ratings.json', 'r') as f:
                data = json.load(f)
                self.ratings = data['ratings']
                self.history = data['history']
                self.games_played = data.get('games_played', {})
