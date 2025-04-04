import copy
import random
import math
import numpy as np
from collections import defaultdict


# -------------------------------
# TODO: Define transformation functions (rotation and reflection), i.e., rot90, rot180, ..., etc.
# -------------------------------

def rotation(pattern, board_size):
  pat = []
  for coordinate in pattern:
    pat.append([coordinate[1], board_size-1-coordinate[0]])
  return pat

def reflection(pattern,board_size):

  pat = []
  for coordinate in pattern:
    pat.append([coordinate[0], board_size-1-coordinate[1]])
  return pat



class NTupleApproximator:
    def __init__(self, board_size, patterns):
        """
        Initializes the N-Tuple approximator.
        Hint: you can adjust these if you want
        """
        self.board_size = board_size
        self.patterns = patterns
        # Create a weight dictionary for each pattern (shared within a pattern group)
        self.weights = [defaultdict(float) for _ in patterns]
        # Generate symmetrical transformations for each pattern
        self.symmetry_patterns = []
        for pattern in self.patterns:
            syms = self.generate_symmetries(pattern)
            # print(syms)
            self.symmetry_patterns.append(syms)
        # print(self.symmetry_patterns)

    def generate_symmetries(self, pattern):
        # TODO: Generate 8 symmetrical transformations of the given pattern.
        syms = []
        syms.append(pattern.copy())
        syms.append(rotation(pattern, self.board_size))
        syms.append(rotation(rotation(pattern, self.board_size), self.board_size))
        syms.append(rotation(rotation(rotation(pattern, self.board_size), self.board_size), self.board_size))
        syms.append(reflection(pattern, self.board_size))
        syms.append(rotation(reflection(pattern, self.board_size), self.board_size))
        syms.append(rotation(rotation(reflection(pattern, self.board_size), self.board_size), self.board_size))
        syms.append(rotation(rotation(rotation(reflection(pattern, self.board_size), self.board_size), self.board_size), self.board_size))

        return syms


    def tile_to_index(self, tile):
        """
        Converts tile values to an index for the lookup table.
        """
        if tile == 0:
            return 0
        else:
            return int(math.log(tile, 2))

    def get_feature(self, board, coords):
        # TODO: Extract tile values from the board based on the given coordinates and convert them into a feature tuple.
        feature = []
        # print(coords)
        # print(board)
        for coord in coords:

            feature.append(self.tile_to_index(board[coord[0], coord[1]]))
        return tuple(feature)

    def value(self, board):
        # TODO: Estimate the board value: sum the evaluations from all patterns.
        value = 0
        # print(len(self.symmetry_patterns))
        for patterns, weight in zip(self.symmetry_patterns, self.weights):

            for pattern in patterns:
                feature = self.get_feature(board, pattern)
                value += weight.get(feature, 0)
        return value


    def update(self, board, delta, alpha):
        # TODO: Update weights based on the TD error.

        # used a set to store features, avoid update same features multiple times
        for patterns, weight in zip(self.symmetry_patterns, self.weights):
            features = set()
            for pattern in patterns:
                feature = self.get_feature(board, pattern)
                features.add(feature)
            for feature in features:
                weight[feature] = weight.get(feature, 0) + alpha * delta / len(self.patterns) / 8

def td_learning(env, approximator, num_episodes=50000, alpha=0.01, gamma=0.99, epsilon=0.1):
    """
    Trains the 2048 agent using TD-Learning.

    Args:
        env: The 2048 game environment.
        approximator: NTupleApproximator instance.
        num_episodes: Number of training episodes.
        alpha: Learning rate.
        gamma: Discount factor.
        epsilon: Epsilon-greedy exploration rate.
    """
    final_scores = []
    success_flags = []
    number_of_trial = 1

    for episode in range(num_episodes):
        state = env.reset()
        trajectory = []  # Store trajectory data if needed
        previous_score = 0
        done = False
        max_tile = np.max(state)
        previous_approx_value = approximator.value(state)

        while not done:
            legal_moves = [a for a in range(4) if env.is_move_legal(a)]
            if not legal_moves:
                break
            # TODO: action selection
            # Note: TD learning works fine on 2048 without explicit exploration, but you can still try some exploration methods.
            action = None
            best_value = -float('inf')
            for move in legal_moves:
                new_env = copy.deepcopy(env)
                new_env.board = state.copy()
                new_env.score = previous_score

                _, new_score, _, after_state = new_env.step(move)
                action_value = new_score + gamma * approximator.value(after_state)
                if action_value > best_value:
                    action = move
                    best_value = action_value


            state = state.copy()
            next_state, new_score, done, after_state = env.step(action)
            incremental_reward = new_score - previous_score
            previous_score = new_score
            max_tile = max(max_tile, np.max(next_state))

            # TODO: Store trajectory or just update depending on the implementation
            trajectory.append((state.copy(), incremental_reward, after_state.copy(), done))

            if done:
              GAME_OVER_REWARD = - max_tile 
              trajectory.append((next_state.copy(), GAME_OVER_REWARD, None, done))



            state = next_state

        # TODO: If you are storing the trajectory, consider updating it now depending on your implementation.
        # update from the end of episode
        i = 0
        next_approx_value = 0
        approx_value = 0
        for state, reward, next_state, done in reversed(trajectory):
            # if i==20 or i == 21 and episode == 0:
            #   # print("reward:, ", reward)
            #   print("state:, ", state)
            #   print("next state:, ", next_state)
            #   # print("features: ", approximator.get_feature(state, approximator.patterns[0]))
            #   # print("next features: ", approximator.get_feature(next_state, approximator.patterns[0]))
            #   # print("state value: ", approximator.value(state))
            #   # print("next state value: ", approximator.value(next_state))
            #   # print("weights: ", approximator.weights)

            #   approx_value = approximator.value(state)
            #   delta = reward + gamma * next_approx_value - approx_value
            #   next_approx_value = approx_value
            #   approximator.update(state, delta, alpha)
            #   # print("delta: ", delta)
            #   # print("updated state value: ", approximator.value(state))
            #   # print("updated weights: ", approximator.weights)
            #   # print()
            #   i += 1
            # else:
              approx_value = approximator.value(state)
              delta = reward + gamma * next_approx_value - approx_value
              next_approx_value = approx_value
              approximator.update(state, delta, alpha)
              # i += 1


        final_scores.append(env.score)
        success_flags.append(1 if max_tile >= 2048 else 0)

        if (episode + 1) % 100 == 0:
            avg_score = np.mean(final_scores[-100:])
            success_rate = np.sum(success_flags[-100:]) / 100
            print(f"Episode {episode+1}/{num_episodes} | Avg Score: {avg_score:.2f} | Success Rate: {success_rate:.2f}")

    return final_scores


# TODO: Define your own n-tuple patterns
# patterns = [
#     [[1,0],[1,1],[2,0],[2,1],[3,0],[3,1]],
#     [[1,1],[1,2],[2,1],[2,2],[3,1],[3,2]],
#     [[1,0],[0,0],[2,0],[2,1],[3,0],[3,1]],
#     [[1,1],[0,1],[2,1],[2,2],[3,1],[3,2]],
#     [[0,0],[1,0],[2,0],[3,0],[3,1],[3,2]],
#     [[0,1],[1,1],[2,1],[3,1],[3,2],[3,3]],
#     [[0,0],[0,1],[0,2],[1,0],[1,1],[2,0]],
#     [[0,1],[0,2],[0,3],[1,1],[1,2],[2,1]]
# ]

# patterns = [
#     [(0, 0), (0, 1), (1, 0), (1, 1)]

# ]
# patterns = [
#     [(0, 0), (0, 1), (1, 0), (1, 1)],
#     [(1, 0), (1, 1), (2, 0), (2, 1)],
#     [(1, 1), (1, 2), (2, 1), (2, 2)],
#     [(0, 0), (1, 0), (2, 0), (3, 0)],
#     [(0, 1), (1, 1), (2, 1), (3, 1)]
# ]
# patterns = [
#     [(0, 0), (0, 1), (1, 0), (1, 1)],   # square
#     [(1, 0), (1, 1), (2, 0), (2, 1)],   # square
#     [(1, 1), (1, 2), (2, 1), (2, 2)],   # square
#     [(0, 0), (1, 0), (2, 0), (3, 0)],   # straight line
#     [(0, 1), (1, 1), (2, 1), (3, 1)],   # straight line
#     [(0, 0), (0, 1), (1, 0), (2, 0)],   # L-shape
#     [(0, 1), (0, 2), (1, 1), (2, 1)]   # L-shape
# ]

# approximator = NTupleApproximator(board_size=4, patterns=patterns)

# env = Game2048Env()



# Run TD-Learning training
# Note: To achieve significantly better performance, you will likely need to train for over 100,000 episodes.
# However, to quickly verify that your implementation is working correctly, you can start by running it for 1,000 episodes before scaling up.

# Mode = 1 # 0: Load, 1: Train new model, 2: Load and train
# Train_num = 100
# Num_episodes = 1000
# Alpha = 0.05

# import pickle
# for i in range(Train_num):
#   if Mode == 0: # load from drive
#     with open('/content/drive/My Drive/DRL_HW2/8x6tuple/approximator_weights_8x6tuple.pkl', 'rb') as f:
#         approximator.weights = pickle.load(f)

#   elif Mode == 1: # train new model
#     final_scores = td_learning(env, approximator, num_episodes=Num_episodes, alpha=Alpha, gamma=0.99, epsilon=0.1)
#     # save the weights of approximator to my drive

#     with open('/content/drive/My Drive/DRL_HW2/8x6tuple/approximator_weights_8x6tuple.pkl', 'wb') as f:
#         pickle.dump(approximator.weights, f)
#     # plot scores vs episode
#     import matplotlib.pyplot as plt
#     plt.plot(final_scores)
#     plt.xlabel('Episode')
#     plt.ylabel('Score')
#     plt.title('Training Scores vs. Episode')
#     #save and show plot
#     plt.savefig(f'/content/drive/My Drive/DRL_HW2/8x6tuple/td_training_scores_8x6tuple_{i+1}.png')
#     plt.show()
#     Mode = 2

#   elif Mode == 2: # load and train
#     with open('/content/drive/My Drive/DRL_HW2/8x6tuple/approximator_weights_8x6tuple.pkl', 'rb') as f:
#         approximator.weights = pickle.load(f)

#     final_scores = td_learning(env, approximator, num_episodes=Num_episodes, alpha=Alpha, gamma=0.99, epsilon=0.1)
#     # save the weights of approximator to my drive

#     with open('/content/drive/My Drive/DRL_HW2/8x6tuple/approximator_weights_8x6tuple.pkl', 'wb') as f:
#         pickle.dump(approximator.weights, f)
#     # plot scores vs episode
#     import matplotlib.pyplot as plt
#     plt.plot(final_scores)
#     plt.xlabel('Episode')
#     plt.ylabel('Score')
#     plt.title('Training Scores vs. Episode')
#     #save and show plot
#     plt.savefig(f'/content/drive/My Drive/DRL_HW2/8x6tuple/td_training_scores_8x6tuple_{i+1}.png')
#     plt.show()




