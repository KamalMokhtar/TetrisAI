from keras.models import Sequential, save_model, load_model
from keras.layers import Dense
from collections import deque
import numpy as np
import random


# Deep Q Learning Agent + Maximin
#
# This version only provides only value per input,
# that indicates the score expected in that state.
# This is because the algorithm will try to find the
# best final state for the combinations of possible states,
# in constrast to the traditional way of finding the best
# action for a particular state.
class DQNAgent:

    '''Deep Q Learning Agent + Maximin

    Args:
        state_size (int): Size of the input domain
        mem_size (int): Size of the replay buffer
        discount (float): How important is the future rewards compared to the immediate ones [0,1]
        epsilon (float): Exploration (probability of random values given) value at the start
        epsilon_min (float): At what epsilon value the agent stops decrementing it
        epsilon_stop_episode (int): At what episode the agent stops decreasing the exploration variable
        n_neurons (list(int)): List with the number of neurons in each inner layer
        activations (list): List with the activations used in each inner layer, as well as the output
        loss (obj): Loss function
        optimizer (obj): Otimizer used
        replay_start_size: Minimum size needed to train
    '''

    def __init__(self, state_size, mem_size=10000, discount=0.95,
                 epsilon=1, epsilon_min=0, epsilon_stop_episode=500,
                 n_neurons=[1600,1600], activations=['relu', 'relu', 'linear'], #last one linear n_neurons=[32,32]
                 loss='mse', optimizer='adam', replay_start_size=None):

        assert len(activations) == len(n_neurons) + 1

        self.state_size = state_size
        self.memory = deque(maxlen=mem_size)
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / (epsilon_stop_episode)
        self.n_neurons = n_neurons
        self.activations = activations
        self.loss = loss
        self.optimizer = optimizer
        if not replay_start_size:
            replay_start_size = mem_size / 2
        self.replay_start_size = replay_start_size
        self.model = self._build_model()


    def _build_model(self):
        '''Builds a Keras deep neural network model'''
        model = Sequential()    # 32  self.n_neurons[0]          #  4   self.state_size                     #relu
        model.add(Dense(self.n_neurons[0], input_dim=200, activation=self.activations[0]))

        for i in range(1, len(self.n_neurons)): # 1 to 2
                            # self.n_neurons[i] 1600
            model.add(Dense(self.n_neurons[i], activation=self.activations[i]))                # the second hidden layer

        model.add(Dense(1, activation=self.activations[-1], name='output'))                                # output layer

        model.compile(loss=self.loss, optimizer=self.optimizer)
        
        return model

    # current_state, next_state,
    def add_to_memory(self, current_board, next_board, reward, done):
        '''Adds a play to the replay memory buffer'''
        # current_state, next_state,
        self.memory.append((current_board, next_board, reward, done))


    def random_value(self):
        '''Random score for a certain action'''
        return random.random()

    # def predict_value(self, state):
    def predict_value(self, board):
        '''Predicts the score for a certain state'''
        # return self.model.predict(state)[0]
        return self.model.predict(board)[0]

    # *K I think this is never used
    # def act(self, state):
    #     '''Returns the expected score of a certain state'''
    #     state = np.reshape(state, [1, self.state_size])
    #     if random.random() <= self.epsilon:
    #         return self.random_value()
    #     else:
    #         return self.predict_value(state)

        # def best_state(self, states):
    def best_board(self, boards): # states is the value of possible moves
        '''Returns the best board for a given collection of boards'''
        max_value = None
        # best_state = None
        best_board = None

        if random.random() <= self.epsilon:
            return random.choice(list(boards))

        else:
            # for state in states:
            for board in boards:
                # print("board1")
                # print(board)
                # value = self.predict_value(np.reshape(state, [1, self.state_size]))
                value = self.predict_value(np.reshape(board, [1, 200]))
                # print("value")
                # print(value)
                if not max_value or value > max_value:
                    max_value = value
                    # best_state = state
                    best_board = board
                    # print("1")
                    # print(best_board)

        # return best_state
        return best_board

    # *k train
    def train(self, batch_size=1, epochs=1, board=[0] * 200, played_blocks=0):  # batch_size=32 epochs=3
        '''Trains the agent'''
        # print("train")
        # print(board)
        n = len(self.memory)  # this increases with the playing number of blocks
        # print(n)
        # print(self.replay_start_size)
        # print(batch_size)
        if n >= self.replay_start_size and n >= batch_size:  # replay_start_size original 2000 changed to 100
            # batch_size changed to 1
            # print("I am in")

            batch = random.sample(self.memory, batch_size)
            # print("batch")
            # print(batch)
            # Get the expected score for the next states, in batch (better performance)
            # print("batch")
            # print(batch)
            # next_states = np.array([x[1] for x in batch])
            next_boards = np.array([x[1] for x in batch])
            # converting next_boards to board_model_input
            # print("next_boards")
            # print(next_boards)

            # # -----------------#
            # # first extracting, maybe can be done better
            # board_converted_array = np.array(next_boards)
            # board_model_input = board_converted_array[0]
            #
            # for row in range(1, len(board_converted_array)):
            #     board_model_input = np.concatenate((board_model_input, board_converted_array[row]), axis=None)
            #
            # # second extracting
            # board_converted_array = board_model_input
            # board_model_input = board_converted_array[0]
            #
            # for row in range(1, len(board_converted_array)):
            #     board_model_input = np.concatenate((board_model_input, board_converted_array[row]), axis=None)
            # # -----------------#

            # next_qs = [x[0] for x in
            #            self.model.predict(next_states)]  # from random samples in predict the score for the next block

            # print("board_model_input")
            # print(board_model_input)
            # k = [board_model_input]
            # print("k")
            # print(k)
            #
            next_board_qs = [x[0] for x in self.model.predict(next_boards)]
            #
            # print("next_qs")
            # print(next_board_qs)

            x = []
            y = []

            # Build xy structure to fit the model in batch (better performance)
            # for i, (state, _, reward, done) in enumerate(batch):
            # def add_to_memory(self, current_board, next_board, reward, done):
            for i, (board, _, reward, done) in enumerate(batch):
                if not done:
                    # Partial Q formula
                    # print("i")
                    # print(i)
                    # new_q = reward + self.discount * next_qs[i]
                    new_q = reward + self.discount * next_board_qs[i]
                    # print("if new_q")
                    # print(new_q)
                else:
                    new_q = reward
                    # print("else new_q")
                    # print(new_q)

                x.append(board)
                y.append(new_q)

            # board_converted = np.array(board)
            # board_model_input = board_converted[0]
            #
            # for row in range(1, len(board_converted)):
            #     board_model_input = np.concatenate((board_model_input, board_converted[row]), axis=None)
            #
            # print("board_model_input")
            # print(board_model_input)
            # print(len(board_model_input))

            # print("x")
            # print(len(np.array(x)))
            # print(np.array(x))
            # print("y")
            # print(len(np.array(y)))

            # print(np.array(y))
            # Fit the model to the given values
            # x are the states ex: for batch of 3 [[ 0  7 20 27] [ 0 12 11 44] [ 0  3  7 15]]
            # y are the rewards ex: for batch of 3 [-3.27968986 -3.80597708 -0.84219939]
            # np.array(x)
            # print("train")
            self.model.fit(np.array(x), np.array(y), batch_size=batch_size, epochs=epochs, verbose=0)

            # Update the exploration variable
            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decay

    # # train
    # def train(self, batch_size=1, epochs=1): # batch_size=32 epochs=3
    #     '''Trains the agent'''
    #     n = len(self.memory)
    #
    #     if n >= self.replay_start_size and n >= batch_size:
    #
    #         batch = random.sample(self.memory, batch_size)
    #
    #         # Get the expected score for the next states, in batch (better performance)
    #         next_states = np.array([x[1] for x in batch])
    #         next_qs = [x[0] for x in self.model.predict(next_states)]
    #
    #         x = []
    #         y = []
    #         # Build xy structure to fit the model in batch (better performance)
    #         for i, (state, _, reward, done) in enumerate(batch):
    #             if not done:
    #                 # Partial Q formula
    #                 new_q = reward + self.discount * next_qs[i]
    #             else:
    #                 new_q = reward
    #
    #             x.append(state) # length 512
    #             y.append(new_q)
    #             print(np.array(y))
    #
    #         # Fit the model to the given values
    #
    #         self.model.fit(np.array(x), np.array(y), batch_size=batch_size, epochs=epochs, verbose=0)
    #
    #         # Update the exploration variable
    #         if self.epsilon > self.epsilon_min:
    #             self.epsilon -= self.epsilon_decay
