from keras.models import Sequential, save_model, load_model
from keras.layers import Dense
from collections import deque
import numpy as np
import random
from datetime import datetime


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
                 n_neurons=[32, 32, 32, 32], activations=['relu', 'relu', 'relu', 'relu', 'linear'],
                 # last one linear n_neurons=[32,32]
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
        self.model = self._build_model(fetch_old_model=False)

    def _build_model(self, fetch_old_model):
        '''Builds a Keras deep neural network model'''
        if not fetch_old_model:
            print("new model made")
            model = Sequential()  # 32  self.n_neurons[0]          #  4   self.state_size                     #relu
            model.add(Dense(self.n_neurons[0], input_dim=200, activation=self.activations[0]))

            for i in range(1, len(self.n_neurons)):  # 1 to 2
                # self.n_neurons[i] 1600
                model.add(Dense(self.n_neurons[i], activation=self.activations[i]))  # the second hidden layer

            model.add(Dense(1, activation=self.activations[-1], name='output'))  # output layer

            model.compile(loss=self.loss, optimizer=self.optimizer)
        else:
            print("old model returned")
            # put the name of the model file you want
            model = load_model('models/my_model.h5')
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
    def best_board(self, boards):  # states is the value of possible moves
        '''Returns the best board for a given collection of boards'''
        max_value = None
        best_board = None

        if random.random() <= self.epsilon:
            return random.choice(list(boards))
        else:
            for board in boards:
                value = self.predict_value(np.reshape(board, [1, 200]))
                if not max_value or value > max_value:
                    max_value = value
                    best_board = board
        return best_board

    # *k train
    def train(self, batch_size=1, epochs=1, board=[0] * 200, played_blocks=0):  # batch_size=32 epochs=3
        '''Trains the agent'''
        n = len(self.memory)  # this increases with the playing number of blocks
        if n >= self.replay_start_size and n >= batch_size:  # replay_start_size original 2000 changed to 100

            batch = random.sample(self.memory, batch_size)
            next_boards = np.array([x[1] for x in batch])
            next_board_qs = [x[0] for x in self.model.predict(next_boards)]

            x = []
            y = []

            # Build xy structure to fit the model in batch (better performance)
            # for i, (state, _, reward, done) in enumerate(batch):
            # def add_to_memory(self, current_board, next_board, reward, done):
            for i, (board, _, reward, done) in enumerate(batch):
                if not done:
                    new_q = reward + self.discount * next_board_qs[i]
                else:
                    new_q = reward

                x.append(board)
                y.append(new_q)

            self.model.fit(np.array(x), np.array(y), batch_size=batch_size, epochs=epochs, verbose=0)

            # Update the exploration variable
            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decay

    def model_save(self, time_frame):
        self.model.save(f'models/my_model-{time_frame}-h5')  # creates a HDF5 file
