from keras.models import Sequential, save_model, load_model
from keras.layers import Dense, Flatten, Conv2D, Input, concatenate
from keras.models import Model
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
                 n_neurons=[64, 64, 64, 64, 32, 32],
                 activations=['relu', 'relu', 'relu', 'relu','relu', 'relu', 'linear'],
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
        model1in = Input(shape=(20,10,1))
        model1out = Conv2D(32, kernel_size=(20,1), strides=(1,1), padding='same',
                           activation=self.activations[0],
                           name='layer1')(model1in)
        model1out2 = Conv2D(kernel_size=(1, 10), strides=(1,1), activation='relu',
                            filters=32, padding='same')(model1out)

        model2in = Input(shape=(20,10,1))
        model2out = Conv2D(64, kernel_size=(8, 8), strides=(1,1), activation='relu',
                           padding='same')(model2in)
        model2out2 = Conv2D(kernel_size=(4, 4), strides=(1,1), activation='relu',
                            filters=16, padding='same')(model2out)
        concmodel = concatenate([model2out2, model1out2])
        fully1 = Dense(256, activation='relu')(concmodel)
        fully2 = Dense(128, activation='relu')(fully1)
        fully3 = Dense(32, activation='relu')(fully2)
        flat = Flatten()(fully3)
        out = Dense(1, activation='linear')(flat)
        merged_model = Model([model1in, model2in], out)
        merged_model.compile(loss=self.loss, optimizer=self.optimizer)
        
        return merged_model


    def add_to_memory(self, current_state, next_state, reward, done):
        '''Adds a play to the replay memory buffer'''
        self.memory.append((current_state, next_state, reward, done))


    def random_value(self):
        '''Random score for a certain action'''
        return random.random()


    def predict_value(self, state):
        '''Predicts the score for a certain state'''
        return self.model.predict([state, state])[0]


    def act(self, state):
        '''Returns the expected score of a certain state'''
        state = np.reshape(state, [1, self.state_size])
        if random.random() <= self.epsilon:
            return self.random_value()
        else:
            return self.predict_value(state)


    def best_state(self, states):
        '''Returns the best state for a given collection of states'''
        max_value = None
        best_state = None

        if random.random() <= self.epsilon:
            return random.choice(list(states))

        else:
            for state in states:
                predictinput = np.reshape(state, (20, 10, 1) )

                value = self.predict_value([predictinput, predictinput])
                if not max_value or value > max_value:
                    max_value = value
                    best_state = state

        return best_state


    def train(self, batch_size=32, epochs=3):
        '''Trains the agent'''
        n = len(self.memory)
    
        if n >= self.replay_start_size and n >= batch_size:

            batch = random.sample(self.memory, batch_size)

            # Get the expected score for the next states, in batch (better performance)
            next_states = np.array([x[1] for x in batch])
            next_states =np.reshape(next_states, (batch_size, 20, 10, 1))

            next_qs = [x[0] for x in self.model.predict([next_states, next_states])]

            x = []
            y = []

            # Build xy structure to fit the model in batch (better performance)

            for i, (state, _, reward, done) in enumerate(batch):
                if not done:
                    # Partial Q formula
                    new_q = reward + self.discount * next_qs[i]
                else:
                    new_q = reward

                x.append(state)
                y.append(new_q)

            # Fit the model to the given values
            x = np.reshape(x, (batch_size, 20, 10, 1))
            self.model.fit([x, x], np.array(y), epochs=epochs, verbose=0)

            # Update the exploration variable
            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decay
