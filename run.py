from dqn_agent import DQNAgent
from tetris import Tetris
from datetime import datetime
from statistics import mean, median
import random
from logs import CustomTensorBoard
from tqdm import tqdm
from keras.models import load_model


# Run dqn with Tetris
def dqn():
    saves = 0
    savespepi = []
    env = Tetris()
    episodes = 200000
    max_steps = None
    epsilon_stop_episode = 190000
    mem_size = 2000000
    discount = 0.95
    batch_size = 512
    epochs = 1
    render_every = 0
    log_every = 50
    replay_start_size = 2000
    train_every = 1
    n_neurons = [256, 32, 16]
    render_delay = None
    activations = ['relu', 'relu', 'relu', 'linear']

    agent = DQNAgent(env.get_state_size(),
                     n_neurons=n_neurons, activations=activations,
                     epsilon_stop_episode=epsilon_stop_episode, mem_size=mem_size,
                     discount=discount, replay_start_size=replay_start_size)

    log_dir = f'logs/tetris-nn={str(n_neurons)}-mem={mem_size}-bs={batch_size}-e={epochs}-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    log = CustomTensorBoard(log_dir=log_dir)

    scores = []

    for episode in tqdm(range(episodes)):
        current_state = env.reset()
        done = False
        steps = 0

        if render_every and episode % render_every == 0:
            render = True
        else:
            render = False

        # Game
        while not done and (not max_steps or steps < max_steps):
            next_states = env.get_next_states()
            best_state = agent.best_state(next_states.values())

            best_action = None
            for action, state in next_states.items():
                if state == best_state:
                    best_action = action
                    break

            score, done, reward = env.play(best_action[0], best_action[1], log_dir, render=render,
                                           render_delay=render_delay)
            saves = saves + 1
            agent.add_to_memory(current_state, next_states[best_action], reward, done)
            current_state = next_states[best_action]
            steps += 1
            if saves % 100000:
                print(saves)
                print(episode)
        scores.append(env.get_game_score())

        # Train
        if episode % train_every == 0:
            agent.train(batch_size=batch_size, epochs=epochs)

        # Logs
        if log_every and episode and episode % log_every == 0:
            avg_score = mean(scores[-log_every:])
            min_score = min(scores[-log_every:])
            max_score = max(scores[-log_every:])

            log.log(episode, avg_score=avg_score, min_score=min_score,
                    max_score=max_score)

        agent.model.save(log_dir + "/model.h5")


if __name__ == "__main__":
    dqn()