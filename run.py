from dqn_agent import DQNAgent
from tetris import Tetris
from datetime import datetime
from statistics import mean, median
import random
from logs import CustomTensorBoard
from tqdm import tqdm
import itertools


# Run dqn with Tetris
def dqn():
    env = Tetris()
    episodes = 150000  # 2000
    max_steps = None
    epsilon_stop_episode = 140000# 1500
    mem_size = 1500000  # 20000
    discount = 0.95
    batch_size = 2500  # 512
    epochs = 1
    render_every = 1  # 50
    log_every = 50  # 50
    replay_start_size = 150000  # 2000
    train_every = 1
    n_neurons = [360, 360, 360, 360, 360]
    render_delay = None
    activations = ['relu', 'relu', 'relu', 'relu', 'relu', 'linear']
    # if model play put agent_train False
    # if model train, put both True
    model_save = True
    # if not training put the right model name you want to retrieve in _build_model function
    agent_train = False
    agent_play = True
    # if you want model original to play set board_state True
    board_state = True
    if board_state:
        input_size = [1, 4]
    else:
        input_size = [1, 200]

    agent = DQNAgent(env.get_board_size(),
                     n_neurons=n_neurons, activations=activations,
                     epsilon_stop_episode=epsilon_stop_episode, mem_size=mem_size,
                     discount=discount, replay_start_size=replay_start_size)

    time_frame = datetime.now().strftime("%Y%m%d-%H%M%S")

    open('lines_logging/' + f'linesfile-{time_frame}.txt', 'w') # creating file for the line logging
    log_dir = f'logs/tetris-nn={str(n_neurons)}-mem={mem_size}-bs={batch_size}-e={epochs}-{time_frame}'
    log = CustomTensorBoard(log_dir=log_dir)

    scores = []

    for episode in tqdm(range(episodes)):

        current_board = env.reset()
        done = False
        steps = 0

        if render_every and episode % render_every == 0:
            render = True
        else:
            render = False

        # *k Game
        while not done and (not max_steps or steps < max_steps, agent_train, input_size):
            next_boards = env.get_next_boards(board_state)  # returns the all possible moves in next_state
            best_board = agent.best_board(next_boards.values(),agent_play ,input_size, board_state)

            best_action = None

            for action, board in next_boards.items():  # Find the corresponding action for the desired board
                if board == best_board:
                    best_action = action
                    break
            # Reward each block placed yields 1 point. When clearing lines, the given score is
            # number_lines_cleared^2 × board_width. Losing a game subtracts 1 point.
            reward, done = env.play(best_action[0], best_action[1], render=render,
                                    render_delay=render_delay, time_frame=time_frame)

            if board_state:
                agent.add_to_memory(current_board, next_boards[best_action], reward, done)
            else:
                agent.add_to_memory(list(itertools.chain.from_iterable(current_board)),
                                    list(itertools.chain.from_iterable(next_boards[best_action])), reward, done)

            current_board = next_boards[best_action]
            steps += 1

        scores.append(env.get_game_score())

        # Train
        if episode % train_every == 0 and agent_train:
            agent.train(batch_size=batch_size, epochs=epochs)
        # Logs
        if log_every and episode and episode % log_every == 0:
            avg_score = mean(scores[-log_every:])
            min_score = min(scores[-log_every:])
            max_score = max(scores[-log_every:])

            log.log(episode, avg_score=avg_score, min_score=min_score,
                    max_score=max_score)
    # save the model
    if model_save and agent_train:
        agent.model_save(time_frame)


if __name__ == "__main__":
    dqn()
