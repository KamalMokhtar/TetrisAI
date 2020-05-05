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
    episodes = 200000  # 2000
    max_steps = None
    epsilon_stop_episode = 190000  # 1500
    mem_size = 2000000  # 20000
    discount = 0.95
    batch_size = 512  # 512
    epochs = 1
    render_every = 1  # 50
    log_every = 50  # 50
    replay_start_size = 7000  # 2000
    train_every = 1
    n_neurons = [160, 160, 160, 160, 160] # [32, 32]
    render_delay = None
    activations = ['relu', 'relu', 'relu', 'relu', 'relu', 'linear'] # ['relu', 'relu', 'linear']

    # -------------------In play mode------------------- #
    # put model name that you want it to play in model_name below
    # set render_every = 1  in line 21
    # set agent_play below to True
    # set fetch_old_model below to True
    # the model will not be trained nor will be saved
    # the model lines clearing scoring will be saved if you let the model finish the episodes set above
    # ------------------- play mode Nuno Faria ------------------- #
    # https://github.com/nuno-faria/tetris-ai
    # same steps as in play mode, then
    # set board_state True
    # model_name = 'models/original'
    # ------------------- training from scratch mode Nuno Faria ------------------- #
    # set fetch_old_model = False, agent_play = false, board_state = True and or
    # set the return parameter to score in th function play in tetris.py
    # ------------------- if training from scratch full board------------------- #
    # set fetch_old_model = False, agent_play = false, board_state = False
    # you can choose sum_model_reward or score in th function play in tetris.py
    # sum_model_reward is the matrix reward
    # score the oridnary Tetris game reward
    # ------------------- continue training ------------------- #
    # can continue the model but keep in mind that it will start exploring in the beginning again depending on epsilon_stop_episode
    # set the model name that you want to continue training from
    # set fetch_old_model = True, agent_play = false, board_state = false
    # you can choose sum_model_reward or score in th function play in tetris.py
    # after the training is done, give the model and the files appropriate name

    # all model names, line_logging and the logs will be save with the same time stamp

    model_name = 'models/my_model-20200504-233443-h5'
    # Rendering false for Peregrine
    rendering = False
    fetch_old_model = False
    agent_play = False
    board_state = True
    if board_state:
        input_size = [1, 4]
        input_width = 4
    else:
        input_size = [1, 200]
        input_width= 200

    agent = DQNAgent(env.get_board_size(),
                     n_neurons=n_neurons, activations=activations,
                     epsilon_stop_episode=epsilon_stop_episode, mem_size=mem_size,
                     discount=discount, replay_start_size=replay_start_size, fetch_old_model=fetch_old_model, model_name=model_name, input_width=input_width)

    time_frame = datetime.now().strftime("%Y%m%d-%H%M%S")

    open("lines_logging/" + f"linesfile-{time_frame}.txt", "w") # creating file for the line logging
    log_dir = f"logs/tetris-nn={str(n_neurons)}-mem={mem_size}-bs={batch_size}-e={epochs}-{time_frame}"
    log = CustomTensorBoard(log_dir=log_dir)

    scores = []

    for episode in tqdm(range(episodes)):

        env.reset()
        if board_state:
            current_board = [0]*4
        else:
            current_board = [0]*200

        done = False
        steps = 0
        if rendering:
            if render_every and episode % render_every == 0:
                render = True
            else:
                render = False
        else:
            render = False
        # *k Game
        while not done and (not max_steps or steps < max_steps): #agent_train, input_size
            next_boards = env.get_next_boards(board_state)  # returns the all possible moves in next_state
            best_board = agent.best_board(next_boards.values(), agent_play, input_size, board_state)

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
        if episode % train_every == 0 and not agent_play:
            agent.train(batch_size=batch_size, epochs=epochs)
        # Logs
        if log_every and episode and episode % log_every == 0:
            avg_score = mean(scores[-log_every:])
            min_score = min(scores[-log_every:])
            max_score = max(scores[-log_every:])

            log.log(episode, avg_score=avg_score, min_score=min_score,
                    max_score=max_score)
    # save the model
    if not agent_play:
        agent.model_save(time_frame)


if __name__ == "__main__":
    dqn()
