import os
import shutil
import yaml
from random import random, randint, sample
import numpy as np
import torch
import torch.nn as nn


from tetris import Tetris
from deep_q_network import DeepQNetwork




def train():
    with open('train_config.yml','r') as config_file:
        params = yaml.safe_load(config_file)
    torch.cuda.set_device(0)
    env = Tetris(width=params['params']['width'], height=params['params']['height'], block_size=params['params']['block_size'])
    model = DeepQNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['params']['learning_rate'])
    criterion = nn.MSELoss()
    replay_memory=[]
    replay_memory_max_length=params['params']['max_memory']
    state=env.reset()

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        model.cuda()
        state=state.cuda()
    num_epoch=params['params']['num_epoch']
    epoch=0
    final_cleared_lines = 0
    while final_cleared_lines<128:
    # while epoch<num_epoch:
        epsilon = params['params']['final_epsilon'] + (max(params['params']['num_decay_epoch']-epoch,0)*(params['params']['initial_epsilon']-params['params']['final_epsilon'])/params['params']['num_decay_epoch'])
        next_steps=env.get_next_states()
        u=random()
        randomize_action=False
        if u<=epsilon:
            randomize_action=True
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states)
        next_states=next_states.cuda()
        model.eval()
        with torch.no_grad():
            predictions=model(next_states)[:,0]
        model.train()
        if randomize_action:
            action_index=randint(0,len(next_steps)-1)
        else:
            action_index=torch.argmax(predictions).item()

        next_state=next_states[action_index,:]
        next_action=next_actions[action_index]
        reward,done=env.step(next_action,render=False)

        next_state=next_state.cuda()
        replay_memory.append([state,reward,next_state,done])

        if done:
            final_score=env.score
            final_tetrominos=env.tetrominoes
            final_cleared_lines=env.cleared_lines
            state=env.reset()
            state=state.cuda()
        else:
            state=next_state
            continue
        if len(replay_memory)<=replay_memory_max_length/10:
            continue
        batch=sample(replay_memory,min(len(replay_memory),params['params']['batch_size']))
        state_batch,reward_batch,next_state_batch,done_batch=zip(*batch)
        state_batch=torch.stack(tuple(state for state in state_batch))
        reward_batch=torch.from_numpy(np.array(reward_batch,dtype=np.float32)[:,None])
        next_state_batch = torch.stack(tuple(state for state in next_state_batch))

        state_batch = state_batch.cuda()
        reward_batch = reward_batch.cuda()
        next_state_batch = next_state_batch.cuda()
        epoch+=1
        Q=model(state_batch)
        model.eval()
        with torch.no_grad():
            next_prediction_batch=model(next_state_batch)
        model.train()

        y_batch = torch.cat(tuple(reward if done else reward + params['params']['gamma'] * prediction for reward, done, prediction in zip(reward_batch, done_batch, next_prediction_batch)))[:, None]
        criterion = nn.MSELoss()

        optimizer.zero_grad()
        loss = criterion(Q, y_batch)
        loss.backward()
        optimizer.step()
        print("Epoch: {}/{}, Action: {}, Score: {}, Tetrominoes {}, Cleared lines: {}".format(
            epoch,
            params['params']['num_epoch'],
            next_action,
            final_score,
            final_tetrominos,
            final_cleared_lines))
        if epoch > 0 and epoch % 200== 0:
            torch.save(model, "{}/tetris_{}".format('./log', epoch))

if __name__ == "__main__":
    train()
