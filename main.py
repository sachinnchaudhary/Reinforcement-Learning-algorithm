import gym
import numpy


#deep q network: FrozenLake-v1

env =gym.make('FrozenLake', render_mode = "human")

if not isinstance(env.action_space, gym.spaces.Discrete):
    raise ValueError("Action space is not Discrete")
if not isinstance(env.observation_space, gym.spaces.Discrete):
    raise ValueError("Observation space is not Discrete")



q_table = numpy.zeros((env.observation_space.n, env.action_space.n ))

print(f"q table shape{q_table.shape}")
print(f"observations:{env.observation_space.n}")
print(f"actions:{env.action_space.n}")


alpha = 0.1 #learning rate

gemma = 0.99 #discount factor

exploration = 0.1 #exploration rate

episodes = 10000

for episode in range(episodes):

    state_tuple =env.reset()
    state = state_tuple[0]
    done = False


    while not done:

        if numpy.random.uniform(0, 1) < exploration:

            action  = env.action_space.sample() #priotirize exploreness

        else:
            action= numpy.argmax(q_table[state])  #exploitness choose the action with highest q value
            print(action)

        next_state , reward , done, info, adii_info =  env.step(action)

        #Q-learning update

        best_next_action  = numpy.argmax(q_table[next_state])
        td_target =  reward + gemma * q_table[next_state][best_next_action]
        td_error =  td_target - q_table[state, action]
        q_table[state, action] += alpha* td_error


        state = next_state

        if episode % 1000 == 0:
             print(f"episode is done{episode}")
print(f"trained q table {q_table}")

env.close()
