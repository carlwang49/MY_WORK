
import numpy as np

class QLearningAgent():

    def __init__(self, num_actions, num_states, eps_start=1.0, eps_decay=.9999, eps_min=1e-08, step_size=0.1, gamma=1):
        # Initialise agent
        self.num_actions = num_actions
        self.num_states = num_states
        self.epsilon = eps_start
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.step_size = step_size
        self.gamma = gamma
        self.rand_generator = np.random.RandomState(1)
        
        # Create an array for action-value estimates and initialize it to zero.
        self.state_dict = {}
        self.q = np.zeros((self.num_states, self.num_actions)) # The array of action-value estimates.

        
    def agent_start(self, state):

        #Update epsilon at the start of each episode
        self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)
        #Add state to dict if new + get index of state
        if state not in self.state_dict.keys():
            self.state_dict[state] = len(self.state_dict)
        state_idx = self.state_dict[state]

        # Choose action using epsilon greedy.
        current_q = self.q[state_idx,:]
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions) # random action selection
        else:
            action = self.argmax(current_q) # greedy action selection

        self.prev_state_idx = self.state_dict[state]
        self.prev_action = action
        return action
    
    def agent_step(self, reward, state):

        #Add state to dict if new + get index of state
        if state not in self.state_dict.keys():
            self.state_dict[state] = len(self.state_dict)
        state_idx = self.state_dict[state]

        # Choose action using epsilon greedy.
        current_q = self.q[state_idx, :]
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(current_q)
        
        # Update q values
        self.q[self.prev_state_idx][self.prev_action] = (1-self.step_size) * self.q[self.prev_state_idx][self.prev_action] + \
                                                     self.step_size * (reward + self.gamma * self.q[state_idx][self.argmax(current_q)])
        
        self.prev_state_idx = self.state_dict[state]
        self.prev_action = action
        return action
    
    def agent_end(self, state, reward):
        #Add state to dict if new + get index
        if state not in self.state_dict.keys():
            self.state_dict[state] = len(self.state_dict)
        state_idx = self.state_dict[state]
        
        # Perform the last q value update
        self.q[self.prev_state_idx][self.prev_action] = (1-self.step_size) * self.q[self.prev_state_idx][self.prev_action] + \
                                                    self.step_size * reward

    # Takes step in testing environment, epsilon=0 and no updates made
    def test_step(self, state):
        
        #Add state to dict if new + get index of state
        if state not in self.state_dict.keys():
            self.state_dict[state] = len(self.state_dict)
        state_idx = self.state_dict[state]

        # Pick greedy action
        current_q = self.q[state_idx, :]
        action = self.argmax(current_q)

        return action

        
    def argmax(self, q_values):
        top = float("-inf")
        ties = []

        for i in range(len(q_values)):
            if q_values[i] > top:
                top = q_values[i]
                ties = []

            if q_values[i] == top:
                ties.append(i)

        return self.rand_generator.choice(ties)