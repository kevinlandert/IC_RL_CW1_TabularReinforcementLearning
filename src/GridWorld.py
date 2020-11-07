
import numpy as np
import random
import matplotlib.pyplot as plt
import math

class GridWorld(object):
    
    def __init__(self,p = 0.65):
        
        ### Attributes defining the Gridworld #######

        # Shape of the gridworld
        self.shape = (6,6)
        
        # Locations of the obstacles
        self.obstacle_locs = [(1,1),(2,3),(2,5),(3,1),(4,1),(4,2),(4,4)]
        
        # Locations for the absorbing states
        self.absorbing_locs = [(1,3),(4,3)]
        
        # Rewards for each of the absorbing states 
        self.special_rewards = [+10,-100] # Corresponds to each of the absorbing_locs
        
        # Reward for all the other states
        self.default_reward = -1
        
        # Action names
        self.action_names = ['N','E','S','W'] # Action 0 is 'N', 1 is 'E' and so on
        
        # Number of actions
        self.action_size = len(self.action_names)
        
        # Randomizing action results: [1 0 0 0] to no Noise in the action results.
        self.action_randomizing_array = [p,(1-p)/3,(1-p)/3,(1-p)/3]
        
        #Get attributes defining the world
        state_size, T, R, absorbing, locs = self.build_grid_world()
        
        #Number of valid states in the world
        self.state_size = state_size
        
        
        # Transition operator (3D tensor)
        self.T = T # T[st+1, st, a] gives the probability that action a will 
                   # transition state st to state st+1
        
        # Reward function (3D tensor)
        self.R = R # R[st+1, st, a ] gives the reward for transitioning to state
                   # st+1 from state st with action a
        
        # Absorbing states
        self.absorbing = absorbing
        
        # The locations of the valid states 
        self.locs = locs # State 0 is at the location self.locs[0] and so on
        
        # Starting location
        #TODO: randomize this
        self.starting_loc =  self.get_random_start()
        
        #Absorbing states
        self.absorbing_states = [self.loc_to_state(state_id,self.locs) for state_id in self.absorbing_locs]

        
        
        # Number of the starting state
        self.starting_state = self.loc_to_state(self.starting_loc, locs);
        
        # Locating the initial state
        self.initial = np.zeros((1,len(locs)));
        self.initial[0,self.starting_state] = 1
        
        # Placing the walls on a bitmap
        self.walls = np.zeros(self.shape);
        for ob in self.obstacle_locs:
            self.walls[ob]=1
            
        # Placing the absorbers on a grid for illustration
        self.absorbers = np.zeros(self.shape)
        for ab in self.absorbing_locs:
            self.absorbers[ab] = -1
        
        # Placing the rewarders on a grid for illustration
        self.rewarders = np.zeros(self.shape)
        for i, rew in enumerate(self.absorbing_locs):
            self.rewarders[rew] = self.special_rewards[i]
        
        #Illustrating the grid world
        self.paint_maps()
        
        
    #takes list of locations and a loc and gives the index corresponding to input loc
    def loc_to_state(self,loc,locs):
            return locs.index(tuple(loc))
    
    def get_random_start(self):
        states = [state for state in self.locs if state not in self.absorbing_locs]
        return random.choice(states)
        
    #Check if a specific location is valid and not an obstacle
    def is_location(self,loc):
        if(loc[0] < 0 or loc[1] < 0 or loc[0] > self.shape[0] - 1 or loc[1] > self.shape[1] -1):
            return False
        elif(loc in self.obstacle_locs):
            return False
        else:
            return True
        
    def get_neighbour(self, loc,direction):
        
        i = loc[0]
        j = loc[1]
        
        nr = (i-1,j)
        ea = (i,j+1)
        so = (i+1,j)
        we = (i,j-1)
        
        #If the neighbour is a valid location, accept it, otherwise, stay put
        if (direction == 'nr' and self.is_location(nr)):
            return nr
        elif (direction == 'ea' and self.is_location(ea)):
            return ea
        elif (direction == 'so' and self.is_location(so)):
            return so
        elif (direction == 'we' and self.is_location(we)):
            return we
        else:
            #Default stay where you are
            return loc
      
    ####### Getters ###########
    
    def get_transition_matrix(self):
        return self.T
    
    def get_reward_matrix(self):
        return self.R
    
    def get_valuefunction(self):
        return self.Value
    
    ########################
    
    
    
    #Returns the locations of our states, the number of their neighbours and the absorbing states
    #locations of our states: list of tuples
    #2D array: number of states x neighbours (same number if wall or blocked)
    #absorbing states: array length number of states 1 for absorbing states
    def get_topology(self):
    
        height = self.shape[0]
        width = self.shape[1]
        
        #index = 1
        locs = []
        neighbour_locs = []
        
        for i in range(height):
            for j in range(width):
                
                #Get the location of each state
                loc = (i,j)
                
                #And append if is is a valid state i.e. not absorbing
                if (self.is_location(loc)):
                    locs.append(loc)
                    
                    #Get an array with all neighbours of each state, in terms of locations
                    local_neighbours = [self.get_neighbour(loc,direction) for direction in ['nr','ea','so','we']]
                    neighbour_locs.append(local_neighbours)
                    
        #translate neighbour lists from locations to states
        num_states = len(locs)
        state_neighbours = np.zeros((num_states,4))
        
        for state in range(num_states):
            
            for direction in range(4):
                #Get neighbour location
                nloc = neighbour_locs[state][direction]
                
                #Turn location into state number 
                nstate = self.loc_to_state(nloc,locs)
                
                #Insert into neighbour matrix
                state_neighbours[state][direction] = nstate
                
        #Translate absorbing locations into state indices
        absorbing = np.zeros((1,num_states))
        for a in self.absorbing_locs:
            absorbing_statenum = self.loc_to_state(a,locs)
            absorbing[0,absorbing_statenum] = 1
        
        return locs,state_neighbours,absorbing
        
    #Builds the Grid world 
    #Return the Number of states, Transition Matrix, Reward matrix, absorbing states, and the
    #locations of all valid states
    def build_grid_world(self):
        
        locations, neighbours,absorbing = self.get_topology()
        #Get number of sttes
        S = len(locations)
        
        self.neighbours = neighbours
        #Initialize the transition matrix
        T = np.zeros((S,S,4))
        
        #Fill the transition Matrix
        for action in range(4):
            for effect in range(4):
                outcome = (action+effect+1) % 4
                if outcome == 0:
                    outcome = 3
                else:
                    outcome -= 1                
                
                
                
                
                
                prob = self.action_randomizing_array[effect]
                for prior_state in range(S):
                    post_state = neighbours[prior_state,outcome]
                    post_state = int(post_state)
                    T[post_state,prior_state,action] = T[post_state,prior_state,action]+prob
                    
    
        R = self.default_reward*np.ones((S,S,4))
        for i, sr in enumerate(self.special_rewards):
            post_state = self.loc_to_state(self.absorbing_locs[i],locations)
            R[post_state,:,:] = sr
        
        
        return S,T,R,absorbing,locations
    
        ########### Internal Drawing Functions #####################

    ## You do not need to understand these functions in detail in order to complete the lab ##


    def draw_deterministic_policy(self, Policy):
        # Draw a deterministic policy
        # The policy needs to be a np array of 22 values between 0 and 3 with
        # 0 -> N, 1->E, 2->S, 3->W
        plt.figure()
        
        plt.imshow(self.walls+self.rewarders +self.absorbers) # Create the graph of the grid
        #plt.hold('on')
        for state, action in enumerate(Policy):
            if(self.absorbing[0,state]): # If it is an absorbing state, don't plot any action
                continue
            arrows = [r"$\uparrow$",r"$\rightarrow$", r"$\downarrow$", r"$\leftarrow$"] # List of arrows corresponding to each possible action
            action_arrow = arrows[action] # Take the corresponding action
            location = self.locs[state] # Compute its location on graph
            plt.text(location[1], location[0], action_arrow, ha='center', va='center') # Place it on graph
    
        plt.show()

    
    def draw_value(self, Value):
        # Draw a policy value function
        # The value need to be a np array of 22 values 
        plt.figure()
        
        plt.imshow(self.walls+self.rewarders +self.absorbers) # Create the graph of the grid
        for state, value in enumerate(Value):
            if(self.absorbing[0,state]): # If it is an absorbing state, don't plot any value
                continue
            location = self.locs[state] # Compute the value location on graph
            plt.text(location[1], location[0], round(value,2), ha='center', va='center') # Place it on graph
    
        plt.show()


    def draw_deterministic_policy_grid(self, Policy, title, n_columns, n_lines):
        # Draw a grid of deterministic policy
        # The policy needs to be an arrya of np array of 22 values between 0 and 3 with
        # 0 -> N, 1->E, 2->S, 3->W
        plt.figure(figsize=(20,8))
        for subplot in range (len(Policy)): # Go through all policies
          ax = plt.subplot(n_columns, n_lines, subplot+1) # Create a subplot for each policy
          ax.imshow(self.walls+self.rewarders +self.absorbers) # Create the graph of the grid
          for state, action in enumerate(Policy[subplot]):
              if(self.absorbing[0,state]): # If it is an absorbing state, don't plot any action
                  continue
              arrows = [r"$\uparrow$",r"$\rightarrow$", r"$\downarrow$", r"$\leftarrow$"] # List of arrows corresponding to each possible action
              action_arrow = arrows[action] # Take the corresponding action
              location = self.locs[state] # Compute its location on graph
              plt.text(location[1], location[0], action_arrow, ha='center', va='center') # Place it on graph
          ax.title.set_text(title[subplot]) # Set the title for the graoh given as argument
        plt.show()

    def draw_value_grid(self, Value, title, n_columns, n_lines):
        # Draw a grid of value function
        # The value need to be an array of np array of 22 values 
        plt.figure(figsize=(20,8))
        for subplot in range (len(Value)): # Go through all values
          ax = plt.subplot(n_columns, n_lines, subplot+1) # Create a subplot for each value
          ax.imshow(self.walls+self.rewarders +self.absorbers) # Create the graph of the grid
          for state, value in enumerate(Value[subplot]):
              if(self.absorbing[0,state]): # If it is an absorbing state, don't plot any value
                  continue
              location = self.locs[state] # Compute the value location on graph
              plt.text(location[1], location[0], round(value,1), ha='center', va='center') # Place it on graph
          ax.title.set_text(title[subplot]) # Set the title for the graoh given as argument
        plt.show()

    ##########################
    
    def paint_maps(self):
        # Helper function to print the grid word used in __init__
        plt.figure()
        plt.subplot(1,3,1)
        plt.imshow(self.walls)
        plt.title('Obstacles')
        plt.subplot(1,3,2)
        plt.imshow(self.absorbers)
        plt.title('Absorbing states')
        plt.subplot(1,3,3)
        plt.imshow(self.rewarders)
        plt.title('Reward states')
        plt.show()
        
    
    
    def get_trace(self,policy,exploring_start = False):
        
        start = self.get_random_start()
        start_num = self.loc_to_state(start,self.locs)
        state_id = start_num
        trace = []
        
        #Creates consequite state,action,reward until reaches terminal state
        while(state_id not in self.absorbing_states):
            
            #Takes prefered action
            if exploring_start:
                action = np.random.randint(4)
            else:
                action = np.argmax(policy[state_id])
            #Probablility of successor states in this combo
            prob = self.T[:,state_id,action]
            state_id_prime = np.random.choice(np.arange(prob.shape[0]),p = prob)
            reward = self.R[state_id_prime,state_id,action]
            trace.append((state_id,action,reward))
            state_id = state_id_prime
        #trace.append((state_id,None,None))
        return trace
    
    
    def get_step(self,state_id,action):
        
        done = False
        prob = self.T[:,state_id,action]
        state_id_prime = np.random.choice(np.arange(prob.shape[0]),p = prob)
        reward = self.R[state_id_prime,state_id,action]
        if state_id_prime in self.absorbing_states:
            done = True
        return reward,state_id_prime,done
        
        
        
    
    #Given optimal estimated action for every state creates a greedy policy
    """
    def choose_epolicy(self,epsilon,policy):
        
        action = np.zeros(policy.shape)
        
        for i in range(policy.shape[0]):
            eaction = self.choose_eaction(epsilon,policy[i])
            action[i,eaction] = 1
        return action
    
    """
    
    
    #Given a row of our Q (state_action matrix) chooses a egreedy action
    def choose_eaction(self,epsilon,state_action_row):
        
        maxindex = np.argmax(state_action_row)
        prob = np.ones(4) * epsilon / 4
        prob[maxindex] = 1 - epsilon + epsilon / 4
        return np.random.choice(np.arange(4),p = prob)
    
    
    def choose_epolicy(self,epsilon,Q):
        
        epolicy = np.zeros(Q.shape)
        
        for i in range(Q.shape[0]):
            maxindex = np.argmax(Q[i,:])
            prob = np.ones(4) * (epsilon / 4)
            prob[maxindex] = 1 - epsilon + (epsilon / 4)
            epolicy[i,:] = prob
        return epolicy
    
    """
    def choose_epolicy(self,epsilon,Q):
        
        epolicy = np.zeros(Q.shape)
        
        for i in range(Q.shape[0]):
            
            if np.random.rand() < epsilon:
                action = np.argmax(Q[i, :])
            else:
                action = np.random.randint(0, Q.shape[1])
            
            epolicy[i,action] = 1
        return epolicy
   """ 
        
    #Given optimal estimated action for every state creates a greedy policy
    def choose_policy(self,Q):
        
        action = np.zeros(Q.shape)
        maxindex = np.argmax(Q,axis = 1)
        for i in range(len(maxindex)):
            action[i][maxindex[i]] = 1
        return action
    
    
    def epsilon(self,i):
        return 1 / i
    
    def alpha(self,i):
        return 1 / (2 ** i)
     


    
    def MC_algorithm(self,discount):
        
        epochs = 0
        Q = np.zeros((self.state_size,self.action_size))
        policy = self.choose_policy(Q)
        count_per_state = np.ones((self.state_size,self.action_size))

        #print(Q)
        #print(policy)
        
        while epochs < 5000:
            
            epochs += 1
            oldpolicy = policy
            
            Q,count_per_state = self.MC_Estimation(Q,policy,discount,count_per_state)
            #print(Q)
            policy = self.choose_policy(Q)
        
        return (policy,Q,epochs)
    
    
    #Takes lists (empty dictionary) and a trace and returns lists with the chain of rewards for every state onward
    def create_lists(self,trace,lists):
    
        for episode in trace:
        
            state_id,action,reward = episode
        
            if (state_id,action) not in lists:
                lists[(state_id,action)] = []
            
            for state_action_pair in lists:
                lists[state_action_pair].append(reward)
    
        return lists
    
    #Computes the discounted result for every state in the trace
    def discount_rewards(self,lists,discount):
    
        for state_action in lists:
            dis_reward = 0 
            for time,reward in enumerate(lists[state_action]):
                dis_reward += (discount ** time) * reward
            lists[state_action] = dis_reward
    
        return lists
    
    def policy_setup(self):
        random = np.random.random((self.state_size,self.action_size))
        self.choose_policy(random)
        return self.choose_policy(random)

    def Q_setup(self,terminal_to_zero):
        Q = np.random.random((self.state_size,self.action_size))
        #Q = np.ones((self.state_size,self.action_size))
        if terminal_to_zero:
            for i in self.absorbing_states:
                Q[i,:] = 0
        return Q
        
    #Monte Carlo estimation
    def mc_estimation(self,alpha,discount,epochs):
    
        policy  = self.policy_setup()
        Q = self.Q_setup(False) -  self.Q_setup(False)
        averages_per_state = self.Q_setup(False)
    
        for i in range(epochs):
        
            trace = self.get_trace(policy,True)
        
            #Gets discounted rewards per state_action pair 
            lists = {}
            lists = self.create_lists(trace,lists)
            lists = self.discount_rewards(lists,discount)
        
            for state,action in lists:
                #Q[state,action] += 1 / averages_per_state[state,action] * (lists[(state,action)] - Q[state, action])
                Q[state,action] += alpha * (lists[(state,action)] - Q[state, action])
            
                averages_per_state[state,action] += 1
          
        policy = self.choose_policy(Q)
        
        return Q,policy,epochs
    
    def mc_egreedy_fv(self,episodes,discount):
        
        Q = self.Q_setup(False)
        
        #Adjust policy here to be e-soft policy
        policy = self.policy_setup()
        Returns = {}
        
        
        for i in range(episodes):
            
            trace = self.get_trace(policy,False)
            
            lists = {}
            lists = self.create_lists(trace,lists)
            lists = self.discount_rewards(lists,discount)
            
            for key in lists:
                if key not in Returns:
                    Returns[key] = []
                Returns[key].append(lists[key])
                
            
            for key in Returns:
                state,action = key
                Q[state,action] = np.mean(Returns[key])
                
            print(Returns)
             
            opt_pol = np.argmax(Q,axis = 1)
            policy = self.choose_epolicy(epsiolon,Q)
    
    
    def get_reward_of_trace(self,trace,discount):
        rewards = [element[2] for element in trace]
        discounts = discount ** np.arange(len(trace))
        total_reward = sum(rewards * discounts)
        return total_reward
    
    
    def reward_log(self,state_id,dis_reward):
        
        self.rewards[state_id] = dis_reward
        
    def get_av_reward(self):
        
        acc = 0
        
        for key in self.rewards:
            acc += self.rewards[key]
            
        return acc / len(self.rewards)
    
    #Question 2
    def value_iteration(self,discount = 0.2,threshold = 0.0001):
    
        T = self.get_transition_matrix()
        R = self.get_reward_matrix()
        
        # Initialisation
        epochs = 0
        delta = threshold # Setting value of delta to go through the first breaking condition
        V = np.zeros(self.state_size) # Initialise values at 0 for each state
        
        while delta >= threshold:
            epochs += 1
            delta = 0
            
            #Loop over all states
            for state_id in range(self.state_size):
                
                
                #Only look at non absorbing states
                if not (self.absorbing[0,state_id]):
                    
                    v = V[state_id]
                    
                    #Compute Q value
                    Q = np.zeros(4)
                    for state_id_prime in range(self.state_size):
                        Q += T[state_id_prime,state_id,:] * (R[state_id_prime,state_id,:] + discount * V[state_id_prime])
                        
                    V[state_id] = np.max(Q)  
                    
                    delta = max(delta,np.abs(v - V[state_id]))
                                
        optimal_policy = np.zeros((self.state_size, self.action_size))  
        
        for state_id in range(self.state_size):
            
            Q = np.zeros(4)
            
            for state_id_prime in range(self.state_size):
                Q += T[state_id_prime,state_id,:] * (R[state_id_prime,state_id,:] + discount * V[state_id_prime])
            
            optimal_policy[state_id,np.argmax(Q)] = 1
            
        self.Value = V
        self.policy = optimal_policy
                            
        return V,optimal_policy, epochs
    
    def policy_iteration(self, discount=0.9, threshold = 0.0001):
        ## Slide 139 of the lecture notes for pseudocode ##
        
        # Transition and reward matrices, both are 3d tensors, c.f. internal state
        T = self.get_transition_matrix()
        R = self.get_reward_matrix()
        
        # Initialisation
        policy = np.zeros((self.state_size, self.action_size)) # Vector of 0
        policy[:,0] = 1 # Initialise policy to choose action 1 systematically
        epochs = 0
        policy_stable = False # Condition to stop the main loop

        while not(policy_stable): 

            # Policy evaluation
            V, epochs_eval = self.policy_evaluation(policy, threshold, discount)
            epochs += epochs_eval # Increment epoch

            # Set the boolean to True, it will be set to False later if the policy prove unstable
            policy_stable = True

            # Policy iteration
            for state_idx in range(policy.shape[0]):
                
                # If not an absorbing state
                if not(self.absorbing[0,state_idx]):
                    
                    # Store the old action
                    old_action = np.argmax(policy[state_idx,:])
                
                    # Compute Q value
                    Q = np.zeros(4) # Initialise with value 0
                    for state_idx_prime in range(policy.shape[0]):
                        Q += T[state_idx_prime,state_idx,:] * (R[state_idx_prime,state_idx, :] + discount * V[state_idx_prime])

                    # Compute corresponding policy
                    new_policy = np.zeros(4)
                    new_policy[np.argmax(Q)] = 1  # The action that maximises the Q value gets probability 1
                    policy[state_idx] = new_policy
                
                    # Check if the policy has converged
                    if old_action != np.argmax(policy[state_idx]):
                        policy_stable = False
           
        self.Value = V
        self.policy = policy
        
        return V, policy, epochs
    
    def policy_evaluation(self, policy, threshold, discount):
        
        # Make sure delta is bigger than the threshold to start with
        delta= 2*threshold
        
        #Get the reward and transition matrices
        R = self.get_reward_matrix()
        T = self.get_transition_matrix()
        
        # The value is initialised at 0
        V = np.zeros(policy.shape[0])
        # Make a deep copy of the value array to hold the update during the evaluation
        Vnew = np.copy(V)
        
        epoch = 0
        # While the Value has not yet converged do:
        while delta>threshold:
            epoch += 1
            for state_idx in range(policy.shape[0]):
                # If it is one of the absorbing states, ignore
                if(self.absorbing[0,state_idx]):
                    continue   
                
                # Accumulator variable for the Value of a state
                tmpV = 0
                for action_idx in range(policy.shape[1]):
                    # Accumulator variable for the State-Action Value
                    tmpQ = 0
                    for state_idx_prime in range(policy.shape[0]):
                        tmpQ = tmpQ + T[state_idx_prime,state_idx,action_idx] * (R[state_idx_prime,state_idx, action_idx] + discount * V[state_idx_prime])
                    
                    tmpV += policy[state_idx,action_idx] * tmpQ
                    
                # Update the value of the state
                Vnew[state_idx] = tmpV
            
            # After updating the values of all states, update the delta
            # Note: The below is our example way of computing delta.
            #       Other stopping criteria may be used (for instance mean squared error).
            #       We encourage you to explore different ways of computing delta to see 
            #       how it can influence outcomes.
            delta =  max(abs(Vnew-V))
            # and save the new value into the old
            V=np.copy(Vnew)
            
        return V, epoch
    
    
    
    #Question 3        
    def mc_iterative(self,epsilon,discount,episodes):
    
        policy  = self.policy_setup()
        state_action_upt_ctr = np.zeros((self.state_size,self.action_size))
        Q = self.Q_setup(False)
        #print(Q)
        learning = []
        mse = []
    
        for i in range(episodes):
        
            trace = self.get_trace(policy,False)
            dis_reward = self.get_reward_of_trace(trace,discount)
            learning.append(dis_reward)
            
            #print(self.get_reward_of_trace(trace,discount))
        
            #Gets discounted rewards per state_action pair 
            lists = {}
            lists = self.create_lists(trace,lists)
            lists = self.discount_rewards(lists,discount)
            #print(lists)
        
            for state,action in lists:
                Q[state,action] += 1/ (state_action_upt_ctr[state,action] + 1) * (lists[(state,action)] - Q[state, action])
                state_action_upt_ctr[state,action] += 1
            
            #policy = self.choose_epolicy(epsilon(i+1),Q)
            if i < 200:
                policy = self.choose_epolicy(1 / (i + 1),Q)
            else:
                policy = self.choose_policy(Q)
            bestpolicy = self.choose_policy(Q)
            Vest = np.sum(np.multiply(bestpolicy,Q),axis = 1)
            mse.append(((Vest - self.Value)**2).mean(axis=None))
         
        #In the end choose optimal policy
        policy = self.choose_policy(Q)
        
        print(state_action_upt_ctr)

        return Q,policy,learning,mse,state_action_upt_ctr
            
        
    #Question 4
    
      #Given a row of our Q (state_action matrix) chooses a egreedy action
    def choose_eaction_sarsa(self,epsilon,state_action_row):
         
        if np.random.rand() < epsilon:
            action = np.argmax(state_action_row)
        else:
            action = np.random.randint(0, len(state_action_row))
        return action
    
    
    def SARSA_Control(self,discount,alpha,epsilon,episodes):
        
        Q = self.Q_setup(True)
        
        #Counts how often we have seen a state-action pair for correct aver
        alpha = np.ones((self.state_size,self.action_size))
        
        #For analysis
        learning = []
        mse = []
        
        
        for episode in range(episodes):
            
            start = self.get_random_start()
            start_num = self.loc_to_state(start,self.locs)
            state_id = start_num
            total_reward = 0
            
            #Chooses action according to epsilon policy
            action = self.choose_eaction_sarsa(1 / (episode + 1),Q[state_id])
            steps = 0
            
            while(True):
                reward,state_id_prime, done = self.get_step(state_id,action)
                total_reward += discount ** steps * reward
                steps+= 1
                action_prime = self.choose_eaction_sarsa(1 / (episode + 1),Q[state_id_prime])
                if done: 
                    Q[state_id,action] = Q[state_id,action] + 1/ alpha[state_id,action] * (reward  - Q[state_id,action])
                    alpha[state_id,action] += 1
                    break
                else:
                    Q[state_id,action] = Q[state_id,action] + 1/ alpha[state_id,action] * (reward + discount * Q[state_id_prime,action_prime] - Q[state_id,action])
                    alpha[state_id,action] += 1
                state_id = state_id_prime
                action = action_prime
                
            learning.append(total_reward)
            
            epolicy = self.choose_policy(Q) #self.choose_epolicy_sarsa(1/episodes,Q)
            Vest = np.sum(np.multiply(epolicy,Q),axis = 1)
            mse.append(((Vest - self.Value)**2).mean(axis=None))
            
        print(alpha)
        epolicy = self.choose_epolicy_sarsa(1/episodes,Q)
        opt_policy = self.choose_policy(Q)
        return Q,opt_policy,learning,mse,alpha