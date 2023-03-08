from collections import defaultdict
import numpy as np
import ot

#DOUBLE CHECK THE LOGIC!!!

#buffer should be intialised once at start of training only to save all trajectories

class TrajectoryBuffer(object):
    '''
        For storing trajectories. Stores only trajectories from the latest play for each level and their masks.
    '''
    def __init__(self, action_space, obs_space, sample_full_distribution):
        #initialise a dictionary mapping from seed to trajectory
        self.action_buffer = defaultdict()
        self.obs_buffer = defaultdict()
        self.action_space_high = action_space.high
        self.action_space_low = action_space.low
        self.obs_space_high = obs_space.high
        self.obs_space_low = obs_space.low
        self.sample_full_distribution = sample_full_distribution

    def retrieve(self, seed):
        #retrieve the latest trajectories and masks from a level
        return self.action_buffer[seed], self.obs_buffer[seed]

    def insert(self, masks, level_seeds, action_trajs, obs_trajs):
        #insert the latest trajectories from a level
        #action_trajs and obs_trajs need to be broken up according to the level seeds
        #CHECK SHAPE OF LEVEL_SEEDS
        #self.level_seeds = torch.zeros(num_steps, num_processes, 1, dtype=torch.int)
        #self.actions = torch.zeros(num_steps, num_processes, action_shape)
        #self.obs = torch.zeros(num_steps + 1, num_processes, *observation_space.shape)
        #self.masks = torch.ones(num_steps + 1, num_processes, 1)

        #for each process we can expect a different seed
        for p in self.level_seeds.shape[1]:
            seed = np.unique(level_seeds[p])
            #ERROR CHECKING IF SEED IS NOT UNIQUE PER PROCESS - delete if confirm correct
            if seed.shape[0]!=1:
                raise Exception("Seed not unique")
            seed = seed.item()
            action_trajs_seed = action_trajs[:,p,:]
            obs_trajs_seed = obs_trajs[:,p,:]
            masks_seed = masks[:,p,:]

            #maybe break into episodes then episodes are stacked?

            #scores calculated by episode ----------
            done = ~(masks_seed > 0)
            start_t = 0
            #CHECK THIS IS CORRECT ----
            total_steps = action_trajs_seed.shape[0]

            done_steps = done.nonzero()[:,0]
            #split into episodes
            #intialise empty list for temporary holding
            temp_list_action = []
            temp_list_obs = []
            num_steps = []

            for t in done_steps:

                #probably are the steps marking conclusion of episode? -------
                if not start_t < total_steps: break

                # if t == 0: # if t is 0, then this done step caused a full update of previous seed last cycle
                #     continue 

                action_traj_episode = action_trajs_seed[start_t:t, :]
                obs_traj_episode = obs_trajs_seed[start_t:t, :]
            
                temp_list_action.append(action_traj_episode)
                temp_list_obs.append(obs_traj_episode)
                num_steps.append(action_traj_episode.shape[0])
                
                start_t = t.item()

            #transform into numpy array with np.nans as buffers
            max_steps = max(num_steps)
            action_traj_array = np.empty([len(temp_list_action), max_steps, action_traj_episode.shape[1]])
            obs_traj_array = np.zeros([len(temp_list_action), max_steps, obs_traj_episode.shape[1]])
            action_traj_array[:] = np.nan
            obs_traj_array[:] = np.nan
            for ep in range(len(temp_list_action)):
                #for each episode
                action_traj_episode = temp_list_action[ep]
                obs_traj_episode = temp_list_obs[ep]
                #insert episode trajectories into np array for smaller storage
                action_traj_array[ep,:action_traj_episode.shape[0],:] = action_traj_episode
                obs_traj_array[ep,:obs_traj_episode.shape[0],:] = obs_traj_episode
            
            #add numpy arrays into dictionary
            self.action_buffer[seed] = action_traj_array
            self.obs_buffer[seed] = obs_traj_array


    def _calculate_wasserstein_distance(self, seed_eval, seeds_compare):
        #SHAPE of each traj array: (#trajs, #timestepsw/buffer, act/obs#dim)
        #full or not distribution should be handled outside this function

        #intialise pairwise wasserstein distance array
        distances = np.array.empty(seeds_compare.shape[0])
        #retrieve current set of trajs
        seed_eval_actions = self.action_buffer[seed_eval]
        seed_eval_obs = self.obs_buffer[seed_eval]
        #process trajectories: normalise, combine
        seed_eval_actions = (seed_eval_actions - self.action_space_low)/(self.action_space_high - self.action_space_low)
        seed_eval_obs = (seed_eval_obs - self.observation_space_low)/(self.observation_space_high - self.observation_space_low)
        eval_state_action_traj = np.concatenate([seed_eval_actions, seed_eval_obs], axis=-1)

        #for each seed we compare each pair of episode trajectories and get the average
        for seed_idx in range(len(seeds_compare)):
            #retrieve seed_compare
            seed = seeds_compare[seed_idx]

            seed_compare_actions = self.action_buffer[seed]
            seed_compare_obs = self.obs_buffer[seed]
            seed_compare_actions = (seed_compare_actions - self.action_space_low)/(self.action_space_high - self.action_space_low)
            seed_compare_obs = (seed_compare_obs - self.observation_space_low)/(self.observation_space_high - self.observation_space_low)
            compare_state_action_traj = np.concatenate([seed_compare_actions, seed_compare_obs], axis=-1)
            #intialise temp array for each eval_traj
            temp_2 = []

            for eval_traj_idx in seed_eval_actions.shape[0]:
                #intialise temp array for collecting comparisons with each seed compare traj
                temp_1 = []

                #process trajectories: normalise, remove np.nans
                #samples array for distance cost mat
                eval_traj = eval_state_action_traj[eval_traj_idx,:,:]
                #remove np.nan buffer timesteps
                eval_traj = eval_traj[~np.isnan(eval_traj)]
                #uniform weight distribution for each timestep
                s_eval = [1/eval_state_action_traj.shape[0]]*eval_state_action_traj.shape[0]

                for compare_traj_idx in seed_compare_actions.shape[0]:
                    compare_traj = compare_state_action_traj[compare_traj_idx,:,:]
                    compare_traj = compare_traj[~np.isnan(compare_traj)]

                    s_compare = [1/len(compare_state_action_traj.shape[0])]*compare_state_action_traj.shape[0]
                    
                    #create cost matrix using euclidean distance as cost
                    M = ot.dist(eval_traj, compare_traj, metric='euclidean')
                    #calculate wasserstein distance
                    W = ot.emd2(s_eval, s_compare, M)
                    temp_1.append(W)

                #average W
                temp_2.append(sum(temp_1)/len(temp_1))

            distances[seed_idx] = sum(temp_2)/len(temp_2)

        #return minimum distance
        return np.min(distances)


    def calculate_wasserstein_distance(self, sampled_seeds):
        #wrapper to handle based on configs (full or partial distribution)
        #should probably be called in update level sampler -> score is updated there
        #seeds that were earlier sampled should be eval_seeds

        if self.sample_full_distribution:
            #retrieve seeds from working set as compare_seeds? (CHECK IF THIS IS CORRECT)
            #then compare the score with staging set (though i think this is handled in level sampler)
            seeds_compare = 
            seeds_eval = 
            
            self._calculate_wasserstein_distance(actor_index, seed_t, score, num_steps)
        else:
            #retrieve seeds from entire seed buffer as compare_seeds? (CHECK IF THIS IS CORRECT)
            self._calculate_wasserstein_distance(actor_index, seed_t, score, max_score, num_steps)

        pass