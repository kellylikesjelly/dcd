from collections import defaultdict
import numpy as np
import ot

#DOUBLE CHECK THE LOGIC!!!

#buffer should be intialised once at start of training only to save all trajectories

class TrajectoryBuffer(object):
    '''
        For storing trajectories. Stores only trajectories from the latest play for each level and their masks.
    '''
    def __init__(self, env, sample_full_distribution):
        #initialise a dictionary mapping from seed to trajectory
        self.action_buffer = defaultdict()
        self.obs_buffer = defaultdict()

        action_space = env.action_space
        self.action_space_high = action_space.high
        self.action_space_low = action_space.low

        #observation can be unbounded (-ve, +ve infinity)
        obs_space = env.observation_space
        self.obs_space_high = obs_space.high
        self.obs_space_low = obs_space.low
        self.sample_full_distribution = sample_full_distribution

    def retrieve(self, seed):
        #retrieve the latest trajectories and masks from a level
        return self.action_buffer[seed], self.obs_buffer[seed]

    def insert(self, storage):
        #insert the latest trajectories from a level
        #action_trajs and obs_trajs need to be broken up according to the level seeds
        #CHECK SHAPE OF LEVEL_SEEDS
        #self.level_seeds = torch.zeros(num_steps, num_processes, 1, dtype=torch.int)
        #self.actions = torch.zeros(num_steps, num_processes, action_shape)
        #self.obs = torch.zeros(num_steps + 1, num_processes, *observation_space.shape)

        level_seeds = storage.level_seeds
        action_trajs = storage.actions
        obs_trajs = storage.obs

        # if storage.is_dict_obs:
        #     np_obs = np.empty((action_trajs.shape[0], action_trajs.shape[1], len(obs_trajs)))
        #     print('HIIIIIIIIIIIIIIIIIIIIIIIIIII')
        #     print(obs_trajs.keys())
        #     print(obs_trajs)
        #     for k in obs_trajs.keys():
        #         idx_np = self.key2idx[k]
        #         np_obs[:,:,idx_np] = obs_trajs[k][1:,:]
        #     obs_trajs = np_obs

        # change to collect based on seeds across processes
        #for each process we can expect multiple seeds if in replay mode

        #get all seeds then use indexing to extract
        seeds = np.unique(level_seeds)
        for seed in seeds:
            action_trajs_seed = action_trajs[np.squeeze(level_seeds==seed),:]
            obs_trajs_seed = obs_trajs[:-1,:,:][np.squeeze(level_seeds==seed),:]

            if action_trajs_seed.shape[0] != obs_trajs_seed.shape[0]:
                raise Exception("Trajs not same length!")

            self.action_buffer[seed] = action_trajs_seed
            self.obs_buffer[seed] = obs_trajs_seed
            

    def _calculate_wasserstein_distance(self, seed_eval, seeds_compare):
        '''CALCULATES MIN WASSERSTEIN OF 1 SEED_EVAL WITH MULTIPLE SEEDS_COMPARE'''

        #SHAPE of each traj array: (#trajs, #timestepsw/buffer, act/obs#dim)
        #full or not distribution should be handled outside this function

        #intialise pairwise wasserstein distance array
        distances = np.empty(seeds_compare.shape[0])
        #retrieve current set of trajs
        seed_eval_actions = self.action_buffer[seed_eval]
        seed_eval_obs = self.obs_buffer[seed_eval]
        #process trajectories: normalise, combine
        seed_eval_actions = (seed_eval_actions - self.action_space_low)/(self.action_space_high - self.action_space_low)
        
        if self.obs_space_high[0]==np.inf:
            seed_eval_obs = np.exp(seed_eval_obs) / (1 + np.exp(seed_eval_obs))
        else:
            seed_eval_obs = (seed_eval_obs - self.obs_space_low)/(self.obs_space_high - self.obs_space_low)
        
        eval_state_action_traj = np.concatenate([seed_eval_actions, seed_eval_obs], axis=-1)

        #for each compare_seed we compare the pair of trajectories
        for seed_idx in range(len(seeds_compare)):
            #retrieve seed_compare
            seed = seeds_compare[seed_idx]

            #don't compare seeds with themselves
            if seed == seed_eval:
                continue

            seed_compare_actions = self.action_buffer[seed]
            seed_compare_obs = self.obs_buffer[seed]
            seed_compare_actions = (seed_compare_actions - self.action_space_low)/(self.action_space_high - self.action_space_low)
            
            #if observations are unbounded, to bound (0...1) range
            if self.obs_space_high[0]==np.inf:
                seed_compare_obs = np.exp(seed_compare_obs) / (1 + np.exp(seed_compare_obs))
            else:
            #use min max normalisation
                seed_compare_obs = (seed_compare_obs - self.obs_space_low)/(self.obs_space_high - self.obs_space_low)
            
            compare_state_action_traj = np.concatenate([seed_compare_actions, seed_compare_obs], axis=-1)

            #uniform weight distribution for each timestep
            s_eval = [1/eval_state_action_traj.shape[0]]*eval_state_action_traj.shape[0]
            s_compare = [1/compare_state_action_traj.shape[0]]*compare_state_action_traj.shape[0]
                    
            #create cost matrix using euclidean distance as cost
            M = ot.dist(eval_state_action_traj, compare_state_action_traj, metric='euclidean')
            #calculate wasserstein distance
            W = ot.emd2(s_eval, s_compare, M)
            distances[seed_idx] = W

        #return minimum distance
        return np.min(distances)


    def calculate_wasserstein_distance(self, seeds, level_sampler):
        #wrapper to handle based on configs (full or partial distribution)
        #should probably be called in update level sampler -> score is updated there
        #seeds that were earlier sampled should be eval_seeds

        #handle when not all seeds have collected trajectories yet -> dont compare with them
        seeds_with_trajs = np.array(list(self.action_buffer.keys()))

        if self.sample_full_distribution:
            #retrieve seeds from working set as compare_seeds? (CHECK IF THIS IS CORRECT)
            #then compare the score with staging set (though i think this is handled in level sampler)
            seeds_compare = np.array(list(level_sampler.working_seed_set))
            print(seeds_compare)
            print(seeds_with_trajs)
            print(seeds_compare.shape)
            print(seeds_with_trajs.shape)
            seeds_compare = seeds_compare[np.isin(seeds_compare, seeds_with_trajs)]
            print(seeds_compare)
        else:
            seeds_compare = seeds_with_trajs
            #retrieve seeds from entire seed buffer as compare_seeds? (CHECK IF THIS IS CORRECT)

        #NEED TO SPLIT SEEDS ACCORDING TO PROCESS AVOID REPEATS!!!!
        seeds = seeds[0,:,0]

        print('staging seeds:', seeds)
        for seed in seeds:
            print('SEEDDDDDDDDDDDDDDDDDD')
            seed = seed.item()
            print(seed)
            #if no seeds to compare, temporarily set seed to 0 diversity (should still fill up working seeds)
            if len(seeds_compare)==0:
                print('NO SEEDS TO COMPARE')
                level_sampler.update_diversity(seed, 0, self.sample_full_distribution)
                continue
            diversity = self._calculate_wasserstein_distance(seed, seeds_compare)
            level_sampler.update_diversity(seed, diversity, self.sample_full_distribution)