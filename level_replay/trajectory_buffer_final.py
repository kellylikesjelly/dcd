from collections import defaultdict
import numpy as np
import ot
#for code checking
import pandas as pd

#buffer should be intialised once at start of training only to save all trajectories

class TrajectoryBuffer(object):
    '''
        For storing trajectories. Stores only trajectories from the latest run for each level.
    '''
    def __init__(self, env, sample_full_distribution):
        #initialise a dictionary mapping from seed to trajectory
        self.action_buffer = dict()
        self.obs_buffer = dict()

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

        level_seeds = storage.level_seeds.cpu()
        action_trajs = storage.actions.cpu()
        obs_trajs = storage.obs.cpu()

        # change to collect based on seeds across processes
        #for each process we can expect multiple seeds if in replay mode

        #get all seeds then use indexing to extract
        seeds = np.unique(level_seeds)
        for seed in seeds:
            action_trajs_seed = action_trajs[np.squeeze(level_seeds==seed),:]
            obs_trajs_seed = obs_trajs[:-1,:,:][np.squeeze(level_seeds==seed),:]

            # if action_trajs_seed.shape[0] != obs_trajs_seed.shape[0]:
            #     raise Exception("Trajs not same length!")

            self.action_buffer[seed] = action_trajs_seed
            self.obs_buffer[seed] = obs_trajs_seed
            

    def _calculate_wasserstein_distance(self, seed_eval, seeds_compare):
        '''CALCULATES MIN WASSERSTEIN OF 1 SEED_EVAL WITH MULTIPLE SEEDS_COMPARE'''
        #DEBUGGING
        # pd.set_option('display.max_columns', 30)

        #SHAPE of each traj array: (#trajs, #timestepsw/buffer, act/obs#dim)

        #intialise pairwise wasserstein distance array
        distances = np.empty(seeds_compare.shape[0])
        #retrieve current set of traj
        seed_eval_actions = self.action_buffer[seed_eval]
        seed_eval_obs = self.obs_buffer[seed_eval]
        #process trajectories: normalise, combine
        seed_eval_actions = (seed_eval_actions - self.action_space_low)/(self.action_space_high - self.action_space_low)
        # print('seed_eval_actions_norm')
        # print(pd.DataFrame(seed_eval_actions).describe())

        # print('seed_eval_obs', seed_eval_obs.shape)
        # pd.DataFrame(seed_eval_obs).describe()
        if self.obs_space_high[0]==np.inf:
            seed_eval_obs = np.exp(seed_eval_obs) / (1 + np.exp(seed_eval_obs))
        else:
            seed_eval_obs = (seed_eval_obs - self.obs_space_low)/(self.obs_space_high - self.obs_space_low)
        # print('seed_eval_obs_norm', seed_eval_obs.shape)
        # print(pd.DataFrame(seed_eval_obs).describe())

        eval_state_action_traj = np.concatenate([seed_eval_actions, seed_eval_obs], axis=-1)
        # print('eval_state_action_traj')
        # print(pd.DataFrame(eval_state_action_traj).describe())

        #for each compare_seed we compare the pair of trajectories
        for seed_idx in range(len(seeds_compare)):
            #retrieve seed_compare
            seed = seeds_compare[seed_idx]
            # print('seed_compare', seed)

            seed_compare_actions = self.action_buffer[seed]
            seed_compare_obs = self.obs_buffer[seed]

            # print('seed_compare_actions', seed_compare_actions, seed_compare_actions.shape)
            seed_compare_actions = (seed_compare_actions - self.action_space_low)/(self.action_space_high - self.action_space_low)
            # print('seed_compare_actions_norm', seed_compare_actions, seed_compare_actions.shape)

            # print('seed_compare_obs', seed_compare_obs, seed_compare_obs.shape)
            #if observations are unbounded, to bound (0...1) range
            if self.obs_space_high[0]==np.inf:
                seed_compare_obs = np.exp(seed_compare_obs) / (1 + np.exp(seed_compare_obs))
            else:
            #use min max normalisation
                seed_compare_obs = (seed_compare_obs - self.obs_space_low)/(self.obs_space_high - self.obs_space_low)
            # print('seed_compare_obs_norm', seed_compare_obs, seed_compare_obs.shape)

            compare_state_action_traj = np.concatenate([seed_compare_actions, seed_compare_obs], axis=-1)
            # print('compare_state_action_traj', compare_state_action_traj, compare_state_action_traj.shape)

            #uniform weight distribution for each timestep
            s_eval = [1/eval_state_action_traj.shape[0]]*eval_state_action_traj.shape[0]
            s_compare = [1/compare_state_action_traj.shape[0]]*compare_state_action_traj.shape[0]
            # print('s_eval', len(s_eval))
            # print('s_compare',len(s_compare))
                    
            #create cost matrix using euclidean distance as cost
            M = ot.dist(eval_state_action_traj, compare_state_action_traj, metric='euclidean')
            # print('M', M)
            #calculate wasserstein distance
            W = ot.emd2(s_eval, s_compare, M, numItermax=200000)
            # print('W', W)
            distances[seed_idx] = W

        #return minimum distance
        return np.min(distances)


    def calculate_wasserstein_distance(self, seeds, level_sampler):
        #wrapper to handle based on configs (full or partial distribution)
        #should probably be called in update level sampler -> score is updated there
        #seeds that were earlier sampled should be eval_seeds

        #handle when not all seeds have collected trajectories yet -> dont compare with them
        seeds_with_trajs = np.array(list(self.action_buffer.keys()))

        #exclude seeds eval

        if self.sample_full_distribution:
            #retrieve seeds from working set as compare_seeds? (CHECK IF THIS IS CORRECT)
            #then compare the score with staging set (though i think this is handled in level sampler)
            seeds_compare = np.array(list(level_sampler.working_seed_set))
            # seeds_compare = seeds_compare[np.isin(seeds_compare, seeds_with_trajs)]
            # print(seeds_compare)
        else:
            seeds_compare = seeds_with_trajs
            #retrieve seeds from entire seed buffer as compare_seeds? (CHECK IF THIS IS CORRECT)

        #exclude seeds eval since they might be from working set/alr have trajs
        # print('SEEDS COMPARE')
        # print(seeds_compare)
        # IN REPLAY YOU SHOULD STILL UPDATE SEEDS. the seeds compare will all be in working seeds
        # resulting in empty list --> need edit

        #get unique list of seeds since seeds is in shape (#timesteps, #process, 1)
        seeds = np.unique(seeds.cpu())

        # print('evaluated seeds:', seeds)
        for seed in seeds:
            # print('SEEDDDDDDDDDDDDDDDDDD')
            seed = seed.item()
            # print(seed)
            seeds_compare_ = seeds_compare[seeds_compare!=seed]
            # print('seeds compare', seeds_compare_)
            #if no seeds to compare, temporarily set seed to 0 diversity (should still fill up working seeds)
            if len(seeds_compare_)==0:
                # print('NO SEEDS TO COMPARE')
                level_sampler.update_diversity(seed, 0, self.sample_full_distribution)
                continue
            #if currently evaluated seed is in working set (i.e. replay) then remove it from seeds compare
            diversity = self._calculate_wasserstein_distance(seed, seeds_compare_)
            level_sampler.update_diversity(seed, diversity, self.sample_full_distribution)