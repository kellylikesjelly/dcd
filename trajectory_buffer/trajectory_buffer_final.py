from collections import defaultdict
import numpy as np
import ot
#for code checking
import pandas as pd

class TrajectoryBuffer():
    '''
        For storing trajectories. Stores only trajectories from the latest run for each level.
    '''
    def __init__(self, obs_space, action_space, sample_full_distribution):
        #initialise a dictionary mapping from seed to trajectory
        self.action_buffer = dict()
        self.obs_buffer = dict()
        self.processed_trajs = dict()

        self.action_space_high = action_space.high
        self.action_space_low = action_space.low

        self.obs_space_high = obs_space.high
        self.obs_space_low = obs_space.low
        self.sample_full_distribution = sample_full_distribution

        # print('initialised')
        # print('action_space',self.action_space_high, self.action_space_low)
        # print('obs space', self.obs_space_high, self.obs_space_low)

    def insert(self, storage):
        #insert the latest trajectories from a level

        #self.level_seeds = torch.zeros(num_steps, num_processes, 1, dtype=torch.int)
        #self.actions = torch.zeros(num_steps, num_processes, action_shape)
        #self.obs = torch.zeros(num_steps + 1, num_processes, *observation_space.shape)

        level_seeds = storage.level_seeds.cpu()
        action_trajs = storage.actions.cpu()
        obs_trajs = storage.obs.cpu()

        #for each process we can expect multiple seeds if in replay mode
        #get all seeds then use indexing to extract
        seeds = np.unique(level_seeds)
        # print(seeds)
        for seed in seeds:
            # print(seed)
            action_trajs_seed = action_trajs[np.squeeze(level_seeds==seed),:]
            obs_trajs_seed = obs_trajs[:-1,:,:][np.squeeze(level_seeds==seed),:]

            self.action_buffer[seed] = action_trajs_seed
            self.obs_buffer[seed] = obs_trajs_seed

            # print(self.action_buffer.keys())
            # print(self.obs_buffer.keys())

        self._process_trajs(seeds)

    def _process_trajs(self, seeds):
        for seed in seeds:
            #process trajectory and store it
            actions = self.action_buffer[seed]
            obs = self.obs_buffer[seed]
            actions = (actions - self.action_space_low)/(self.action_space_high - self.action_space_low)
            obs = (obs - self.obs_space_low)/(self.obs_space_high - self.obs_space_low)
            action_obs_samples = np.concatenate([actions, obs], axis=-1)
            if action_obs_samples.shape[0]>2000:
                #to manage computation speed sample timesteps if traj is v long
                sampled_timesteps = np.random.randint(action_obs_samples.shape[0],size=2000)
                action_obs_samples = action_obs_samples[sampled_timesteps, :]
            self.processed_trajs[seed] = action_obs_samples
    

    # def _distance_working_buffer(self, level_sampler):
    #     #calculate distances only for last 5 seeds in staleness + regret ranking

    #     #get last ranking 5 seeds
    #     weights = level_sampler.sample_weights()
    #     num_rank = 5
    #     lowest_seeds = level_sampler.seeds[np.argsort(weights)<num_rank]

    #     working_seeds = list(lowest_seeds)
    #     working_seeds.sort()

    #     # working_seeds = list(level_sampler.working_seed_set)
    #     # working_seeds.sort()

    #     distance_matrix = np.zeros(shape=(len(working_seeds),len(working_seeds)))
    #     distance_matrix[:,:] = np.nan

    #     i = 0

    #     for working_seed_idx in range(len(working_seeds)):
    #         working_seed = working_seeds[working_seed_idx]
    #         working_traj = self.processed_trajs[working_seed]

    #         for compare_seed_idx in range(working_seed_idx+1, len(working_seeds)):
    #             compare_seed = working_seeds[compare_seed_idx]
    #             compare_traj = self.processed_trajs[compare_seed]

    #             s_working = [1/working_traj.shape[0]]*working_traj.shape[0]
    #             s_compare = [1/compare_traj.shape[0]]*compare_traj.shape[0]
    #             M = ot.dist(working_traj, compare_traj, metric='euclidean')
    #             W = ot.emd2(s_working, s_compare, M, numItermax=100000)
    #             distance_matrix[working_seed_idx][compare_seed_idx] = W
    #             distance_matrix[compare_seed_idx][working_seed_idx] = W

    #             # i+=1

    #     # print(i)

    #     # print(distance_matrix)

    #     for seed_idx in range(len(working_seeds)):
    #         seed = working_seeds[seed_idx]
    #         distance = np.nanmin(distance_matrix[seed_idx])
    #         level_sampler.update_diversity(seed, distance)


    # def _distance_staging_buffer(self, level_sampler):
    #     staging_seeds = list(level_sampler.staging_seed_set)
    #     working_seeds = list(level_sampler.working_seed_set)

    #     i = 0
    #     for staging_seed in staging_seeds:
    #         staging_traj = self.processed_trajs[staging_seed]
    #         distances = np.zeros(shape=len(working_seeds))-1
    #         for working_idx in range(len(working_seeds)):
    #             working_seed = working_seeds[working_idx]
    #             working_traj = self.processed_trajs[working_seed]

    #             s_staging = [1/staging_traj.shape[0]]*staging_traj.shape[0]
    #             s_working = [1/working_traj.shape[0]]*working_traj.shape[0]
    #             M = ot.dist(staging_traj, working_traj, metric='euclidean')
    #             W = ot.emd2(s_staging, s_working, M, numItermax=100000)
    #             distances[working_idx] = W

    #             i+=1
    #         level_sampler.update_diversity(staging_seed, np.min(distances))

    #     # print(i)

    def _distance(self, eval_seeds, highest_seeds, level_sampler):
        #place check for same seed (highly unlikely)

        for eval_seed in eval_seeds:
            eval_traj = self.processed_trajs[eval_seed]
            distances = np.zeros(shape=len(highest_seeds))-1
            for working_idx in range(len(highest_seeds)):
                working_seed = highest_seeds[working_idx]
                working_traj = self.processed_trajs[working_seed]

                #empty list means uniform distribution
                s_staging = []
                s_working = []
                M = ot.dist(eval_traj, working_traj, metric='euclidean')
                W = ot.emd2(s_staging, s_working, M, numItermax=100000)
                distances[working_idx] = W

            level_sampler.update_diversity(eval_seed, np.min(distances))

    def calculate_wasserstein_distance(self, level_sampler, discard_grad):

        #in first iteration diversity score is 0
        if len(level_sampler.working_seed_set)==0:
            for seed in level_sampler.staging_seed_set:
                level_sampler.update_diversity(seed, 0)
            return

        # if discard_grad:
        #     self._distance_staging_buffer(level_sampler)
        # else:
        #     self._distance_working_buffer(level_sampler)
    
        #get highest ranking 5 seeds
        #rmbr to change sample weights accordingly
        weights = level_sampler.sample_weights()
        seen_seeds = level_sampler.seeds[level_sampler.unseen_seed_weights==0]
        seen_weights = weights[level_sampler.unseen_seed_weights==0]

        num_rank = 5
        
        highest_seeds = list(np.argsort(seen_weights)[-num_rank:])
        highest_seeds = [seen_seeds[seed_idx] for seed_idx in highest_seeds]

        if discard_grad:
            eval_seeds = list(level_sampler.staging_seed_set)
        else:
            lowest_seeds = list(np.argsort(seen_weights)[:num_rank])
            lowest_seeds = [seen_seeds[seed_idx] for seed_idx in lowest_seeds]
            eval_seeds = list(lowest_seeds)

        print(len(eval_seeds), len(highest_seeds))
        print(eval_seeds, highest_seeds)
        self._distance(eval_seeds, highest_seeds, level_sampler)