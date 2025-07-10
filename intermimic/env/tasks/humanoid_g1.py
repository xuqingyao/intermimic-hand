import torch
import os

from isaacgym import gymapi
from isaacgym.torch_utils import *

from utils import torch_utils
from env.tasks.humanoid import *


class Humanoid_G1(Humanoid_SMPLX):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self._key_body_ids_gt = to_torch(cfg["env"]["keyIndex"], device='cuda', dtype=torch.long)
        self._contact_body_ids_gt = to_torch(cfg["env"]["contactIndex"], device='cuda', dtype=torch.long)
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        return

    def _setup_character_props(self, key_bodies):
        self._dof_obs_size = self.cfg["env"]["numDoF"]
        self._num_actions = self.cfg["env"]["numDoF"]
        self._num_actions_hand = self.cfg["env"]["numDoFHand"]
        self._num_actions_wrist = self.cfg["env"]["numDoFWrist"]
        self._num_obs = self.cfg["env"]["numObs"]
        return

    def _build_termination_heights(self):
        super()._build_termination_heights()
        self._termination_heights_init = 0.5
        self._termination_heights_init = to_torch(self._termination_heights_init, device=self.device)
        return
    
    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = self.cfg["env"]["asset"]["assetRoot"]
        asset_file = self.robot_type

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        humanoid_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        self.num_humanoid_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)
        self.num_humanoid_shapes = self.gym.get_asset_rigid_shape_count(humanoid_asset)

        self.torso_index = 0
        self.num_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)
        self.num_dof = self.gym.get_asset_dof_count(humanoid_asset)
        self.num_joints = self.gym.get_asset_joint_count(humanoid_asset)

        self.humanoid_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []

        max_agg_bodies = self.num_humanoid_bodies + 2
        max_agg_shapes = self.num_humanoid_shapes + 65
        
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            self._build_env(i, env_ptr, humanoid_asset)

            self.gym.end_aggregate(env_ptr)
            self.envs.append(env_ptr)

        dof_prop = self.gym.get_actor_dof_properties(self.envs[0], self.humanoid_handles[0])
        for j in range(self.num_dof):
            if dof_prop['lower'][j] > dof_prop['upper'][j]:
                self.dof_limits_lower.append(dof_prop['upper'][j])
                self.dof_limits_upper.append(dof_prop['lower'][j])
            else:
                self.dof_limits_lower.append(dof_prop['lower'][j])
                self.dof_limits_upper.append(dof_prop['upper'][j])

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)

        if (self._pd_control):
            self._build_pd_action_offset_scale()

        return
    
    def _build_env(self, env_id, env_ptr, humanoid_asset):
        col_group = env_id
        col_filter = self._get_humanoid_collision_filter()
        segmentation_id = 0

        start_pose = gymapi.Transform()
        asset_file = self.robot_type
        char_h = 0.89

        start_pose.p = gymapi.Vec3(*get_axis_params(char_h, self.up_axis_idx))
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        humanoid_handle = self.gym.create_actor(env_ptr, humanoid_asset, start_pose, "humanoid", col_group, col_filter, segmentation_id)

        self.gym.enable_actor_dof_force_sensors(env_ptr, humanoid_handle)

        if (self._pd_control):
            dof_prop = self.gym.get_asset_dof_properties(humanoid_asset)
            dof_prop["driveMode"] = gymapi.DOF_MODE_POS
            dof_prop["stiffness"] = [500 for _ in range(12)] + [300 for _ in range(7)] + [200 for _ in range(self._num_actions_wrist)] + [100 for _ in range(self._num_actions_hand)] + [300 for _ in range(4)] + [200 for _ in range(self._num_actions_wrist)] + [100 for _ in range(self._num_actions_hand)]
            dof_prop["damping"] = [50 for _ in range(12)] + [30 for _ in range(7)] + [20 for _ in range(self._num_actions_wrist)] + [10 for _ in range(self._num_actions_hand)] + [30 for _ in range(4)] + [20 for _ in range(self._num_actions_wrist)] + [10 for _ in range(self._num_actions_hand)]
            self.gym.set_actor_dof_properties(env_ptr, humanoid_handle, dof_prop)
                    
        props = self.gym.get_actor_rigid_shape_properties(env_ptr, humanoid_handle)
        names = self.gym.get_actor_rigid_body_names(env_ptr, humanoid_handle)

        for p_idx in range(len(props)):
            if 'left_hand' in names[p_idx]:
                props[p_idx].filter = 1
            if 'right_hand' in names[p_idx]:
                props[p_idx].filter = 128
            if 'right' in names[p_idx]:
                if 'ankle' in names[p_idx]:
                    props[p_idx].filter = 2
                elif 'knee' in names[p_idx]:
                    props[p_idx].filter = 6
                elif 'hip' in names[p_idx]:
                    props[p_idx].filter = 12
            if 'left' in names[p_idx]:
                if 'ankle' in names[p_idx]:
                    props[p_idx].filter = 16
                elif 'knee' in names[p_idx]:
                    props[p_idx].filter = 48
                elif 'hip' in names[p_idx]:
                    props[p_idx].filter = 96

        self.gym.set_actor_rigid_shape_properties(env_ptr, humanoid_handle, props)
        self.humanoid_handles.append(humanoid_handle)

        return

    def _build_pd_action_offset_scale(self):
        
        lim_low = self.dof_limits_lower.cpu().numpy()
        lim_high = self.dof_limits_upper.cpu().numpy()

        self._pd_action_offset = 0.5 * (lim_high + lim_low)
        self._pd_action_scale = 0.5 * (lim_high - lim_low)
        self._pd_action_offset = to_torch(self._pd_action_offset, device=self.device)
        self._pd_action_scale = to_torch(self._pd_action_scale, device=self.device)
        return

    def _get_humanoid_collision_filter(self):
        return 0

    def _compute_reward(self, actions):
        return
    
    def _compute_reset(self):
        self.reset_buf[:], self._terminate_buf[:] = self.compute_humanoid_reset(self.reset_buf, self.progress_buf, self.obs_buf,
                                                                                self._rigid_body_pos, self.max_episode_length[self.data_id],
                                                                                self._enable_early_termination, self._termination_heights, self.start_times, 
                                                                                self.rollout_length
                                                                                )
        return

    def _compute_humanoid_obs(self, env_ids=None, ref_obs=None, next_ts=None):
        if (env_ids is None):
            body_pos = self._rigid_body_pos
            body_rot = self._rigid_body_rot
            body_vel = self._rigid_body_vel
            body_ang_vel = self._rigid_body_ang_vel
            contact_forces = self._contact_forces
        else:
            body_pos = self._rigid_body_pos[env_ids]
            body_rot = self._rigid_body_rot[env_ids]
            body_vel = self._rigid_body_vel[env_ids]
            body_ang_vel = self._rigid_body_ang_vel[env_ids]
            contact_forces = self._contact_forces[env_ids]
        
        obs = self.compute_humanoid_observations_max(body_pos, body_rot, body_vel, body_ang_vel, self._local_root_obs,
                                                self._root_height_obs,
                                                contact_forces, self._contact_body_ids, ref_obs, self._key_body_ids, 
                                                self._key_body_ids_gt, self._contact_body_ids_gt)
        # obs = self.compute_humanoid_observations_max(body_pos, body_rot, body_vel, body_ang_vel, self._local_root_obs,
        #                                         self._root_height_obs,
        #                                         contact_forces, self._contact_body_ids, ref_obs, self._key_body_ids, 
        #                                         )

        return obs
