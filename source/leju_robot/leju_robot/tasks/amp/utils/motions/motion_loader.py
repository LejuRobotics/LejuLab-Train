# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import os
import torch
from typing import Optional, Union
from collections import Counter

class MotionLoader:
    """
    Helper class to load and sample motion data from NumPy-file format.
    """

    def __init__(self, motion_file: Union[str, list[str], dict], device: torch.device, 
             motion_weights: Optional[list[float]] = None) -> None:
        """Load a motion file and initialize the internal variables.

        Args:
            motion_file: Motion file path to load.
            device: The device to which to load the data.

        Raises:
            AssertionError: If the specified motion file doesn't exist.
        """
        # Convert single file to list for uniform handling

        print(f"[DEBUG] MotionLoader.__init__ called")
        print(f"[DEBUG] motion_file received - type: {type(motion_file)}")
        print(f"[DEBUG] motion_file content: {motion_file}")
        
        # Convert single file to list for uniform handling
        if isinstance(motion_file, dict):
            # 字典格式：{文件路径: 权重}
            motion_files = list(motion_file.keys())
            weights = list(motion_file.values())
        elif isinstance(motion_file, str):
            # 单文件
            motion_files = [motion_file]
            weights = [1.0]
        else:
            # 文件列表
            motion_files = motion_file
            weights = motion_weights if motion_weights else [1.0] * len(motion_files)
        
        # 归一化权重
        weights = np.array(weights)
        self.motion_weights = weights / np.sum(weights)

        print(f"[DEBUG] Motion weights:")
        for i, (file, weight) in enumerate(zip(motion_files, self.motion_weights)):
            print(f"[DEBUG]   Motion {i}: weight={weight:.3f}, file={os.path.basename(file)}")

        for i, file in enumerate(motion_files):
            print(f"[DEBUG]   file[{i}]: type={type(file)}, value='{file}'")
        
        print(f"[DEBUG] Starting file existence checks...")
        for i, file in enumerate(motion_files):
            print(f"[DEBUG] Checking file {i}: {file}")
            try:
                exists = os.path.isfile(file)
                print(f"[DEBUG] File {i} exists: {exists}")
                if not exists:
                    print(f"[DEBUG] File does not exist: {file}")
                assert exists, f"Invalid file path: {file}"
            except Exception as e:
                print(f"[DEBUG] Error checking file {i}: {e}")
                print(f"[DEBUG] File variable type: {type(file)}")
                print(f"[DEBUG] File variable repr: {repr(file)}")
                raise
        
        print(f"[DEBUG] All files exist, continuing with initialization...")
        # if isinstance(motion_file, str):
        #     motion_files = [motion_file]
        # else:
        #     motion_files = motion_file

        # for file in motion_files:
        #     assert os.path.isfile(file), f"Invalid file path: {file}"


        self.device = device
        self.motion_files = motion_files
        self.num_motions = len(motion_files)
        
        # Load all motion data
        self.motions = []
        for i, file in enumerate(motion_files):
            data = np.load(file, allow_pickle=True)  # 允许加载对象数组（字符串）
            
            # Store motion data
            motion_data = {
                'dof_names': data["dof_names"].tolist(),
                'body_names': data["body_names"].tolist(),
                'dof_positions': torch.tensor(data["dof_positions"], dtype=torch.float32, device=self.device),
                'dof_velocities': torch.tensor(data["dof_velocities"], dtype=torch.float32, device=self.device),
                'body_positions': torch.tensor(data["body_positions"], dtype=torch.float32, device=self.device),
                'body_rotations': torch.tensor(data["body_rotations"], dtype=torch.float32, device=self.device),
                'body_linear_velocities': torch.tensor(data["body_linear_velocities"], dtype=torch.float32, device=self.device),
                'body_angular_velocities': torch.tensor(data["body_angular_velocities"], dtype=torch.float32, device=self.device),
                'dt': 1.0 / data["fps"],
                'num_frames': data["dof_positions"].shape[0],
            }
            motion_data['duration'] = motion_data['dt'] * (motion_data['num_frames'] - 1)
            self.motions.append(motion_data)
            
            print(f"Motion loaded ({file}): duration: {motion_data['duration']} sec, frames: {motion_data['num_frames']}")

        # Use first motion's skeleton info as reference (assuming all motions have same skeleton)
        self._dof_names = self.motions[0]['dof_names']
        self._body_names = self.motions[0]['body_names']
        
        # For backward compatibility, use first motion's properties
        self.dof_positions = self.motions[0]['dof_positions']
        self.dof_velocities = self.motions[0]['dof_velocities'] 
        self.body_positions = self.motions[0]['body_positions']
        self.body_rotations = self.motions[0]['body_rotations']
        self.body_linear_velocities = self.motions[0]['body_linear_velocities']
        self.body_angular_velocities = self.motions[0]['body_angular_velocities']
        self.dt = self.motions[0]['dt']
        self.num_frames = self.motions[0]['num_frames']
        self.duration = self.motions[0]['duration']

    @property
    def dof_names(self) -> list[str]:
        """Skeleton DOF names."""
        return self._dof_names

    @property
    def body_names(self) -> list[str]:
        """Skeleton rigid body names."""
        return self._body_names

    @property
    def num_dofs(self) -> int:
        """Number of skeleton's DOFs."""
        return len(self._dof_names)

    @property
    def num_bodies(self) -> int:
        """Number of skeleton's rigid bodies."""
        return len(self._body_names)
    
    def _sample_motion_index(self, num_samples: int) -> np.ndarray:
        """Randomly sample motion indices with weights."""
        return np.random.choice(
            self.num_motions, 
            size=num_samples, 
            p=self.motion_weights,
            replace=True
        )
    def _interpolate(
        self,
        a: torch.Tensor,
        *,
        b: Optional[torch.Tensor] = None,
        blend: Optional[torch.Tensor] = None,
        start: Optional[np.ndarray] = None,
        end: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """Linear interpolation between consecutive values.

        Args:
            a: The first value. Shape is (N, X) or (N, M, X).
            b: The second value. Shape is (N, X) or (N, M, X).
            blend: Interpolation coefficient between 0 (a) and 1 (b).
            start: Indexes to fetch the first value. If both, ``start`` and ``end` are specified,
                the first and second values will be fetches from the argument ``a`` (dimension 0).
            end: Indexes to fetch the second value. If both, ``start`` and ``end` are specified,
                the first and second values will be fetches from the argument ``a`` (dimension 0).

        Returns:
            Interpolated values. Shape is (N, X) or (N, M, X).
        """
        if start is not None and end is not None:
            return self._interpolate(a=a[start], b=a[end], blend=blend)
        if a.ndim >= 2:
            blend = blend.unsqueeze(-1)
        if a.ndim >= 3:
            blend = blend.unsqueeze(-1)
        return (1.0 - blend) * a + blend * b

    def _slerp(
        self,
        q0: torch.Tensor,
        *,
        q1: Optional[torch.Tensor] = None,
        blend: Optional[torch.Tensor] = None,
        start: Optional[np.ndarray] = None,
        end: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """Interpolation between consecutive rotations (Spherical Linear Interpolation).

        Args:
            q0: The first quaternion (wxyz). Shape is (N, 4) or (N, M, 4).
            q1: The second quaternion (wxyz). Shape is (N, 4) or (N, M, 4).
            blend: Interpolation coefficient between 0 (q0) and 1 (q1).
            start: Indexes to fetch the first quaternion. If both, ``start`` and ``end` are specified,
                the first and second quaternions will be fetches from the argument ``q0`` (dimension 0).
            end: Indexes to fetch the second quaternion. If both, ``start`` and ``end` are specified,
                the first and second quaternions will be fetches from the argument ``q0`` (dimension 0).

        Returns:
            Interpolated quaternions. Shape is (N, 4) or (N, M, 4).
        """
        if start is not None and end is not None:
            return self._slerp(q0=q0[start], q1=q0[end], blend=blend)
        if q0.ndim >= 2:
            blend = blend.unsqueeze(-1)
        if q0.ndim >= 3:
            blend = blend.unsqueeze(-1)

        qw, qx, qy, qz = 0, 1, 2, 3  # wxyz
        cos_half_theta = (
            q0[..., qw] * q1[..., qw]
            + q0[..., qx] * q1[..., qx]
            + q0[..., qy] * q1[..., qy]
            + q0[..., qz] * q1[..., qz]
        )

        neg_mask = cos_half_theta < 0
        q1 = q1.clone()
        q1[neg_mask] = -q1[neg_mask]
        cos_half_theta = torch.abs(cos_half_theta)
        cos_half_theta = torch.unsqueeze(cos_half_theta, dim=-1)

        half_theta = torch.acos(cos_half_theta)
        sin_half_theta = torch.sqrt(1.0 - cos_half_theta * cos_half_theta)

        ratio_a = torch.sin((1 - blend) * half_theta) / sin_half_theta
        ratio_b = torch.sin(blend * half_theta) / sin_half_theta

        new_q_x = ratio_a * q0[..., qx : qx + 1] + ratio_b * q1[..., qx : qx + 1]
        new_q_y = ratio_a * q0[..., qy : qy + 1] + ratio_b * q1[..., qy : qy + 1]
        new_q_z = ratio_a * q0[..., qz : qz + 1] + ratio_b * q1[..., qz : qz + 1]
        new_q_w = ratio_a * q0[..., qw : qw + 1] + ratio_b * q1[..., qw : qw + 1]

        new_q = torch.cat([new_q_w, new_q_x, new_q_y, new_q_z], dim=len(new_q_w.shape) - 1)
        new_q = torch.where(torch.abs(sin_half_theta) < 0.001, 0.5 * q0 + 0.5 * q1, new_q)
        new_q = torch.where(torch.abs(cos_half_theta) >= 1, q0, new_q)
        return new_q

    def _compute_frame_blend(self, times: np.ndarray, motion_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the indexes of the first and second values, as well as the blending time
        to interpolate between them and the given times.

        Args:
            times: Times, between 0 and motion duration, to sample motion values.
                Specified times will be clipped to fall within the range of the motion duration.
            motion_idx: Index of the motion to use.

        Returns:
            First value indexes, Second value indexes, and blending time between 0 (first value) and 1 (second value).
        """
        motion = self.motions[motion_idx]
        phase = np.clip(times / motion['duration'], 0.0, 1.0)
        index_0 = np.floor(phase * (motion['num_frames'] - 1)).astype(int)
        index_1 = np.minimum(index_0 + 1, motion['num_frames'] - 1)
        blend   = (times - index_0 * motion['dt']) / motion['dt']
        return index_0, index_1, blend

    def sample_times(self, num_samples: int, duration: float | None = None) -> np.ndarray:
        min_duration = min(motion['duration'] for motion in self.motions)
        duration = min_duration if duration is None else duration
        assert (
            duration <= min_duration
        ), f"The specified duration ({duration}) is longer than the shortest motion duration ({min_duration})"
        return duration * np.random.uniform(low=0.0, high=1.0, size=num_samples)
    
    # def sample(
    #     self, num_samples: int, times: Optional[np.ndarray] = None, duration: float | None = None
    # ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    #     """Sample motion data.

    #     Args:
    #         num_samples: Number of time samples to generate. If ``times`` is defined, this parameter is ignored.
    #         times: Motion time used for sampling.
    #             If not defined, motion data will be random sampled uniformly in time.
    #         duration: Maximum motion duration to sample.
    #             If not defined, samples will be within the range of the motion duration.
    #             If ``times`` is defined, this parameter is ignored.

    #     Returns:
    #         Sampled motion DOF positions (with shape (N, num_dofs)), DOF velocities (with shape (N, num_dofs)),
    #         body positions (with shape (N, num_bodies, 3)), body rotations (with shape (N, num_bodies, 4), as wxyz quaternion),
    #         body linear velocities (with shape (N, num_bodies, 3)) and body angular velocities (with shape (N, num_bodies, 3)).
    #     """
    #     # print(f"\n[DEBUG] ===== sample() called =====")
    #     # print(f"[DEBUG] Parameters: num_samples={num_samples}, times={times is not None}, duration={duration}")
    #     times = self.sample_times(num_samples, duration) if times is None else times
    #     # Randomly select one motion for this sampling call
    #     motion_idx = self._sample_motion_index(1)[0]
    #     motion = self.motions[motion_idx]
    #     # print(f"[DEBUG] Selected motion {motion_idx}: duration={motion['duration']:.3f}s, frames={motion['num_frames']}, dt={motion['dt']:.4f}s")
    #     # print(f"[DEBUG] Motion file: {self.motion_files[motion_idx]}")


    #     index_0, index_1, blend = self._compute_frame_blend(times, motion_idx)
    #     blend = torch.tensor(blend, dtype=torch.float32, device=self.device)
    #     # print(f"[DEBUG] Interpolating motion data...")
    #     # print(f"[DEBUG] Motion data shapes:")
    #     # print(f"[DEBUG]   dof_positions: {motion['dof_positions'].shape}")
    #     # print(f"[DEBUG]   dof_velocities: {motion['dof_velocities'].shape}")
    #     # print(f"[DEBUG]   body_positions: {motion['body_positions'].shape}")
    #     # print(f"[DEBUG]   body_rotations: {motion['body_rotations'].shape}")

    #     return (
    #         self._interpolate(motion['dof_positions'], blend=blend, start=index_0, end=index_1),
    #         self._interpolate(motion['dof_velocities'], blend=blend, start=index_0, end=index_1),
    #         self._interpolate(motion['body_positions'], blend=blend, start=index_0, end=index_1),
    #         self._slerp(motion['body_rotations'], blend=blend, start=index_0, end=index_1),
    #         self._interpolate(motion['body_linear_velocities'], blend=blend, start=index_0, end=index_1),
    #         self._interpolate(motion['body_angular_velocities'], blend=blend, start=index_0, end=index_1),
    #     )


    def sample(
        self,
        num_samples: int,                    
        times: Optional[np.ndarray] = None,
        duration: float | None = None,
        num_observation: int =2
    ):
        # DEBUG 
        DEBUG = os.getenv("MOTION_DEBUG", "0") != "0"
        if DEBUG:
            print("\n" + "=" * 60)
            print(f"MotionLoader.sample DEBUG  |  num_samples(arg)={num_samples}"
                  f",  times={'given' if times is not None else 'None'}")

        # ---------- 1) 反推窗口帧数 num_obs 与环境数 num_envs ----------
        if times is not None:                       # 已给 flatten times
            total_frames = len(times)
            assert total_frames % num_samples == 0, \
                "len(times) 不是 num_samples 的整数倍"
            num_obs  = total_frames // num_samples  # 窗口长度
            num_envs = num_samples                  # 传进来的就是 env 数
            num_samples = total_frames              # 真正的行数
        else:                                       # 调用方没给 times
            num_obs  = getattr(self, "num_obs", 5)  # 若 __init__ 未传则默 5
            
            if num_observation is not None:
                num_obs = num_observation
            num_envs = num_samples
            num_samples = num_envs * num_obs
            total_frames = num_samples

        

        if DEBUG:
            print(f"• num_envs={num_envs}, num_obs={num_obs}, "
                  f"num_samples={num_samples}")

        # ---------- 2) env 级抽 motion，再 repeat num_obs 次 ----------
        env_motion_ids = self._sample_motion_index(num_envs)        # (num_envs,)
        motion_ids     = np.repeat(env_motion_ids, num_obs)         # (num_samples,)

        if DEBUG:
            from collections import Counter
            cnt = Counter(motion_ids)
            print("• Motion selection  (id | weight | count)")
            for m_id, c in sorted(cnt.items()):
                print(f"  {m_id:2d} | {self.motion_weights[m_id]:.3f} | ×{c}")

        # ---------- 3) 如果没给 times，就各轨迹内 Uniform 采样 ----------
        if times is None:
            times = np.empty(num_samples, dtype=np.float32)
            for i ,m_id in enumerate(env_motion_ids):
                max_dur = self.motions[m_id]["duration"]
                
                if duration is not None:
                    max_dur = min(duration, max_dur)
                random_time = np.random.uniform(self.dt*num_obs, max_dur)
                # times[i*num_obs:(i+1)*num_obs] = np.expand_dims(random_time, axis=-1)- self.dt * np.arange(0, num_obs)
                times[i*num_obs:(i+1)*num_obs] = np.expand_dims(random_time, axis=-1) - self.dt * np.arange(0, num_obs, dtype=np.float32)

        times = np.clip(times, 0.0, None)           # 裁负值
        if DEBUG:
            print(f"• times range    : [{times.min():.4f}, {times.max():.4f}]"
                  f"  mean={times.mean():.4f}")

        # ---------- 4) 预分配输出张量 ---------------------------------
        n_dof, n_body, dev = self.num_dofs, self.num_bodies, self.device
        out_pos  = torch.empty((num_samples, n_dof),     device=dev)
        out_vel  = torch.empty_like(out_pos)
        out_bpos = torch.empty((num_samples, n_body, 3), device=dev)
        out_brot = torch.empty((num_samples, n_body, 4), device=dev)
        out_blin = torch.empty_like(out_bpos)
        out_bang = torch.empty_like(out_bpos)

        # ---------- 5) 按轨迹分组插值 ---------------------------------
        for m_id in np.unique(motion_ids):
            idx  = np.where(motion_ids == m_id)[0]
            t_sel = times[idx]
            i0, i1, blend_np = self._compute_frame_blend(t_sel, m_id)
            blend = torch.tensor(blend_np, dtype=torch.float32, device=dev)

            mot = self.motions[m_id]
            out_pos[idx]  = self._interpolate(mot["dof_positions"],  blend=blend, start=i0, end=i1)
            out_vel[idx]  = self._interpolate(mot["dof_velocities"], blend=blend, start=i0, end=i1)
            out_bpos[idx] = self._interpolate(mot["body_positions"], blend=blend, start=i0, end=i1)
            out_brot[idx] = self._slerp(      mot["body_rotations"], blend=blend, start=i0, end=i1)
            out_blin[idx] = self._interpolate(mot["body_linear_velocities"],  blend=blend, start=i0, end=i1)
            out_bang[idx] = self._interpolate(mot["body_angular_velocities"], blend=blend, start=i0, end=i1)

            if DEBUG:
                print(f"  ↳ m_id {m_id:2d} | {len(idx):3d} samples | "
                      f"dur={mot['duration']:.3f}s | "
                      f"t[min,max]=({t_sel.min():.3f}, {t_sel.max():.3f})")

        if DEBUG:
            print("=" * 60)

        return out_pos, out_vel, out_bpos, out_brot, out_blin, out_bang


    def sample_by_motion_idx(
        self,
        motion_idx: int,
        num_samples: int,
        times: Optional[np.ndarray] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
            , torch.Tensor]:
        """Sample motion data by motion index.

        Args:
            motion_idx: Index of the motion to sample.
            num_samples: Number of time samples to generate. If ``times`` is defined, this parameter is ignored.
            times: Motion time used for sampling.
                If not defined, motion data will be random sampled uniformly in time.
            duration: Maximum motion duration to sample.
                If not defined, samples will be within the range of the motion duration.
                If ``times`` is defined, this parameter is ignored.

        Returns:
            Sampled motion DOF positions (with shape (N, num_dofs)), DOF velocities (with shape (N, num_dofs)),
            body positions (with shape (N, num_bodies, 3)), body rotations (with shape (N, num_bodies, 4), as wxyz quaternion),
            body linear velocities (with shape (N, num_bodies, 3)) and body angular velocities (with shape (N, num_bodies, 3)).
        """
        assert 0 <= motion_idx < self.num_motions, f"Invalid motion index: {motion_idx}"
             # DEBUG 
        
        # ---------- 1) 反推窗口帧数 num_obs 与环境数 num_envs ----------
        if times is not None:                       # 已给 flatten times
            total_frames = len(times)
            assert total_frames % num_samples == 0, \
                "len(times) 不是 num_samples 的整数倍"
            num_obs  = total_frames // num_samples  # 窗口长度
            num_envs = num_samples                  # 传进来的就是 env 数
            num_samples = total_frames              # 真正的行数
        else:                                       # 调用方没给 times
            num_obs  = getattr(self, "num_obs", 5)  # 若 __init__ 未传则默 5
            num_envs = num_samples
            num_samples = num_envs * num_obs
            total_frames = num_samples

        if times is None:                     # 调用方没给 times
            times = self.sample_times(num_samples, self.motions[motion_idx]["duration"]) if times is None else times
        times = np.clip(times, 0.0, None)           # 裁负值


        # ---------- 4) 预分配输出张量 ---------------------------------
        n_dof, n_body, dev = self.num_dofs, self.num_bodies, self.device
        out_pos  = torch.empty((num_samples, n_dof),     device=dev)
        out_vel  = torch.empty_like(out_pos)
        out_bpos = torch.empty((num_samples, n_body, 3), device=dev)
        out_brot = torch.empty((num_samples, n_body, 4), device=dev)
        out_blin = torch.empty_like(out_bpos)
        out_bang = torch.empty_like(out_bpos)

        # ---------- 5) 按轨迹分组插值 ---------------------------------

        i0, i1, blend_np = self._compute_frame_blend(times, motion_idx)
        blend = torch.tensor(blend_np, dtype=torch.float32, device=dev)

        mot = self.motions[motion_idx]
        out_pos  = self._interpolate(mot["dof_positions"],  blend=blend, start=i0, end=i1)
        out_vel  = self._interpolate(mot["dof_velocities"], blend=blend, start=i0, end=i1)
        out_bpos = self._interpolate(mot["body_positions"], blend=blend, start=i0, end=i1)
        out_brot = self._slerp(      mot["body_rotations"], blend=blend, start=i0, end=i1)
        out_blin = self._interpolate(mot["body_linear_velocities"],  blend=blend, start=i0, end=i1)
        out_bang = self._interpolate(mot["body_angular_velocities"], blend=blend, start=i0, end=i1)

        return out_pos, out_vel, out_bpos, out_brot, out_blin, out_bang


    def get_dof_index(self, dof_names: list[str]) -> list[int]:
        """Get skeleton DOFs indexes by DOFs names.

        Args:
            dof_names: List of DOFs names.

        Raises:
            AssertionError: If the specified DOFs name doesn't exist.

        Returns:
            List of DOFs indexes.
        """
        indexes = []
        for name in dof_names:
            assert name in self._dof_names, f"The specified DOF name ({name}) doesn't exist: {self._dof_names}"
            indexes.append(self._dof_names.index(name))
        return indexes

    def get_body_index(self, body_names: list[str]) -> list[int]:
        """Get skeleton body indexes by body names.

        Args:
            dof_names: List of body names.

        Raises:
            AssertionError: If the specified body name doesn't exist.

        Returns:
            List of body indexes.
        """
        indexes = []
        for name in body_names:
            assert name in self._body_names, f"The specified body name ({name}) doesn't exist: {self._body_names}"
            indexes.append(self._body_names.index(name))
        return indexes


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Motion file")
    args, _ = parser.parse_known_args()

    motion = MotionLoader(args.file, "cpu")

    print("- number of frames:", motion.num_frames)
    print("- number of DOFs:", motion.num_dofs)
    print("- number of bodies:", motion.num_bodies)
