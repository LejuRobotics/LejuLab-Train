# Copyright (c) 2022-2024, The Berkeley Humanoid Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

from .height_field.hf_terrains_cfg import HfRandomUniformDifficultyTerrainCfg 


HARD_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.0, 0.15),
            step_width=0.30,
            platform_width=3.0,
            border_width=0.25,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.0, 0.15),
            step_width=0.30,
            platform_width=3.0,
            border_width=0.25,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.75, grid_height_range=(0.0, 0.15), platform_width=3.0
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.0, 0.05), noise_step=0.01, border_width=0.25, downsampled_scale=0.2,
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=3.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=3.0, border_width=0.25
        ),
    },
)

RANDOM_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.0, 0.02), noise_step=0.02, border_width=0.25
        )
    },
)

MIX_RANDOM_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "random_rough_l0": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.1, noise_range=(0.0, 0.00), noise_step=0.02, border_width=0.25
        ),
        "random_rough_l1": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.3, noise_range=(0.0, 0.02), noise_step=0.02, border_width=0.25
        ),
        "random_rough_l2": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.3, noise_range=(0.0, 0.04), noise_step=0.02, border_width=0.25
        ),
        "random_rough_l3": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.3, noise_range=(-0.02, 0.04), noise_step=0.02, border_width=0.25
        ),
    },
)


ROUGH_TERRAINS_CFG_test = TerrainGeneratorCfg(
    curriculum = True,
    difficulty_range = (0,1),
    size=(10.0, 10.0),
    border_width=20.0,
    num_rows=10,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={

        "random_rough_h_1_s_1": HfRandomUniformDifficultyTerrainCfg(
            proportion=0.1,
            noise_range=(0.0, 0.00),  
            noise_step=0.01,         
            border_width=0.25,
            downsampled_scale=1,    
        ),
    },
)


ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum = True,
    difficulty_range = (0,1),
    size=(10.0, 10.0),
    border_width=20.0,
    num_rows=10,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={

        "random_rough_h_1_s_1": HfRandomUniformDifficultyTerrainCfg(
            proportion=0.1,
            noise_range=(0.0, 0.00),  
            noise_step=0.01,         
            border_width=0.25,
            downsampled_scale=1,    
        ),
        

        "random_rough_h_1_s_2": HfRandomUniformDifficultyTerrainCfg(
            proportion=0.1,
            noise_range=(0.0, 0.04),  
            noise_step=0.0125,         
            border_width=0.25,
            downsampled_scale=0.2,    
        ),
        
        "random_rough_h_1_s_3": HfRandomUniformDifficultyTerrainCfg(
            proportion=0.1,
            noise_range=(0.0, 0.04),  
            noise_step=0.015,         
            border_width=0.25,
            downsampled_scale=0.1,    
        ),


        "random_rough_h_1_s_4": HfRandomUniformDifficultyTerrainCfg(
            proportion=0.1,
            noise_range=(0.0, 0.04),  
            noise_step=0.02,         
            border_width=0.25,
            downsampled_scale=0.2,    
        ),

        

        "random_rough_h_2_l_1": HfRandomUniformDifficultyTerrainCfg(
            proportion=0.1,
            noise_range=(0, 0.2),   # 较大的整体高度变化
            noise_step=0.025,           # 较大的步长产生更平缓的变化，导致局部斜率更小
            border_width=0.25,
            downsampled_scale=1.,    # 较大的下采样尺度平滑地形
        ),
        

        "random_rough_h_2_l_2": HfRandomUniformDifficultyTerrainCfg(
            proportion=0.1,
            noise_range=(0, 0.2),   # 较大的整体高度变化
            noise_step=0.05,           # 较大的步长产生更平缓的变化，导致局部斜率更小
            border_width=0.25,
            downsampled_scale=1,    # 较大的下采样尺度平滑地形
        ),

        "random_rough_h_2_l_3": HfRandomUniformDifficultyTerrainCfg(
            proportion=0.1,
            noise_range=(0, 0.2),   # 较大的整体高度变化
            noise_step=0.1,           # 较大的步长产生更平缓的变化，导致局部斜率更小
            border_width=0.25,
            downsampled_scale=1,    # 较大的下采样尺度平滑地形
        ),


        # 10% 金字塔斜坡，最大斜率0.3
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1,
            slope_range=(0.0, 0.3),   # 最大斜率为0.3
            platform_width=3.0,
            border_width=0.25
        ),
        
        # 10% 反金字塔斜坡，最大斜率0.3
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1,
            slope_range=(0.0, 0.3),   # 最大斜率为0.3
            platform_width=3.0,
            border_width=0.25
        ),

        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.1, grid_width=0.75, grid_height_range=(0.0, 0.15), platform_width=3.0
        ),
    },
)