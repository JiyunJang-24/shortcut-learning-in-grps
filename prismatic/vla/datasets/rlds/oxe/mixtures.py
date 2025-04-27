"""
mixtures.py

Defines a registry of dataset mixtures and weights for the Open-X Embodiment Datasets. Each dataset is associated with
a float "sampling weight"
"""

from typing import Dict, List, Tuple

# fmt: off
OXE_NAMED_MIXTURES: Dict[str, List[Tuple[str, float]]] = {
    # === Bridge V2 Dataset ===
    "bridge": [
        # ("bridge_oxe", 1.0),                                    # Version of Bridge V2 in Open-X GCP Bucket
        ("bridge_orig", 1.0),                                   # Original Version of Bridge V2 from Project Website
    ],


    # === [Moderate-Scale] Bridge++ Mixtures ===
    "bridge_rt_1": [
        # ("bridge_oxe", 1.0)                                   # Version of Bridge V2 in Open-X GCP Bucket
        ("bridge_orig", 1.0),                                   # Original Version of Bridge V2 from Project Website

        ("fractal20220817_data", 1.0),                          # Google RT-1 Robot Data (Large-Scale)
    ],

    # === RT-X Mixtures ===
    "rtx": [
        ("fractal20220817_data", 0.54087122203),                # Google RT-1 Robot Data (Large-Scale)
        ("kuka", 0.8341046294),
        # ("bridge_oxe", 1.0)                                   # Version of Bridge V2 in Open-X GCP Bucket
        ("bridge_orig", 1.0),                                   # Original Version of Bridge V2 from Project Website
        ("taco_play", 2.0),
        ("jaco_play", 2.0),
        ("berkeley_cable_routing", 3.0),
        ("roboturk", 1.0),
        # ("nyu_door_opening_surprising_effectiveness", 5.0),   # Note --> only contains wrist camera images (skip?)
        ("viola", 2.0),
        ("berkeley_autolab_ur5", 1.0),
        ("toto", 1.0),
    ],

    "rtx_franka": [
        ("fractal20220817_data", 0.54087122203),                # Google RT-1 Robot Data (Large-Scale)
        ("kuka", 0.8341046294),
        # ("bridge_oxe", 1.0)                                   # Version of Bridge V2 in Open-X GCP Bucket
        ("bridge_orig", 1.0),                                   # Original Version of Bridge V2 from Project Website
        ("taco_play", 2.0),
        ("jaco_play", 2.0),
        ("berkeley_cable_routing", 3.0),
        ("roboturk", 1.0),
        # ("nyu_door_opening_surprising_effectiveness", 5.0),   # Note --> only contains wrist camera images (skip?)
        ("viola", 2.0),
        ("berkeley_autolab_ur5", 1.0),
        ("toto", 1.0),

        ("taco_play", 1.0),
        ("berkeley_cable_routing", 1.0),
        ("viola", 1.0),
        ("toto", 1.0),
        ("stanford_hydra_dataset_converted_externally_to_rlds", 1.0),
        ("austin_buds_dataset_converted_externally_to_rlds", 3.0),
        ("nyu_franka_play_dataset_converted_externally_to_rlds", 3.0),
        ("maniskill_dataset_converted_externally_to_rlds", 0.1),
        ("furniture_bench_dataset_converted_externally_to_rlds", 0.1),
        ("cmu_franka_exploration_dataset_converted_externally_to_rlds", 5.0),
        ("austin_sailor_dataset_converted_externally_to_rlds", 1.0),
        ("austin_sirius_dataset_converted_externally_to_rlds", 1.0),
        ("berkeley_rpt_converted_externally_to_rlds", 1.0),
        ("kaist_nonprehensile_converted_externally_to_rlds", 3.0),
        ("stanford_robocook_converted_externally_to_rlds", 1.0),
        ("iamlab_cmu_pickup_insert_converted_externally_to_rlds", 1.0),
        ("utaustin_mutex", 1.0),
        ("cmu_play_fusion", 1.0),
    ],

    # === Open-X Magic Soup ===
    "oxe_magic_soup": [
        ("fractal20220817_data", 0.54087122203),                # Google RT-1 Robot Data (Large-Scale)
        ("kuka", 0.8341046294),
        # ("bridge_oxe", 1.0)                                   # Version of Bridge V2 in Open-X GCP Bucket
        ("bridge_orig", 1.0),                                   # Original Version of Bridge V2 from Project Website
        ("taco_play", 2.0),
        ("jaco_play", 1.0),
        ("berkeley_cable_routing", 1.0),
        ("roboturk", 2.0),
        # ("nyu_door_opening_surprising_effectiveness", 1.0),   # Note --> only contains wrist camera images (skip?)
        ("viola", 2.0),
        ("berkeley_autolab_ur5", 2.0),
        ("toto", 1.0),
        ("language_table", 0.1),
        ("stanford_hydra_dataset_converted_externally_to_rlds", 2.0),
        ("austin_buds_dataset_converted_externally_to_rlds", 1.0),
        ("nyu_franka_play_dataset_converted_externally_to_rlds", 3.0),
        ("furniture_bench_dataset_converted_externally_to_rlds", 0.1),
        ("ucsd_kitchen_dataset_converted_externally_to_rlds", 2.0),
        ("austin_sailor_dataset_converted_externally_to_rlds", 1.0),
        ("austin_sirius_dataset_converted_externally_to_rlds", 1.0),
        # ("bc_z", 0.2),                                        # Note --> raw data is broken!
        ("dlr_edan_shared_control_converted_externally_to_rlds", 1.0),
        ("iamlab_cmu_pickup_insert_converted_externally_to_rlds", 1.0),
        # ("uiuc_d3field", 1.0),                                # Note --> raw data is broken!
        ("utaustin_mutex", 1.0),
        ("berkeley_fanuc_manipulation", 2.0),
        ("cmu_stretch", 1.0),
    ],

    # === Open-X Magic Soup++ ===
    "oxe_magic_soup_plus": [
        ("fractal20220817_data", 0.54087122203),                # Google RT-1 Robot Data (Large-Scale)
        ("kuka", 0.8341046294),
        ("bridge_orig", 1.0),                                   # Original Version of Bridge V2 from Project Website
        ("taco_play", 2.0),
        ("jaco_play", 1.0),
        ("berkeley_cable_routing", 1.0),
        ("roboturk", 2.0),
        ("viola", 2.0),
        ("berkeley_autolab_ur5", 2.0),
        ("toto", 1.0),
        ("language_table", 0.1),
        ("stanford_hydra_dataset_converted_externally_to_rlds", 2.0),
        ("austin_buds_dataset_converted_externally_to_rlds", 1.0),
        ("nyu_franka_play_dataset_converted_externally_to_rlds", 3.0),
        ("furniture_bench_dataset_converted_externally_to_rlds", 0.1),
        ("ucsd_kitchen_dataset_converted_externally_to_rlds", 2.0),
        ("austin_sailor_dataset_converted_externally_to_rlds", 1.0),
        ("austin_sirius_dataset_converted_externally_to_rlds", 1.0),
        ("dlr_edan_shared_control_converted_externally_to_rlds", 1.0),
        ("iamlab_cmu_pickup_insert_converted_externally_to_rlds", 1.0),
        ("utaustin_mutex", 1.0),
        ("berkeley_fanuc_manipulation", 2.0),
        ("cmu_stretch", 1.0),
        ## New Datasets in MagicSoup++
        ("bc_z", 0.2),                                          # Note: use v0.1.0 --> later versions broken
        ("fmb_dataset", 1.0),
        ("dobbe", 0.2),
        ("droid", 0.06),
    ],

    "oxe_magic_soup_plus_minus": [
        ("fractal20220817_data", 1.0),                          # Google RT-1 Robot Data (Large-Scale)
        ("kuka", 0.8341046294),
        ("bridge_orig", 1.0),                                   # Original Version of Bridge V2 from Project Website
        ("taco_play", 2.0),
        ("jaco_play", 1.0),
        ("berkeley_cable_routing", 1.0),
        ("roboturk", 2.0),
        ("viola", 2.0),
        ("berkeley_autolab_ur5", 2.0),
        ("toto", 1.0),
        # ("language_table", 0.1),
        ("stanford_hydra_dataset_converted_externally_to_rlds", 2.0),
        ("austin_buds_dataset_converted_externally_to_rlds", 1.0),
        ("nyu_franka_play_dataset_converted_externally_to_rlds", 3.0),
        ("furniture_bench_dataset_converted_externally_to_rlds", 0.1),
        ("ucsd_kitchen_dataset_converted_externally_to_rlds", 2.0),
        ("austin_sailor_dataset_converted_externally_to_rlds", 1.0),
        ("austin_sirius_dataset_converted_externally_to_rlds", 1.0),
        ("dlr_edan_shared_control_converted_externally_to_rlds", 1.0),
        ("iamlab_cmu_pickup_insert_converted_externally_to_rlds", 1.0),
        ("utaustin_mutex", 1.0),
        ("berkeley_fanuc_manipulation", 2.0),
        ("cmu_stretch", 1.0),
        ## New Datasets in MagicSoup++
        ("bc_z", 0.2),                                          # Note: use v0.1.0 --> later versions broken
        ("fmb_dataset", 1.0),
        ("dobbe", 0.2),
        # ("droid", 0.06),
    ],

    # === T-DROID Dataset ===
    "tdroid_carrot_in_bowl": [
        ("tdroid_carrot_in_bowl", 1.0),
    ],
    "tdroid_pour_corn_in_pot": [
        ("tdroid_pour_corn_in_pot", 1.0),
    ],
    "tdroid_flip_pot_upright": [
        ("tdroid_flip_pot_upright", 1.0),
    ],
    "tdroid_move_object_onto_plate": [
        ("tdroid_move_object_onto_plate", 1.0),
    ],
    "tdroid_knock_object_over": [
        ("tdroid_knock_object_over", 1.0),
    ],
    "tdroid_cover_object_with_towel": [
        ("tdroid_cover_object_with_towel", 1.0),
    ],

    # === DROID Finetuning Datasets ===
    "droid_wipe": [
        ("droid_wipe", 1.0),
    ],

    # === LIBERO Datasets (Modified Versions) ===
    "libero_spatial_no_noops": [
        ("libero_spatial_no_noops", 1.0),
    ],
    "libero_object_no_noops": [
        ("libero_object_no_noops", 1.0),
    ],
    "libero_goal_no_noops": [
        ("libero_goal_no_noops", 1.0),
    ],
    "libero_10_no_noops": [
        ("libero_10_no_noops", 1.0),
    ],
    "libero_90": [
        ("libero_90", 1.0),
    ],
    
    # ==== LIBERO Shortcut Datasets (xyg Versions) ===
    "libero-spatial-island-viewpoint-400400": [
        ("v-0.400-0.400_num1/libero_spatial", 1.0),   
        ("v-0.600-0.600_num5/libero_spatial", 1.0),   
    ],
    
    "minivla-spatial-check-dataset-1": [
        ("v-0.500-0.500_num1/libero_spatial", 1.0),   
        ("v-0.500-0.500_num5/libero_spatial", 1.0),   
    ],
    
    "minivla-spatial-check-dataset-2": [
        ("v-0.400-0.400_num1/libero_spatial", 1.0),   
        ("v-0.600-0.600_num5/libero_spatial", 1.0),   
    ],
    
    # minivla viewpoint diversity group 1
    "minivla-spatial-split-dataset-400400-600600": [
        ("v-0.400-0.400_0,1,3,5,8/libero_spatial", 1.0),   
        ("v-0.600-0.600_2,4,6,7,9/libero_spatial", 1.0),  
    ],
    "minivla-spatial-split-dataset-400500-500600": [
        ("v-0.400-0.500_0,1,3,5,8/libero_spatial", 1.0),   
        ("v-0.500-0.600_2,4,6,7,9/libero_spatial", 1.0),  
    ],
    "minivla-spatial-split-dataset-400550-450600": [
        ("v-0.400-0.550_0,1,3,5,8/libero_spatial", 1.0),   
        ("v-0.450-0.600_2,4,6,7,9/libero_spatial", 1.0),  
    ],
    
    # minivla viewpoint diversity group 2
    "minivla-spatial-split-dataset-200200-800800": [
        ("v-0.200-0.200_0,1,3,5,8/libero_spatial", 1.0),   
        ("v-0.800-0.800_2,4,6,7,9/libero_spatial", 1.0),  
    ],
    "minivla-spatial-split-dataset-200350-650800": [
        ("v-0.200-0.350_0,1,3,5,8/libero_spatial", 1.0),   
        ("v-0.650-0.800_2,4,6,7,9/libero_spatial", 1.0),  
    ],
    "minivla-spatial-split-dataset-200500-500800": [
        ("v-0.200-0.500_0,1,3,5,8/libero_spatial", 1.0),   
        ("v-0.500-0.800_2,4,6,7,9/libero_spatial", 1.0),  
    ],
    "minivla-spatial-split-dataset-200650-350800": [
        ("v-0.200-0.650_0,1,3,5,8/libero_spatial", 1.0),   
        ("v-0.350-0.800_2,4,6,7,9/libero_spatial", 1.0),  
    ],
    
    # minivla viewpoint diversity group 3
    "minivla-spatial-split-dataset-300300-700700": [
        ("v-0.300-0.300_0,1,3,5,8/libero_spatial", 1.0),   
        ("v-0.700-0.700_2,4,6,7,9/libero_spatial", 1.0),  
    ],
    "minivla-spatial-split-dataset-300450-550700": [
        ("v-0.300-0.450_0,1,3,5,8/libero_spatial", 1.0),   
        ("v-0.550-0.700_2,4,6,7,9/libero_spatial", 1.0),  
    ],
    "minivla-spatial-split-dataset-300550-450700": [
        ("v-0.300-0.550_0,1,3,5,8/libero_spatial", 1.0),   
        ("v-0.450-0.700_2,4,6,7,9/libero_spatial", 1.0),  
    ],
    
    # minivla viewpoint diversity group 4
    "minivla-spatial-split-dataset-350350-650650": [
        ("v-0.350-0.350_0,1,3,5,8/libero_spatial", 1.0),   
        ("v-0.650-0.650_2,4,6,7,9/libero_spatial", 1.0),  
    ],
    
    "minivla-spatial-split-dataset-400400-600600": [
        ("v-0.400-0.400_0,1,3,5,8/libero_spatial", 1.0),   
        ("v-0.600-0.600_2,4,6,7,9/libero_spatial", 1.0),  
    ],
    
    # combine d diversity and c diversity into data island
    "minivla-spatial-split-dataset-400400-600600-specify_tmp": [
        ("v-0.400-0.400_0,1,3,5,8_specify_tmp/libero_spatial", 1.0),   
        ("v-0.600-0.600_2,4,6,7,9_specify_tmp/libero_spatial", 1.0),  
    ],
    
    # minivla viewpoint distance group 1
    "minivla-spatial-split-dataset-050350-650950": [
        ("v-0.050-0.350_0,1,3,5,8/libero_spatial", 1.0),   
        ("v-0.650-0.950_2,4,6,7,9/libero_spatial", 1.0),  
    ],
    
    # minivla viewpoint distance new group
    "minivla-spatial-split-dataset-050150-850950": [
        ("v-0.050-0.150_0,1,3,5,8/libero_spatial", 1.0),   
        ("v-0.850-0.950_2,4,6,7,9/libero_spatial", 1.0),  
    ],
    "minivla-spatial-split-dataset-050200-800950": [
        ("v-0.050-0.200_0,1,3,5,8/libero_spatial", 1.0),   
        ("v-0.800-0.950_2,4,6,7,9/libero_spatial", 1.0),  
    ],
    "minivla-spatial-split-dataset-050250-750950": [
        ("v-0.050-0.250_0,1,3,5,8/libero_spatial", 1.0),   
        ("v-0.750-0.950_2,4,6,7,9/libero_spatial", 1.0),  
    ],
    
    "minivla-spatial-split-dataset-100200-800900": [
        ("v-0.100-0.200_0,1,3,5,8/libero_spatial", 1.0),   
        ("v-0.800-0.900_2,4,6,7,9/libero_spatial", 1.0),  
    ],
    "minivla-spatial-split-dataset-150250-750850": [
        ("v-0.150-0.250_0,1,3,5,8/libero_spatial", 1.0),   
        ("v-0.750-0.850_2,4,6,7,9/libero_spatial", 1.0),  
    ],
    "minivla-spatial-split-dataset-200300-700800": [
        ("v-0.200-0.300_0,1,3,5,8/libero_spatial", 1.0),   
        ("v-0.700-0.800_2,4,6,7,9/libero_spatial", 1.0),  
    ],
    "minivla-spatial-split-dataset-250350-650750": [
        ("v-0.250-0.350_0,1,3,5,8/libero_spatial", 1.0),   
        ("v-0.650-0.750_2,4,6,7,9/libero_spatial", 1.0),  
    ],
    "minivla-spatial-split-dataset-300400-600700": [
        ("v-0.300-0.400_0,1,3,5,8/libero_spatial", 1.0),   
        ("v-0.600-0.700_2,4,6,7,9/libero_spatial", 1.0),  
    ],
    "minivla-spatial-split-dataset-350450-550650": [
        ("v-0.350-0.450_0,1,3,5,8/libero_spatial", 1.0),   
        ("v-0.550-0.650_2,4,6,7,9/libero_spatial", 1.0),  
    ],
    "minivla-spatial-split-dataset-400500-500600": [
        ("v-0.400-0.500_0,1,3,5,8/libero_spatial", 1.0),   
        ("v-0.500-0.600_2,4,6,7,9/libero_spatial", 1.0),  
    ],
    "minivla-spatial-split-dataset-425525-475575": [
        ("v-0.425-0.525_0,1,3,5,8/libero_spatial", 1.0),   
        ("v-0.475-0.575_2,4,6,7,9/libero_spatial", 1.0),  
    ],
    
}
# fmt: on
