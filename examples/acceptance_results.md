# Release acceptance results

Validated all four models on three examples with deposited and inferred receptors (24 runs).

Runtime: win32; NVIDIA GeForce RTX 4060 Laptop GPU; PyTorch 2.4.1; CUDA 12.1; seed 0; batch size 1.

Dice and AP compare predictions with the ground truth. Native values are the paper reference results. These examples support quick trial runs; the full dataset splits are listed in the repository.

| Example | Receptor | Model | Native Dice | Release Dice | Native AP | Release AP |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 8Z0F | deposited | unet_raw | 0.820825 | 0.820825 | 0.877785 | 0.877791 |
| 8Z0F | deposited | unet_diff | 0.820937 | 0.820937 | 0.890138 | 0.890124 |
| 8Z0F | deposited | unet_full | 0.835513 | 0.835513 | 0.880614 | 0.880603 |
| 8Z0F | deposited | ligandseek_pv | 0.838043 | 0.837113 | 0.870514 | 0.870311 |
| 8Z0F | inferred | unet_raw | 0.821067 | 0.821067 | 0.877966 | 0.877971 |
| 8Z0F | inferred | unet_diff | 0.817754 | 0.817754 | 0.890955 | 0.890935 |
| 8Z0F | inferred | unet_full | 0.868343 | 0.867506 | 0.906922 | 0.906924 |
| 8Z0F | inferred | ligandseek_pv | 0.870112 | 0.870112 | 0.924218 | 0.923998 |
| 7Z24 | deposited | unet_raw | 0.779343 | 0.779343 | 0.871748 | 0.871749 |
| 7Z24 | deposited | unet_diff | 0.799087 | 0.799087 | 0.895618 | 0.895626 |
| 7Z24 | deposited | unet_full | 0.849383 | 0.849383 | 0.937161 | 0.937137 |
| 7Z24 | deposited | ligandseek_pv | 0.863741 | 0.866359 | 0.941736 | 0.941487 |
| 7Z24 | inferred | unet_raw | 0.779343 | 0.779343 | 0.871768 | 0.871769 |
| 7Z24 | inferred | unet_diff | 0.800915 | 0.799087 | 0.893466 | 0.893465 |
| 7Z24 | inferred | unet_full | 0.850633 | 0.850633 | 0.919504 | 0.919528 |
| 7Z24 | inferred | ligandseek_pv | 0.840376 | 0.840376 | 0.935894 | 0.935837 |
| 9BXC | deposited | unet_raw | 0.727702 | 0.727702 | 0.748039 | 0.747622 |
| 9BXC | deposited | unet_diff | 0.755589 | 0.755522 | 0.776969 | 0.776973 |
| 9BXC | deposited | unet_full | 0.803810 | 0.803810 | 0.817974 | 0.817951 |
| 9BXC | deposited | ligandseek_pv | 0.828740 | 0.829245 | 0.841755 | 0.841608 |
| 9BXC | inferred | unet_raw | 0.727358 | 0.727530 | 0.747948 | 0.747567 |
| 9BXC | inferred | unet_diff | 0.780625 | 0.780625 | 0.798138 | 0.798137 |
| 9BXC | inferred | unet_full | 0.791771 | 0.791771 | 0.816558 | 0.816589 |
| 9BXC | inferred | ligandseek_pv | 0.819421 | 0.819139 | 0.834054 | 0.833963 |
