# Ligandseek

**Ligandseek: Receptor-Aware Ligand-Region Localization in Cryo-EM Density Maps**, accepted at **IEEE BIBM 2026**.

Ligandseek predicts ligand regions in cryo-EM maps and exports their probabilities, filtered voxel masks, candidate instances, and world coordinates. This repository provides the four paper models, tensor-only weights, EMDB–PDB split lists, and three prepared examples. The examples include deposited and CryoAtom2-inferred receptor inputs, so all four models can run without installing Chimera or CryoAtom2.

Source code and released weights use the [MIT license](LICENSE). Third-party notices are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Software citation metadata is provided in [CITATION.cff](CITATION.cff).

## Try an example

Use Python 3.10 and an NVIDIA CUDA environment. The commands below describe the paper's Linux reference environment. Install the base package from the repository root:

```bash
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e .
```

The three voxel models use the base dependencies. Ligandseek-PV additionally uses Point Transformer V3 and the Linux CUDA dependencies below. Install the matching scatter/cluster wheels before the other PV dependencies:

```bash
python -m pip install torch-scatter==2.1.2 torch-cluster==1.6.3 -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
python -m pip install torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install packaging ninja
python -m pip install -e '.[pv]' --no-build-isolation
```

FlashAttention installation requires a compatible CUDA toolkit and build environment; see its [installation instructions](https://github.com/Dao-AILab/flash-attention). The PyTorch commands follow the [official previous-version instructions](https://pytorch.org/get-started/previous-versions/). Match the scatter/cluster wheels to your PyTorch and CUDA versions.

The original Linux reference environment uses Python 3.10, PyTorch 2.4.1 with CUDA 12.1, NumPy 1.26.4, SciPy 1.11.4, Biopython 1.83, mrcfile 1.5.4, and einops 0.8.2. Its PV backend uses FlashAttention 2.8.3, torch-scatter 2.1.2, and torch-cluster 1.6.3.

The release examples were run on Windows with an NVIDIA GeForce RTX 4060 Laptop GPU, PyTorch 2.4.1 and CUDA 12.1, using torch-scatter 2.1.2, torch-cluster 1.6.3, and ordinary PyTorch attention in place of FlashAttention. The comparison covers all four models on three examples with both receptor settings (24 runs), plus six seeded PV repetitions; see [example validation results](examples/acceptance_results.md). This does not establish full Linux or cross-platform validation. A fresh installation of all dependencies has not been tested.

Download all four weights and 7Z24, the smallest prepared example, then run the four models:

```bash
python scripts/download_assets.py --example 7z24
python scripts/run_example.py --example 7z24 --source deposited --model all
```

The downloader needs no account or token. It verifies the sizes and SHA-256 values in [assets_manifest.json](assets_manifest.json). Existing verified downloads are reused. Weights are placed in `artifacts/checkpoints/`, and example inputs in `artifacts/examples/7z24/`.

Run the inferred receptor setting, or both settings:

```bash
python scripts/run_example.py --example 7z24 --source inferred --model all
python scripts/run_example.py --example 7z24 --source both --model all
```

Use `--model unet_raw` to download or run just the raw model; choose `8z0f` or `9bxc` with `--example` for the other cases. Example results appear in `outputs/<example>/<source>/<model>/`, with a combined `outputs/<example>/summary.json`. CUDA inference defaults to batch size 1 and seed 0. `--device`, `--batch-size`, `--seed`, and `--threshold` are explicit overrides. The reference values below come from the original paper runs; the linked validation results report the release runs.

## Predict a new map

`unet_raw` requires only experimental density. It does not require a receptor structure, simulated density, or CryoAtom2:

```bash
python scripts/predict.py --map experimental.mrc --output outputs/raw
```

Its threshold defaults to **0.86**. A supplied threshold takes precedence:

```bash
python scripts/predict.py --map experimental.mrc --threshold 0.90 --output outputs/raw_custom
```

For the paper's receptor-constrained output, additionally supply `--receptor receptor.cif`. Receptor atoms are mapped to their home voxels, whose ligand probabilities are set to zero. The network still receives only experimental density. Map-only prediction omits this constraint, so its final output has a different interpretation from a receptor-constrained paper run.

The other three variants require a prepared receptor-only simulated map:

```bash
python scripts/predict.py --model ligandseek_pv --map experimental.mrc --receptor receptor.cif --sim simulated.mrc --output outputs/pv
```

For a CryoAtom2-inferred structure, add `--receptor-mode inferred` and supply its matching simulated map. `unet_full` and Ligandseek-PV require receptor atoms; `unet_diff` can run from the two density inputs alone, while a supplied receptor retains the paper output constraint. Experimental and simulated maps must describe the same grid before preprocessing. Inputs are Fourier-resampled using the paper method to approximately 1 Å spacing; actual resampled spacing is preserved in the outputs.

## Four models and frozen parameters

| CLI model | Paper variant | Network inputs | Deposited threshold | Inferred threshold |
| --- | --- | --- | ---: | ---: |
| `unet_raw` | Raw-density U-Net | Experimental density, 1 channel | 0.86 | 0.86 |
| `unet_diff` | Difference-density U-Net | Experimental and difference density, 2 channels | 0.87 | 0.87 |
| `unet_full` | Receptor-aware U-Net | Full 56 density channels | 0.95 | 0.98 |
| `ligandseek_pv` | Ligandseek-PV | Full density channels and receptor point/voxel embedding | 0.79 | 0.88 |

No example-specific threshold search is performed. `--threshold` overrides the frozen value for the selected model and receptor setting. All variants use 80³ windows, stride 40, an internal crop of 5 voxels, and Gaussian merging with sigma ratio 0.5. Basic postprocessing thresholds with `>=`, removes six-connected components smaller than 10 voxels, and transitively merges component centers at a strict distance below 5 Å. Candidate confidence is the mean probability over its voxels.

Ligandseek-PV uses shuffled point serialization. The released commands fix the random seed, but changing the seed, attention backend, GPU, or batch size can introduce numerical differences. Compare probability maps, masks, and instance matches when reproducing a reported result; do not rely on instance numbers alone.

## Prepared examples and reference values

| Example | Map | Original deposited raw Dice | Original deposited PV Dice | Original inferred PV Dice |
| --- | --- | ---: | ---: | ---: |
| [`8z0f`](https://www.rcsb.org/structure/8Z0F) | [EMD-39703](https://www.ebi.ac.uk/emdb/EMD-39703) | 0.820825 | 0.838043 | 0.870112 |
| [`7z24`](https://www.rcsb.org/structure/7Z24) | [EMD-14457](https://www.ebi.ac.uk/emdb/EMD-14457) | 0.779343 | 0.863741 | 0.840376 |
| [`9bxc`](https://www.rcsb.org/structure/9BXC) | [EMD-45004](https://www.ebi.ac.uk/emdb/EMD-45004) | 0.727702 | 0.828740 | 0.819421 |

The cases include two protein receptors (8Z0F and 9BXC) and one protein–nucleic acid complex (7Z24, from the nucleic-acid benchmark cohort). In the original paper runs, all four variants reached Dice above 0.70 under both receptor settings, and Ligandseek-PV reached Dice above 0.80. They demonstrate successful predictions; they do not estimate average test-set performance or guarantee accuracy on new maps. Full original metrics for all 24 model/setting/case combinations are in [examples/reference_metrics.json](examples/reference_metrics.json).

Downloaded inputs follow this layout; the same filenames are used for each of the three cases:

```text
artifacts/
├── checkpoints/                         # four tensor-only checkpoints
└── examples/
    └── 8z0f/
        ├── experimental.map             # original experimental map
        ├── ground_truth.npz             # fixed-grid ligand mask and instance labels
        ├── deposited/
        │   ├── receptor.cif             # deposited structure, parsed for receptor atoms
        │   └── simulated.mrc            # prepared receptor-only simulated density
        └── inferred/
            ├── receptor.cif             # CryoAtom2-inferred receptor structure
            └── simulated.mrc            # matching prepared simulated density
```

`ground_truth.npz` contains `gt_ligand_mask` (boolean ZYX array), `gt_instance_label` (integer ZYX array; 0 is background), `origin` (XYZ grid origin in Å), and `voxel_size` (XYZ spacing in Å). The two label arrays align with the resampled prediction grid. Ground-truth ligand regions use the paper's 1.7 Å atom-distance definition. Ground truth is used only for evaluation, and is not a model input.

## Prediction outputs

For example, `outputs/8z0f/deposited/unet_raw/` contains:

| File | Contents |
| --- | --- |
| `ligand_pred.npz` | `ligand_pred`: float32 `(Z,Y,X)` probability after optional receptor clearing |
| `ligand_mask_filtered.npz` | `binary_mask_filtered`: boolean `(Z,Y,X)` accepted ligand voxels; `score_map`: float32 probabilities on the same grid |
| `instance_label_filtered.npz` | `instance_label_filtered`: int32 `(Z,Y,X)` candidate labels, with 0 as background |
| `voxel_candidates.json` | One record per positive instance ID, with voxel count, mean/max probability, center and bounding box |
| `grid_metadata.npz` | `origin`: float32 XYZ origin in Å; `voxel_size`: float32 XYZ spacing in Å; `hardmask`: integer ZYX receptor occupancy, 1 occupied and 0 unoccupied |
| `run_metadata.json` | Model, receptor setting, effective threshold and its source, clearing mode, grid and inference parameters |
| `metrics.json` | Example evaluation: voxel Dice/precision/recall/AP, directional and one-to-one instance coverage at 0.3/0.6, and top-3 success at 0.6 |

`metrics.json` is written by `run_example.py`; prediction of an unlabelled new map does not produce it. Average Precision excludes receptor home voxels in receptor-constrained evaluations. One-to-one instance matching maximizes the geometric mean of predicted-side and GT-side coverage, then checks both coverages against the threshold. Top-3 success means at least one of the three highest mean-probability candidates reaches both coverages at 0.6.

Candidate JSON fields are `instance_id`, `voxel_count`, `score_mean`, `score_max`, `center_voxel_zyx`, `center_world_xyz`, `bbox_min_zyx`, and `bbox_max_zyx`. Bounds are inclusive. World centers use voxel centers: `origin + (index_xyz + 0.5) * voxel_size`. Empty predictions produce an empty candidate list and all-zero instance labels. For example, an instance ID of `2` refers to exactly the voxels labelled `2` in `instance_label_filtered.npz`; its ID carries no ranking meaning.

This constructed format example represents 12 voxels with unit spacing and zero origin; it is not a measured prediction:

```json
{
  "instance_id": 2,
  "voxel_count": 12,
  "score_mean": 0.90,
  "score_max": 0.95,
  "center_voxel_zyx": [1.5, 3.0, 3.5],
  "center_world_xyz": [4.0, 3.5, 2.0],
  "bbox_min_zyx": [1, 2, 3],
  "bbox_max_zyx": [2, 4, 4]
}
```

## Data splits and paper evaluation

| Manifest | Purpose | EMDB–PDB pairs |
| --- | --- | ---: |
| [train.json](train.json) | Training | 4,152 |
| [val.json](val.json) | Checkpoint selection | 269 |
| [eval_protein_40.json](eval_protein_40.json) | Protein calibration | 40 |
| [eval_protein_110.json](eval_protein_110.json) | Protein test | 110 |
| [eval_nucleic_40.json](eval_nucleic_40.json) | Nucleic-acid test | 40 |

Each file is a JSON list of EMDB-to-PDB mappings, for example `[{"emd_39703": "8Z0F"}]`. These five identifier sets are mutually disjoint. The calibration and two test sets are subsets of a 969-pair held-out pool; training, validation, and the full held-out pool total 5,390 pairs. Identifier disjointness does not imply a sequence-homology split. The release provides the split lists and three complete examples; it does not distribute the entire evaluation pool.

For a locally prepared split, `scripts/run_paper_eval.py` requires a JSON list supplied with `--manifest`. The identifier lists above are not input-path manifests. A complete one-case path template is provided in [examples/evaluation_manifest.json](examples/evaluation_manifest.json); it matches the downloaded 8Z0F files:

```bash
python scripts/run_paper_eval.py --manifest examples/evaluation_manifest.json --data-root artifacts --model unet_raw --setting deposited --output-dir outputs/paper_eval --device cuda:0
```

For another prepared split, copy the template and add one record per case. All six fields are paths relative to `--data-root`: `map_path`, `labels_npz_path`, `cif_gt_path` (deposited receptor), `cif_path` (inferred receptor), `sim_map_path` (deposited simulation), and `sim_map_path_cryoatom` (inferred simulation). Labels may use the prepared `ground_truth.npz` schema described above. The selected model and setting determine which receptor and simulation are read. The evaluator uses the same prediction pipeline and frozen thresholds; `--threshold` supplies an explicit override.

## Preparing additional receptor inputs

The example simulations are already prepared. For a new receptor-aware input, use the same external workflow:

1. Use the deposited receptor, or obtain an inferred receptor with [CryoAtom2](https://github.com/YangLab-SDU/CryoAtom).
2. Remove ligands from the structure used to create simulated density, retaining the receptor atoms.
3. In [UCSF Chimera 1.18](https://www.rbvi.ucsf.edu/chimera/), simulate density at the experimental map's reported resolution on its grid. With receptor model `#0` and experimental map `#1`, the command is `molmap #0 <resolution_in_Angstroms> onGrid #1`; for a 3 Å map, use `molmap #0 3.0 onGrid #1`.
4. Export the simulated map and supply it with the matching receptor to `scripts/predict.py`.

Chimera and CryoAtom2 are external dependencies of this preparation step. Their packages and environments are not bundled here, and ChimeraX has not been substituted for the Chimera method.
