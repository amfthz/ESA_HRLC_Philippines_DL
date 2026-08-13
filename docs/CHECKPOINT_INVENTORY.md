# Checkpoint Inventory

## CC50 Clean Baseline

Status: CURRENT PHILIPPINES BASELINE

Checkpoint:
`/home/tlcrs/Philippines_Project/09_Luigi Russo/DATASET_PHILIPPINES/checkpoints_cc50_clean_baseline/philippines/best_model.pt`

SHA-256:
`e13160c52cf6e65c6463788823258db96781af3ce76ce353f6a2641a1253a4ce`

Size: approximately 289 MB

Best checkpoint epoch: 53

Early stopping epoch: 78

Monitor: val_miou

Best training metric: 0.3696774013914169


## CC50 WeightedRandomSampler 10x

Status: EXPERIMENTAL - NOT BASELINE

Checkpoint:
`/home/tlcrs/Philippines_Project/09_Luigi Russo/DATASET_PHILIPPINES/checkpoints_cc50_weighted_sampler_10x/philippines/best_model.pt`

SHA-256:
`d871a654889019a91373ab4fe808e20c29da6e0c77aa5bb2eebae35960caa666`

Size: approximately 289 MB

Best checkpoint epoch: 20

Early stopping epoch: 45

Monitor: val_miou

Best training metric: 0.3524741870332935


## Storage note

The checkpoint binaries are not stored in the GitHub repository because of their size.

They remain on the UNIPV laboratory workstation under the paths documented above.

The final reference checkpoint to be used for production inference will be confirmed during the repository review with Luigi.
