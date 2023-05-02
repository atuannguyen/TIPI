# TIPI: Test Time Adaptation with Transformation Invariance

This repository is the official implementation for the CVPR 2023 paper TIPI: Test Time Adaptation with Transformation Invariance.

Please consider citing our paper as

```
@inproceedings{
nguyen2023tipi,
title={{TIPI}: Test Time Adaptation with Transformation Invariance},
author={A. Tuan Nguyen and Thanh Nguyen-Tang and Ser-Nam Lim and Philip Torr},
booktitle={Conference on Computer Vision and Pattern Recognition 2023},
year={2023},
url={https://openreview.net/forum?id=NVh1cy37Ge}
}
```

## Notes:
- For a fair comparison with TENT (our main baseline), this implementation only optimizes over the affine paramters (consistent with the paper). However, we also found that optimizing over all network parameters except for the last layer (similar to SHOT) gives much better performance. You can change this in the `collect_params` function in `tipi.py`. I will consider adding support for this soon.
- This implementation doesn't use a datapoint selection strategy. As stated in the paper's supplementary material, using a datapoint selection such as EATA significantly improves the performance.

## Requirements:
python3, pytorch 1.7.0 or higher, torchvision 0.8.0 or higher

## How to run:
An example is provided in run.sh. 
