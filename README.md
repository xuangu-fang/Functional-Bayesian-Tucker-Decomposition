# Functional-Bayesian-Tucker-Tensor

This authors' official PyTorch implementation for paper:["Functional Bayesian Tucker Decomposition for Continuous-indexed Tensor Data"](https://openreview.net/forum?id=ZWyZeqE928&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2024%2FConference%2FAuthors%23your-submissions)) (ICLR 2024).



## Requirements:
The project is mainly built with pytorch 1.10.1 under python 3. Besides that, make sure to install [tqdm](https://tqdm.github.io/) and [tensorly](http://tensorly.org/stable/index.html) before running the project. The detailed package info can be found in requirement.txt (some packages in the list are redundant).

## Data

We offer the [raw data](https://drive.google.com/drive/folders/1DQJFZ9IkKw9pzr_vBSCLnrzqn4dp4kBd?usp=drive_link), [processed scripts](https://github.com/xuangu-fang/Functional-Bayesian-Tucker-Decomposition/tree/master/data/process_script), and processed data([.npy format](https://github.com/xuangu-fang/Functional-Bayesian-Tucker-Decomposition/tree/master/data/process_script)) for all four datasets used in paper. 

## Citation
Please cite our work if you would like it
```
@misc{fang2023streaming,
      title={Streaming Factor Trajectory Learning for Temporal Tensor Decomposition}, 
      author={Shikai Fang and Xin Yu and Shibo Li and Zheng Wang and Robert Kirby and Shandian Zhe},
      year={2023},
      eprint={2310.17021},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

```

to-do-list:
- [ ] 1. Add baseline
- [ ] 2. Add raw data -done
- [ ] 3. finish readme 
- [ ] 4. update arxiv link