# Graph CNNs with Motif and Variable Temporal Block for Skeleton-Based Action Recognition

Hierarchical structure and different semantic roles of joints in human skeleton convey important information for action recognition. Conventional graph convolution methods for modeling skeleton structure consider only physically connected neighbors of each joint, and the joints of the same type, thus failing to capture highorder information. In this work, we propose a novel model with motif-based graph convolution to encode hierarchical spatial structure, and a variable temporal dense block to exploit local temporal information over different ranges of human skeleton sequences. Moreover, we employ a non-local block to capture global dependencies of temporal domain in an attention mechanism. Our model achieves improvements over the state-of-the-art methods on two large-scale datasets.

The Kinetics-M datasets contains 'belly dancing', 'punching bag', 'capoeira', 'squat', 'windsurfing', 'skipping rope', 'swimming backstroke', 'hammer throw', 'throwing discus', 'tobogganing', 'hopscotch',
               'hitting baseball', 'roller skating', 'arm wrestling', 'snatch weight lifting', 'tai chi', 'riding mechanical bull', 'salsa dancing', 'hurling (sport)', 'lunge',
               'skateboarding', 'country line dancing', 'juggling balls', 'surfing crowd', 'deadlifting', 'clean and jerk', 'crawling baby', 'push up', 'front raises', 'pull ups'. 
## Citation
Please cite the following paper if you use this repository in your reseach.
```
@inproceedings{Wen2019GraphCW,
  title={Graph CNNs with Motif and Variable Temporal Block for Skeleton-Based Action Recognition},
  author={Yu-hui Wen and Lin Gao and Hongbo Fu and Fang-Lue Zhang and Shihong Xia},
  booktitle={AAAI},
  year={2019}
}
```

## Contact
For any question, feel free to contact
```
Yuhui Wen: wenyh1616@gmail.com
```

## Special Thanks
Our work is based on [ST-GCN](https://github.com/yysijie/st-gcn). Special thanks to Sijie Yan, Yuanjun Xiong for their outstanding work.
