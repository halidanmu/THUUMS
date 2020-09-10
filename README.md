# THUUMS: An Open Source Toolkit for Uyghur Morphological Segmentation
## Contents
* [Introduction](#introduction)
* [Usage](#usage)
* [Experimental Results](#experimental-results)
* [License](#License)
* [Citation](#Citation)
* [Development Team](#development-team)
* [Contributors](#contributors)
* [Contact](#contact)

## Introduction

Morphological segmentation is a natural language processing task that aims to segment to its' corresponding morphemes, which are smallest meaning unit. 
THUUMS(Tsinghua University Uyghur Morphology Segmenter) is an open-source toolkit for neural machine translation developed by [the Natural Language Processing Group at Tsinghua University](http://nlp.csai.tsinghua.edu.cn/site2/index.php?lang=en).

## Usage
In the trunk folder

Preprocessing

```
 ./my_scripts/run_preprocess.sh 35 ../data/input.train ../data/output.train ../param/
```

Training

```
nohup ./my_scripts/run_exp.sh ../param/my_state.py gpu3 >log 2>&1 &
```

Testing

```
./transIter_seg_dev.sh
```

## Experimental Results

| Method | P | R | F |
| :------------: | :---: | :--------------: | :----------------: |
| Morfessor       |  73.10 | 73.10 | 73.25 | 
| CRF       |  94.20 | 95.23 | 94.66 | 
| FGRU       |  89.48 | 86.76 | 88.10 | 
| BGRU       |  84.76 | 81.74 | 83.22 | 
| BiGRU |  **96.74** | **97.39** | **97.06** |

## License

The source code is dual licensed. Open source licensing is under the [BSD-3-Clause](https://opensource.org/licenses/BSD-3-Clause), which allows free use for research purposes. For commercial licensing, please email [thumt17@gmail.com](mailto:thumt17@gmail.com).

## Citation

Please cite the following paper:

> Abudukelimu Halidanmu, Yong Cheng, Yang Liu, and Maosong Sun. 2017. [Uyghur Morphological Segmentation with Bidirectional GRU Neural Networks](http://jst.tsinghuajournals.com/EN/10.16511/j.cnki.qhdxxb.2017.21.001#1). Journal of Tsinghua University (Science and Technology).(in Chinese)

## Development Team

Project leaders: [Maosong Sun](http://www.thunlp.org/site2/index.php/zh/people?id=16), [Yang Liu](http://nlp.csai.tsinghua.edu.cn/~ly/), Huanbo Luan

Project members: Abudukelimu Halidanmu, Yong Cheng

## Contributors 
* [Abudukelimu Halidanmu](mailto:abdklmhldm@gmail.com) (Tsinghua University)

## Contact

If you have questions, suggestions and bug reports, please email [abdklmhldm@gmail.com](mailto:thumt17@gmail.com).
