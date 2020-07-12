# Robot Exploration with Deep Reinforcement Learning
This repository contains code for robot exploration training with Deep Reinforcement Learning (DRL). The agent utilize the local structure of the environment to predict robot’s optimal sensing action. A demonstration can be found here -> www.youtube.com/watch?v=2gNF6efv12s

<p align='center'>
    <img src="/doc/exploration.png" alt="drawing" width="1000"/>
</p>

<p align='center'>
    <img src="/doc/policy.gif" alt="drawing" width="1000"/>
</p>

## Dependency
- Python 3
- [scikit-image](https://scikit-image.org/)
- [tensorboardX](https://github.com/lanpa/tensorboardX)
- [TensorFlow](https://www.tensorflow.org/install) (code is writen under TF1.x but it is modified to be compatible with TF2)
- [pybind11](https://github.com/pybind/pybind11) (pybind11 — Seamless operability between C++11 and Python)
  ```
  wget -O ~/Downloads/pybind11.zip https://github.com/pybind/pybind11/archive/master.zip
  cd ~/Downloads/ && unzip pybind11.zip -d ~/Downloads/
  cd ~/Downloads/pybind11-master/
  mkdir build && cd build
  cmake ..
  sudo make install
  ```
## Compile

You can use the following commands to download and compile the package.
```
git clone https://github.com/RobustFieldAutonomyLab/DRL_robot_exploration.git
cd DRL_robot_exploration
mkdir build && cd build
cmake ..
make
```

## How to Run?
- For CNN policy
    ```
    cd DRL_robot_exploration/scripts
    python3 tf_policy_cnn.py
    ```
- For RNN policy
    ```
    cd DRL_robot_exploration/scripts
    python3 tf_policy_rnn.py
    ```
- To select running mode, at the beginning of the tf_policy code:
    ```
    # select mode
    TRAIN = True
    PLOT = False
    ```
  Set ``TRAIN=False`` to run saved policy. You can train your own policy by set ``TRAIN=True``. Set `` PLOT=True `` to turn on visualization plots.
  
## Cite

Please cite [our paper](https://www.researchgate.net/profile/Fanfei_Chen/publication/330200308_Self-Learning_Exploration_and_Mapping_for_Mobile_Robots_via_Deep_Reinforcement_Learning/links/5d6e7ad4a6fdccf93d381d2e/Self-Learning-Exploration-and-Mapping-for-Mobile-Robots-via-Deep-Reinforcement-Learning.pdf) if you use any of this code: 
```
@inproceedings{ExplorDRL2019,
  title={Self-Learning Exploration and Mapping for Mobile Robots via Deep Reinforcement Learning},
  author={Chen, Fanfei and Bai, Shi and Shan, Tixiao and Englot, Brendan},
  booktitle={AIAA SciTech Forum},
  pages={0396},
  year={2019},
}
```

## Reference
- [DeepRL-Agents](https://github.com/awjuliani/DeepRL-Agents)
- [DeepLearningFlappyBird](https://github.com/yenchenlin/DeepLearningFlappyBird)