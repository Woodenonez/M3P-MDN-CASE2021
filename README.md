# Multimodal Motion Prediction (Micro-version for Model Evaluation)
This is a repository presenting the evaluation on the MDN model in "[Motion Prediction Based on Multiple Futures for Dynamic Obstacle Avoidance of Mobile Robots](https://ieeexplore.ieee.org/document/9551463)", which is published on IEEE CASE2021. <br />

## Related Code
1. [Trajectory Generator](https://github.com/Woodenonez/TrajGenAvo_NMPC_OpEn)
2. [Motion prediction](https://github.com/Woodenonez/SimMotionPred_MDN_Pytorch)

#### Requirements
- pytorch
- matplotlib 

#### Data
Two evaluation sets are provided. <br />
Eval 1: Pedestrians and forklifts, T=10,...,20. <br />
Eval 2: Forklifts, T=20.

#### Model
The model is pre-trained.

#### Test run
Two 'main' files are meant to be run. The 'evaluation' file shows the evaluation histogram. The 'animation' file shows a visualized comparison between MDN and KF.
