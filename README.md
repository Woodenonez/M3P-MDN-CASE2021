# Multimodal Motion Prediction (Micro-version for Model Evaluation)
This is a repository presenting the evaluation on the MDN model in "Motion Prediction Based on Multiple Futures for Dynamic Obstacle Avoidance of Mobile Robots".

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
