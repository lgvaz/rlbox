# GymmeForce  
*Work in progress...*  
## Installation
```bash
git clone https://github.com/lgvaz/gymmeforce  
cd gymmeforce  
pip install -e .  
```
## About  
[TensorFlow](https://www.tensorflow.org/) implementation of [DQN](https://www.nature.com/nature/journal/v518/n7540/full/nature14236.html?foxtrotcallback=true) for solving [OpenAi-Gym](https://gym.openai.com/) discrete environments.  
![](assets/ep0_nolegend.gif)
![](assets/ep3500_nolegend.gif)
![](assets/ep6000_nolegend.gif)
![](assets/ep7500_nolegend.gif)
![](assets/ep21500_nolegend.gif)  
<img src="assets/cart_pole.gif" width="280" height="200" />
<img src="assets/acrobot.gif" width="280" height="200" />
<img src="assets/lunar_lander.gif" width="280" height="200" />  
**Standard DQN run on Breakout**  
Mean reward after training: 421 (100 episodes)  
Dark blue: Standard DQN  
Light blue: Double DQN  
![Breakout reward](assets/breakout_plots.png)  

