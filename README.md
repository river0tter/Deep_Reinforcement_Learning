# ADL HW3
Please don't revise test.py, environment.py,  atari_wrapper.py,  agent_dir/agent.py

## Installation
Type the following command to install OpenAI Gym Atari environment.

`$ pip3 install opencv-python gym gym[atari]`

Please refer to [OpenAI's page](https://github.com/openai/gym) if you have any problem while installing.

## How to run :
training policy gradient:
* `$ python3 main.py --train_pg`

testing policy gradient:
* `$ python3 test.py --test_pg`

training DQN:
* `$ python3 main.py --train_dqn`

testing DQN:
* `$ python3 test.py --test_dqn`

If you want to see your agent playing the game,
* `$ python3 test.py --test_[pg|dqn] --do_render`


## Code structure

```
.
├── agent_dir (all agents are placed here)
│   ├── agent.py (defined 4 required functions of the agent. DO NOT MODIFY IT)
│   ├── agent_dqn.py (DQN agent sample code)
│   └── agent_pg.py (PG agent sample code)
├── argument.py (you can add your arguments in here. we will use the default value when running test.py)
├── atari_wrapper.py (wrap the atari environment. DO NOT MODIFY IT)
├── environment.py (define the game environment in HW3, DO NOT MODIFY IT)
├── main.py (main function)
└── test.py (test script. we will use this script to test your agents. DO NOT MODIFY IT)

```
