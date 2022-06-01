# rl-paraphrasing

Paraphrasing using mT5 and self-critic ppo

```sh
python preprocess.py
python actor_supervised_train.py
python critic_supervised_train.py
python ppo_data.py
python actor_critic_ppo.py
```