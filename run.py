import wandb
from utils.preprocess import generate_data
from train.critic_supervised_train import critic_train
from train.actor_supervised_train import actor_train
from train.actor_critic_ppo import ppo_trainer


def main():
    wandb.init(project="rl-paraphrasing")
    generate_data()

    critic_train([
        "--num_train_epochs", "20",
    ])
    
    actor_train([
        "--num_train_epochs", "20",
    ])

    ppo_trainer([
        "--num_train_epochs", "20",
        #"--actor", "t5-small",
        #"--critic", "distilbert-base-uncased",
    ])

if __name__ == "__main__":
    main()

# TODO:
# try to use ppo from stable baselines