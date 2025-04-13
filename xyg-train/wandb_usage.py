# train.py
import wandb
import random  # for demo script
import torch

def main1():
    wandb.login()

    epochs = 10
    lr = 0.01

    run = wandb.init(
        # Set the project where this run will be logged
        project="my-awesome-project",
        # Track hyperparameters and run metadata
        config={
            "learning_rate": lr,
            "epochs": epochs,
        },
    )

    offset = random.random() / 5
    print(f"lr: {lr}")

    # simulating a training run
    for epoch in range(2, epochs):
        acc = 1 - 2**-epoch - random.random() / epoch - offset
        loss = 2**-epoch + random.random() / epoch + offset
        print(f"epoch={epoch}, accuracy={acc}, loss={loss}")
        wandb.log({"accuracy": acc, "loss": loss})
    # run.log_code()


def main2():
    x = torch.load("123.pt")
    print(x)
    
    import ipdb; ipdb.set_trace()
    # ['name', 'dir', 'config', 'project', 'entity', 'group']
    del x['entity']
    x['entity'] = '1207481522'
    wandb.login()
    run1 = wandb.init(**x)


if __name__ == "__main__":
    main2()