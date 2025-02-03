import torch
import os

# handling older model version loading
import sys
import hvae_backbone, hvae_backbone.block
sys.modules['custom'] = hvae_backbone
sys.modules['custom.blocks'] = hvae_backbone.block

class Checkpoint:
    """
    Checkpoint class for saving and loading experiments
    """
    def __init__(self, epoch=-1, model=None, optimizer=None, scheduler=None, params=None):
        try:
            self.epoch: int = epoch
            self.model = model
            self.params = params

            self.scheduler_state_dict = scheduler.state_dict() if scheduler is not None else None
            self.optimizer_state_dict = optimizer.state_dict() if optimizer is not None else None
        except TypeError:
            print("Error loading experiment")

    def save(self, path, save_locally=False):
        checkpoint_dir = os.path.join(path, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        if save_locally:
            local_checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{self.epoch}.pth")
            checkpoint_path = os.path.join(checkpoint_dir, "ref.pth")
            torch.save(local_checkpoint_path, checkpoint_path)
            #print('saving:\n', local_checkpoint_path, self.model)
            torch.save(self, local_checkpoint_path)
            description = f"epoch: {self.epoch}  \n"\
                          f"path: {local_checkpoint_path}"
        else:
            checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
            torch.save(self, checkpoint_path)
            description = f"epoch: {self.epoch}"
        
        architecture_path = os.path.join(checkpoint_dir, "architecture.txt")
        if not os.path.exists(architecture_path):
            architecture = open(architecture_path, "w")
            architecture.write(str(self.model))
            architecture.close()

        return checkpoint_path

    def save_migration(self, path):
        os.makedirs(path, exist_ok=True)
        checkpoint_path = os.path.join(path, f"migrated_checkpoint.pth")
        torch.save(self, checkpoint_path)
        return checkpoint_path

    @staticmethod
    def load(path):
        experiment: Checkpoint = torch.load(path, map_location='cpu')
        return experiment

    def get_model(self):
        return self.model

    def __getstate__(self):
        return {
                "epoch": self.epoch,
                "model":       self.model.serialize(),
                "scheduler_state_dict": self.scheduler_state_dict,
                "optimizer_state_dict": self.optimizer_state_dict,
                }

    def __setstate__(self, state):
        from .hvae import hVAE

        self.epoch = state["epoch"]
        self.model = hVAE.deserialize(state["model"])
        self.scheduler_state_dict = state["scheduler_state_dict"]
        self.optimizer_state_dict = state["optimizer_state_dict"]


