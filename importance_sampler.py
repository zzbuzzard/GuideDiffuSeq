import torch
import matplotlib.pyplot as plt


class ImportanceSampler:
    """Class to handle importance sampling. Note: assumes t=0 is never used."""
    def __init__(self, timesteps: int, n: int = 10, device="cpu"):
        self.timesteps = timesteps
        self.n = 10
        self.device = device

        self.sq_losses = torch.zeros((timesteps, n), device=device)
        self.lengths = torch.zeros((timesteps,), dtype=torch.long, device=device)
        self.is_ready = False
        self.ps = None

    def register(self, timesteps: torch.LongTensor, losses: torch.FloatTensor):
        # It would be nice to vectorize this, but it's a bit fiddly and not too expensive
        #  so have not implemented for now
        (batch_size,) = losses.shape
        for i in range(batch_size):
            t = timesteps[i]
            self.sq_losses[t, self.lengths[t] % self.n] = losses[i].detach() ** 2
            self.lengths[t] += 1

        ps = torch.mean(self.sq_losses, dim=1).sqrt()
        self.ps = ps / torch.sum(ps)

    def scale_losses(self, timesteps: torch.LongTensor, losses: torch.FloatTensor):
        if self.check_ready():
            my_ps = self.ps[timesteps]
            return losses / my_ps / self.timesteps
        else:
            return losses

    def check_ready(self):
        if self.is_ready:
            return True
        # 1: to exclude timestep 0 which is not used during training
        if torch.all(self.lengths[1:] >= self.n).item():
            self.is_ready = True
            print("Importance sampling begins now!")
            return True
        return False

    def sample(self, batch_size: int):
        if not self.is_ready:
            return torch.randint(low=1, high=self.timesteps, size=(batch_size,), device=self.device)
        dist = torch.distributions.categorical.Categorical(probs=self.ps)
        return dist.sample((batch_size,)).to(self.device)

    def visualise(self):
        if not self.is_ready:
            return
        xs = torch.arange(1, self.timesteps)
        plt.plot(xs, self.ps.detach().cpu()[1:])
        plt.xlabel("Timestep")
        plt.ylabel("Probability")
        plt.show()
