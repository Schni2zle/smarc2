import torch
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution


class GP2DModel(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GP2DTrainer:
    def __init__(self, bounds=(0, 100), resolution=2, inducing_count=50, training_iter=1000):
        self.bounds = bounds
        self.bound_range = bounds[1] - bounds[0]
        self.resolution = resolution
        self.training_iter = training_iter

        self.train_grid_points, self.train_grid_points_stdv, self.train_outputs = self.generate_training_data()

        # Randomly sample inducing points
        self.inducing_points = self.train_grid_points[torch.randperm(self.train_grid_points.size(0))[:inducing_count]]
        self.model = GP2DModel(inducing_points=self.inducing_points)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

        self.optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
            {'params': self.likelihood.parameters()},
        ], lr=0.01)

        self.mll = gpytorch.mlls.VariationalELBO(
            self.likelihood, self.model, num_data=self.train_outputs.size(0)
        )

    def generate_training_data(self):
        tx = torch.linspace(self.bounds[0], self.bounds[1], self.resolution)
        ty = torch.linspace(self.bounds[0], self.bounds[1], self.resolution)
        xx, yy = torch.meshgrid(tx, ty, indexing="ij")
        points = torch.stack([xx.ravel(), yy.ravel()], dim=-1)

        tx_std = torch.linspace(0.03, 0.01, self.resolution)
        ty_std = torch.linspace(0.03, 0.01, self.resolution)
        xx_std, yy_std = torch.meshgrid(tx_std, ty_std, indexing="ij")
        stdv = torch.stack([xx_std.ravel(), yy_std.ravel()], dim=-1)

        outputs = torch.sin(points[:, 0] * 2 * math.pi / self.bound_range) + \
                  torch.cos(points[:, 1] * 2 * math.pi / self.bound_range) + \
                  torch.randn(points.size(0)) * 0.2

        return points, stdv, outputs

    def train(self):
        self.model.train()
        self.likelihood.train()

        for i in range(self.training_iter):
            sample = torch.distributions.Normal(self.train_grid_points, self.train_grid_points_stdv).rsample()

            self.optimizer.zero_grad()
            output = self.model(sample)
            loss = -self.mll(output, self.train_outputs)
            loss.backward()
            self.optimizer.step()

            if i % 50 == 0 or i == self.training_iter - 1:
                print(f"Iter {i}/{self.training_iter}, Loss: {loss.item():.4f}")

    def predict(self, test_resolution=100):
        self.model.eval()
        self.likelihood.eval()

        test_x = torch.linspace(self.bounds[0], self.bounds[1], test_resolution)
        test_y = torch.linspace(self.bounds[0], self.bounds[1], test_resolution)
        xx, yy = torch.meshgrid(test_x, test_y, indexing="ij")
        test_points = torch.stack([xx.flatten(), yy.flatten()], dim=-1)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(test_points))
            pred_mean = observed_pred.mean.reshape(xx.shape)
            pred_std = observed_pred.stddev.reshape(xx.shape)

        return xx, yy, pred_mean, pred_std 
    
    def cost_at_sample(self, x, y):
        x_sample = torch.tensor([x, y], dtype=torch.float32).unsqueeze(0)
        # Get predictive distribution
        pred = self.likelihood(self.model(x_sample))

        mean = pred.mean.item()
        stddev = pred.stddev.item()

        # print(f"Point: {x_sample.numpy()}, Mean: {mean:.3f}, Stddev: {stddev:.3f}")

        return mean, stddev

    def plot_results(self, xx, yy, pred_mean, pred_std):
        true_function = torch.sin(2 * math.pi * xx / self.bounds[1]) + \
                        torch.sin(2 * math.pi * yy / self.bounds[1])

        fig, axs = plt.subplots(1, 3, figsize=(18, 5))  # Now 3 subplots

        gp_plot = axs[0].contourf(xx.numpy(), yy.numpy(), pred_mean.numpy(), levels=20, cmap="viridis")
        axs[0].set_title("GP Predictive Mean")
        fig.colorbar(gp_plot, ax=axs[0])

        true_plot = axs[1].contourf(xx.numpy(), yy.numpy(), true_function.numpy(), levels=20, cmap="viridis")
        axs[1].set_title("True Function")
        fig.colorbar(true_plot, ax=axs[1])

        std_plot = axs[2].contourf(xx.numpy(), yy.numpy(), pred_std.numpy(), levels=20, cmap="magma")
        axs[2].set_title("GP Predictive Stddev (Uncertainty)")
        fig.colorbar(std_plot, ax=axs[2])

        plt.tight_layout()
        plt.show()

    def plot_training_data(self):
        f, ax = plt.subplots(1, 1, figsize=(6, 6))
        sc = ax.scatter(self.train_grid_points[:, 0], self.train_grid_points[:, 1],
                        c=self.train_outputs, cmap='viridis', s=40, label='Train Output')

        for (x, y), (sx, sy) in zip(self.train_grid_points, self.train_grid_points_stdv):
            ellipse = Ellipse((x.item(), y.item()), width=sx.item()*2, height=sy.item()*2,
                              edgecolor='gray', facecolor='none', lw=0.5, alpha=0.6)
            ax.add_patch(ellipse)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('2D Training Data with Uncertainty')
        plt.colorbar(sc, label='Train Output')
        ax.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

def main():
    trainer = GP2DTrainer(training_iter=1000)
    trainer.plot_training_data()
    trainer.train()
    xx, yy, pred_mean, pred_std = trainer.predict()
    trainer.plot_results(xx, yy, pred_mean, pred_std)
    trainer.cost_at_sample(25, 50)

if __name__ == "__main__":
    main()
