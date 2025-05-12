import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt

class SKIGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, grid_size=32, grid_bounds=(0., 100.)):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        base_kernel = gpytorch.kernels.RBFKernel()
        self.covar_module = gpytorch.kernels.GridInterpolationKernel(
            base_kernel, grid_size=grid_size, num_dims=2, grid_bounds=[grid_bounds, grid_bounds]
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class SKIGaussianProcessEstimator:
    def __init__(self, field_func=None, bounds=(0., 100.), resolution=50, noise_std=0.2, grid_size=32):
        self.bounds = bounds
        self.resolution = resolution
        self.noise_std = noise_std
        self.grid_size = grid_size

        self.positions = []
        self.measurements = []

        self.field_func = field_func if field_func else self.default_field
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self.model = None

    def default_field(self, x, y):
        return 10 + 5 * np.sin(0.1 * x) + 3 * np.cos(0.1 * y)
    
    def synthetic_uncertainty(self, x, y):
        cx,cy = 50, 50
        sigma = 10
        return 10 * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))


    def sample_at(self, x, y):
        z = self.field_func(x, y) + np.random.normal(0, self.noise_std)
        # z = self.field_func(x, y) + np.random.normal(0, self.synthetic_uncertainty(x, y))
        self.positions.append([x, y])
        self.measurements.append(z)
        return z

    def sample_path(self, path):
        for wp in path:
            x, y = wp
            self.sample_at(x, y)

    def fit_gp(self):
        if not self.positions:
            raise ValueError("No samples to train on.")

        train_x = torch.tensor(self.positions, dtype=torch.float32).to(self.device)
        train_y = torch.tensor(self.measurements, dtype=torch.float32).to(self.device)

        self.model = SKIGPModel(train_x, train_y, self.likelihood, grid_size=self.grid_size, grid_bounds=self.bounds).to(self.device)
        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for i in range(50):
            optimizer.zero_grad()
            output = self.model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

    def predict_field(self):
        self.model.eval()
        self.likelihood.eval()

        x = np.linspace(self.bounds[0], self.bounds[1], self.resolution)
        y = np.linspace(self.bounds[0], self.bounds[1], self.resolution)
        xx, yy = np.meshgrid(x, y)
        test_points = np.vstack([xx.ravel(), yy.ravel()]).T
        test_x = torch.tensor(test_points, dtype=torch.float32).to(self.device)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = self.likelihood(self.model(test_x))
            mean = predictions.mean.cpu().numpy()
            std = predictions.stddev.cpu().numpy()

        return xx, yy, mean.reshape(xx.shape), std.reshape(xx.shape) 

    def model_update(self):
        x, y = np.random.uniform(self.bounds[0], self.bounds[1], 2)
        self.sample_at(x, y)
        self.fit_gp()
        return self.predict_field()

    def plot_field(self, xx, yy, mean, std=None, samples=None, title="SKI GP Estimated Field"):
        plt.figure(figsize=(8, 6))
        # cp = plt.contourf(xx, yy, mean, cmap='viridis')
        # plt.colorbar(cp, label="Mean Prediction")

        if std is not None:
            plt.contour(xx, yy, std, levels=10, cmap='Reds', alpha=0.6)
            plt.colorbar(label="Uncertainty")

        if samples is not None:
            px, py = zip(*samples)
            plt.scatter(px, py, c='red', s=10, label='Samples')

        plt.title(title)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def main():
    estimator = SKIGaussianProcessEstimator()
    path = [(x, y) for x in range(0, 101, 10) for y in range(0, 101, 10)]
    estimator.sample_path(path)
    estimator.fit_gp()
    xx, yy, mean, std = estimator.predict_field()
    estimator.plot_field(xx, yy, mean, std, estimator.positions, title="SKI GP Estimated Field")

if __name__ == "__main__":
    main()