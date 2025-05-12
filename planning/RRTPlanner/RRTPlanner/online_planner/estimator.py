import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import gpytorch

class EnvironmentalFieldGP:
    def __init__(self, field_func=None, bounds=(0, 100), resolution=100, noise_std=0.5):
        self.bounds = bounds  # (min, max)
        self.resolution = resolution
        self.noise_std = noise_std

        self.positions = []
        self.measurements = []

        self.field_func = field_func if field_func else self.default_field
        self.kernel = C(1.0) * RBF(length_scale=10.0)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=5)

    def default_field(self, x, y):
        return 10 + 5 * np.sin(0.1 * x) + 3 * np.cos(0.1 * y)

    def sample_at(self, x, y):
        z = self.field_func(x, y) + np.random.normal(0, self.noise_std)
        self.positions.append([x, y])
        self.measurements.append(z)
        return z

    def sample_path(self, path):
        for wp in path:
            x, y = wp
            self.sample_at(x, y)

    def fit_gp(self):
        if not self.positions:
            raise ValueError("No samples collected. Use sample_at() or sample_path() first.")
        X = np.array(self.positions)
        y = np.array(self.measurements)
        self.gp.fit(X, y)

    def predict_field(self):
        x = np.linspace(self.bounds[0], self.bounds[1], self.resolution)
        y = np.linspace(self.bounds[0], self.bounds[1], self.resolution)
        xx, yy = np.meshgrid(x, y)
        grid_points = np.vstack([xx.ravel(), yy.ravel()]).T
        mean, std = self.gp.predict(grid_points, return_std=True)
        return xx, yy, mean.reshape(xx.shape), std.reshape(xx.shape)

    def plot_field(self, mean, std=None):
        plt.figure(figsize=(8, 6))
        plt.contourf(*mean[:2], mean[2], cmap='viridis')
        plt.colorbar(label="Estimated Field")
        if self.positions:
            px, py = zip(*self.positions)
            plt.scatter(px, py, c='red', s=10, label='Samples')
        plt.title("GP-Approximated Field")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.grid(True)
        plt.show()

    def run(self, path, visualize=True):
        self.sample_path(path)
        self.fit_gp()
        pred = self.predict_field()
        if visualize:
            self.plot_field(pred)
        return pred  # returns (xx, yy, mean, std)


    def surrogate_model(self):
        # Placeholder for surrogate model implementation
        # Sample a path
        path = [(x, y) for x in range(0, 101, 10) for y in range(0, 101, 10)]
        # Run the GP estimator
        pred = self.run(path)
        # what is stored?
    
    def model_update(self):
        sample = self.sample_at(np.random.uniform(0, 100), np.random.uniform(0, 100))
        self.fit_gp()
        pred = self.predict_field()
        # self.plot_field(pred)
        return pred

def main():
    # Example usage
    field_func = lambda x, y: 10 + 10 * np.sin(0.1 * x) + 10 * np.cos(0.1 * y)
    # field_func = lambda x, y: np.sin(2 * x) * np.cos(2 * y)

    env_field_gp = EnvironmentalFieldGP(field_func=field_func, bounds=(0, 100), resolution=100, noise_std = 0.2)
    env_field_gp.surrogate_model()
    for i in range(500):
        pred = env_field_gp.model_update()
    env_field_gp.plot_field(pred)

if __name__ == "__main__":
    main()