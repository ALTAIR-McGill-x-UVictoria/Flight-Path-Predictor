"""Validate the EKF by comparing estimated and actual trajectories."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

from yorgo_predictor import optimize_noise, run_ekf


def plot_results(est, actual, lat0, lon0, file_name):
    """Plot actual GPS path and EKF estimated path."""
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(actual[:, 0], actual[:, 1], actual[:, 2], label="Actual GPS")
    ax.plot(est[:, 0], est[:, 1], est[:, 2], label="EKF Estimate")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Altitude (m)")
    ax.legend()
    plt.title(f"Trajectory Comparison: {file_name}")
    plt.tight_layout()
    plt.show()


def plot_metrics(times, pred_pos, actual_future, nis, nees, file_name):
    """Plot EKF error metrics."""
    errors = np.linalg.norm(pred_pos - actual_future, axis=1)
    rmse = float(np.sqrt(np.mean(errors ** 2))) if len(errors) > 0 else np.nan

    dof = 3
    nis_thresh = chi2.ppf(0.95, dof)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    axes[0].plot(times, errors, label="5s Prediction Error")
    axes[0].set_ylabel("Error (m)")
    axes[0].set_title(f"Prediction Error (RMSE={rmse:.2f} m)")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(times, nis, label="NIS")
    axes[1].axhline(nis_thresh, color="r", linestyle="--", label="95% threshold")
    axes[1].set_ylabel("NIS")
    axes[1].set_title("Normalized Innovation Squared")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(times, nees, label="NEES")
    axes[2].axhline(nis_thresh, color="r", linestyle="--", label="95% threshold")
    axes[2].set_ylabel("NEES")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title("Normalized Estimation Error Squared")
    axes[2].grid(True)
    axes[2].legend()

    plt.suptitle(f"EKF Performance Metrics: {file_name}")
    plt.tight_layout()
    plt.show()


def main():
    file_base = "flight_07-24/flight_07-24"  # change if needed

    (
        times,
        est_pos,
        pred_pos,
        actual_pos,
        actual_future,
        nis,
        nees,
        lat0,
        lon0,
        _,
    ) = run_ekf(file_base)

    plot_results(est_pos, actual_pos, lat0, lon0, file_base)
    plot_metrics(times, pred_pos, actual_future, nis, nees, file_base)


if __name__ == "__main__":
    main()