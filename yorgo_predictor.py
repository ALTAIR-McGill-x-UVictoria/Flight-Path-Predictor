# Core Extended Kalman Filter implementation used by validation.py
# Handles data loading, state propagation, and noise optimization.

from typing import Tuple
import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp


def latlon_to_xy(lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float) -> Tuple[np.ndarray, np.ndarray]:
    """Convert latitude/longitude to local Cartesian coordinates in meters."""
    R = 6371000.0
    x = np.radians(lon - lon0) * R * np.cos(np.radians(lat0))
    y = np.radians(lat - lat0) * R
    return x, y


def xy_to_latlon(x: np.ndarray, y: np.ndarray, lat0: float, lon0: float) -> Tuple[np.ndarray, np.ndarray]:
    """Convert local Cartesian coordinates back to latitude/longitude."""
    R = 6371000.0
    lat = np.degrees(y / R) + lat0
    lon = np.degrees(x / (R * np.cos(np.radians(lat0)))) + lon0
    return lat, lon


def load_pixhawk_data(file_name: str):
    """Load Pixhawk acceleration and GPS data only up to the flight apogee."""
    accel_path = os.path.join("Pixhawk_Data", f"{file_name}_vehicle_acceleration_0.csv")
    gps_path = os.path.join("Pixhawk_Data", f"{file_name}_vehicle_global_position_0.csv")

    acceleration = pd.read_csv(accel_path)
    gps = pd.read_csv(gps_path)

    # Determine apogee timestamp and truncate data
    apogee_idx = gps["alt"].idxmax()
    apogee_time = gps.loc[apogee_idx, "timestamp"]
    acceleration = acceleration[acceleration["timestamp"] <= apogee_time]
    gps = gps[gps["timestamp"] <= apogee_time]

    accel_x = acceleration["xyz[0]"].values
    accel_y = acceleration["xyz[1]"].values
    accel_z = acceleration["xyz[2]"].values - 9.81
    accel_time = acceleration["timestamp"].values

    downsample_factor = 20  # if original is 20Hz
    accel_time = accel_time[::downsample_factor]
    accel_x = accel_x[::downsample_factor]
    accel_y = accel_y[::downsample_factor]
    accel_z = accel_z[::downsample_factor]

    lat = gps["lat"].values
    lon = gps["lon"].values
    alt = gps["alt"].values
    gps_time = gps["timestamp"].values

    return (accel_time, accel_x, accel_y, accel_z), (gps_time, lat, lon, alt)


def run_ekf(file_name: str, q_scale: float = 100.0, r_scale: float = 0.01):
    """Run the EKF on a Pixhawk dataset and return results plus RMSE."""
    (accel_time, ax, ay, az), (gps_time, lat, lon, alt) = load_pixhawk_data(file_name)
    print(gps_time)
    # Convert timestamps to seconds from start
    accel_time = (accel_time - accel_time[0]) / 1e6
    gps_time = (gps_time - gps_time[0]) / 1e6

    lat0, lon0 = lat[0], lon[0]
    x_meas, y_meas = latlon_to_xy(lat, lon, lat0, lon0)
    z_meas = alt

    ax_i = interp1d(accel_time, ax, fill_value='extrapolate')
    ay_i = interp1d(accel_time, ay, fill_value='extrapolate')
    az_i = interp1d(accel_time, az, fill_value='extrapolate')

    def dyn(t, state):
        x, y, z, vx, vy, vz = state
        return [vx, vy, vz, float(ax_i(t)), float(ay_i(t)), float(az_i(t))]

    state = np.array([x_meas[0], y_meas[0], z_meas[0], 0.0, 0.0, 0.0])
    P = np.eye(6)

    Q = np.eye(6) * q_scale
    R = np.eye(3) * r_scale
    H = np.hstack([np.eye(3), np.zeros((3, 3))])

    est_positions = [state[:3]]
    pred_positions = []
    actual_positions = [np.array([x_meas[0], y_meas[0], z_meas[0]])]
    times = [gps_time[0]]
    nis_values = []
    nees_values = []

    max_t = accel_time[-1]

    for i in range(25, len(gps_time)):
        t_prev = gps_time[i - 25]
        t_curr = gps_time[i]

        sol = solve_ivp(dyn, (t_prev, t_curr), state, t_eval=[t_curr], method='RK45')
        state = sol.y[:, -1]

        dt = t_curr - t_prev
        A = np.eye(6)
        A[0, 3] = dt
        A[1, 4] = dt
        A[2, 5] = dt
        P = A @ P @ A.T + Q

        z = np.array([x_meas[i], y_meas[i], z_meas[i]])
        y_res = z - H @ state
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        state = state + K @ y_res
        P = (np.eye(6) - K @ H) @ P

        nis_values.append(float(y_res.T @ np.linalg.inv(S) @ y_res))
        err = state[:3] - z
        nees_values.append(float(err.T @ np.linalg.inv(P[:3, :3]) @ err))

        times.append(t_curr)
        est_positions.append(state[:3])
        actual_positions.append(z)

        # Predict 5 seconds ahead (GPS is 5Hz, so 25 timesteps)
        pred_idx = i + 25
        if pred_idx < len(gps_time):
            t_pred = gps_time[pred_idx]
        else:
            t_pred = gps_time[-1]
        sol_pred = solve_ivp(dyn, (t_curr, t_pred), state, t_eval=[t_pred], method='RK45')
        pred_positions.append(sol_pred.y[:3, -1])

    est_positions = np.array(est_positions)
    pred_positions = np.array(pred_positions)
    actual_positions = np.array(actual_positions)

    pred_times = np.array(times[1:]) + 5.0*1e6
    actual_future = np.stack([
        np.interp(pred_times, gps_time, x_meas),
        np.interp(pred_times, gps_time, y_meas),
        np.interp(pred_times, gps_time, z_meas)
    ], axis=1)

    if pred_positions.shape[0] == 0 or actual_future.shape[0] == 0:
        errors = np.array([])
        rmse = np.nan
    else:
        errors = np.linalg.norm(pred_positions - actual_future, axis=1)
        rmse = float(np.sqrt(np.mean(errors ** 2))) if len(errors) > 0 else np.nan
    print(f"Q: {q_scale:.2f}, R: {r_scale:.2f}, RMSE: {rmse:.2f} m")
    return (
        np.array(times[1:]),
        est_positions[1:],
        pred_positions,
        actual_positions[1:],
        actual_future,
        np.array(nis_values),
        np.array(nees_values),
        lat0,
        lon0,
        rmse,
    )


def optimize_noise(file_name: str):
    """Grid search over Q and R scales to minimize prediction RMSE."""
    q_scales = [100.0, 110.0, 150.0, 200.0]
    r_scales = [0.01, 0.1, 0.4]

    best_rmse = float("inf")
    best_result = None
    best_q = None
    best_r = None

    for q in q_scales:
        for r in r_scales:
            result = run_ekf(file_name, q_scale=q, r_scale=r)
            rmse = result[-1]
            if rmse < best_rmse:
                best_rmse = rmse
                best_result = result
                best_q = q
                best_r = r

    print(f"Best Q scale: {best_q}, R scale: {best_r}, RMSE={best_rmse:.2f}")
    return best_result