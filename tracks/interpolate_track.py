"""Interpolation and raycasting for a track's boundary, loaded from Tracks/<map>/points-offset.csv."""
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d


@dataclass
class Vec3:
    x: float
    y: float
    z: float

    def __add__(self, other):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar):
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other):
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def length(self):
        return (self.x**2 + self.y**2 + self.z**2) ** 0.5

    def normalized(self):
        l = self.length()
        return Vec3(self.x / l, self.y / l, self.z / l) if l > 0 else Vec3(0, 0, 0)

    def to_array(self):
        return np.array([self.x, self.y, self.z])


class TrackInterpolator:
    def __init__(self, csv_path=None, map_name="A03-Race"):
        if csv_path is None:
            csv_path = Path(__file__).parent / map_name / "points-offset.csv"
        else:
            csv_path = Path(csv_path)
        self.csv_path = csv_path
        self.left_points = []
        self.right_points = []
        self.block_points = []
        self._load_points()
        self._build_interpolators()
        self._build_wall_polylines()

    def _load_points(self):
        """Load points from CSV, separated by type."""
        with open(self.csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                p = Vec3(float(row["x"]), float(row["y"]), float(row["z"]))
                if row["type"] == "left":
                    self.left_points.append(p)
                elif row["type"] == "right":
                    self.right_points.append(p)
                elif row["type"] == "block":
                    self.block_points.append(p)

    def _build_interpolators(self):
        """Build scipy interpolators for each boundary (left/right)."""
        def build_interp(points):
            if len(points) < 2:
                return None
            t = np.arange(len(points))
            xs = np.array([p.x for p in points])
            ys = np.array([p.y for p in points])
            zs = np.array([p.z for p in points])
            return (
                interp1d(t, xs, kind="cubic", fill_value="extrapolate"),
                interp1d(t, ys, kind="cubic", fill_value="extrapolate"),
                interp1d(t, zs, kind="cubic", fill_value="extrapolate"),
            )

        self.left_interp = build_interp(self.left_points)
        self.right_interp = build_interp(self.right_points)

    def _build_wall_polylines(self, n_samples: int = 2000, oversample: int = 10):
        """Pre-sample interpolated walls into XZ polylines, evenly spaced by arc length."""
        def sample_wall(interp, max_t, n_points):
            if interp is None:
                return None
            fx, _, fz = interp
            # Densely sample in parameter space, then re-parameterize by arc length
            # so the final points are evenly spaced by distance, not by CSV index.
            fine_t = np.linspace(0, max_t, n_points * oversample)
            fine_xz = np.column_stack([fx(fine_t), fz(fine_t)])
            seg_lengths = np.linalg.norm(np.diff(fine_xz, axis=0), axis=1)
            cum_length = np.concatenate([[0.0], np.cumsum(seg_lengths)])
            total_length = cum_length[-1]
            if total_length <= 0:
                return fine_xz[:n_points]
            target_lengths = np.linspace(0, total_length, n_points)
            target_t = np.interp(target_lengths, cum_length, fine_t)
            return np.column_stack([fx(target_t), fz(target_t)])

        self._left_wall_xz = sample_wall(self.left_interp, len(self.left_points) - 1, n_samples)
        self._right_wall_xz = sample_wall(self.right_interp, len(self.right_points) - 1, n_samples)

    def cast_rays(
        self,
        car_pos: Vec3,
        car_direction: Vec3,
        angles: list,
        max_dist: float = 500.0,
    ) -> list:
        """Cast rays from the car at given angles and return distances to walls.

        Args:
            car_pos: Car position in world space
            car_direction: Car forward direction (normalized)
            angles: List of angles in degrees. 0 = forward, positive = left, negative = right
            max_dist: Maximum ray distance

        Returns:
            List of (distance, side) tuples per angle. distance is None if no hit within max_dist.
        """
        # Work entirely in XZ plane — Y is ignored
        origin = np.array([car_pos.x, car_pos.z])
        fwd = np.array([car_direction.x, car_direction.z])
        fwd_len = np.linalg.norm(fwd)
        if fwd_len < 1e-6:
            return [(None, None)] * len(angles)
        fwd /= fwd_len

        # Rotate forward direction by each angle (positive = left in XZ plane)
        angles_rad = np.radians(angles)
        cos_a, sin_a = np.cos(angles_rad), np.sin(angles_rad)
        ray_x = fwd[0] * cos_a - fwd[1] * sin_a
        ray_z = fwd[0] * sin_a + fwd[1] * cos_a

        best_dist = np.full(len(angles), np.inf)
        best_side = [None] * len(angles)

        for wall_xz, side in ((self._left_wall_xz, "left"), (self._right_wall_xz, "right")):
            if wall_xz is None:
                continue
            dists = self._rays_polyline_dist(origin, ray_x, ray_z, wall_xz, max_dist)
            for i in np.nonzero(dists < best_dist)[0]:
                best_dist[i] = dists[i]
                best_side[i] = side

        return [
            (float(dist) if side is not None else None, side)
            for dist, side in zip(best_dist, best_side)
        ]

    def _rays_polyline_dist(
        self,
        origin: np.ndarray,
        ray_x: np.ndarray,
        ray_z: np.ndarray,
        wall_xz: np.ndarray,
        max_dist: float,
    ) -> np.ndarray:
        """Find each ray's nearest intersection distance with a polyline (XZ plane), inf if none.

        Uses vectorized Cramer's rule across all rays and segments simultaneously.
        """
        A = wall_xz[:-1]   # segment starts (N, 2)
        B = wall_xz[1:]    # segment ends   (N, 2)
        AB = B - A         # segment vectors (N, 2)
        AP = A - origin    # origin-to-segment-start (N, 2)

        dx, dz = ray_x[:, None], ray_z[:, None]  # (R, 1), broadcast against the segments

        # Cramer's rule for: O + t*d = A + s*AB
        # det = AB.x*dz - dx*AB.z
        det = AB[:, 0] * dz - dx * AB[:, 1]  # (R, N)

        # Parallel segments (det ~ 0) divide by inf instead, giving t = s = 0;
        # `usable` below rejects those.
        usable = np.abs(det) > 1e-10
        safe_det = np.where(usable, det, np.inf)

        # t = (AB.x*AP.z - AB.z*AP.x) / det
        t = (AB[:, 0] * AP[:, 1] - AB[:, 1] * AP[:, 0]) / safe_det
        # s = (dx*AP.z - dz*AP.x) / det
        s = (dx * AP[:, 1] - dz * AP[:, 0]) / safe_det

        # Valid hits: ray goes forward (t >= 0), hit is within segment (0 <= s <= 1)
        valid = usable & (t >= 0) & (s >= 0) & (s <= 1) & (t <= max_dist)

        return np.min(np.where(valid, t, np.inf), axis=1)

    def get_position(self, t: float, side: str = "left") -> Optional[Vec3]:
        """Interpolate position at parameter t (0 to len(points)-1).

        Args:
            t: Parameter along the track (0 to num_points-1)
            side: "left" or "right" wall

        Returns:
            Interpolated Vec3 position, or None if invalid
        """
        interp = self.left_interp if side == "left" else self.right_interp
        if interp is None:
            return None
        fx, fy, fz = interp
        return Vec3(float(fx(t)), float(fy(t)), float(fz(t)))

    def get_direction(self, t: float, delta: float = 0.1) -> Optional[Vec3]:
        """Get forward direction at parameter t by finite difference.

        Args:
            t: Parameter along the track
            delta: Finite difference step size

        Returns:
            Normalized direction vector
        """
        p1 = self.get_position(t - delta, "left")
        p2 = self.get_position(t + delta, "left")
        if p1 is None or p2 is None:
            return None
        direction = (p2 - p1).normalized()
        return direction if direction.length() > 0.01 else None

    def find_nearest_point(self, position: Vec3, side: str = "left") -> Tuple[int, float]:
        """Find nearest discrete point and its parameter value.

        Args:
            position: Query position in world space
            side: "left" or "right" wall

        Returns:
            (index, parameter_t) of nearest point
        """
        points = self.left_points if side == "left" else self.right_points
        min_dist = float("inf")
        min_idx = 0

        for i, p in enumerate(points):
            dist = (p - position).length()
            if dist < min_dist:
                min_dist = dist
                min_idx = i

        return min_idx, float(min_idx)


if __name__ == "__main__":
    interp = TrackInterpolator()

    # Example: car at t=100 along the left boundary
    t = 100.0
    car_pos = interp.get_position(t, "left")
    direction = interp.get_direction(t)

    if car_pos and direction:
        angles = [-90, -45, -10, 0, 10, 45, 90]
        hits = interp.cast_rays(car_pos, direction, angles)

        print(f"Car pos: {car_pos.x:.2f}, {car_pos.y:.2f}, {car_pos.z:.2f}")
        print(f"Forward: {direction.x:.3f}, {direction.y:.3f}, {direction.z:.3f}")
        print()
        for angle, (dist, side) in zip(angles, hits):
            label = f"{angle:+4d}°"
            if dist is not None:
                print(f"  {label}: {dist:6.2f} units  ({side} wall)")
            else:
                print(f"  {label}: no hit")
