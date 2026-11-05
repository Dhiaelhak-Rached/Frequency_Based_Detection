#!/usr/bin/env python3
"""Acoustic Line-of-Bearing (LoB) 2D triangulation solver using weighted least squares.

Fuses directional bearings from multiple microphone arrays, weights each array
by its spectral prominence/confidence, and computes the 2D intersection point,
target range, and residual geometric error.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from geometry import RigGeometry, vector_to_bearing


@dataclass
class TriangulationResult:
    success: bool
    x: Optional[float]  # meters
    y: Optional[float]  # meters
    range_m: Optional[float]  # distance from rig origin (m)
    bearing_deg: Optional[float]  # global bearing from rig origin (deg)
    residual_error_m: float  # geometric residual error (m)
    active_rays: int
    rays: Dict[str, Tuple[np.ndarray, float]]  # array_id -> (origin [x, y], bearing_deg)


class Triangulator:
    """Overdetermined weighted least-squares 2D ray intersection engine."""

    def __init__(
        self,
        geometry: RigGeometry,
        min_prominence: float = 1.4,
        max_residual_m: float = 0.25,
        max_relative_residual: float = 0.1,
    ):
        self.geo = geometry
        self.min_prominence = min_prominence
        self.max_residual_m = max_residual_m
        self.max_relative_residual = max_relative_residual

    def triangulate(
        self,
        bearings: Dict[str, Optional[float]],
        prominences: Dict[str, float],
    ) -> TriangulationResult:
        """Intersect bearing rays from arrays with valid detections.

        bearings: {array_id: global_bearing_deg}
        prominences: {array_id: prominence_score}
        """
        origins = []
        angles = []
        weights = []
        ray_dict = {}

        for arr_id, angle in bearings.items():
            if angle is None or arr_id not in self.geo.arrays:
                continue
            prom = prominences.get(arr_id, 1.0)
            if prom < self.min_prominence:
                continue

            arr = self.geo.arrays[arr_id]
            origins.append(arr.center)
            angles.append(angle)
            weights.append(prom)
            ray_dict[arr_id] = (arr.center.copy(), angle)

        n_rays = len(origins)
        if n_rays < 2:
            return TriangulationResult(
                success=False,
                x=None,
                y=None,
                range_m=None,
                bearing_deg=None,
                residual_error_m=0.0,
                active_rays=n_rays,
                rays=ray_dict,
            )

        # Formulate line equations: cos(alpha)*x - sin(alpha)*y = x0*cos(alpha) - y0*sin(alpha)
        A = np.zeros((n_rays, 2), dtype=float)
        b = np.zeros(n_rays, dtype=float)

        for i, (orig, alpha_deg) in enumerate(zip(origins, angles)):
            rad = np.deg2rad(alpha_deg)
            c = np.cos(rad)
            s = np.sin(rad)
            A[i, 0] = c
            A[i, 1] = -s
            b[i] = orig[0] * c - orig[1] * s

        # Weighted least squares: (A.T @ W @ A) p = A.T @ W @ b
        W = np.diag(weights)
        AtW = A.T @ W
        AtWA = AtW @ A
        AtWb = AtW @ b

        cond = np.linalg.cond(AtWA)
        if cond > 1e4 or np.isnan(cond):
            # Rays are nearly parallel (cannot intersect reliably)
            return TriangulationResult(
                success=False,
                x=None,
                y=None,
                range_m=None,
                bearing_deg=None,
                residual_error_m=999.0,
                active_rays=n_rays,
                rays=ray_dict,
            )

        try:
            sol = np.linalg.solve(AtWA, AtWb)
            x_est, y_est = float(sol[0]), float(sol[1])

            # Forward check: verify the intersection is along the positive ray direction
            forward_votes = 0
            for orig, alpha_deg in zip(origins, angles):
                rad = np.deg2rad(alpha_deg)
                u = np.array([np.sin(rad), np.cos(rad)])  # unit ray direction
                vec_to_point = sol - orig
                if np.dot(u, vec_to_point) > 0:
                    forward_votes += 1

            if forward_votes != n_rays:
                return TriangulationResult(
                    success=False,
                    x=None,
                    y=None,
                    range_m=None,
                    bearing_deg=None,
                    residual_error_m=999.0,
                    active_rays=n_rays,
                    rays=ray_dict,
                )

            residuals = np.abs(A @ sol - b)
            avg_residual = float(np.average(residuals, weights=weights))
            range_m = float(np.sqrt(x_est ** 2 + y_est ** 2))
            residual_limit = max(self.max_residual_m, self.max_relative_residual * range_m)
            if not np.isfinite(range_m) or range_m <= 0.0 or avg_residual > residual_limit:
                return TriangulationResult(
                    success=False,
                    x=None,
                    y=None,
                    range_m=None,
                    bearing_deg=None,
                    residual_error_m=avg_residual,
                    active_rays=n_rays,
                    rays=ray_dict,
                )

            bearing_deg = vector_to_bearing(sol)
            return TriangulationResult(
                success=True,
                x=x_est,
                y=y_est,
                range_m=range_m,
                bearing_deg=bearing_deg,
                residual_error_m=avg_residual,
                active_rays=n_rays,
                rays=ray_dict,
            )

        except np.linalg.LinAlgError:
            return TriangulationResult(
                success=False,
                x=None,
                y=None,
                range_m=None,
                bearing_deg=None,
                residual_error_m=999.0,
                active_rays=n_rays,
                rays=ray_dict,
            )
