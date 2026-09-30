"""Projected geometry and shape metrics for deformable mesh particles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import alphashape
import numpy as np
from numpy.typing import NDArray
from shapely.geometry.base import BaseGeometry

from .frames import extract_mesh_vertices

FloatArray = NDArray[np.floating]


@dataclass(frozen=True)
class ProjectedShape:
    """A particle's xy alpha shape and its projected two-dimensional metrics."""

    geometry: BaseGeometry
    area: float
    perimeter: float
    iq: float


def _project_xy(vertices: FloatArray) -> FloatArray:
    points = np.asarray(vertices, dtype=float)
    if points.ndim != 2 or points.shape[1] not in (2, 3):
        raise ValueError("vertices must have shape (N, 2) or (N, 3)")
    if points.shape[0] < 3:
        raise ValueError("at least three vertices are required for an alpha shape")
    if not np.all(np.isfinite(points)):
        raise ValueError("vertices must contain only finite coordinates")
    return points[:, :2]


def projected_alpha_shape(
    vertices: FloatArray,
    *,
    alpha: float | None = None,
) -> BaseGeometry:
    """Construct the two-dimensional alpha shape of xy-projected vertices."""
    points = _project_xy(vertices)
    if alpha is not None and alpha < 0:
        raise ValueError("alpha must be non-negative")

    if alpha is None:
        shape = alphashape.alphashape(points)
    else:
        shape = alphashape.alphashape(points, alpha)

    if shape.is_empty:
        raise ValueError("alpha-shape construction produced an empty geometry")
    if shape.area <= 0 or shape.length <= 0:
        raise ValueError(
            "alpha-shape construction did not produce a finite-area outline"
        )
    return shape


def projected_shape_metrics(
    vertices: FloatArray,
    *,
    alpha: float | None = None,
) -> ProjectedShape:
    """Calculate projected area, perimeter, and 2D IQ for one mesh.

    The isoperimetric quotient is ``4 * pi * area / perimeter**2``. This is a
    projected xy metric and is not a three-dimensional mesh sphericity.
    """
    shape = projected_alpha_shape(vertices, alpha=alpha)
    area = float(shape.area)
    perimeter = float(shape.length)
    iq = float(4.0 * np.pi * area / perimeter**2)
    return ProjectedShape(shape, area, perimeter, iq)


def particle_shape_metrics(
    frame: Any,
    n_mesh: int,
    n_particles: int | None = None,
    *,
    alpha: float | None = None,
) -> tuple[ProjectedShape, ...]:
    """Calculate projected shape metrics for every mesh in a frame."""
    vertices = extract_mesh_vertices(
        frame, n_mesh, n_particles=n_particles, unwrap=True
    )
    return tuple(
        projected_shape_metrics(mesh, alpha=alpha) for mesh in vertices
    )


def mesh_centers(vertices: FloatArray) -> FloatArray:
    """Return the vertex centroid of each unwrapped mesh.

    This is an unweighted arithmetic mean of vertex coordinates. It is not a
    volume centroid or a triangle-area-weighted surface centroid.
    """
    points = np.asarray(vertices, dtype=float)
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError("vertices must have shape (n_particles, n_mesh, 3)")
    if points.shape[0] == 0 or points.shape[1] == 0:
        raise ValueError("vertices must contain at least one non-empty mesh")
    return np.mean(points, axis=1)


def projected_packing_efficiency(
    frame: Any,
    n_mesh: int,
    n_particles: int | None = None,
    *,
    alpha: float | None = None,
) -> float:
    """Return summed projected particle area divided by the xy box area.

    Projected overlaps are counted once for each particle, matching the
    original notebook definition; this is not the area of the union of all
    projected particle shapes.
    """
    metrics = particle_shape_metrics(
        frame, n_mesh, n_particles=n_particles, alpha=alpha
    )
    box_values = np.asarray(frame.configuration.box, dtype=float)
    box_area = float(box_values[0] * box_values[1])
    if box_area <= 0:
        raise ValueError("the frame must have positive Lx and Ly")
    return float(sum(metric.area for metric in metrics) / box_area)

