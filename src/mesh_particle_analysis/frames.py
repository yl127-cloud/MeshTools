"""Helpers for reading consecutive membrane meshes from HOOMD/GSD frames."""

from __future__ import annotations

from typing import Any

import freud
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]


def frame_box(frame: Any) -> freud.box.Box:
    """Return a freud box constructed from a HOOMD/GSD-like frame."""
    if not hasattr(frame, "configuration") or not hasattr(
        frame.configuration, "box"
    ):
        raise TypeError("frame must provide configuration.box")
    return freud.box.Box.from_box(frame.configuration.box)


def _mesh_counts(
    total_vertices: int,
    n_mesh: int,
    n_particles: int | None,
) -> tuple[int, int]:
    if not isinstance(n_mesh, (int, np.integer)) or n_mesh <= 0:
        raise ValueError("n_mesh must be a positive integer")

    if n_particles is None:
        if total_vertices % n_mesh != 0:
            raise ValueError(
                "the frame particle count is not divisible by n_mesh; "
                "pass n_particles explicitly if the frame contains other particles"
            )
        n_particles = total_vertices // n_mesh
    elif not isinstance(n_particles, (int, np.integer)) or n_particles <= 0:
        raise ValueError("n_particles must be a positive integer")

    used_vertices = int(n_particles) * int(n_mesh)
    if used_vertices > total_vertices:
        raise ValueError(
            f"requested {used_vertices} mesh vertices, but the frame has "
            f"only {total_vertices} particles"
        )
    return int(n_particles), used_vertices


def extract_mesh_vertices(
    frame: Any,
    n_mesh: int,
    n_particles: int | None = None,
    *,
    unwrap: bool = True,
) -> FloatArray:
    """Extract consecutive mesh vertices from a HOOMD/GSD-like frame.

    The first ``n_particles * n_mesh`` frame particles are interpreted as mesh
    vertices, with all vertices for mesh 0 first, followed by mesh 1, and so on.
    The returned array has shape ``(n_particles, n_mesh, 3)``.

    When ``unwrap`` is true, the function uses each vertex's stored GSD image
    and the frame box. This preserves a contiguous particle when it crosses a
    periodic boundary.
    """
    if not hasattr(frame, "particles") or not hasattr(
        frame.particles, "position"
    ):
        raise TypeError("frame must provide particles.position")

    positions = np.asarray(frame.particles.position, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("frame.particles.position must have shape (N, 3)")

    particle_count, used_vertices = _mesh_counts(
        positions.shape[0], n_mesh, n_particles
    )
    vertices = positions[:used_vertices].copy()

    if unwrap:
        if not hasattr(frame.particles, "image"):
            raise TypeError(
                "frame.particles.image is required when unwrap=True"
            )
        images = np.asarray(frame.particles.image)
        if images.shape != positions.shape:
            raise ValueError("frame.particles.image must have shape (N, 3)")
        vertices = frame_box(frame).unwrap(
            vertices, images[:used_vertices]
        )

    return np.asarray(vertices, dtype=float).reshape(
        particle_count, int(n_mesh), 3
    )

