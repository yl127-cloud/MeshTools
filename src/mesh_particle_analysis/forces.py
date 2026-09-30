"""Harmonic contact forces and per-mesh force aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .frames import extract_mesh_vertices, frame_box

FloatArray = NDArray[np.floating]


@dataclass(frozen=True)
class MeshPairForce:
    """Reconstructed harmonic contact force for one pair of meshes.

    ``scalar_load`` is the sum of vertex-pair force magnitudes, preserving the
    quantity calculated in the notebooks. ``force_on_i`` is their vector sum
    on mesh ``i``; the force on mesh ``j`` is its negative.
    """

    mesh_i: int
    mesh_j: int
    vertex_pair_contacts: int
    scalar_load: float
    force_on_i: FloatArray


def harmonic_potential(
    distance: ArrayLike,
    *,
    k: float,
    sigma: float,
) -> FloatArray:
    """Evaluate ``k/2 * (1 - distance/sigma)**2`` without applying a cutoff."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    values = np.asarray(distance, dtype=float)
    return np.asarray(k / 2.0 * (1.0 - values / sigma) ** 2)


def harmonic_force_magnitude(
    distance: ArrayLike,
    *,
    k: float,
    sigma: float,
) -> FloatArray:
    """Evaluate ``k/sigma * (1 - distance/sigma)`` without applying a cutoff."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    values = np.asarray(distance, dtype=float)
    return np.asarray(k / sigma * (1.0 - values / sigma))


def mesh_pair_contact_forces(
    frame: Any,
    n_mesh: int,
    *,
    sigma: float,
    k: float,
    n_particles: int | None = None,
) -> dict[tuple[int, int], MeshPairForce]:
    """Reconstruct harmonic vertex and mesh-pair forces from frame positions.

    All mesh pairs are returned. Non-contacting pairs have zero contacts, load,
    and vector force. Pair interactions use full 3D minimum-image distances and
    the strict ``distance < sigma`` cutoff used by the simulation table.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    meshes = extract_mesh_vertices(
        frame, n_mesh, n_particles=n_particles, unwrap=False
    )
    box = frame_box(frame)
    results: dict[tuple[int, int], MeshPairForce] = {}

    for i in range(meshes.shape[0]):
        for j in range(i + 1, meshes.shape[0]):
            displacements = meshes[i, :, None, :] - meshes[j, None, :, :]
            minimum_image = box.wrap(displacements.reshape(-1, 3))
            distances = np.linalg.norm(minimum_image, axis=1)
            contact_mask = distances < sigma

            if not np.any(contact_mask):
                results[(i, j)] = MeshPairForce(
                    i, j, 0, 0.0, np.zeros(3, dtype=float)
                )
                continue

            contact_distances = distances[contact_mask]
            if np.any(contact_distances == 0):
                raise ValueError(
                    "cannot determine force direction for exactly overlapping vertices"
                )

            magnitudes = harmonic_force_magnitude(
                contact_distances, k=k, sigma=sigma
            )
            directions = (
                minimum_image[contact_mask] / contact_distances[:, None]
            )
            force_on_i = np.sum(magnitudes[:, None] * directions, axis=0)
            results[(i, j)] = MeshPairForce(
                mesh_i=i,
                mesh_j=j,
                vertex_pair_contacts=int(np.count_nonzero(contact_mask)),
                scalar_load=float(np.sum(magnitudes)),
                force_on_i=np.asarray(force_on_i, dtype=float),
            )

    return results


def aggregate_mesh_forces(
    forces: ArrayLike,
    n_mesh: int,
    n_particles: int | None = None,
    *,
    dimensions: int = 2,
) -> FloatArray:
    """Sum per-vertex force vectors into one vector for each mesh."""
    values = np.asarray(forces, dtype=float)
    if values.ndim != 2 or values.shape[1] < dimensions:
        raise ValueError("forces must have shape (N, D) with D >= dimensions")
    if dimensions not in (2, 3):
        raise ValueError("dimensions must be 2 or 3")
    if n_mesh <= 0:
        raise ValueError("n_mesh must be positive")

    if n_particles is None:
        if values.shape[0] % n_mesh != 0:
            raise ValueError("force count is not divisible by n_mesh")
        n_particles = values.shape[0] // n_mesh
    used_vertices = n_particles * n_mesh
    if used_vertices > values.shape[0]:
        raise ValueError("requested more mesh forces than the array contains")

    grouped = values[:used_vertices, :dimensions].reshape(
        n_particles, n_mesh, dimensions
    )
    return np.sum(grouped, axis=1)

