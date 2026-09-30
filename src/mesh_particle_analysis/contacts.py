"""Periodic contact detection and particle contact networks."""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from .frames import extract_mesh_vertices, frame_box

FloatArray = NDArray[np.floating]


def _pair_distances(
    mesh_i: FloatArray,
    mesh_j: FloatArray,
    box: Any,
) -> FloatArray:
    displacements = mesh_i[:, None, :] - mesh_j[None, :, :]
    minimum_image = box.wrap(displacements.reshape(-1, 3))
    return np.linalg.norm(minimum_image, axis=1).reshape(
        mesh_i.shape[0], mesh_j.shape[0]
    )


def _contact_records(
    frame: Any,
    n_mesh: int,
    cutoff: float,
    n_particles: int | None,
) -> tuple[int, list[tuple[int, int, float, int]]]:
    if cutoff <= 0:
        raise ValueError("cutoff must be positive")

    meshes = extract_mesh_vertices(
        frame, n_mesh, n_particles=n_particles, unwrap=False
    )
    box = frame_box(frame)
    records: list[tuple[int, int, float, int]] = []

    for i in range(meshes.shape[0]):
        for j in range(i + 1, meshes.shape[0]):
            distances = _pair_distances(meshes[i], meshes[j], box)
            within_cutoff = distances < cutoff
            if np.any(within_cutoff):
                records.append(
                    (
                        i,
                        j,
                        float(np.min(distances)),
                        int(np.count_nonzero(within_cutoff)),
                    )
                )
    return meshes.shape[0], records


def contact_pairs(
    frame: Any,
    n_mesh: int,
    cutoff: float,
    n_particles: int | None = None,
) -> tuple[tuple[int, int], ...]:
    """Return mesh pairs having a 3D minimum-image vertex distance below cutoff."""
    _, records = _contact_records(frame, n_mesh, cutoff, n_particles)
    return tuple((i, j) for i, j, _, _ in records)


def contact_graph(
    frame: Any,
    n_mesh: int,
    cutoff: float,
    n_particles: int | None = None,
) -> nx.Graph:
    """Build an undirected graph of periodic mesh contacts.

    Each edge stores ``minimum_distance`` and ``vertex_pair_contacts``.
    Contact uses the same strict ``distance < cutoff`` criterion as the
    harmonic interaction in the original simulations.
    """
    particle_count, records = _contact_records(
        frame, n_mesh, cutoff, n_particles
    )
    graph = nx.Graph()
    graph.add_nodes_from(range(particle_count))
    for i, j, minimum_distance, vertex_pair_contacts in records:
        graph.add_edge(
            i,
            j,
            minimum_distance=minimum_distance,
            vertex_pair_contacts=vertex_pair_contacts,
        )
    return graph


def contact_degrees(graph: nx.Graph) -> NDArray[np.integer]:
    """Return contact degree in ascending node order."""
    nodes = sorted(graph.nodes)
    return np.asarray([graph.degree(node) for node in nodes], dtype=int)

