"""Physical xy visualizations of mesh outlines and contact networks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from .contacts import contact_graph
from .frames import extract_mesh_vertices, frame_box
from .geometry import mesh_centers, projected_alpha_shape

FloatArray = NDArray[np.floating]


@dataclass(frozen=True)
class NetworkPlotData:
    """Geometry and graph data used to draw a particle-network plot."""

    centers: FloatArray
    outlines: tuple[BaseGeometry, ...]
    graph: nx.Graph


def _polygon_parts(shape: BaseGeometry) -> tuple[Polygon, ...]:
    if isinstance(shape, Polygon):
        return (shape,)
    if isinstance(shape, MultiPolygon):
        return tuple(shape.geoms)
    if isinstance(shape, GeometryCollection):
        polygons: list[Polygon] = []
        for geometry in shape.geoms:
            polygons.extend(_polygon_parts(geometry))
        return tuple(polygons)
    return ()


def _draw_outline(
    ax: Axes,
    shape: BaseGeometry,
    *,
    color: str,
    linewidth: float,
) -> None:
    polygons = _polygon_parts(shape)
    if not polygons:
        raise ValueError("an alpha shape must contain a polygon to be plotted")
    for polygon in polygons:
        x, y = polygon.exterior.xy
        ax.plot(x, y, color=color, linewidth=linewidth)
        for interior in polygon.interiors:
            x, y = interior.xy
            ax.plot(x, y, color=color, linewidth=linewidth)


def _draw_periodic_edge(
    ax: Axes,
    center_i: FloatArray,
    center_j: FloatArray,
    box: Any,
    *,
    color: str,
    linewidth: float,
    alpha: float,
) -> None:
    displacement_ij = np.asarray(box.wrap(center_j - center_i), dtype=float)
    displacement_ji = np.asarray(box.wrap(center_i - center_j), dtype=float)
    midpoint_i = center_i + 0.5 * displacement_ij
    midpoint_j = center_j + 0.5 * displacement_ji
    ax.plot(
        [center_i[0], midpoint_i[0]],
        [center_i[1], midpoint_i[1]],
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        zorder=1,
    )
    ax.plot(
        [center_j[0], midpoint_j[0]],
        [center_j[1], midpoint_j[1]],
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        zorder=1,
    )


def _full_outline_limits(
    outlines: tuple[BaseGeometry, ...],
    box: Any,
    padding: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    if padding < 0:
        raise ValueError("view_padding must be non-negative")

    minimum_x = min(-box.Lx / 2.0, *(shape.bounds[0] for shape in outlines))
    minimum_y = min(-box.Ly / 2.0, *(shape.bounds[1] for shape in outlines))
    maximum_x = max(box.Lx / 2.0, *(shape.bounds[2] for shape in outlines))
    maximum_y = max(box.Ly / 2.0, *(shape.bounds[3] for shape in outlines))

    x_margin = max((maximum_x - minimum_x) * padding, np.finfo(float).eps)
    y_margin = max((maximum_y - minimum_y) * padding, np.finfo(float).eps)
    return (
        (minimum_x - x_margin, maximum_x + x_margin),
        (minimum_y - y_margin, maximum_y + y_margin),
    )


def plot_particle_network(
    frame: Any,
    n_mesh: int,
    *,
    contact_cutoff: float,
    n_particles: int | None = None,
    alpha: float | None = None,
    ax: Axes | None = None,
    outline_color: str = "C0",
    center_color: str = "black",
    edge_color: str = "0.35",
    outline_linewidth: float = 1.5,
    edge_linewidth: float = 1.0,
    center_size: float = 18.0,
    show_labels: bool = False,
    show_full_outlines: bool = True,
    view_padding: float = 0.04,
) -> tuple[Figure, Axes, NetworkPlotData]:
    """Plot projected mesh outlines, vertex centroids, and contact edges.

    Meshes are unwrapped before their centers and outlines are calculated. Each
    complete mesh is then translated so its center lies in the primary box.
    Contacts use full 3D minimum-image distances, even though the visualization
    is a two-dimensional xy projection.

    The current plotting bounds support orthorhombic boxes. Contact detection
    itself remains valid for triclinic boxes through freud's wrapping rules.
    """
    box = frame_box(frame)
    if any(abs(value) > 1e-12 for value in (box.xy, box.xz, box.yz)):
        raise NotImplementedError(
            "plot_particle_network currently supports orthorhombic boxes only"
        )

    unwrapped_vertices = extract_mesh_vertices(
        frame, n_mesh, n_particles=n_particles, unwrap=True
    )
    unwrapped_centers = mesh_centers(unwrapped_vertices)
    centers = np.asarray(box.wrap(unwrapped_centers), dtype=float)
    translations = centers - unwrapped_centers
    display_vertices = unwrapped_vertices + translations[:, None, :]
    outlines = tuple(
        projected_alpha_shape(mesh, alpha=alpha) for mesh in display_vertices
    )

    graph = contact_graph(
        frame,
        n_mesh,
        contact_cutoff,
        n_particles=unwrapped_vertices.shape[0],
    )
    for node, center in enumerate(centers):
        graph.nodes[node]["center"] = center.copy()

    if ax is None:
        figure, ax = plt.subplots()
    else:
        figure = ax.figure

    for shape in outlines:
        _draw_outline(
            ax,
            shape,
            color=outline_color,
            linewidth=outline_linewidth,
        )

    for i, j in graph.edges:
        _draw_periodic_edge(
            ax,
            centers[i],
            centers[j],
            box,
            color=edge_color,
            linewidth=edge_linewidth,
            alpha=0.7,
        )

    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        s=center_size,
        color=center_color,
        zorder=3,
    )
    if show_labels:
        for index, center in enumerate(centers):
            ax.annotate(
                str(index),
                (center[0], center[1]),
                xytext=(4, 4),
                textcoords="offset points",
            )

    if show_full_outlines:
        x_limits, y_limits = _full_outline_limits(
            outlines, box, view_padding
        )
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
    else:
        ax.set_xlim(-box.Lx / 2.0, box.Lx / 2.0)
        ax.set_ylim(-box.Ly / 2.0, box.Ly / 2.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    data = NetworkPlotData(
        centers=centers,
        outlines=outlines,
        graph=graph,
    )
    return figure, ax, data
