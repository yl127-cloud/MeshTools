import matplotlib
import numpy as np

matplotlib.use("Agg")

from mesh_particle_analysis import plot_particle_network


def test_plot_returns_geometry_centers_and_contact_graph(make_frame):
    positions = [
        [-1.0, -0.5, 0],
        [0.0, -0.5, 0],
        [0.0, 0.5, 0],
        [-1.0, 0.5, 0],
        [0.2, -0.5, 0],
        [1.2, -0.5, 0],
        [1.2, 0.5, 0],
        [0.2, 0.5, 0],
    ]
    frame = make_frame(positions)

    figure, axes, data = plot_particle_network(
        frame,
        n_mesh=4,
        contact_cutoff=0.3,
        alpha=0.0,
    )

    assert axes.figure is figure
    assert data.centers.shape == (2, 3)
    assert len(data.outlines) == 2
    assert list(data.graph.edges) == [(0, 1)]
    figure.clear()


def test_plot_limits_include_complete_boundary_crossing_outline(make_frame):
    frame = make_frame(
        [
            [4.0, -1.0, 0],
            [-5.0, -1.0, 0],
            [-5.0, 1.0, 0],
            [4.0, 1.0, 0],
        ],
        images=[
            [0, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [0, 0, 0],
        ],
    )

    figure, axes, data = plot_particle_network(
        frame,
        n_mesh=4,
        contact_cutoff=0.3,
        alpha=0.0,
    )

    outline_maximum_x = data.outlines[0].bounds[2]
    assert outline_maximum_x > 5.0
    assert axes.get_xlim()[1] > outline_maximum_x
    figure.clear()
