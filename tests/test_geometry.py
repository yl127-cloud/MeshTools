import numpy as np

from mesh_particle_analysis import (
    mesh_centers,
    projected_packing_efficiency,
    projected_shape_metrics,
)


def test_square_projected_shape_metrics():
    square = np.asarray(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        dtype=float,
    )

    metrics = projected_shape_metrics(square, alpha=0.0)

    assert np.isclose(metrics.area, 1.0)
    assert np.isclose(metrics.perimeter, 4.0)
    assert np.isclose(metrics.iq, np.pi / 4.0)


def test_mesh_center_uses_unwrapped_vertex_mean():
    vertices = np.asarray(
        [[[4.8, 0, 0], [5.2, 0, 0]], [[0, 1, 0], [2, 3, 0]]]
    )

    centers = mesh_centers(vertices)

    np.testing.assert_allclose(centers, [[5.0, 0, 0], [1.0, 2.0, 0]])


def test_projected_packing_efficiency_sums_particle_areas(make_frame):
    positions = [
        [-2, -0.5, 0],
        [-1, -0.5, 0],
        [-1, 0.5, 0],
        [-2, 0.5, 0],
        [1, -0.5, 0],
        [2, -0.5, 0],
        [2, 0.5, 0],
        [1, 0.5, 0],
    ]
    frame = make_frame(positions, box=(10, 10, 10, 0, 0, 0))

    efficiency = projected_packing_efficiency(frame, n_mesh=4, alpha=0.0)

    assert np.isclose(efficiency, 0.02)

