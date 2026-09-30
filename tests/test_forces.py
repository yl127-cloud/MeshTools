import numpy as np

from mesh_particle_analysis import (
    aggregate_mesh_forces,
    mesh_pair_contact_forces,
)


def test_known_pair_force(make_frame):
    frame = make_frame([[0, 0, 0], [0.5, 0, 0]])

    result = mesh_pair_contact_forces(
        frame, n_mesh=1, sigma=1.0, k=2.0
    )[(0, 1)]

    assert result.vertex_pair_contacts == 1
    assert np.isclose(result.scalar_load, 1.0)
    # freud performs box wrapping in single precision.
    np.testing.assert_allclose(
        result.force_on_i, [-1.0, 0.0, 0.0], atol=2e-6
    )


def test_noncontact_pair_has_zero_force(make_frame):
    frame = make_frame([[0, 0, 0], [2, 0, 0]])

    result = mesh_pair_contact_forces(
        frame, n_mesh=1, sigma=1.0, k=2.0
    )[(0, 1)]

    assert result.vertex_pair_contacts == 0
    assert result.scalar_load == 0.0
    np.testing.assert_array_equal(result.force_on_i, [0, 0, 0])


def test_aggregate_mesh_forces_preserves_xy_definition():
    forces = np.asarray(
        [[1, 2, 10], [3, 4, 20], [-1, 1, 30], [2, -2, 40]],
        dtype=float,
    )

    totals = aggregate_mesh_forces(forces, n_mesh=2, dimensions=2)

    np.testing.assert_allclose(totals, [[4, 6], [1, -1]])
