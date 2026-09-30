import numpy as np
import pytest

from mesh_particle_analysis import extract_mesh_vertices


def test_extracts_multiple_consecutive_meshes(make_frame):
    positions = np.arange(18, dtype=float).reshape(6, 3)
    frame = make_frame(positions)

    vertices = extract_mesh_vertices(frame, n_mesh=2, unwrap=False)

    assert vertices.shape == (3, 2, 3)
    np.testing.assert_array_equal(vertices[1], positions[2:4])


@pytest.mark.parametrize("n_particles", [1, 2, 4])
def test_extracts_explicit_particle_counts(make_frame, n_particles):
    positions = np.arange(24, dtype=float).reshape(8, 3)
    frame = make_frame(positions)

    vertices = extract_mesh_vertices(
        frame, n_mesh=2, n_particles=n_particles, unwrap=False
    )

    assert vertices.shape == (n_particles, 2, 3)


def test_unwraps_mesh_vertices_using_images(make_frame):
    frame = make_frame(
        [[4.8, 0, 0], [-4.8, 0, 0]],
        images=[[0, 0, 0], [1, 0, 0]],
    )

    vertices = extract_mesh_vertices(frame, n_mesh=2)

    np.testing.assert_allclose(vertices[0, :, 0], [4.8, 5.2])


def test_requires_divisible_particle_count_when_inferred(make_frame):
    frame = make_frame(np.zeros((5, 3)))

    with pytest.raises(ValueError, match="not divisible"):
        extract_mesh_vertices(frame, n_mesh=2)

