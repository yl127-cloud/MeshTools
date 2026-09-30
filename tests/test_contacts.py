import numpy as np

from mesh_particle_analysis import contact_degrees, contact_graph, contact_pairs


def test_no_contact_case(make_frame):
    frame = make_frame([[-2, 0, 0], [-1.8, 0, 0], [2, 0, 0], [2.2, 0, 0]])

    graph = contact_graph(frame, n_mesh=2, cutoff=0.5)

    assert graph.number_of_edges() == 0
    np.testing.assert_array_equal(contact_degrees(graph), [0, 0])


def test_known_contact_case(make_frame):
    frame = make_frame([[0, 0, 0], [0, 1, 0], [0.4, 0, 0], [0.4, 1, 0]])

    graph = contact_graph(frame, n_mesh=2, cutoff=0.5)

    assert list(graph.edges) == [(0, 1)]
    assert graph.edges[0, 1]["vertex_pair_contacts"] == 2
    np.testing.assert_array_equal(contact_degrees(graph), [1, 1])


def test_periodic_contact_across_box_boundary(make_frame):
    frame = make_frame(
        [[-4.9, 0, 0], [-4.9, 1, 0], [4.9, 0, 0], [4.9, 1, 0]],
        box=(10, 10, 10, 0, 0, 0),
    )

    pairs = contact_pairs(frame, n_mesh=2, cutoff=0.3)

    assert pairs == ((0, 1),)

