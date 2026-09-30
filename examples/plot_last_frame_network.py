"""Plot mesh outlines and the contact network from a trajectory's last frame."""

import gsd.hoomd
import matplotlib.pyplot as plt

from mesh_particle_analysis import plot_particle_network


trajectory = gsd.hoomd.open("trajectory.gsd", mode="r")
frame = trajectory[-1]

plot_particle_network(
    frame,
    n_mesh=256,
    contact_cutoff=2.09,
    alpha=0.7,
    show_labels=True,
)
plt.show()

