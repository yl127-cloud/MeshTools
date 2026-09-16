# %%
import math
import itertools

import gsd.hoomd
import hoomd
import numpy as np
import packaging.version
from scipy.spatial import ConvexHull

import matplotlib.pyplot as plt
import time

# %% [markdown]
# ### TO DO:
#
# 1. Experiment with N_mesh and sigma: How does this affect the time it takes to run the simulation? The shape of the particles when fully compressed? The structure of the system? First examine changes visually, via OVITO. Then examine changes quantitatively: g(r) of center-to-center distances, and isoperimetric quotient (?) of particle shapes.
# 2. Experiment with k_bend, k_vol, k_area: How does this affect the time it takes to run the simulation? The shape of the particles when fully compressed? The structure of the system? First examine changes visually, via OVITO. Then examine changes quantitatively: g(r) of center-to-center distances, and isoperimetric quotient (?) of particle shapes.

# %%


def harmonic_U(k, sigma, r):
    V = k/2 * pow((1. - r/sigma), 2)
    return V


def harmonic_F(k, sigma, r):
    F = k/sigma * (1. - r/sigma)
    return F


def set_up_forces_WCA(epsilon, sigma, N_part):
    cell = hoomd.md.nlist.Cell(buffer=0.4)
    lj = hoomd.md.pair.LJ(nlist=cell, mode="shift")

    for i in range(N_part):
        # we want vertices on same mesh not to talk to each other...
        lj.params[("vertex%g" % i, "vertex%g" % i)] = dict(epsilon=0, sigma=0)
        lj.r_cut[("vertex%g" % i, "vertex%g" % i)] = 2 ** (1.0 / 6.0) * 0

        for j in range(i+1, N_part):
            lj.params[("vertex%g" % i, "vertex%g" % j)] = dict(
                epsilon=epsilon, sigma=sigma)
            lj.r_cut[("vertex%g" % i, "vertex%g" %
                      j)] = 2 ** (1.0 / 6.0) * sigma

    return lj


def set_up_walls_WCA(wall_epsilon, wall_sigma, location, N_part):
    top = hoomd.wall.Plane(origin=(0, 0, location), normal=(0, 0, -1))
    bottom = hoomd.wall.Plane(origin=(0, 0, -location), normal=(0, 0, 1))
    wall_lj = hoomd.md.external.wall.LJ(walls=[top, bottom])

    for j in range(N_part):
        wall_lj.params["vertex%g" % j] = dict(
            sigma=wall_sigma, epsilon=wall_epsilon, r_cut=2 ** (1.0 / 6.0) * wall_sigma)

    return wall_lj


def set_up_forces_harmonic(k, sigma, N_part):

    table = hoomd.md.pair.Table(
        nlist=hoomd.md.nlist.Cell(buffer=0.4, exclusions=['bond']))

    # endpoint = True? (used to be False)
    rs = np.linspace(0., sigma, num=1000)

    for i in range(N_part):
        # we want vertices on same mesh not to talk to each other...
        table.params[("vertex%g" % i, "vertex%g" % i)
                     ] = dict(r_min=0, U=[0], F=[0])
        table.r_cut[("vertex%g" % i, "vertex%g" % i)] = 0

        for j in range(i+1, N_part):
            table.params[("vertex%g" % i, "vertex%g" % j)] = dict(
                r_min=0, U=harmonic_U(k, sigma, rs), F=harmonic_F(k, sigma, rs))
            table.r_cut[("vertex%g" % i, "vertex%g" % j)] = sigma

    return table


def set_up_meshes(mesh_objs, k_bend, k_vol, vol, k_area, area, integrator):
    # k_bend: how difficult it is to bend the surface

    mesh_forces = []
    for mesh_obj in mesh_objs:

        bending_rigidity = hoomd.md.mesh.bending.BendingRigidity(mesh_obj)
        bending_rigidity.params["mesh"] = dict(k=k_bend)
        integrator.forces.append(bending_rigidity)
        mesh_forces.append(bending_rigidity)

        area_potential = hoomd.md.mesh.conservation.TriangleArea(mesh_obj)
        area_potential.params["mesh"] = dict(k=k_area, A0=area)
        integrator.forces.append(area_potential)
        mesh_forces.append(area_potential)

        volume_potential = hoomd.md.mesh.conservation.Volume(mesh_obj)
        volume_potential.params["mesh"] = dict(k=k_vol, V0=vol)
        integrator.forces.append(volume_potential)
        mesh_forces.append(volume_potential)

    print("done setting up meshes", flush=True)

    return mesh_forces


def create_system_flat(phi_init, N_part, N_mesh, R, sigma):
    Lz = 2.2*R
    Lx = math.ceil(pow((N_part * np.pi*(R+sigma/2.)**2)/phi_init, 1./2.))
    print(Lx, Lz)

    # Adapted from Catherine's thesis code.. is this efficient?
    spacing = 2*R + 2*sigma
    centers = np.zeros([N_part, 3])
    # creates one mesh at each center
    for i in np.arange(N_part):
        overlap = True
        while overlap:
            # buffer zones are necessary so that we don't have to worry about PBC right now
            x = np.round(np.random.uniform(-Lx/2 + R +
                         sigma, Lx/2 - R - sigma), 3)
            y = np.round(np.random.uniform(-Lx/2 + R +
                         sigma, Lx/2 - R - sigma), 3)
            z = 0

            centers[i, :] = np.array([x, y, z])

            dist = np.sqrt((centers[i, 0] - centers[0:i, 0])**2
                           + (centers[i, 1] - centers[0:i, 1])**2
                           + (centers[i, 2] - centers[0:i, 2])**2)

            if all(np.less(np.array([spacing]*i), dist)):
                print(i)
                overlap = False

    global_positions = []
    mesh_objs = []

    for i in range(N_part):
        vertex_positions, triangle_tags = create_particle(
            N_mesh, R, centers[i])
        # update particle tags
        triangle_tags += i*N_mesh
        # add to global list
        global_positions.extend(vertex_positions)

        mesh_obj = hoomd.mesh.Mesh()
        mesh_obj.types = ["mesh"]
        mesh_obj.triangulation = dict(
            type_ids=[0] * triangle_tags.shape[0], triangles=triangle_tags)
        # add to global list
        mesh_objs.append(mesh_obj)

    global_positions = np.asarray(global_positions)
    snapshot = hoomd.Snapshot()
    snapshot.particles.N = len(global_positions)
    snapshot.particles.position[:] = global_positions
    snapshot.particles.types = ["vertex%g" % n for n in range(N_part)]
    typeid = []
    for n in range(N_part):
        typeid.extend([n]*N_mesh)
    snapshot.particles.typeid[:] = typeid
    snapshot.configuration.box = [Lx, Lx, Lz, 0, 0, 0]

    snapshot.wrap()

    return snapshot, mesh_objs


def create_particle(N_mesh, R, center):
    # N_mesh: Number of vertices in the mesh
    # R: radius of the vesicle
    # center: center of the vesicle

    indices = np.arange(0, N_mesh, dtype=float) + 0.5

    # fibonacci sphere: phi and theta are the spherical coordinates
    phi = np.arccos(1 - 2 * indices / N_mesh)
    theta = np.pi * (1 + 5**0.5) * indices

    # converting spherical coordinates to cartesian coordinates
    x, y, z = (
        R * np.cos(theta) * np.sin(phi),
        R * np.sin(theta) * np.sin(phi),
        R * np.cos(phi),
    )

    vertex_positions = np.column_stack((x, y, z))

    # rs = [np.dot(r, r) for r in (vertex_positions-vertex_positions0)-center]
    # plt.hist(rs, bins=40)
    # plt.show()

    # figures out which particles should be connected to create triangles
    hull = ConvexHull(vertex_positions)
    triangle_tags = hull.simplices

    # orient triangles so all normals are facing outward from surface

    for i in range(len(triangle_tags)):
        normal = np.cross(
            vertex_positions[triangle_tags[i, 0]] -
            vertex_positions[triangle_tags[i, 2]],
            vertex_positions[triangle_tags[i, 0]] -
            vertex_positions[triangle_tags[i, 1]],
        )
        if np.dot(vertex_positions[triangle_tags[i, 0]], normal) < 0:
            triangle_tags[i, 1], triangle_tags[i, 2] = (
                triangle_tags[i, 2],
                triangle_tags[i, 1],
            )

    # must be added after orienting triangles...
    # otherwise, some triangles are oriented facing inward from surface for some reason
    # and there is a pinching effect on the mesh
    vertex_positions += np.asarray(center)

    return vertex_positions, triangle_tags


def find_mesh_distances(N_mesh, R):
    indices = np.arange(0, N_mesh, dtype=float) + 0.5
    # fibonacci sphere?
    phi = np.arccos(1 - 2 * indices / N_mesh)
    theta = np.pi * (1 + 5**0.5) * indices

    x, y, z = (
        R * np.cos(theta) * np.sin(phi),
        R * np.sin(theta) * np.sin(phi),
        R * np.cos(phi),
    )

    vertex_positions = np.column_stack((x, y, z))
    distances = []
    hull = ConvexHull(vertex_positions)
    triangle_tags = hull.simplices

    # orient triangles so all normals are facing outward from surface

    for i in range(len(triangle_tags)):
        d1 = vertex_positions[triangle_tags[i, 0]] - \
            vertex_positions[triangle_tags[i, 2]]
        d2 = vertex_positions[triangle_tags[i, 0]] - \
            vertex_positions[triangle_tags[i, 1]]
        distances.append(np.sqrt(np.dot(d1, d1)))
        distances.append(np.sqrt(np.dot(d2, d2)))

    return distances


def set_up_logger(name, period, sim, forces, mesh_forces):
    gsd_writer = hoomd.write.GSD(
        filename=name,
        trigger=hoomd.trigger.Periodic(period),
        mode="wb",
        filter=hoomd.filter.All(),
        dynamic=['property', 'momentum']
    )

    thermo = hoomd.md.compute.ThermodynamicQuantities(
        filter=hoomd.filter.All())
    sim.operations.add(thermo)

    logger = hoomd.logging.Logger()
    logger.add(sim, quantities=['timestep'])
    logger.add(thermo, quantities=[
               'kinetic_temperature', 'kinetic_energy', 'potential_energy', 'volume'])
    logger.add(forces, quantities=["energies", "forces", "torques"])
    # for mesh_force in mesh_forces:
    #     logger.add(mesh_force, quantities=["energies", "forces", "torques"])
    gsd_writer.logger = logger

    return gsd_writer

# %% [markdown]
# ## Wall compression


# %%
N_part = 5   # the number of deformable particles (mesh)
N_mesh = 256   # the number of particles that make up one deformable particle
R = 4    # radius of one undeformed mesh
phi_init = 0.1
phi_fin = 1.5   # final density, can reduce to less than 1
Lz_fin = R    # final box size
kT = 0.1
gamma = 1

# choose an appropriate sigma: controls the interaction range (larger sigma = particles touch each other earlier)
distances = find_mesh_distances(N_mesh, R)
print("Min distance: ", min(distances))
print("Max distance: ", max(distances))
sigma = 1.6 * max(distances)
print("chosen sigma: ", sigma)

# plt.figure(figsize=(6,4))
# plt.hist(distances, bins=50)
# plt.xlabel("Edge length")
# plt.ylabel("Count")
# plt.title("Mesh edge lengths")
# plt.axvline(
#     sigma,
#     linestyle="--",
#     label = f"sigma={sigma:.2f}"
# )
# plt.legend()
# plt.show

# %%
dump_period = int(1e4)
wall_period = int(1e4)
compress_period = int(5e5)

simseed = 1

snapshot, mesh_objs = create_system_flat(phi_init, N_part, N_mesh, R, sigma)

CPU = hoomd.device.CPU()
sim = hoomd.Simulation(device=CPU, seed=simseed)
state = sim.create_state_from_snapshot(snapshot)

# integrator = hoomd.md.Integrator(dt=0.001)
integrator = hoomd.md.minimize.FIRE(dt=0.001,
                                    force_tol=1e-2,
                                    angmom_tol=1e-2,
                                    energy_tol=1e-2
                                    )

sim.operations.integrator = integrator

####
# all mesh objects have same triangulation
N_triangles = len(mesh_objs[0].triangulation["triangles"])

# Define target area
Area = 4 * np.pi * R**2 / N_triangles

# Define target volume
Volume = 4 / 3 * np.pi * R**3

k_bend = 1
print("k_bend: ", k_bend)
k_vol = 100000 / Volume
print("k_vol: ", k_vol)
k_area = 150 / Area
print("k_area: ", k_area)
k_harm = 100
####

filename = (f"Npart{N_part}"
            f"_karea{k_area:.2f}"
            f"_phifin{phi_fin}"
            )
print("filename: ", filename)

# %%
start = time.time()
mesh_forces = set_up_meshes(
    mesh_objs, k_bend, k_vol, Volume, k_area, Area, integrator)

nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
integrator.methods.append(nve)

# forces = set_up_forces_WCA(1, sigma, N_part)
# integrator.forces.append(forces)

forces = set_up_forces_harmonic(k_harm, sigma, N_part)
integrator.forces.append(forces)

# create walls (wall_sigma = 1, wall_epsilon = 1)
wall_lj = set_up_walls_WCA(1, 1, sim.state.box.Lz/2, N_part)
sim.operations.integrator.forces.append(wall_lj)

gsd_writer = set_up_logger(
    filename+".gsd", dump_period, sim, forces, mesh_forces)
# so that OVITO knows particle size
gsd_writer.write_diameter = True
sim.operations.writers.append(gsd_writer)

# minimize energy
while not integrator.converged:
    sim.run(100)

print("Minimization runtime: ", time.time()-start)

# SQUISH THE PARTICLES
print("squishing!")

init_volume = sim.state.box.Lx*sim.state.box.Ly*sim.state.box.Lz
init_L = sim.state.box.Lz
slope_Lz = ((Lz_fin/2)-(init_L/2))/wall_period

sim.run(0)
for i in range(1, wall_period, 10):
    if i % 1000 == 0:
        print(i)

    wall_position = init_L/2 + i*slope_Lz
    top = hoomd.wall.Plane(origin=(0, 0, wall_position), normal=(0, 0, -1))
    bottom = hoomd.wall.Plane(origin=(0, 0, -wall_position), normal=(0, 0, 1))
    wall_lj.walls[0] = top
    wall_lj.walls[1] = bottom

    integrator.reset()

    while not integrator.converged:
        sim.run(100)

print("Squish runtime: ", time.time()-start)

# COMPRESS THE BOX
print("compressing!")

final_volume = (N_part * 4/3*np.pi*(R+sigma/2.)**3)/phi_fin
final_area = final_volume/Lz_fin
final_Lx = np.sqrt(final_area)

volume_ramp = hoomd.variant.box.Interpolate(
    initial_box=sim.state.box,
    final_box=[final_Lx, final_Lx, sim.state.box.Lz, 0, 0, 0],
    variant=hoomd.variant.Power(
        0, 1, power=0.1, t_start=0, t_ramp=compress_period),
)

for i in range(0, compress_period, 10):
    if i % 10000 == 0:
        print(i)
    new_box = volume_ramp(i)
    hoomd.update.BoxResize.update(sim.state, box=hoomd.Box(Lx=new_box[0], Ly=new_box[1], Lz=new_box[2],
                                                           xy=new_box[3], xz=new_box[4], yz=new_box[5]), filter=hoomd.filter.All())

    integrator.reset()

    while not (integrator.converged):
        sim.run(100)

sim.run(0)

print("Wall compression runtime: ", time.time()-start)

# must be included to close the gsd file so that it is readable without terminating this kernel
sim.operations.writers.remove(gsd_writer)
del gsd_writer

end = time.time()

# %%
print(f"Runtime = {end-start:.2f} seconds")

# %%
print("final timestep:", sim.timestep)
print("final timestep % dump_period", sim.timestep % dump_period)

# %%
traj = gsd.hoomd.open("Npart5_karea378.99.gsd")
print("last frame timestep", traj[-1].configuration.step)

# %%
