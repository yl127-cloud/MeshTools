# Mesh Particle Analysis

> **For Teich Lab members:** This package is intended for members of the
> [Wellesley College Soft Matter Lab](https://sites.google.com/wellesley.edu/teichlab)

`mesh-particle-analysis` provides reusable Python tools for analyzing deformable
mesh particles generated with HOOMD-blue and stored in GSD trajectories. The
package separates general scientific calculations from Signac job selection,
study-specific filtering, and one-off notebook plotting.

## Contents

- [What the package does](#what-the-package-does)
- [Scientific conventions](#scientific-conventions)
- [Repository structure](#repository-structure)
- [Installation](#installation)
  - [Clone the repository](#1-clone-the-repository)
  - [Use an existing HOOMD environment](#2a-use-an-existing-hoomd-environment)
  - [Create a separate analysis environment](#2b-create-a-separate-analysis-environment)
  - [Install the local package](#3-install-the-local-package)
  - [Configure VS Code and Jupyter](#4-configure-vs-code-and-jupyter)
  - [Verify the installation](#5-verify-the-installation)
- [Expected simulation data](#expected-simulation-data)
- [Basic usage](#basic-usage)
  - [Open the last GSD frame](#open-the-last-gsd-frame)
  - [Plot particle outlines and contacts](#plot-particle-outlines-and-contacts)
  - [Calculate shape metrics](#calculate-shape-metrics)
  - [Calculate packing efficiency](#calculate-packing-efficiency)
  - [Build a contact network](#build-a-contact-network)
  - [Reconstruct pairwise contact forces](#reconstruct-pairwise-contact-forces)
  - [Aggregate logged forces by mesh](#aggregate-logged-forces-by-mesh)
- [Working with Signac](#working-with-signac)
- [Running the tests](#running-the-tests)
- [Important assumptions and limitations](#important-assumptions-and-limitations)
- [Files that should not be committed](#files-that-should-not-be-committed)

## What the package does

The package currently supports:

- Extracting consecutively stored mesh vertices from a HOOMD/GSD frame.
- Unwrapping mesh vertices through periodic boundaries using the frame's image
  vectors and simulation box.
- Calculating a vertex-centroid center for each mesh particle.
- Projecting mesh vertices onto the xy plane.
- Constructing projected two-dimensional alpha shapes.
- Calculating projected area, perimeter, and isoperimetric quotient (IQ).
- Calculating projected packing efficiency.
- Detecting mesh contacts with minimum-image vertex distances.
- Building NetworkX contact graphs and calculating contact degree.
- Reconstructing harmonic vertex-pair and mesh-pair contact forces.
- Aggregating logged per-vertex forces into per-mesh net forces.
- Plotting mesh outlines, centers, labels, and contact-network edges.

The core package accepts a simulation frame directly. It does not require a
Signac job object.

The intended workflow is:

```text
Signac locates a simulation
        ↓
GSD opens trajectory.gsd
        ↓
A frame is selected
        ↓
mesh_particle_analysis analyzes the frame
```

## Scientific conventions

### Projected IQ

IQ is calculated from the two-dimensional alpha shape of mesh vertices
projected onto the xy plane:

```text
IQ = 4πA / P²
```

where `A` is projected alpha-shape area and `P` is projected alpha-shape
perimeter. This is a **projected two-dimensional shape metric**, not a
three-dimensional mesh sphericity.

### Mesh center

The center of a mesh is the arithmetic mean of its unwrapped vertex
coordinates. This is a vertex centroid, not a volume centroid or a
triangle-area-weighted surface centroid.

### Contact

Two meshes are in contact when at least one inter-mesh vertex pair satisfies:

```text
minimum-image 3D distance < contact cutoff
```

For the existing harmonic simulations, the contact cutoff is normally the
interaction parameter `sigma`.

### Packing efficiency

Projected packing efficiency is:

```text
sum of individual projected particle areas / xy box area
```

Overlapping projected regions are counted once for each particle. This is not
the area of the union of every projected shape.

### Forces

The force API distinguishes between:

- A harmonic force for one interacting vertex pair.
- The sum of force magnitudes for all vertex contacts between two meshes.
- The vector resultant force between two meshes.
- A net force obtained by summing logged vertex forces over an entire mesh.

## Repository structure

```text
Mesh_code/
├── pyproject.toml
├── README.md
├── LICENSE
├── src/
│   └── mesh_particle_analysis/
│       ├── __init__.py
│       ├── frames.py
│       ├── geometry.py
│       ├── contacts.py
│       ├── forces.py
│       └── visualization.py
├── tests/
│   ├── conftest.py
│   ├── test_frames.py
│   ├── test_geometry.py
│   ├── test_contacts.py
│   ├── test_forces.py
│   └── test_visualization.py
└── examples/
    └── plot_last_frame_network.py
```

The modules have the following responsibilities:

- `frames.py`: frame validation, box conversion, mesh extraction, and vertex
  unwrapping.
- `geometry.py`: alpha shapes, projected metrics, centers, and packing
  efficiency.
- `contacts.py`: periodic contact detection, graphs, and contact degrees.
- `forces.py`: harmonic interactions, pair forces, and mesh-force aggregation.
- `visualization.py`: projected outlines, centers, labels, and contact edges.

The original notebooks and simulation scripts are retained as research
provenance, but they are not imported by the package.

## Installation

### 1. Clone the repository

Replace the placeholder with the repository URL:

```bash
git clone https://github.com/yl127-cloud/MeshTools.git Mesh_code
cd Mesh_code
```

If the repository is already on your computer, open its root directory in VS
Code. The root directory is the one containing `pyproject.toml`.

### 2A. Use an existing HOOMD environment

This is the recommended option when the environment is also used to run the
simulations.

```bash
conda activate hoomd

conda install -c conda-forge \
    numpy gsd freud alphashape shapely networkx matplotlib \
    pytest hatchling ipykernel
```

The Conda package is named `freud`, while the corresponding Python import is:

```python
import freud
```

### 2B. Create a separate analysis environment

HOOMD itself is not required when analyzing an existing GSD trajectory. A
smaller analysis-only environment can be created with:

```bash
conda create -n mesh-analysis -c conda-forge \
    python=3.12 numpy gsd freud alphashape shapely networkx \
    matplotlib pytest hatchling ipykernel

conda activate mesh-analysis
```

Use either the existing HOOMD environment or the separate analysis
environment. There is no need to install the same dependencies into both.

### 3. Install the local package

From the repository root, with the selected environment active:

```bash
python -m pip install -e . --no-deps
```

The `-e` flag installs the package in editable mode. Changes made under
`src/mesh_particle_analysis/` become available without reinstalling the
package. The installation points to the source directory and does not create a
second copy of the package.

Alternatively, in an environment where the dependencies have not already been
installed through Conda, pip can install the package and optional GSD support:

```bash
python -m pip install -e ".[io]"
```

Conda is recommended for compiled scientific dependencies such as freud.

### 4. Configure VS Code and Jupyter

In VS Code:

1. Open the Command Palette.
2. Select **Python: Select Interpreter**.
3. Choose the environment used during installation.
4. For a notebook, click the kernel name in the upper-right corner.
5. Select the same environment as the notebook kernel.
6. Restart the notebook kernel after changing environments.

If an environment does not appear as a named Jupyter kernel, register it:

```bash
python -m ipykernel install \
    --user \
    --name mesh-analysis \
    --display-name "Python (mesh-analysis)"
```

When using the existing HOOMD environment, the names can instead be:

```bash
python -m ipykernel install \
    --user \
    --name hoomd \
    --display-name "Python (hoomd)"
```

Confirm the interpreter from inside a notebook:

```python
import sys

print(sys.executable)
```

### 5. Verify the installation

From a terminal:

```bash
python -c "import mesh_particle_analysis; print(mesh_particle_analysis.__file__)"
```

Then verify the principal dependencies and plotting function:

```bash
python -c "import gsd.hoomd, freud, alphashape, shapely, networkx, matplotlib; from mesh_particle_analysis import plot_particle_network; print('Installation successful')"
```

If the terminal import works but a notebook import fails, the notebook is using
a different Python kernel. Print `sys.executable` in both places and select the
same interpreter in VS Code.

## Expected simulation data

The first version assumes that each deformable particle has `n_mesh` vertices
and that the vertices are stored consecutively:

```text
mesh 0: vertices 0 through n_mesh - 1
mesh 1: vertices n_mesh through 2*n_mesh - 1
mesh 2: vertices 2*n_mesh through 3*n_mesh - 1
...
```

When the frame contains only mesh vertices, the number of meshes can be
inferred:

```python
n_particles = frame.particles.N // n_mesh
```

If a frame contains additional non-mesh particles, pass `n_particles`
explicitly and ensure the mesh vertices are the first particles in the frame.

## Basic usage

### Open the last GSD frame

```python
from pathlib import Path

import gsd.hoomd

gsd_path = Path("Npart13_area125.gsd")
trajectory = gsd.hoomd.open(gsd_path, mode="r")
frame = trajectory[-1]

print("Frames:", len(trajectory))
print("Last timestep:", frame.configuration.step)
print("Stored vertices:", frame.particles.N)
```

If the data directory is the parent of the code directory:

```python
gsd_path = Path("..") / "Npart13_area125.gsd"
trajectory = gsd.hoomd.open(gsd_path.resolve(), mode="r")
```

Using `pathlib.Path` safely handles spaces in filenames.

### Plot particle outlines and contacts

```python
import matplotlib.pyplot as plt

from mesh_particle_analysis import plot_particle_network

figure, axes, data = plot_particle_network(
    frame,
    n_mesh=256,
    contact_cutoff=2.09,
    alpha=0.7,
    show_labels=True,
    view_padding=0.04,
)

axes.set_title("Final-frame particle contact network")
plt.show()
```

The returned `data` object contains:

```python
print(data.centers)
print(data.outlines)
print(list(data.graph.edges))
print(dict(data.graph.degree))
```

The plot automatically expands its limits to display complete outlines for
particles that cross a periodic boundary. Set `show_full_outlines=False` to
restrict the view to the primary simulation box.

### Calculate shape metrics

```python
from mesh_particle_analysis import particle_shape_metrics

metrics = particle_shape_metrics(
    frame,
    n_mesh=256,
    alpha=0.7,
)

for index, shape in enumerate(metrics):
    print(
        f"Mesh {index}: "
        f"area={shape.area:.3f}, "
        f"perimeter={shape.perimeter:.3f}, "
        f"IQ={shape.iq:.3f}"
    )
```

### Calculate packing efficiency

```python
from mesh_particle_analysis import projected_packing_efficiency

efficiency = projected_packing_efficiency(
    frame,
    n_mesh=256,
    alpha=0.7,
)

print("Projected packing efficiency:", efficiency)
```

### Build a contact network

```python
from mesh_particle_analysis import contact_degrees, contact_graph

graph = contact_graph(
    frame,
    n_mesh=256,
    cutoff=2.09,
)

degrees = contact_degrees(graph)

print("Contact pairs:", list(graph.edges))
print("Contact degrees:", degrees)
```

Each graph edge also stores the minimum vertex distance and the number of
interacting vertex pairs:

```python
for mesh_i, mesh_j, edge_data in graph.edges(data=True):
    print(mesh_i, mesh_j, edge_data)
```

### Reconstruct pairwise contact forces

For a harmonic interaction with `sigma=2.09` and `k=100`:

```python
from mesh_particle_analysis import mesh_pair_contact_forces

pair_forces = mesh_pair_contact_forces(
    frame,
    n_mesh=256,
    sigma=2.09,
    k=100,
)

for pair, result in pair_forces.items():
    if result.vertex_pair_contacts > 0:
        print(
            pair,
            "vertex-pair contacts:", result.vertex_pair_contacts,
            "scalar load:", result.scalar_load,
            "force on first mesh:", result.force_on_i,
        )
```

`scalar_load` preserves the notebook calculation: it is the sum of interacting
vertex-pair force magnitudes. `force_on_i` is the vector sum of those forces on
the first mesh in the pair.

### Aggregate logged forces by mesh

If the GSD frame contains logged per-vertex pair forces:

```python
from mesh_particle_analysis import aggregate_mesh_forces

vertex_forces = frame.log["particles/md/pair/Table/forces"]

mesh_forces_xy = aggregate_mesh_forces(
    vertex_forces,
    n_mesh=256,
    dimensions=2,
)

print(mesh_forces_xy)
```

This sums the supplied force field. It should only be described as a total
physical force if the supplied log field includes every force contribution of
interest.

## Working with Signac

Signac belongs in study scripts and notebooks rather than the core package:

```python
import gsd.hoomd
import signac

from mesh_particle_analysis import particle_shape_metrics

project = signac.get_project()

for job in project.find_jobs({"N_part": 13, "k_area_scale": 125}):
    trajectory = gsd.hoomd.open(job.fn("trajectory.gsd"), mode="r")
    frame = trajectory[-1]

    metrics = particle_shape_metrics(
        frame,
        n_mesh=job.sp.N_mesh,
        n_particles=job.sp.N_part,
        alpha=0.7,
    )
```

This keeps job discovery and statepoint filtering separate from reusable
scientific calculations.

## Running the tests

Activate the environment in which the package and pytest are installed, then
run from the repository root:

```bash
python -m pytest
```

The tests use small deterministic synthetic frames. Full research trajectories
are not required.

Current tests cover:

- Consecutive mesh extraction.
- Multiple particle counts.
- Image-based periodic unwrapping.
- Known projected area, perimeter, and IQ.
- Vertex-centroid mesh centers.
- No-contact and known-contact cases.
- Contact across a periodic boundary.
- Contact graph construction and degrees.
- Known harmonic pair forces.
- Noncontacting force pairs.
- Aggregation of vertex forces by mesh.
- Visualization output and boundary-crossing outlines.

## Important assumptions and limitations

- Mesh vertices must be stored consecutively and in a known order.
- `n_mesh` must match the number of vertices belonging to each particle.
- Alpha shapes and IQ are calculated after projection onto xy.
- A fixed alpha value, such as `0.7`, has units relative to the coordinate
  scale and is not automatically scale-independent.
- Contact detection uses full three-dimensional distances, not projected xy
  distances.
- Contacts and force reconstruction use periodic minimum-image displacements.
- Mesh centers rely on correct GSD image vectors for unwrapping.
- The current plotting viewport supports orthorhombic simulation boxes. Contact
  calculations use freud wrapping and can support triclinic boxes.
- Force reconstruction assumes the harmonic interaction implemented in the
  original simulations.

## Files that should not be committed

Full GSD trajectories should remain outside Git or be ignored:

```text
*.gsd
```

Do not commit generated or machine-specific files:

```text
.DS_Store
.pytest_cache/
.vscode/
__pycache__/
*.pyc
dist/
*.egg-info/
```

Before committing, inspect the staged files:

```bash
git status --short
git diff --cached --name-only
```

Package source, tests, examples, `pyproject.toml`, `README.md`, `LICENSE`, and
`.gitignore` should be committed. Full trajectories, local test notebooks,
editor settings, and generated caches should not be committed.
