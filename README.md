# ACE: Adaptive Computation and Eye-movements

> An Active Perception and Cognitive Modeling Architecture in Julia & JAX.

ACE is a computational cognitive model designed to simulate how human observers allocate visual attention, plan eye movements (saccades and fixations), and perceive dynamic physical environments under resource constraints.

---

## Table of Contents

- [Overview & Scientific Background](#overview--scientific-background)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Repository Structure](#repository-structure)
- [Prerequisites & Installation](#prerequisites--installation)
- [Quickstart & Running Notebooks](#quickstart--running-notebooks)
- [Core Concepts for New Contributors](#core-concepts-for-new-contributors)
- [Citation](#citation)
- [Contributing](#contributing)

---

## Overview & Scientific Background

The human visual system does not process the entire visual field in uniform high resolution. Instead:

1. **Foveated Vision:** Photoreceptor density is concentrated in the central 1–3% of the visual field (the fovea), dropping off steeply into the periphery.
2. **Active Information Sampling:** To understand the world, the brain must make saccadic eye movements to actively sample task-relevant regions.
3. **Probabilistic World Models:** Perception operates as approximate Bayesian inference (Analysis-by-Synthesis), combining internal generative physical models with noisy sensory inputs.

ACE models this perception-action loop using:

- **Probabilistic Programming (Gen.jl):** Formulates state estimation and particle filtering over physical object states.
- **Hardware-Accelerated Forward Rendering (JAX):** Rapidly evaluates receptive field statistics (means and variances across foveal and peripheral channels).
- **Resource-Rational Attention & Planning:** Determines where to allocate gaze next based on information gain and task goals.

---

## Key Features

- **Forward-Rendering Receptive Field Model:** JAX-accelerated differentiable receptive field sensor array with customizable foveal concentration and eccentricity scaling.
- **Probabilistic World Model:** Physics-based generative simulations of 2D dynamic scenes with kinematic and collision dynamics.
- **Modular Mental Modules:** Decoupled submodules for **Perception**, **Attention**, **Memory**, and **Planning**.
- **Interactive Visualizations:** Built-in 2D rendering engine via Luxor.jl and reactive Pluto.jl notebooks.

---

## System Architecture

```
                 +-------------------------------+
                 |        Physical World         |
                 |  (Objects, Boundaries, Motion)|
                 +---------------+---------------+
                                 | (Photons / Ground Truth)
                                 v
+-----------------------------------------------------------------+
|                         ACE Agent                               |
|                                                                 |
|   +---------------------------------------------------------+   |
|   | Sensory Engine (JAX / PythonCall)                       |   |
|   | - Foveated Receptive Fields                             |   |
|   | - Gaussian-weighted color statistics                    |   |
|   +----------------------------+----------------------------+   |
|                                | (Observed RGB μ & σ)           |
|                                v                                |
|   +---------------------------------------------------------+   |
|   | Perception Module (Gen.jl Particle Filter)              |   |
|   | - Infer 2D Object Positions, Velocities & Classes       |   |
|   +----------------------------+----------------------------+   |
|                                | (Belief State)                 |
|                                v                                |
|   +---------------------------------------------------------+   |
|   | Attention & Planning Modules                            |   |
|   | - Adaptive Computation Allocation                       |   |
|   | - Fixation Planning (Next Saccade Target)               |   |
|   +----------------------------+----------------------------+   |
|                                |                                |
+--------------------------------+--------------------------------+
                                 | (Fixation Update: [x, y])
                                 v
                 +-------------------------------+
                 |        Camera / Gaze          |
                 +-------------------------------+
```

---

## Repository Structure

```text
ACE/
├── CondaPkg.toml            # Python dependencies managed via CondaPkg / PythonCall
├── Project.toml             # Julia project configuration & dependencies
├── notebooks/
│   └── perception.jl        # Interactive Pluto notebook for perception & RF debugging
├── python/
│   └── jax_script.py        # JAX kernels for receptive field geometry & forward rendering
├── scripts/
│   ├── test_perception.jl   # CLI integration test for perception pipeline
│   └── world_model.jl       # World model simulation and trial generator
├── src/
│   ├── ACE.jl               # Package root entry point
│   ├── python.jl            # Julia-Python interop initialization
│   ├── agent/               # Cognitive agent modules
│   │   ├── agent.jl         # Agent loop integration
│   │   ├── attention/       # Attention mechanisms & visual routing
│   │   ├── inference/       # State inference & particle filter chains
│   │   ├── memory/          # Short-term and episodic memory buffers
│   │   ├── perception/      # Sensory observation models
│   │   └── planning/        # Goal designation & saccade trajectory planning
│   ├── world_model/         # Generative scene & physics simulator
│   │   ├── gen.jl           # Gen.jl generative function definitions
│   │   ├── graphics.jl      # RFGraphics interface & buffer synchronization
│   │   ├── motion.jl        # Inertial physics and bounce mechanics
│   │   ├── visuals.jl       # Luxor.jl paint! and drawing routines
│   │   └── world_model.jl   # World state container definitions
│   └── utils/               # Math, geometry, fill utilities, and data structures
```

---

## Prerequisites & Installation

### 1. Prerequisites
- **Julia (>= 1.9)**: [Download Julia](https://julialang.org/downloads/)
- **Python (>= 3.9)** with CUDA (if using GPU acceleration) or CPU.

### 2. Setup

Clone the repository:
```bash
git clone --recurse-submodules https://github.com/CNCLgithub/ACE.git
cd ACE
```

Start Julia and instantiate the environment (this will automatically configure both Julia dependencies and Python packages via `CondaPkg`):
```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

---

## Quickstart & Running Notebooks

### Interactive Pluto Notebook

The easiest way to experiment with receptive field properties and visual perception is via Pluto.jl:

```julia
using Pluto
Pluto.run(notebook="notebooks/perception.jl")
```

Once open in your browser, you can:

- Interactively drag fixation coordinates $(x, y)$.
- Inspect the four-panel rendering:
  1. **Ground Truth Scene:** Exact object states.
  2. **Receptive Field Mean ($\mu$):** Color predictions per receptive field ring.
  3. **Receptive Field Variance ($\sigma^2$):** Predicted uncertainty/variance.
  4. **Perceptual Belief:** Particle filter inferences overlaid in real-time.

### Running Headless Tests

To run the perceptual test harness from the command line:
```bash
julia --project=. scripts/test_perception.jl
```

---

## Core Concepts for New Contributors

### 1. Multiple Dispatch & `paint!` API
All visual elements implement the `paint!` interface in `src/world_model/visuals.jl`:
```julia
paint!(object::WorldObject)                  # Paints a scene entity
paint!(rf::AbstractVector, color::S3V)       # Paints a receptive field disc
paint_state(graphics::RFGraphics, scene)     # Paints full RF layer
```

### 2. Zero-Copy Julia-JAX Bridge
To run high-throughput simulations:

- Julia allocates pre-pinned host arrays (`scene_buf`, `fixation_buf` in `RFGraphics`).
- JAX wraps these arrays using Python buffer protocol views (`np.frombuffer`).
- Forward calculations execute as JIT-compiled kernels (`jax_script.py`).

---

## Citation

```bibtex
@article{belledonne_ace,
  author   = "Belledonne, Mario, Dursun, Oktay, and Yildirim, Ilker",
  title    = "ACE: Adaptive Computation guided Eye-movements in Active Perception",
  journal  = "TBD",
  year     = 2026,
}
```

---

## Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/new-analysis`.
3. Commit your changes: `git commit -m "Add feature"`.
4. Push to your fork and submit a Pull Request.
