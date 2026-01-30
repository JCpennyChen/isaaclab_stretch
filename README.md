# Isaac Lab Stretch Project

## Overview

This repository contains the Isaac Lab extension for the Hello Robot Stretch.
It enables motion planning integration using CuRobo and provides a platform for Reinforcement Learning
tasks specific to the Stretch platform.

**Key Features:**

- `Isolation`: Work outside the core Isaac Lab repository, ensuring that your development efforts remain self-contained.
- `CuRobo Integration`: Seamless motion planning for the Stretch robot using CUDA-accelerated kernels.
- `Flexibility`: Designed to run both as a standalone script or as an Omniverse Extension.

**Keywords:** extension, stretch, isaaclab, curobo, robotics

## Installation

1. **Install Isaac Lab**: Follow the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). We recommend the Conda installation.

2. **Clone the Project**:

   ```bash
   git clone https://github.com/JCpennyChen/isaaclab_stretch.git
   cd stretch
   ```

3. **Install the Extension**:
   Using the python interpreter from your Isaac Lab environment:
   ```bash
   python -m pip install -e source/stretch
   ```

## Usage

### Verify Installation

List the available Stretch environments to ensure the task is registered correctly:

```bash
python scripts/list_envs.py
```
