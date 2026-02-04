# Isaac Lab Stretch Project

![Stretch Robot Demo](Example.gif)

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
Create a virtual environment (optional but highly recommended). As a pre-requisite, `isaacsim` and `curobo` must be installed first. 
It's very important that the `Curobo` and the `isaacsim` sync,you can refer the installation guide from `Curobo`'s website.

2. **Clone the Project**:

   ```bash
   git clone [https://github.com/JCpennyChen/isaaclab_stretch.git](https://github.com/JCpennyChen/isaaclab_stretch.git)
   cd stretch
