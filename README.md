## Overview

This repository contains implementations of a variety of Direction of Arrival (DoA) estimation algorithms, along with a flexible simulation framework for generating signals, array configurations, and noise environments to test and compare their performance.

The project provides:

- **Modular implementations of classic and modern DoA algorithms**  
  (e.g., MUSIC, Minimum Variance Distortionless Response, beamformers, and related subspace-based methods)

- **End-to-end simulation tools**  
  for generating multi-sensor array data, steering vectors, propagation models, and measurement noise

- **Experiment and evaluation scripts**  
  to benchmark algorithms under different SNRs, array geometries, and signal conditions

This structure allows rapid experimentation, algorithm development, and reproducible analysis for array signal processing and DoA research.

## Environment Setup

```bash
# Create conda environment
conda create -n df_env python=3.12

# Activate environment
conda activate df_env

# Install Python dependencies
pip install -r requirements.txt

# Install local library in editable mode
pip install -e lib/