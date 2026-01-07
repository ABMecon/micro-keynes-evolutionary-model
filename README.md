# micro-keynes-evolutionary-model
This repository provides a minimal computational prototype for the model developed in  Ichiro Takahashi, Microfoundations of Keynesian Economics, Springer.  The code is intended for transparency and reproducibility, not for production use.

# Microfoundations of Keynesian Economics – Computational Prototype

This repository provides a minimal computational prototype for the model developed in

Ichiro Takahashi  
*Microfoundations of Keynesian Economics:  
An Evolutionary Approach to Stabilizing an Unstable Economy*  
(Springer)

## Purpose

The purpose of this code is to enhance the transparency and reproducibility of the theoretical results presented in the book.  
It illustrates how the Markov process for the capital stock is constructed, simulated, and analyzed numerically.

The code is intentionally kept simple and modular.  
It is not intended to be production-ready software.

## Contents
-----

## Contents

- `transition_kernel.py`: constructs the discrete Markov transition kernel \(P(K' \mid K)\) on a user-supplied grid.
- `diagnostics.py`: lightweight diagnostics (row-sum checks) and plotting utilities for kernel sanity checks.
- `RUN_DIAGNOSTICS.py`: minimal entry point that builds \(P(K'\mid K)\) and produces diagnostic row-overlay plots.

## Usage

The code is written in Python and relies only on standard scientific libraries (NumPy, SciPy, Matplotlib).

Readers are encouraged to modify parameter values and experiment with the model in order to build intuition about the underlying dynamics.

## Disclaimer

This code is provided for educational and illustrative purposes only.  
No guarantee is given regarding numerical efficiency or robustness beyond the scope of the book.

## License

The code is released for academic and non-commercial use.

