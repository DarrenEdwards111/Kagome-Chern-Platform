# A Large-Gap Kagome Chern Platform for Room-Temperature Topological Quantum Architectures

Theoretical architecture for room-temperature topological quantum computation based on the intrinsic Kagome flat band with uniform magnetic flux.

## Key Results

| Criterion | Target | Result |
|-----------|--------|--------|
| C1: Flat band | W ≤ 5 meV | **W = 0** (intrinsic Kagome) |
| C2: Interactions | U/t ≥ 50 meV | **U/t = 9–17** |
| C3: Topological gap | ≥ 25 meV | **173 meV** (7× target) |
| C4: Topology | C ≠ 0 + edge modes | **C = −1**, 2 chiral edge modes |
| C5: Disorder | survives 2 meV | **96% retained at 30 meV** |

**Architecture:** 192 physical qubits, T_max = 363 K, X-Cube fracton code

## Scripts

| Script | What it does |
|--------|-------------|
| `kagome_uniform_flux.py` | Chern band calculation: C = −1, Δ = 173 meV |
| `kagome_edge_modes.py` | Ribbon calculation: chiral edge modes confirmed |
| `kagome_haldane.py` | Haldane NNN scan: C = 0 (negative result) |
| `kagome_chern2.py` | Non-Abelian composite Chern number |
| `kagome_chern3.py` | Comprehensive 64-parameter Chern scan |
| `chern_debug.py` | Convention I/II periodicity diagnostic |
| `kagome_intrinsic_gap.py` | Charge gap via ED (30 meV) |
| `clrc_model.py` | X-Cube fracton architecture (192 qubits, 363 K) |

## Requirements

- Python 3.8+
- NumPy, SciPy

All scripts run in under 30 seconds. No GPU required.

## Paper

- `kagome-chern-platform.tex` — LaTeX source
- `kagome-chern-platform.pdf` — Compiled paper

## Materials Candidates

- **Fe₃Sn₂** — Ferromagnetic Kagome metal, Dirac gap 30–60 meV, T_C ≈ 657 K
- **Mn₃Sn** — Non-collinear AFM, T_N = 430 K (above room temperature)
- **Co₃Sn₂S₂** — Magnetic Weyl semimetal, gap ~110 meV

## Author

D. J. Edwards

## License

Apache 2.0
