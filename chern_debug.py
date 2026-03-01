#!/usr/bin/env python3
"""
chern_debug.py — Debug Chern number calculation
=================================================
Strategy:
1. Test Fukui method on KNOWN model: Haldane honeycomb (C = ±1)
2. Test on trivial model (C = 0)
3. If honeycomb works, the bug is in Kagome BZ / Hamiltonian
4. If honeycomb fails, the bug is in Fukui implementation

Author: D. J. Edwards (2026)
"""

import numpy as np
from numpy import linalg as LA

# ═══════════════════════════════════════════════
# HALDANE HONEYCOMB (known: C = ±1 for |m| < 3√3 t2 sin(φ))
# ═══════════════════════════════════════════════

def haldane_H(kx, ky, t1=1.0, t2=0.1, phi=np.pi/2, M=0.0):
    """
    Standard Haldane model on honeycomb.
    2-band model: sublattices A, B.
    
    NN vectors: δ1=(0,1/√3), δ2=(1/2,-1/(2√3)), δ3=(-1/2,-1/(2√3))
    NNN vectors: ±a1, ±a2, ±(a1-a2)
    """
    # Honeycomb lattice vectors
    a1_h = np.array([1.0, 0.0])
    a2_h = np.array([0.5, np.sqrt(3)/2])
    
    k = np.array([kx, ky])
    
    # NN: A→B displacements
    d1 = np.array([0.0, 1/np.sqrt(3)])
    d2 = np.array([0.5, -1/(2*np.sqrt(3))])
    d3 = np.array([-0.5, -1/(2*np.sqrt(3))])
    
    f_k = sum(np.exp(1j * k.dot(d)) for d in [d1, d2, d3])
    
    # NNN: same sublattice, vectors ±a1, ±a2, ±(a1-a2)
    # Phase: +φ for A sublattice, -φ for B (Haldane convention)
    nnn_vecs = [a1_h, a2_h, a2_h - a1_h]
    g_k = sum(2 * np.cos(k.dot(v) + phi) for v in nnn_vecs)
    g_k_minus = sum(2 * np.cos(k.dot(v) - phi) for v in nnn_vecs)
    
    H = np.zeros((2, 2), dtype=complex)
    H[0, 0] = M + t2 * g_k         # A sublattice
    H[1, 1] = -M + t2 * g_k_minus  # B sublattice  
    H[0, 1] = t1 * f_k
    H[1, 0] = H[0, 1].conj()
    
    return H


def fukui_chern(H_func, b1, b2, nk=60, band_idx=0, n_bands=2):
    """
    Generic Fukui-Hatsugai-Suzuki Chern number.
    k-grid: k = (i1/nk)*b1 + (i2/nk)*b2, i1,i2 = 0..nk-1
    """
    # Precompute eigenstates
    states = np.zeros((nk, nk, n_bands, n_bands), dtype=complex)
    for i1 in range(nk):
        for i2 in range(nk):
            k = (i1 / nk) * b1 + (i2 / nk) * b2
            H = H_func(k[0], k[1])
            _, vecs = LA.eigh(H)
            states[i1, i2] = vecs
    
    total = 0.0
    for i1 in range(nk):
        for i2 in range(nk):
            j1 = (i1 + 1) % nk
            j2 = (i2 + 1) % nk
            
            u = states[:, :, :, band_idx]
            U1 = np.vdot(u[i1, i2], u[j1, i2])
            U2 = np.vdot(u[j1, i2], u[j1, j2])
            U3 = np.vdot(u[j1, j2], u[i1, j2])
            U4 = np.vdot(u[i1, j2], u[i1, i2])
            
            total += np.angle(U1 * U2 * U3 * U4)
    
    return total / (2 * np.pi)


def fukui_chern_v2(H_func, b1, b2, nk=60, band_idx=0, n_bands=2):
    """
    Alternative: use np.angle of each link separately (more numerically stable).
    F_12 = Im[ln(U1 * U2 * U3 * U4)]
         = arg(U1) + arg(U2) + arg(U3) + arg(U4)   [mod 2π done per link]
    """
    states = np.zeros((nk, nk, n_bands, n_bands), dtype=complex)
    for i1 in range(nk):
        for i2 in range(nk):
            k = (i1 / nk) * b1 + (i2 / nk) * b2
            _, vecs = LA.eigh(H)
            states[i1, i2] = vecs
    
    total = 0.0
    for i1 in range(nk):
        for i2 in range(nk):
            j1 = (i1 + 1) % nk
            j2 = (i2 + 1) % nk
            u = states[:, :, :, band_idx]
            
            # Individual link phases
            p1 = np.angle(np.vdot(u[i1,i2], u[j1,i2]))
            p2 = np.angle(np.vdot(u[j1,i2], u[j1,j2]))
            p3 = np.angle(np.vdot(u[j1,j2], u[i1,j2]))
            p4 = np.angle(np.vdot(u[i1,j2], u[i1,i2]))
            
            total += p1 + p2 + p3 + p4
    
    return total / (2 * np.pi)


print("="*60)
print("TEST 1: Haldane honeycomb — KNOWN C = +1 or -1")
print("="*60)

# Honeycomb reciprocal lattice
a1_h = np.array([1.0, 0.0])
a2_h = np.array([0.5, np.sqrt(3)/2])

area = abs(np.cross(a1_h, a2_h))
b1_h = 2*np.pi / area * np.array([a2_h[1], -a2_h[0]])
b2_h = 2*np.pi / area * np.array([-a1_h[1], a1_h[0]])

# Verify: a_i · b_j = 2π δ_ij
print(f"  a1·b1 = {a1_h.dot(b1_h):.4f} (expect {2*np.pi:.4f})")
print(f"  a1·b2 = {a1_h.dot(b2_h):.4f} (expect 0)")
print(f"  a2·b1 = {a2_h.dot(b1_h):.4f} (expect 0)")
print(f"  a2·b2 = {a2_h.dot(b2_h):.4f} (expect {2*np.pi:.4f})")

# Test: t2=0.1, phi=π/2, M=0 → topological phase, C = ±1
# Condition: |M| < 3√3 t2 sin(φ) = 3√3 × 0.1 × 1 ≈ 0.52
H_haldane = lambda kx, ky: haldane_H(kx, ky, t1=1.0, t2=0.1, phi=np.pi/2, M=0.0)

for nk in [20, 40, 60, 80]:
    c0 = fukui_chern(H_haldane, b1_h, b2_h, nk=nk, band_idx=0, n_bands=2)
    c1 = fukui_chern(H_haldane, b1_h, b2_h, nk=nk, band_idx=1, n_bands=2)
    print(f"  nk={nk:3d}: C0={c0:+.4f}  C1={c1:+.4f}  Σ={c0+c1:+.4f}")

print()

# Test: trivial phase (M large) → C = 0
print("TEST 2: Haldane honeycomb — trivial phase (M=2.0) → C = 0")
H_triv = lambda kx, ky: haldane_H(kx, ky, t1=1.0, t2=0.1, phi=np.pi/2, M=2.0)
c0 = fukui_chern(H_triv, b1_h, b2_h, nk=60, band_idx=0, n_bands=2)
c1 = fukui_chern(H_triv, b1_h, b2_h, nk=60, band_idx=1, n_bands=2)
print(f"  C0={c0:+.4f}  C1={c1:+.4f}  Σ={c0+c1:+.4f}")

print()

# Test: TRS preserved (phi=0) → C = 0
print("TEST 3: Haldane honeycomb — TRS (phi=0) → C = 0")
H_trs = lambda kx, ky: haldane_H(kx, ky, t1=1.0, t2=0.1, phi=0.0, M=0.0)
c0 = fukui_chern(H_trs, b1_h, b2_h, nk=60, band_idx=0, n_bands=2)
c1 = fukui_chern(H_trs, b1_h, b2_h, nk=60, band_idx=1, n_bands=2)
print(f"  C0={c0:+.4f}  C1={c1:+.4f}  Σ={c0+c1:+.4f}")

# ═══════════════════════════════════════════════
# KAGOME — check BZ and Chern
# ═══════════════════════════════════════════════
print(f"\n{'='*60}")
print("TEST 4: Kagome BZ verification")
print("="*60)

a1_k = np.array([1.0, 0.0])
a2_k = np.array([0.5, np.sqrt(3)/2])

area_k = abs(np.cross(a1_k, a2_k))
b1_k = 2*np.pi / area_k * np.array([a2_k[1], -a2_k[0]])
b2_k = 2*np.pi / area_k * np.array([-a1_k[1], a1_k[0]])

print(f"  a1·b1 = {a1_k.dot(b1_k):.4f} (expect {2*np.pi:.4f})")
print(f"  a1·b2 = {a1_k.dot(b1_k):.4f} (expect 0... wait)")

# Proper check
print(f"  a1·b1 = {a1_k.dot(b1_k):.6f}")
print(f"  a1·b2 = {a1_k.dot(b2_k):.6f}")
print(f"  a2·b1 = {a2_k.dot(b1_k):.6f}")
print(f"  a2·b2 = {a2_k.dot(b2_k):.6f}")

# Compare with hardcoded values in main script
b1_old = 2*np.pi * np.array([1.0, -1/np.sqrt(3)])
b2_old = 2*np.pi * np.array([0.0, 2/np.sqrt(3)])

print(f"\n  b1 (computed): {b1_k}")
print(f"  b1 (hardcoded): {b1_old}")
print(f"  b2 (computed): {b2_k}")
print(f"  b2 (hardcoded): {b2_old}")
print(f"  b1 match: {np.allclose(b1_k, b1_old)}")
print(f"  b2 match: {np.allclose(b1_k, b1_old)}")

# Test: Kagome with TRS (t2=0, no flux) → ALL Chern = 0
print(f"\n{'='*60}")
print("TEST 5: Kagome TRS (t2=0, no flux) → C = 0,0,0")
print("="*60)

from kagome_haldane import H_kagome

H_k_trs = lambda kx, ky: H_kagome(kx, ky, t1=1.0, t2=0.0, phi=0.0, Phi_nn=0.0, m_sub=0.0)

# But bands 0,1 touch! Only band 2 is isolated.
# So only C2 should be well-defined.
for nk in [20, 40, 60]:
    cs = []
    for b in range(3):
        c = fukui_chern(H_k_trs, b1_k, b2_k, nk=nk, band_idx=b, n_bands=3)
        cs.append(c)
    print(f"  nk={nk}: C = [{cs[0]:+.4f}, {cs[1]:+.4f}, {cs[2]:+.4f}]  Σ={sum(cs):+.4f}")
    print(f"         (note: bands 0,1 touch at Γ → C0,C1 may be ill-defined, but C2 and Σ should be 0)")

# Test 6: Kagome with sublattice mass only (TRS preserved → C must be 0)
print(f"\n{'='*60}")
print("TEST 6: Kagome + sublattice mass (TRS → C = 0)")
print("="*60)

H_k_mass = lambda kx, ky: H_kagome(kx, ky, t1=1.0, t2=0.0, phi=0.0, Phi_nn=0.0, m_sub=0.2)

# Check gaps first
from kagome_haldane import band_info
p_mass = {'t1': 1.0, 't2': 0.0, 'phi': 0.0, 'Phi_nn': 0.0, 'm_sub': 0.2}
W, gaps, _, _ = band_info(p_mass, nk=80)
print(f"  Gaps: {gaps[0]:.4f}, {gaps[1]:.4f}")
print(f"  Widths: {W}")

for nk in [40, 60, 80]:
    cs = []
    for b in range(3):
        c = fukui_chern(H_k_mass, b1_k, b2_k, nk=nk, band_idx=b, n_bands=3)
        cs.append(c)
    print(f"  nk={nk}: C = [{cs[0]:+.4f}, {cs[1]:+.4f}, {cs[2]:+.4f}]  Σ={sum(cs):+.4f}")

# Test 7: Check Hamiltonian periodicity H(k + b1) = H(k)
print(f"\n{'='*60}")
print("TEST 7: Hamiltonian periodicity H(k+G) = H(k)")
print("="*60)

k_test = np.array([0.3, 0.7])
H0 = H_kagome(k_test[0], k_test[1], t1=1.0, t2=0.1, phi=0.5, Phi_nn=0.0, m_sub=0.0)
H1 = H_kagome(k_test[0]+b1_k[0], k_test[1]+b1_k[1], t1=1.0, t2=0.1, phi=0.5, Phi_nn=0.0, m_sub=0.0)
H2 = H_kagome(k_test[0]+b2_k[0], k_test[1]+b2_k[1], t1=1.0, t2=0.1, phi=0.5, Phi_nn=0.0, m_sub=0.0)

print(f"  ||H(k+b1) - H(k)|| = {LA.norm(H1 - H0):.2e}")
print(f"  ||H(k+b2) - H(k)|| = {LA.norm(H2 - H0):.2e}")

# With complex NN:
H0f = H_kagome(k_test[0], k_test[1], t1=1.0, t2=0.0, phi=0.0, Phi_nn=0.5, m_sub=0.0)
H1f = H_kagome(k_test[0]+b1_k[0], k_test[1]+b1_k[1], t1=1.0, t2=0.0, phi=0.0, Phi_nn=0.5, m_sub=0.0)
H2f = H_kagome(k_test[0]+b2_k[0], k_test[1]+b2_k[1], t1=1.0, t2=0.0, phi=0.0, Phi_nn=0.5, m_sub=0.0)

print(f"  ||H(k+b1) - H(k)|| (Φ_nn) = {LA.norm(H1f - H0f):.2e}")
print(f"  ||H(k+b2) - H(k)|| (Φ_nn) = {LA.norm(H2f - H0f):.2e}")

print(f"\n{'='*60}")
print("DONE")
print("="*60)
