#!/usr/bin/env python3
"""
kagome_chern3.py — Non-Abelian composite Chern for Kagome
===========================================================
Computes Chern of the occupied 2-band subspace {u0, u1} below the
gap between bands 1 and 2, using the gauge-invariant determinant method.

Method (Fukui-Hatsugai-Suzuki, non-Abelian):
  M_μ(k) = Ψ(k)† Ψ(k+Δ_μ)   [2×2 overlap matrix]
  U_μ(k) = det(M_μ) / |det(M_μ)|   [normalized U(1) link]
  F_12(k) = arg(U_1(k) · U_2(k+Δ1) · U_1(k+Δ2)^{-1} · U_2(k)^{-1})
  C = (1/2π) Σ_k F_12(k)

Author: D. J. Edwards (2026)
"""

import numpy as np
from numpy import linalg as LA
import json, time

a1 = np.array([1.0, 0.0])
a2 = np.array([0.5, np.sqrt(3)/2])
area = a1[0]*a2[1] - a1[1]*a2[0]
b1 = (2*np.pi / area) * np.array([a2[1], -a2[0]])
b2 = (2*np.pi / area) * np.array([-a1[1], a1[0]])


def H_kagome(kx, ky, t1=1.0, t2=0.0, phi=0.0):
    """Convention I Kagome Hamiltonian (periodic)."""
    k = np.array([kx, ky])
    H = np.zeros((3, 3), dtype=complex)
    
    H[0, 1] = t1 * (1 + np.exp(-1j * k.dot(a1)))
    H[0, 2] = t1 * (1 + np.exp(-1j * k.dot(a2)))
    H[1, 2] = t1 * (1 + np.exp(1j * k.dot(a1 - a2)))
    H[1, 0] = H[0, 1].conj()
    H[2, 0] = H[0, 2].conj()
    H[2, 1] = H[1, 2].conj()
    
    if abs(t2) > 1e-12:
        da = a2 - a1
        H[0, 0] += t2 * 2 * (np.cos(k.dot(a1) + phi) + np.cos(k.dot(a2) - phi))
        H[1, 1] += t2 * 2 * (np.cos(k.dot(a1) - phi) + np.cos(k.dot(da) + phi))
        H[2, 2] += t2 * 2 * (np.cos(k.dot(a2) + phi) + np.cos(k.dot(da) - phi))
    
    return H


def composite_chern(H_func, nk=80, n_occ=2, n_bands=3):
    """
    Non-Abelian Chern of the occupied n_occ-band subspace.
    
    Returns: C (float), direct_gap (float), details (dict)
    """
    # Precompute eigenstates
    states = np.zeros((nk, nk, n_bands, n_bands), dtype=complex)
    energies = np.zeros((nk, nk, n_bands))
    
    for i1 in range(nk):
        for i2 in range(nk):
            k = (i1/nk)*b1 + (i2/nk)*b2
            evals, vecs = LA.eigh(H_func(k[0], k[1]))
            states[i1, i2] = vecs
            energies[i1, i2] = evals
    
    # Direct gap: min_k (E_{n_occ} - E_{n_occ-1})
    direct_gap = np.min(energies[:, :, n_occ] - energies[:, :, n_occ - 1])
    
    # Compute plaquette Berry phases
    total = 0.0
    det_min = np.inf  # track smallest |det| for diagnostics
    
    for i1 in range(nk):
        for i2 in range(nk):
            j1 = (i1 + 1) % nk
            j2 = (i2 + 1) % nk
            
            # Occupied subspace Ψ at 4 corners (n_bands × n_occ)
            P00 = states[i1, i2, :, :n_occ]
            P10 = states[j1, i2, :, :n_occ]
            P11 = states[j1, j2, :, :n_occ]
            P01 = states[i1, j2, :, :n_occ]
            
            # Overlap matrices (n_occ × n_occ)
            M1 = P00.conj().T @ P10   # link along b1
            M2 = P10.conj().T @ P11   # link along b2
            M3 = P11.conj().T @ P01   # link along -b1
            M4 = P01.conj().T @ P00   # link along -b2
            
            # Determinants
            d1 = LA.det(M1)
            d2 = LA.det(M2)
            d3 = LA.det(M3)
            d4 = LA.det(M4)
            
            for d in [d1, d2, d3, d4]:
                det_min = min(det_min, abs(d))
            
            # Normalized links
            U1 = d1 / abs(d1) if abs(d1) > 1e-15 else 1.0
            U2 = d2 / abs(d2) if abs(d2) > 1e-15 else 1.0
            U3 = d3 / abs(d3) if abs(d3) > 1e-15 else 1.0
            U4 = d4 / abs(d4) if abs(d4) > 1e-15 else 1.0
            
            # Plaquette = U1 · U2 · U3 · U4
            # (going around: (i1,i2)→(j1,i2)→(j1,j2)→(i1,j2)→(i1,i2))
            plaq = U1 * U2 * U3 * U4
            total += np.angle(plaq)
    
    C = total / (2 * np.pi)
    return C, direct_gap, {'det_min': det_min}


def single_chern(H_func, nk=80, band_idx=0, n_bands=3):
    """Standard single-band Chern for isolated bands."""
    states = np.zeros((nk, nk, n_bands, n_bands), dtype=complex)
    for i1 in range(nk):
        for i2 in range(nk):
            k = (i1/nk)*b1 + (i2/nk)*b2
            _, vecs = LA.eigh(H_func(k[0], k[1]))
            states[i1, i2] = vecs
    
    total = 0.0
    for i1 in range(nk):
        for i2 in range(nk):
            j1 = (i1+1)%nk; j2 = (i2+1)%nk
            u = states[:,:,:,band_idx]
            U = (np.vdot(u[i1,i2], u[j1,i2]) * np.vdot(u[j1,i2], u[j1,j2]) *
                 np.vdot(u[j1,j2], u[i1,j2]) * np.vdot(u[i1,j2], u[i1,i2]))
            total += np.angle(U)
    return total / (2*np.pi)


# ═══════════════════════════════════════════════════════
# Haldane honeycomb for sanity check
# ═══════════════════════════════════════════════════════
a1_h = np.array([1.0, 0.0])
a2_h = np.array([0.5, np.sqrt(3)/2])
area_h = a1_h[0]*a2_h[1] - a1_h[1]*a2_h[0]
b1_h = (2*np.pi/area_h) * np.array([a2_h[1], -a2_h[0]])
b2_h = (2*np.pi/area_h) * np.array([-a1_h[1], a1_h[0]])

def H_haldane(kx, ky, t1=1.0, t2=0.1, phi=np.pi/2, M=0.0):
    k = np.array([kx, ky])
    d1 = np.array([0.0, 1/np.sqrt(3)])
    d2 = np.array([0.5, -1/(2*np.sqrt(3))])
    d3 = np.array([-0.5, -1/(2*np.sqrt(3))])
    f_k = sum(np.exp(1j*k.dot(d)) for d in [d1,d2,d3])
    nnn = [a1_h, a2_h, a2_h - a1_h]
    H = np.zeros((2,2), dtype=complex)
    H[0,0] = M + t2*sum(2*np.cos(k.dot(v)+phi) for v in nnn)
    H[1,1] = -M + t2*sum(2*np.cos(k.dot(v)-phi) for v in nnn)
    H[0,1] = t1*f_k
    H[1,0] = H[0,1].conj()
    return H


def composite_chern_generic(H_func, b1g, b2g, nk=80, n_occ=1, n_bands=2):
    """Generic composite Chern with specified BZ vectors."""
    states = np.zeros((nk, nk, n_bands, n_bands), dtype=complex)
    energies = np.zeros((nk, nk, n_bands))
    
    for i1 in range(nk):
        for i2 in range(nk):
            k = (i1/nk)*b1g + (i2/nk)*b2g
            evals, vecs = LA.eigh(H_func(k[0], k[1]))
            states[i1, i2] = vecs
            energies[i1, i2] = evals
    
    direct_gap = np.min(energies[:,:,n_occ] - energies[:,:,n_occ-1])
    
    total = 0.0
    for i1 in range(nk):
        for i2 in range(nk):
            j1 = (i1+1)%nk; j2 = (i2+1)%nk
            P00 = states[i1,i2,:,:n_occ]
            P10 = states[j1,i2,:,:n_occ]
            P11 = states[j1,j2,:,:n_occ]
            P01 = states[i1,j2,:,:n_occ]
            
            d1 = LA.det(P00.conj().T @ P10)
            d2 = LA.det(P10.conj().T @ P11)
            d3 = LA.det(P11.conj().T @ P01)
            d4 = LA.det(P01.conj().T @ P00)
            
            U1 = d1/abs(d1) if abs(d1)>1e-15 else 1.0
            U2 = d2/abs(d2) if abs(d2)>1e-15 else 1.0
            U3 = d3/abs(d3) if abs(d3)>1e-15 else 1.0
            U4 = d4/abs(d4) if abs(d4)>1e-15 else 1.0
            
            total += np.angle(U1*U2*U3*U4)
    
    return total/(2*np.pi), direct_gap


def main():
    t0 = time.time()
    results = {}
    
    # ═══ SANITY 1: Haldane honeycomb (n_occ=1, reduces to Abelian) ═══
    print("="*60)
    print("SANITY 1: Haldane honeycomb — composite (n_occ=1) = single-band")
    print("="*60)
    
    H_hal = lambda kx,ky: H_haldane(kx, ky, t1=1.0, t2=0.1, phi=np.pi/2)
    C_comp, gap = composite_chern_generic(H_hal, b1_h, b2_h, nk=60, n_occ=1, n_bands=2)
    print(f"  Composite C(occ={{0}}) = {C_comp:+.4f}  (expect -1)  gap={gap:.4f}")
    
    H_triv = lambda kx,ky: H_haldane(kx, ky, t1=1.0, t2=0.1, phi=np.pi/2, M=2.0)
    C_triv, gap_t = composite_chern_generic(H_triv, b1_h, b2_h, nk=60, n_occ=1, n_bands=2)
    print(f"  Trivial  C(occ={{0}}) = {C_triv:+.4f}  (expect 0)   gap={gap_t:.4f}")
    
    # ═══ SANITY 2: Kagome TRS → C = 0 ═══
    print(f"\n{'='*60}")
    print("SANITY 2: Kagome TRS (t2=0) — composite C must = 0")
    print("="*60)
    
    H_trs = lambda kx,ky: H_kagome(kx, ky, t1=1.0, t2=0.0)
    C_trs, gap_trs, _ = composite_chern(H_trs, nk=60, n_occ=2)
    print(f"  C(occ={{0,1}}) = {C_trs:+.4f}  (expect 0)  direct_gap(1→2)={gap_trs:.4f}")
    
    # ═══ MAIN: NNN Haldane on Kagome — composite Chern ═══
    print(f"\n{'='*60}")
    print("MAIN: Kagome NNN Haldane — composite Chern C({0,1})")
    print("="*60)
    print(f"  {'t2':>4s} {'φ/π':>5s} {'gap_dir':>8s} {'C_comp':>8s} {'C_2':>8s} {'Σ':>8s} {'det_min':>8s}")
    print(f"  {'-'*4} {'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    
    main_results = []
    for t2 in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
        for phi_f in [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75]:
            p_phi = phi_f * np.pi
            H_f = lambda kx, ky, _t2=t2, _p=p_phi: H_kagome(kx, ky, t1=1.0, t2=_t2, phi=_p)
            
            C_comp, gap_dir, info = composite_chern(H_f, nk=60, n_occ=2)
            
            # Only compute C_2 if gap is positive
            if gap_dir > 0.005:
                C_2 = single_chern(H_f, nk=60, band_idx=2)
                sigma = C_comp + C_2
                
                marker = ""
                if abs(round(C_comp)) > 0:
                    marker = " ★★★ TOPOLOGICAL"
                
                print(f"  {t2:4.2f} {phi_f:5.2f} {gap_dir:8.4f} {C_comp:+8.4f} {C_2:+8.4f} "
                      f"{sigma:+8.4f} {info['det_min']:8.4f}{marker}")
                
                main_results.append({
                    't2': t2, 'phi_pi': phi_f, 'gap_dir': round(gap_dir, 5),
                    'C_comp': round(C_comp, 4), 'C_2': round(C_2, 4),
                    'det_min': round(info['det_min'], 6)
                })
    
    results['main'] = main_results
    
    # ═══ Convergence check on best candidate ═══
    if main_results:
        # Find one with largest gap
        best = max(main_results, key=lambda e: e['gap_dir'])
        
        if abs(round(best['C_comp'])) > 0:
            print(f"\n{'='*60}")
            print(f"CONVERGENCE: t2={best['t2']}, φ/π={best['phi_pi']}")
            print("="*60)
            
            H_b = lambda kx,ky: H_kagome(kx, ky, t1=1.0, t2=best['t2'], phi=best['phi_pi']*np.pi)
            for nk in [40, 60, 80, 100, 120]:
                C, g, _ = composite_chern(H_b, nk=nk, n_occ=2)
                print(f"  nk={nk:3d}: C={C:+.6f}  gap={g:.4f}")
        else:
            print(f"\n  Best gap: t2={best['t2']}, φ/π={best['phi_pi']}, gap={best['gap_dir']:.4f}")
            print(f"  C_comp = {best['C_comp']:+.4f} — still trivial")
            
            # Do convergence on best-gap anyway
            print(f"\n{'='*60}")
            print(f"CONVERGENCE CHECK (best gap): t2={best['t2']}, φ/π={best['phi_pi']}")
            print("="*60)
            H_b = lambda kx,ky,_t=best['t2'],_p=best['phi_pi']*np.pi: H_kagome(kx,ky,t1=1.0,t2=_t,phi=_p)
            for nk in [40, 60, 80, 100]:
                C, g, info = composite_chern(H_b, nk=nk, n_occ=2)
                C2 = single_chern(H_b, nk=nk, band_idx=2)
                print(f"  nk={nk:3d}: C_comp={C:+.6f}  C_2={C2:+.6f}  Σ={C+C2:+.6f}  "
                      f"gap={g:.4f}  |det|_min={info['det_min']:.6f}")
    
    elapsed = time.time() - t0
    results['elapsed_s'] = round(elapsed, 1)
    print(f"\n{'='*60}")
    print(f"COMPLETE in {elapsed:.1f}s")
    print("="*60)
    
    with open('kagome_chern3.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("→ kagome_chern3.json")


if __name__ == '__main__':
    main()
