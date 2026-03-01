#!/usr/bin/env python3
"""
kagome_chern2.py — Multi-band Chern number for Kagome
======================================================
When bands 0,1 overlap, individual Chern is ill-defined.
The PHYSICAL Chern number for a gap between bands 1 and 2 is:
  C_{0+1} = multi-band Chern of the 2-band occupied subspace.

Method: Non-Abelian Fukui-Hatsugai-Suzuki
  U_μ(k) = det M_μ(k)  where  [M_μ]_{mn} = ⟨u_m(k)|u_n(k+δ_μ)⟩
  F(k) = arg(U_1 · U_2 · U_3 · U_4)
  C = Σ_k F(k) / (2π)

Also test: does adding spin-orbit (Kane-Mele type) give Z2 or QAH?

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


def H_kagome(kx, ky, t1=1.0, t2=0.0, phi=0.0, Phi_nn=0.0, m_sub=0.0):
    """Convention I Kagome Hamiltonian (periodic in k)."""
    k = np.array([kx, ky])
    H = np.zeros((3, 3), dtype=complex)
    
    p_up = np.exp(1j * Phi_nn / 3)
    p_dn = np.exp(-1j * Phi_nn / 3)
    
    H[0, 1] = t1 * (p_up + p_dn * np.exp(-1j * k.dot(a1)))
    H[0, 2] = t1 * (p_up + p_dn * np.exp(-1j * k.dot(a2)))
    H[1, 2] = t1 * (p_up + p_dn * np.exp(1j * k.dot(a1 - a2)))
    H[1, 0] = H[0, 1].conj()
    H[2, 0] = H[0, 2].conj()
    H[2, 1] = H[1, 2].conj()
    
    if abs(t2) > 1e-12:
        da = a2 - a1
        H[0, 0] += t2 * 2 * (np.cos(k.dot(a1) + phi) + np.cos(k.dot(a2) - phi))
        H[1, 1] += t2 * 2 * (np.cos(k.dot(a1) - phi) + np.cos(k.dot(da) + phi))
        H[2, 2] += t2 * 2 * (np.cos(k.dot(a2) + phi) + np.cos(k.dot(da) - phi))
    
    if abs(m_sub) > 1e-12:
        H[0, 0] += m_sub
        H[1, 1] -= m_sub / 2
        H[2, 2] -= m_sub / 2
    
    return H


def multiband_chern(H_func, nk=80, n_occ=2, n_bands=3):
    """
    Multi-band Chern number for n_occ occupied bands.
    
    Non-Abelian Fukui method:
    U_μ = det(M_μ) where M_mn = ⟨u_m(k)|u_n(k+δ_μ)⟩ (m,n in occupied set)
    """
    # Precompute eigenstates
    states = np.zeros((nk, nk, n_bands, n_bands), dtype=complex)
    for i1 in range(nk):
        for i2 in range(nk):
            k = (i1/nk)*b1 + (i2/nk)*b2
            _, vecs = LA.eigh(H_func(k[0], k[1]))
            states[i1, i2] = vecs
    
    total = 0.0
    for i1 in range(nk):
        for i2 in range(nk):
            j1 = (i1 + 1) % nk
            j2 = (i2 + 1) % nk
            
            # Occupied subspace at 4 corners of plaquette
            P00 = states[i1, i2, :, :n_occ]  # shape (n_bands, n_occ)
            P10 = states[j1, i2, :, :n_occ]
            P11 = states[j1, j2, :, :n_occ]
            P01 = states[i1, j2, :, :n_occ]
            
            # Link matrices (n_occ × n_occ)
            M1 = P00.conj().T @ P10
            M2 = P10.conj().T @ P11
            M3 = P11.conj().T @ P01
            M4 = P01.conj().T @ P00
            
            # Plaquette = det(M1) · det(M2) · det(M3) · det(M4)
            plaq = LA.det(M1) * LA.det(M2) * LA.det(M3) * LA.det(M4)
            total += np.angle(plaq)
    
    return total / (2 * np.pi)


def single_chern(H_func, nk=80, band_idx=0, n_bands=3):
    """Single-band Chern (for isolated bands)."""
    states = np.zeros((nk, nk, n_bands, n_bands), dtype=complex)
    for i1 in range(nk):
        for i2 in range(nk):
            k = (i1/nk)*b1 + (i2/nk)*b2
            _, vecs = LA.eigh(H_func(k[0], k[1]))
            states[i1, i2] = vecs
    
    total = 0.0
    for i1 in range(nk):
        for i2 in range(nk):
            j1 = (i1+1) % nk
            j2 = (i2+1) % nk
            u = states[:,:,:,band_idx]
            
            U = (np.vdot(u[i1,i2], u[j1,i2]) *
                 np.vdot(u[j1,i2], u[j1,j2]) *
                 np.vdot(u[j1,j2], u[i1,j2]) *
                 np.vdot(u[i1,j2], u[i1,i2]))
            total += np.angle(U)
    
    return total / (2*np.pi)


def band_info(p, nk=100):
    bmin = np.full(3, np.inf)
    bmax = np.full(3, -np.inf)
    for i1 in range(nk):
        for i2 in range(nk):
            k = (i1/nk)*b1 + (i2/nk)*b2
            e = LA.eigvalsh(H_kagome(k[0], k[1], **p))
            bmin = np.minimum(bmin, e)
            bmax = np.maximum(bmax, e)
    return bmax - bmin, [bmin[i+1] - bmax[i] for i in range(2)]


def main():
    t0 = time.time()
    results = {}
    
    # ═══ 1. SANITY: Haldane honeycomb with multi-band ═══
    print("="*60)
    print("1. SANITY: Multi-band Chern on Haldane honeycomb")
    print("="*60)
    
    a1_h = np.array([1.0, 0.0])
    a2_h = np.array([0.5, np.sqrt(3)/2])
    area_h = a1_h[0]*a2_h[1] - a1_h[1]*a2_h[0]
    b1_h = (2*np.pi/area_h) * np.array([a2_h[1], -a2_h[0]])
    b2_h = (2*np.pi/area_h) * np.array([-a1_h[1], a1_h[0]])
    
    def H_haldane(kx, ky):
        k = np.array([kx, ky])
        d1 = np.array([0.0, 1/np.sqrt(3)])
        d2 = np.array([0.5, -1/(2*np.sqrt(3))])
        d3 = np.array([-0.5, -1/(2*np.sqrt(3))])
        f_k = sum(np.exp(1j*k.dot(d)) for d in [d1,d2,d3])
        nnn = [a1_h, a2_h, a2_h-a1_h]
        H = np.zeros((2,2), dtype=complex)
        H[0,0] = 0.1 * sum(2*np.cos(k.dot(v)+np.pi/2) for v in nnn)
        H[1,1] = 0.1 * sum(2*np.cos(k.dot(v)-np.pi/2) for v in nnn)
        H[0,1] = f_k
        H[1,0] = H[0,1].conj()
        return H
    
    # Single-band: C0 should be ±1
    # Need to use honeycomb BZ
    def chern_hc(band_idx):
        nk = 60
        states = np.zeros((nk,nk,2,2), dtype=complex)
        for i1 in range(nk):
            for i2 in range(nk):
                k = (i1/nk)*b1_h + (i2/nk)*b2_h
                _, v = LA.eigh(H_haldane(k[0], k[1]))
                states[i1,i2] = v
        total = 0.0
        for i1 in range(nk):
            for i2 in range(nk):
                j1 = (i1+1)%nk; j2 = (i2+1)%nk
                u = states[:,:,:,band_idx]
                U = (np.vdot(u[i1,i2],u[j1,i2]) * np.vdot(u[j1,i2],u[j1,j2]) *
                     np.vdot(u[j1,j2],u[i1,j2]) * np.vdot(u[i1,j2],u[i1,i2]))
                total += np.angle(U)
        return total/(2*np.pi)
    
    c0 = chern_hc(0)
    c1 = chern_hc(1)
    print(f"  Single-band: C0={c0:+.4f}, C1={c1:+.4f}, Σ={c0+c1:+.4f}")
    print(f"  Expected: C0=-1, C1=+1")
    
    # ═══ 2. Multi-band Chern on Kagome ═══
    print(f"\n{'='*60}")
    print("2. MULTI-BAND CHERN: Kagome NNN Haldane")
    print("="*60)
    
    test_params = [
        {'t2': 0.1, 'phi_pi': 0.25, 'label': 't2=0.1, φ=π/4'},
        {'t2': 0.1, 'phi_pi': 0.50, 'label': 't2=0.1, φ=π/2'},
        {'t2': 0.2, 'phi_pi': 0.50, 'label': 't2=0.2, φ=π/2'},
        {'t2': 0.3, 'phi_pi': 0.50, 'label': 't2=0.3, φ=π/2'},
        {'t2': 0.3, 'phi_pi': 0.40, 'label': 't2=0.3, φ=0.4π (TMW)'},
        {'t2': 0.5, 'phi_pi': 0.25, 'label': 't2=0.5, φ=π/4'},
    ]
    
    for tp in test_params:
        p = {'t1':1, 't2':tp['t2'], 'phi':tp['phi_pi']*np.pi, 'Phi_nn':0, 'm_sub':0}
        W, gaps = band_info(p, nk=80)
        
        H_f = lambda kx, ky, pp=p: H_kagome(kx, ky, **pp)
        
        # Multi-band Chern of bands {0,1} (below gap12)
        C_01 = multiband_chern(H_f, nk=60, n_occ=2)
        # Single-band Chern of band 2 (above gap12)  
        C_2 = single_chern(H_f, nk=60, band_idx=2)
        
        ok = "✅" if abs(round(C_01) - round(C_2)) > 0 else ""
        print(f"  {tp['label']:25s}  gap12={gaps[1]:+.4f}  C_{{0+1}}={C_01:+.4f}  C_2={C_2:+.4f}  "
              f"Σ={C_01+C_2:+.4f}  {ok}")
    
    # ═══ 3. Try with both NNN + sublattice mass ═══
    print(f"\n{'='*60}")
    print("3. NNN + SUBLATTICE MASS (TRS broken by NNN, C3 broken by mass)")
    print("="*60)
    
    for t2 in [0.1, 0.2, 0.3]:
        for phi_f in [0.25, 0.5]:
            for m in [0.1, 0.3, 0.5, 1.0]:
                p = {'t1':1, 't2':t2, 'phi':phi_f*np.pi, 'Phi_nn':0, 'm_sub':m}
                W, gaps = band_info(p, nk=60)
                
                if gaps[1] > 0.01:
                    H_f = lambda kx, ky, pp=p: H_kagome(kx, ky, **pp)
                    C_01 = multiband_chern(H_f, nk=50, n_occ=2)
                    C_2 = single_chern(H_f, nk=50, band_idx=2)
                    
                    if abs(round(C_01)) > 0:
                        marker = " ★★★ TOPOLOGICAL"
                    else:
                        marker = ""
                    
                    print(f"  t2={t2:.1f} φ/π={phi_f:.2f} m={m:.1f}  gap12={gaps[1]:.4f}  "
                          f"C_{{0+1}}={C_01:+.4f}  C_2={C_2:+.4f}{marker}")
    
    # ═══ 4. Try with BOTH gap01 and gap12 open ═══
    print(f"\n{'='*60}")
    print("4. FULLY GAPPED CONFIGURATIONS")
    print("="*60)
    
    for t2 in [0.1, 0.2, 0.3, 0.5]:
        for phi_f in [0.0, 0.25, 0.5]:
            for m in [0.5, 1.0, 1.5, 2.0]:
                p = {'t1':1, 't2':t2, 'phi':phi_f*np.pi, 'Phi_nn':0, 'm_sub':m}
                W, gaps = band_info(p, nk=60)
                
                if gaps[0] > 0.01 and gaps[1] > 0.01:
                    H_f = lambda kx, ky, pp=p: H_kagome(kx, ky, **pp)
                    C = [single_chern(H_f, nk=50, band_idx=b) for b in range(3)]
                    
                    marker = " ★★★" if any(abs(round(c)) > 0 for c in C) else ""
                    print(f"  t2={t2:.1f} φ/π={phi_f:.2f} m={m:.1f}  gaps=[{gaps[0]:.3f},{gaps[1]:.3f}]  "
                          f"C=[{C[0]:+.3f},{C[1]:+.3f},{C[2]:+.3f}]  Σ={sum(C):+.3f}{marker}")
    
    # ═══ 5. Complex NN + NNN ═══
    print(f"\n{'='*60}")
    print("5. COMPLEX NN + NNN (double TRS breaking)")
    print("="*60)
    
    for Phi_f in [0.3, 0.5, 0.7]:
        for t2 in [0.1, 0.2, 0.3]:
            for phi_f in [0.25, 0.5]:
                for m in [0.0, 0.5, 1.0]:
                    p = {'t1':1, 't2':t2, 'phi':phi_f*np.pi, 'Phi_nn':Phi_f*np.pi, 'm_sub':m}
                    W, gaps = band_info(p, nk=50)
                    
                    for gi in range(2):
                        if gaps[gi] > 0.02:
                            H_f = lambda kx, ky, pp=p: H_kagome(kx, ky, **pp)
                            if gi == 0:
                                C_below = single_chern(H_f, nk=40, band_idx=0)
                                C_check = C_below
                            else:
                                C_below = multiband_chern(H_f, nk=40, n_occ=2)
                                C_check = C_below
                            
                            if abs(round(C_check)) > 0:
                                C_all = [single_chern(H_f, nk=40, band_idx=b) for b in range(3)]
                                print(f"  ★ Φ={Phi_f:.1f}π t2={t2:.1f} φ={phi_f:.2f}π m={m:.1f}  "
                                      f"gap[{gi}]={gaps[gi]:.4f}  C=[{C_all[0]:+.3f},{C_all[1]:+.3f},{C_all[2]:+.3f}]")
    
    elapsed = time.time() - t0
    results['elapsed_s'] = round(elapsed, 1)
    print(f"\n{'='*60}")
    print(f"COMPLETE in {elapsed:.1f}s")
    print("="*60)
    
    with open('kagome_chern2.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("→ kagome_chern2.json")


if __name__ == '__main__':
    main()
