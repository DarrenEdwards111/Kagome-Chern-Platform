#!/usr/bin/env python3
"""
kagome_haldane.py — Kagome Chern insulator (FIXED Bloch convention)
====================================================================
BUG FIX: Previous version used Convention II Bloch phases (exp(ik·δ) with
δ = R + τ_β - τ_α). This is NOT periodic: H(k+G) ≠ H(k).
Fukui method requires Convention I: H_αβ(k) = Σ_R t_αβ(R) exp(ik·R)
which IS periodic.

Convention I: phase factors use LATTICE vectors R only.
  H[α,β] = t × Σ_R exp(ik·R)  where R are lattice vectors connecting
  unit cells in which α is NN/NNN to β.

Author: D. J. Edwards (2026)
"""

import numpy as np
from numpy import linalg as LA
import json, time

a1 = np.array([1.0, 0.0])
a2 = np.array([0.5, np.sqrt(3)/2])

# Reciprocal lattice (computed properly)
area = a1[0]*a2[1] - a1[1]*a2[0]  # = √3/2
b1 = (2*np.pi / area) * np.array([a2[1], -a2[0]])
b2 = (2*np.pi / area) * np.array([-a1[1], a1[0]])

tau = np.array([[0.0, 0.0], [0.5, 0.0], [0.25, np.sqrt(3)/4]])
Gamma = np.array([0.0, 0.0])
M = (b1 + b2) / 2
K = (2*b1 + b2) / 3


def H_kagome(kx, ky, t1=1.0, t2=0.0, phi=0.0, Phi_nn=0.0, m_sub=0.0):
    """
    Kagome Hamiltonian — Convention I (periodic in k).
    
    H[α,β](k) = Σ_R t_αβ exp(ik·R)  where R = lattice vector of unit cell.
    
    NN connectivity (lattice vectors R connecting unit cells):
      A(0) ↔ B(0):  R = 0      (intra-cell)
      A(0) ↔ B(-a1): R = -a1   (B in cell -a1 is NN to A in cell 0)
      A(0) ↔ C(0):  R = 0
      A(0) ↔ C(-a2): R = -a2
      B(0) ↔ C(0):  R = 0
      B(0) ↔ C(a1-a2): R = a1-a2
    
    Complex NN: staggered flux Φ per up-triangle.
      Intra-cell bonds (up-triangle): phase +Φ/3
      Inter-cell bonds (down-triangle): phase -Φ/3
    """
    k = np.array([kx, ky])
    H = np.zeros((3, 3), dtype=complex)
    
    p_up = np.exp(1j * Phi_nn / 3)
    p_dn = np.exp(-1j * Phi_nn / 3)
    
    # NN A-B: R = 0 (up-tri) and R = -a1 (down-tri)
    H[0, 1] = t1 * (p_up * 1.0 + p_dn * np.exp(-1j * k.dot(a1)))
    
    # NN A-C: R = 0 (up-tri) and R = -a2 (down-tri)
    H[0, 2] = t1 * (p_up * 1.0 + p_dn * np.exp(-1j * k.dot(a2)))
    
    # NN B-C: R = 0 (up-tri) and R = a1-a2 (down-tri)
    H[1, 2] = t1 * (p_up * 1.0 + p_dn * np.exp(1j * k.dot(a1 - a2)))
    
    # Hermitian conjugate
    H[1, 0] = H[0, 1].conj()
    H[2, 0] = H[0, 2].conj()
    H[2, 1] = H[1, 2].conj()
    
    # NNN Haldane (sublattice-dependent, Convention I uses R = lattice vectors)
    if abs(t2) > 1e-12:
        da = a2 - a1
        # A↔A: via ±a1 (through B, ν=+1), via ±a2 (through C, ν=-1)
        H[0, 0] += t2 * 2 * (np.cos(k.dot(a1) + phi) + np.cos(k.dot(a2) - phi))
        # B↔B: via ±a1 (through A, ν=-1), via ±(a2-a1) (through C, ν=+1)
        H[1, 1] += t2 * 2 * (np.cos(k.dot(a1) - phi) + np.cos(k.dot(da) + phi))
        # C↔C: via ±a2 (through A, ν=+1), via ±(a2-a1) (through B, ν=-1)
        H[2, 2] += t2 * 2 * (np.cos(k.dot(a2) + phi) + np.cos(k.dot(da) - phi))
    
    # Sublattice mass
    if abs(m_sub) > 1e-12:
        H[0, 0] += m_sub
        H[1, 1] -= m_sub / 2
        H[2, 2] -= m_sub / 2
    
    return H


def fukui_chern(H_func, nk=60, band_idx=0):
    """Fukui-Hatsugai-Suzuki Chern number on BZ torus."""
    states = np.zeros((nk, nk, 3, 3), dtype=complex)
    for i1 in range(nk):
        for i2 in range(nk):
            k = (i1 / nk) * b1 + (i2 / nk) * b2
            _, vecs = LA.eigh(H_func(k[0], k[1]))
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


def band_info(params, nk=100):
    bmin = np.full(3, np.inf)
    bmax = np.full(3, -np.inf)
    for i1 in range(nk):
        for i2 in range(nk):
            k = (i1/nk) * b1 + (i2/nk) * b2
            e = LA.eigvalsh(H_kagome(k[0], k[1], **params))
            bmin = np.minimum(bmin, e)
            bmax = np.maximum(bmax, e)
    W = bmax - bmin
    gaps = [bmin[i+1] - bmax[i] for i in range(2)]
    return W, gaps, bmin, bmax


def disorder_test(params, nk=40, W_d_values=None, n_samples=30, gap_idx=0):
    if W_d_values is None:
        W_d_values = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]
    results = []
    for W_d in W_d_values:
        samples = []
        for _ in range(n_samples):
            eps = np.random.uniform(-W_d, W_d, 3)
            gmin = np.inf
            for i1 in range(nk):
                for i2 in range(nk):
                    k = (i1/nk)*b1 + (i2/nk)*b2
                    H = H_kagome(k[0], k[1], **params)
                    H[0,0] += eps[0]; H[1,1] += eps[1]; H[2,2] += eps[2]
                    e = LA.eigvalsh(H)
                    gmin = min(gmin, e[gap_idx+1] - e[gap_idx])
            samples.append(gmin)
        results.append({
            'W_d': W_d, 'gap_mean': float(np.mean(samples)),
            'gap_std': float(np.std(samples)), 'gap_min': float(np.min(samples))
        })
    return results


def main():
    t0 = time.time()
    results = {}
    
    # ═══ VERIFY PERIODICITY ═══
    print("="*60)
    print("0. PERIODICITY CHECK")
    print("="*60)
    
    k_test = np.array([1.3, 0.7])
    for name, p in [("NNN only", {'t1':1,'t2':0.1,'phi':0.5,'Phi_nn':0,'m_sub':0}),
                     ("NN flux", {'t1':1,'t2':0,'phi':0,'Phi_nn':0.5,'m_sub':0}),
                     ("combined", {'t1':1,'t2':0.1,'phi':0.5,'Phi_nn':0.3,'m_sub':0.2})]:
        H0 = H_kagome(k_test[0], k_test[1], **p)
        H1 = H_kagome(k_test[0]+b1[0], k_test[1]+b1[1], **p)
        H2 = H_kagome(k_test[0]+b2[0], k_test[1]+b2[1], **p)
        err = max(LA.norm(H1-H0), LA.norm(H2-H0))
        ok = "✅" if err < 1e-10 else "❌"
        print(f"  {name:15s}: ||H(k+G)-H(k)|| = {err:.2e}  {ok}")
    
    # ═══ CHERN SANITY: TRS → C=0 ═══
    print(f"\n{'='*60}")
    print("1. CHERN SANITY: TRS preserved → C must = 0")
    print("="*60)
    
    # Pure Kagome (bands touch, but sum should still ~0 if periodic)
    H_trs = lambda kx,ky: H_kagome(kx,ky, t1=1, t2=0, phi=0, Phi_nn=0, m_sub=0)
    cs = [fukui_chern(H_trs, nk=40, band_idx=b) for b in range(3)]
    print(f"  Pure Kagome: C = [{cs[0]:+.3f}, {cs[1]:+.3f}, {cs[2]:+.3f}]  Σ={sum(cs):+.3f}")
    print(f"  (bands touch → individual C ill-defined, but Σ should = 0)")
    
    # With mass (TRS, gapped)
    H_mass = lambda kx,ky: H_kagome(kx,ky, t1=1, t2=0, phi=0, Phi_nn=0, m_sub=0.5)
    p_mass = {'t1':1, 't2':0, 'phi':0, 'Phi_nn':0, 'm_sub':0.5}
    W, gaps, _, _ = band_info(p_mass, nk=80)
    print(f"  Mass m=0.5: gaps = [{gaps[0]:.4f}, {gaps[1]:.4f}]")
    cs = [fukui_chern(H_mass, nk=60, band_idx=b) for b in range(3)]
    print(f"  Mass m=0.5: C = [{cs[0]:+.3f}, {cs[1]:+.3f}, {cs[2]:+.3f}]  Σ={sum(cs):+.3f}")
    
    # ═══ NNN HALDANE SCAN ═══
    print(f"\n{'='*60}")
    print("2. NNN HALDANE SCAN (both gaps)")
    print("="*60)
    
    scan = []
    for t2 in [0.1, 0.2, 0.3, 0.5]:
        for phi_f in [0.25, 0.5, 0.75]:
            p = {'t1':1, 't2':t2, 'phi':phi_f*np.pi, 'Phi_nn':0, 'm_sub':0}
            W, gaps, _, _ = band_info(p, nk=60)
            
            tag = ""
            if gaps[0] > 0.01: tag += " ★BOT"
            if gaps[1] > 0.01: tag += " ★TOP"
            
            print(f"  t2={t2:.1f} φ/π={phi_f:.2f}  gap01={gaps[0]:+.4f}  gap12={gaps[1]:+.4f}{tag}")
            scan.append({'t2':t2, 'phi_pi':phi_f, 'gaps':gaps, 'W':W.tolist()})
    
    results['scan_nnn'] = scan
    
    # ═══ COMPLEX NN SCAN ═══
    print(f"\n{'='*60}")
    print("3. COMPLEX NN (staggered flux)")
    print("="*60)
    
    for Phi_f in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
        p = {'t1':1, 't2':0, 'phi':0, 'Phi_nn':Phi_f*np.pi, 'm_sub':0}
        W, gaps, _, _ = band_info(p, nk=80)
        tag = ""
        if gaps[0] > 0.01: tag += " ★BOT"
        if gaps[1] > 0.01: tag += " ★TOP"
        print(f"  Φ/π={Phi_f:.1f}  gap01={gaps[0]:+.4f}  gap12={gaps[1]:+.4f}  W={W.round(3)}{tag}")
    
    # ═══ COMBINED SCAN ═══
    print(f"\n{'='*60}")
    print("4. COMBINED SCAN (NN flux + NNN + mass)")
    print("="*60)
    
    best = None
    best_gap = -np.inf
    candidates = []
    
    for Phi_f in [0.0, 0.3, 0.5, 0.7, 1.0]:
        for t2 in [0.0, 0.1, 0.2, 0.3]:
            for phi_f in [0.0, 0.25, 0.5]:
                for m in [0.0, 0.1, 0.3, 0.5, 1.0]:
                    if Phi_f == 0 and t2 == 0 and m == 0:
                        continue
                    p = {'t1':1, 't2':t2, 'phi':phi_f*np.pi, 'Phi_nn':Phi_f*np.pi, 'm_sub':m}
                    W, gaps, _, _ = band_info(p, nk=50)
                    
                    for gi in range(2):
                        if gaps[gi] > 0.01:
                            candidates.append({
                                'Phi_pi':Phi_f, 't2':t2, 'phi_pi':phi_f, 'm_sub':m,
                                'gap_idx':gi, 'gap':round(gaps[gi],4),
                                'W':[round(w,4) for w in W],
                                'params': p
                            })
                            if gaps[gi] > best_gap:
                                best_gap = gaps[gi]
                                best = candidates[-1]
    
    candidates.sort(key=lambda e: -e['gap'])
    print(f"  Found {len(candidates)} gapped configs. Top 15:")
    for e in candidates[:15]:
        print(f"    Φ={e['Phi_pi']:.1f}π t2={e['t2']:.1f} φ={e['phi_pi']:.2f}π m={e['m_sub']:.1f}  "
              f"gap[{e['gap_idx']}]={e['gap']:.4f}  W=[{e['W'][0]:.3f},{e['W'][1]:.3f},{e['W'][2]:.3f}]")
    
    results['candidates'] = [dict((k,v) for k,v in c.items() if k != 'params') for c in candidates[:30]]
    
    if best:
        bp = best['params']
        gi = best['gap_idx']
        
        print(f"\n★ BEST: gap[{gi}]={best['gap']:.4f}t")
        print(f"  Φ/π={best['Phi_pi']}, t2={best['t2']}, φ/π={best['phi_pi']}, m={best['m_sub']}")
        
        # ═══ CHERN NUMBERS ═══
        print(f"\n{'='*60}")
        print("5. CHERN NUMBERS")
        print("="*60)
        
        H_best = lambda kx,ky: H_kagome(kx,ky, **bp)
        for nk in [30, 50, 70, 100]:
            cs = [fukui_chern(H_best, nk=nk, band_idx=b) for b in range(3)]
            print(f"  nk={nk:3d}: C = [{cs[0]:+.4f}, {cs[1]:+.4f}, {cs[2]:+.4f}]  Σ={sum(cs):+.4f}")
        results['chern'] = cs
        
        # ═══ PHYSICAL SCALES ═══
        t_phys = 100.0
        gap_meV = best['gap'] * t_phys
        print(f"\n{'='*60}")
        print(f"6. PHYSICAL SCALES (t = {t_phys} meV)")
        print("="*60)
        print(f"  Gap: {gap_meV:.1f} meV")
        print(f"  Widths: {[round(w*t_phys,1) for w in best['W']]} meV")
        
        results['physical'] = {'gap_meV': gap_meV}
        
        # ═══ DISORDER ═══
        print(f"\n{'='*60}")
        print(f"7. DISORDER (C5) — gap[{gi}]")
        print("="*60)
        
        W_d_vals = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]
        dis = disorder_test(bp, nk=30, W_d_values=W_d_vals, n_samples=30, gap_idx=gi)
        for d in dis:
            meV = d['W_d'] * t_phys
            gmeV = d['gap_mean'] * t_phys
            ok = "✅" if d['gap_mean'] > 0.05 else "❌"
            print(f"  W_d={meV:5.1f} meV  gap={gmeV:6.1f} ± {d['gap_std']*t_phys:.1f} meV  "
                  f"min={d['gap_min']*t_phys:.1f}  {ok}")
        results['disorder'] = dis
        
        # ═══ Now test topological candidates specifically ═══
        # Find ones with TRS-breaking (Phi_nn ≠ 0 or phi ≠ 0) and gap > 0
        print(f"\n{'='*60}")
        print("8. TOPOLOGICAL CANDIDATES (TRS broken + gap)")
        print("="*60)
        
        topo_cands = [c for c in candidates if c['Phi_pi'] != 0 or (c['t2'] > 0 and c['phi_pi'] != 0)]
        topo_cands.sort(key=lambda e: -e['gap'])
        
        for e in topo_cands[:5]:
            p = e['params'] if 'params' in e else {
                't1':1, 't2':e['t2'], 'phi':e['phi_pi']*np.pi,
                'Phi_nn':e['Phi_pi']*np.pi, 'm_sub':e['m_sub']
            }
            H_t = lambda kx,ky,pp=p: H_kagome(kx,ky, **pp)
            cs = [fukui_chern(H_t, nk=60, band_idx=b) for b in range(3)]
            print(f"  Φ={e['Phi_pi']:.1f}π t2={e['t2']:.1f} φ={e['phi_pi']:.2f}π m={e['m_sub']:.1f}  "
                  f"gap[{e['gap_idx']}]={e['gap']:.4f}  C=[{cs[0]:+.3f},{cs[1]:+.3f},{cs[2]:+.3f}]  Σ={sum(cs):+.3f}")
    
    elapsed = time.time() - t0
    results['elapsed_s'] = round(elapsed, 1)
    print(f"\n{'='*60}")
    print(f"COMPLETE in {elapsed:.1f}s")
    print("="*60)
    
    with open('kagome_haldane.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("→ kagome_haldane.json")


if __name__ == '__main__':
    main()
