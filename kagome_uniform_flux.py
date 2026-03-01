#!/usr/bin/env python3
"""
kagome_uniform_flux.py — Uniform NN flux on Kagome
====================================================
Ohgushi-Murakami-Nagaosa mechanism: uniform flux Φ through every triangle.

Key difference from staggered Haldane:
  - Staggered: up-triangle +Φ, down-triangle -Φ → TRS broken but C=0
  - Uniform: ALL triangles +Φ → total flux = 2Φ per unit cell
  - For Φ = nπ/q (rational), magnetic unit cell may enlarge
  - For Φ = π: total flux = 2π per unit cell → NO enlargement needed

The NN hoppings become genuinely complex off-diagonal terms:
  H[A,B] = t exp(iα_AB) × geometric phase factors

This creates Berry curvature that can give C ≠ 0.

Gauge choice: distribute phase Φ/3 per bond going counterclockwise
around each triangle. For uniform flux, BOTH up and down triangles
get +Φ/3 per bond (same orientation).

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

tau = np.array([[0.0, 0.0], [0.5, 0.0], [0.25, np.sqrt(3)/4]])


def H_kagome_flux(kx, ky, t1=1.0, Phi=0.0, t2=0.0, phi_nnn=0.0):
    """
    Kagome with UNIFORM flux Φ per triangle (Convention I).
    
    Gauge choice for uniform flux:
      Up-triangle (A→B→C→A counterclockwise): each bond gets +Φ/3
      Down-triangle: need consistent gauge.
      
    For uniform flux, the gauge on each bond depends on which triangles
    it borders. Each NN bond borders one up and one down triangle.
    
    Gauge: Assign vector potential A along bonds.
    For uniform flux Φ per triangle:
      - Intra-cell A→B: phase α₁
      - Intra-cell B→C: phase α₂  
      - Intra-cell C→A: phase α₃
      with α₁ + α₂ + α₃ = Φ (up-triangle)
      
    The inter-cell bonds must give Φ for the down-triangle too.
    Down-triangle vertices: B(0), A(a1), C(a1-a2)
    Going B→A(a1)→C(a1-a2)→B counterclockwise:
      Phase of B(0)→A(a1): this is the reverse of inter-cell A→B
      Phase of A(a1)→C(a1-a2): inter-cell A→C type  
      Phase of C(a1-a2)→B(0): inter-cell C→B type
    
    Simplest gauge: put ALL phase on one bond per triangle.
      α₁ = 0, α₂ = 0, α₃ = Φ  (up-triangle: C→A gets all phase)
    Then for down-triangle consistency, the inter-cell bonds
    must carry specific phases.
    
    Actually, for uniform flux the cleanest approach is the
    symmetric gauge: α₁ = α₂ = α₃ = Φ/3.
    
    Convention I with sublattice-dependent gauge:
    The key insight is that for uniform flux, the phase assignment
    on inter-cell bonds DIFFERS from the staggered case.
    
    Let me use the Peierls substitution with a specific vector potential.
    For uniform B-field giving flux Φ per triangle:
    
    A = (Φ/S_tri) × (-y, 0) / 2  [Landau-like gauge]
    where S_tri = area of one triangle.
    
    Phase on bond i→j: φ_ij = (e/ℏ) ∫_i^j A·dl
    
    For straight-line paths:
    φ_ij = (Φ/S_tri) × (-1/2)(y_i + y_j)(x_j - x_i)  ... messy.
    
    Let me just use the explicit approach.
    """
    k = np.array([kx, ky])
    H = np.zeros((3, 3), dtype=complex)
    
    # For uniform flux, I'll use a direct Peierls construction.
    # Triangle area: S_tri = |a1 × a2| / 2 = √3/4
    # But there are 2 triangles per unit cell.
    
    # Instead: parametrize by phases on each distinct bond.
    # There are 6 NN bonds in the unit cell (3 types × 2 each).
    # I need: sum of phases around each triangle = Φ.
    
    # Bond types and their triangles:
    # Type AB_intra: A(0)→B(0) — in up-triangle
    # Type AB_inter: A(0)→B(-a1) — in down-triangle  
    # Type BC_intra: B(0)→C(0) — in up-triangle
    # Type BC_inter: B(0)→C(a1-a2) — in down-triangle
    # Type CA_intra: C(0)→A(0) — in up-triangle
    # Type CA_inter: C(0)→A(a2) — in down-triangle (C→A across a2)
    
    # Wait, let me identify the down-triangle properly.
    # Down-triangle shares edge AB_inter with up-triangle at cell (0):
    # Vertices: A(0), B(-a1), and which C?
    # A(0) is NN to B(-a1) ✓
    # B(-a1) is NN to C(-a1) and C(-a2) 
    # A(0) is NN to C(0) and C(-a2)
    # Common: C(-a2)
    # Down-triangle: A(0), B(-a1), C(-a2)
    # Going counterclockwise: need to determine orientation
    # A(0)=(0,0), B(-a1)=(-1/2,0), C(-a2)=(-1/4,-√3/4)
    # CCW: A→C(-a2)→B(-a1)→A  (checking: cross products positive)
    
    # Actually there are two down-triangles per unit cell. Let me
    # just enumerate all triangles touching the A(0) site.
    
    # Up-triangle at origin: A(0), B(0), C(0)
    #   CCW: A→B→C→A
    #   Bonds: AB_intra, BC_intra, CA_intra
    #   Constraint: φ_AB_i + φ_BC_i + φ_CA_i = Φ
    
    # Down-triangle 1: A(0), B(-a1), C(-a2)  
    #   CCW order: A(0)=(0,0), C(-a2)=(1/4-1/2, √3/4-√3/2)=(-1/4,-√3/4), 
    #              B(-a1)=(-1/2,0)
    #   Hmm, let me check: is A→C(-a2)→B(-a1) counterclockwise?
    #   Cross product of (C-A) × (B-C): 
    #   (C-A) = (-1/4, -√3/4), (B-C) = (-1/4, √3/4)
    #   Cross = (-1/4)(√3/4) - (-√3/4)(-1/4) = -√3/16 - √3/16 = -√3/8 < 0
    #   So this is CLOCKWISE. CCW is: A→B(-a1)→C(-a2)→A
    #   Bonds: A→B(-a1) [AB_inter], B(-a1)→C(-a2) [BC_inter?], C(-a2)→A [CA_inter]
    
    # Let me just assign phases with a simple symmetric gauge.
    # Symmetric: all intra-cell bonds get phase Φ/3 (CCW in up-tri),
    #            all inter-cell bonds get phase Φ/3 (CCW in down-tri).
    
    # For Convention I:
    # H[A,B] = t₁[exp(iφ_AB_intra) × 1 + exp(iφ_AB_inter) × exp(-ik·a1)]
    
    # Up-triangle CCW: A→B→C→A, each +Φ/3
    # So φ_AB_intra = Φ/3, φ_BC_intra = Φ/3, φ_CA_intra = Φ/3
    
    # Down-triangle: vertices A(0), B(-a1), C(-a2)
    # CCW: A→B(-a1)→C(-a2)→A
    # Bonds:
    #   A→B(-a1): inter-cell AB bond, lattice vector R=-a1
    #   B(-a1)→C(-a2): this is B(R=-a1)→C(R=-a2), so inter-cell BC
    #     lattice displacement for B→C: R_C - R_B = -a2-(-a1) = a1-a2
    #     So this is the BC bond with R = a1-a2 (same type as BC_inter)
    #   C(-a2)→A(0): inter-cell CA bond, C(R=-a2)→A(R=0), R = a2
    
    # Down-triangle CCW constraint: φ_AB_inter + φ_BC_inter + φ_CA_inter = Φ
    # With symmetric gauge: φ_AB_inter = φ_BC_inter = φ_CA_inter = Φ/3
    
    p = np.exp(1j * Phi / 3)
    
    # H[A,B] Convention I:
    #   R=0 (intra, up-tri, A→B): phase +Φ/3
    #   R=-a1 (inter, down-tri, A→B(-a1)): phase +Φ/3
    H[0, 1] = t1 * p * (1 + np.exp(-1j * k.dot(a1)))
    
    # H[B,C] Convention I:
    #   R=0 (intra, up-tri, B→C): phase +Φ/3
    #   R=a1-a2 (inter, down-tri, B→C(a1-a2)): phase +Φ/3
    H[1, 2] = t1 * p * (1 + np.exp(1j * k.dot(a1 - a2)))
    
    # H[C,A] = (H[A,C])†, so I need H[A,C].
    # C→A bonds:
    #   R=0 (intra, up-tri, C→A): phase +Φ/3, so A←C has phase -Φ/3
    #   R=a2 (inter, down-tri, C(-a2)→A(0)): phase +Φ/3, so A←C has -Φ/3
    # H[C,A] = t₁ × exp(+iΦ/3) × [1 + exp(ik·a2)]  (C→A direction)
    # H[A,C] = H[C,A]† = t₁ × exp(-iΦ/3) × [1 + exp(-ik·a2)]
    # But wait: Convention I for H[A,C]: contributions from A→C bonds
    #   A→C is REVERSE of C→A. If C→A has phase +Φ/3, then A→C has -Φ/3.
    # H[A,C] = t₁ × exp(-iΦ/3) × Σ exp(ik·R) for C→A bonds
    # Hmm, this is getting confusing. Let me be more careful.
    
    # Convention I: H[α,β](k) = Σ_R t_αβ(R) exp(ik·R)
    # where t_αβ(R) = hopping from β at cell R to α at cell 0
    # (or equivalently, from α at 0 to β at R... need to be consistent)
    
    # Let me define: H[α,β](k) = Σ_R <α,0|H|β,R> exp(ik·R)
    # = Σ_R t_{α←β}(R) exp(ik·R)
    
    # For A←B hopping: 
    #   B at R=0 → A at 0: this is the reverse of A→B intra, phase = -Φ/3
    #   Wait no. If A→B has hopping t×exp(iΦ/3), then B→A has t×exp(-iΦ/3).
    #   <A|H|B,R=0> = t×exp(iΦ/3) if A→B bond has phase +Φ/3? No...
    
    # OK let me just be very explicit.
    # The hopping is: H_hop = Σ_{<ij>} t_ij c†_i c_j
    # where t_ij = t × exp(iφ_ij) and φ_ij = -φ_ji.
    
    # In Bloch basis: c†_α,k = (1/√N) Σ_R exp(-ik·R) c†_α,R
    # H = Σ_k Σ_{αβ} h_αβ(k) c†_α,k c_β,k
    # h_αβ(k) = Σ_R t_{α0,βR} exp(ik·R)
    # where t_{α0,βR} = <α,0|H|β,R> is the hopping amplitude from β at R to α at 0.
    
    # For A(0)→B(0) with phase +Φ/3:
    # H contains: t exp(iΦ/3) c†_B,0 c_A,0
    # So <B,0|H|A,0> = t exp(iΦ/3)
    # And <A,0|H|B,0> = t exp(-iΦ/3) = [t exp(iΦ/3)]*
    
    # h[A,B](k) = Σ_R <A,0|H|B,R> exp(ik·R)
    # For B at R=0: <A,0|H|B,0> = t exp(-iΦ/3)  [reverse of A→B]
    # For B at R=a1: <A,0|H|B,a1> comes from the bond B(a1)→A(0),
    #   which is the inter-cell AB bond. If A(0)→B(-a1) has phase +Φ/3
    #   in the down-triangle, then B(-a1)→A(0) has phase -Φ/3,
    #   equivalently B(a1)→A(0) with R-shift: <A,0|H|B,a1> = ... 
    
    # I'm going in circles. Let me just use a clean formulation.
    
    # CLEAN APPROACH: Define hopping amplitudes directly.
    # For each directed bond (α,R_α) → (β,R_β), define t_{αR_α → βR_β}.
    # Then h[α,β](k) = Σ_R t_{α,0 ← β,R} exp(ik·R)
    #                = Σ_R [t_{β,R → α,0}]* exp(ik·R)  ... no
    # Actually h[α,β](k) = Σ_R <α,0|H|β,R> exp(ik·R)
    # <α,0|H|β,R> = hopping from (β,R) to (α,0) = t_{β,R→α,0}
    
    # Hopping from A(0) to B(0): t exp(iΦ/3). So:
    #   <B,0|H|A,0> = t exp(iΦ/3)
    #   <A,0|H|B,0> = t exp(-iΦ/3)
    
    # Hopping from A(0) to B(-a1): t exp(iΦ/3) in down-tri. So:
    #   <B,-a1|H|A,0> = t exp(iΦ/3)
    #   <A,0|H|B,-a1> = t exp(-iΦ/3)
    # In Bloch: h[A,B] gets contribution from R=-a1:
    #   <A,0|H|B,-a1> exp(-ik·a1) = t exp(-iΦ/3) exp(-ik·a1)
    
    # Wait, h[A,B](k) = Σ_R <A,0|H|B,R> exp(ik·R)
    #   R=0: <A,0|H|B,0> exp(0) = t exp(-iΦ/3)
    #   R=-a1: <A,0|H|B,-a1> exp(-ik·a1) = t exp(-iΦ/3) exp(-ik·a1)
    # So h[A,B] = t exp(-iΦ/3) (1 + exp(-ik·a1))
    
    # And h[B,A] = h[A,B]* = t exp(+iΦ/3) (1 + exp(+ik·a1))
    
    # Similarly for B→C:
    # B(0)→C(0): phase +Φ/3 (up-tri CCW)
    # <C,0|H|B,0> = t exp(iΦ/3)
    # <B,0|H|C,0> = t exp(-iΦ/3)
    
    # B(0)→C(a1-a2): phase +Φ/3 (down-tri CCW)
    # <C,a1-a2|H|B,0> = t exp(iΦ/3)  
    # In h[B,C]: we need <B,0|H|C,R> exp(ik·R)
    #   R=0: <B,0|H|C,0> = t exp(-iΦ/3)
    #   R=a1-a2: <B,0|H|C,a1-a2> = t exp(-iΦ/3)
    #     (because C(a1-a2)→B(0) has phase -Φ/3)
    # h[B,C] = t exp(-iΦ/3) (1 + exp(ik·(a1-a2)))
    
    # For C→A:
    # C(0)→A(0): phase +Φ/3 (up-tri CCW, C→A direction)
    # <A,0|H|C,0> = t exp(iΦ/3)
    # In h[C,A]: <C,0|H|A,0> = t exp(-iΦ/3)
    
    # C(-a2)→A(0): phase +Φ/3 (down-tri CCW)
    # <A,0|H|C,-a2> = t exp(iΦ/3)
    # In h[C,A]: <C,0|H|A,R>... we need h[C,A](k) = Σ_R <C,0|H|A,R> exp(ik·R)
    #   R=0: <C,0|H|A,0> = t exp(-iΦ/3) (reverse of A(0)→C(0))
    #   What other R? A(a2)→C(0)? Is A(a2) NN to C(0)?
    #   A(a2) is at position τ_A + a2 = a2 = (1/2, √3/2)
    #   C(0) is at τ_C = (1/4, √3/4)
    #   Distance: |a2 - τ_C| = |(1/4, √3/4)| = 1/2 ✓ — yes, NN!
    #   So C(0)→A(a2) is a bond. This is the reverse of C(-a2)→A(0) shifted.
    #   Actually: <C,0|H|A,a2> = t exp(-iΦ/3) 
    #   (A(a2)→C(0) has phase +Φ/3 for C receiving, so C→A(a2) is -Φ/3 reversed)
    #   Wait: the bond C(-a2)→A(0) has phase +Φ/3. 
    #   Shifting by a2: C(0)→A(a2) has phase +Φ/3 (translation invariant).
    #   So <A,a2|H|C,0> = t exp(iΦ/3)
    #   And <C,0|H|A,a2> = t exp(-iΦ/3)
    
    # h[C,A](k) = t exp(-iΦ/3) [1 + exp(ik·a2)]
    # h[A,C](k) = h[C,A]* = t exp(+iΦ/3) [1 + exp(-ik·a2)]
    
    # FINAL RESULT (uniform flux, symmetric gauge, Convention I):
    # h[A,B] = t exp(-iΦ/3) (1 + exp(-ik·a1))
    # h[B,C] = t exp(-iΦ/3) (1 + exp(ik·(a1-a2)))
    # h[C,A] = t exp(-iΦ/3) (1 + exp(ik·a2))
    # h[A,C] = t exp(+iΦ/3) (1 + exp(-ik·a2))
    
    pm = np.exp(-1j * Phi / 3)  # minus because h[α,β] = t exp(-iΦ/3) × ...
    
    H[0, 1] = t1 * pm * (1 + np.exp(-1j * k.dot(a1)))        # A←B
    H[1, 2] = t1 * pm * (1 + np.exp(1j * k.dot(a1 - a2)))    # B←C  
    H[2, 0] = t1 * pm * (1 + np.exp(1j * k.dot(a2)))          # C←A
    
    H[1, 0] = H[0, 1].conj()
    H[2, 1] = H[1, 2].conj()
    H[0, 2] = H[2, 0].conj()
    
    # Optional NNN
    if abs(t2) > 1e-12:
        da = a2 - a1
        H[0, 0] += t2 * 2 * (np.cos(k.dot(a1) + phi_nnn) + np.cos(k.dot(a2) - phi_nnn))
        H[1, 1] += t2 * 2 * (np.cos(k.dot(a1) - phi_nnn) + np.cos(k.dot(da) + phi_nnn))
        H[2, 2] += t2 * 2 * (np.cos(k.dot(a2) + phi_nnn) + np.cos(k.dot(da) - phi_nnn))
    
    return H


def composite_chern(H_func, nk=80, n_occ=2, n_bands=3):
    """Non-Abelian Fukui Chern for occupied subspace."""
    states = np.zeros((nk, nk, n_bands, n_bands), dtype=complex)
    energies = np.zeros((nk, nk, n_bands))
    
    for i1 in range(nk):
        for i2 in range(nk):
            k = (i1/nk)*b1 + (i2/nk)*b2
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
            
            plaq = (d1/abs(d1)) * (d2/abs(d2)) * (d3/abs(d3)) * (d4/abs(d4))
            total += np.angle(plaq)
    
    return total/(2*np.pi), direct_gap


def single_chern(H_func, nk=80, band_idx=0, n_bands=3):
    """Single-band Fukui Chern."""
    states = np.zeros((nk, nk, n_bands, n_bands), dtype=complex)
    for i1 in range(nk):
        for i2 in range(nk):
            k = (i1/nk)*b1 + (i2/nk)*b2
            _, vecs = LA.eigh(H_func(k[0], k[1]))
            states[i1, i2] = vecs
    total = 0.0
    for i1 in range(nk):
        for i2 in range(nk):
            j1=(i1+1)%nk; j2=(i2+1)%nk
            u = states[:,:,:,band_idx]
            U = (np.vdot(u[i1,i2],u[j1,i2]) * np.vdot(u[j1,i2],u[j1,j2]) *
                 np.vdot(u[j1,j2],u[i1,j2]) * np.vdot(u[i1,j2],u[i1,i2]))
            total += np.angle(U)
    return total/(2*np.pi)


def band_info(H_func, nk=100):
    bmin = np.full(3, np.inf); bmax = np.full(3, -np.inf)
    for i1 in range(nk):
        for i2 in range(nk):
            k = (i1/nk)*b1 + (i2/nk)*b2
            e = LA.eigvalsh(H_func(k[0], k[1]))
            bmin = np.minimum(bmin, e); bmax = np.maximum(bmax, e)
    return bmax - bmin, [bmin[i+1]-bmax[i] for i in range(2)], bmin, bmax


def disorder_test(H_func, nk=40, W_d_values=None, n_samples=30, gap_idx=0):
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
                    H = H_func(k[0], k[1])
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
    
    # ═══ 0. PERIODICITY CHECK ═══
    print("="*60)
    print("0. PERIODICITY CHECK")
    print("="*60)
    k_test = np.array([1.3, 0.7])
    for Phi_val in [0.0, np.pi/4, np.pi/2, np.pi]:
        H_f = lambda kx,ky,_P=Phi_val: H_kagome_flux(kx,ky,Phi=_P)
        H0 = H_f(k_test[0], k_test[1])
        H1 = H_f(k_test[0]+b1[0], k_test[1]+b1[1])
        H2 = H_f(k_test[0]+b2[0], k_test[1]+b2[1])
        err = max(LA.norm(H1-H0), LA.norm(H2-H0))
        ok = "✅" if err < 1e-10 else "❌"
        print(f"  Φ/π={Phi_val/np.pi:.2f}: ||H(k+G)-H(k)|| = {err:.2e}  {ok}")
    
    # ═══ 1. FLUX SCAN ═══
    print(f"\n{'='*60}")
    print("1. UNIFORM FLUX SCAN — Φ per triangle")
    print("="*60)
    
    scan = []
    for Phi_f in np.arange(0.05, 1.01, 0.05):
        Phi_val = Phi_f * np.pi
        H_f = lambda kx,ky,_P=Phi_val: H_kagome_flux(kx,ky,Phi=_P)
        W, gaps, bmin, bmax = band_info(H_f, nk=80)
        
        tag = ""
        if gaps[0] > 0.01: tag += " ★BOT"
        if gaps[1] > 0.01: tag += " ★TOP"
        
        scan.append({'Phi_pi': round(Phi_f,2), 'W': W.tolist(), 
                     'gaps': [float(g) for g in gaps]})
        
        print(f"  Φ/π={Phi_f:.2f}  gap01={gaps[0]:+.4f}  gap12={gaps[1]:+.4f}  "
              f"W=[{W[0]:.3f},{W[1]:.3f},{W[2]:.3f}]{tag}")
    
    results['scan'] = scan
    
    # ═══ 2. CHERN for gapped configurations ═══
    print(f"\n{'='*60}")
    print("2. CHERN NUMBERS for gapped Φ values")
    print("="*60)
    
    chern_results = []
    for entry in scan:
        gaps = entry['gaps']
        Phi_val = entry['Phi_pi'] * np.pi
        H_f = lambda kx,ky,_P=Phi_val: H_kagome_flux(kx,ky,Phi=_P)
        
        computed = False
        for gi in range(2):
            if gaps[gi] > 0.01:
                if gi == 0:
                    # Individual Chern of band 0
                    C0 = single_chern(H_f, nk=60, band_idx=0)
                    print(f"  Φ/π={entry['Phi_pi']:.2f}  gap[0]={gaps[0]:.4f}  "
                          f"C_0={C0:+.4f}")
                    chern_results.append({'Phi_pi': entry['Phi_pi'], 'gap_idx': 0,
                                         'gap': gaps[0], 'C': round(C0, 4)})
                    computed = True
                
                if gi == 1:
                    # Composite Chern of {0,1}
                    C01, gdirect = composite_chern(H_f, nk=60, n_occ=2)
                    C2 = single_chern(H_f, nk=60, band_idx=2)
                    marker = " ★★★ TOPOLOGICAL" if abs(round(C01)) > 0 else ""
                    print(f"  Φ/π={entry['Phi_pi']:.2f}  gap[1]={gaps[1]:.4f}  "
                          f"C_{{0+1}}={C01:+.4f}  C_2={C2:+.4f}  Σ={C01+C2:+.4f}{marker}")
                    chern_results.append({'Phi_pi': entry['Phi_pi'], 'gap_idx': 1,
                                         'gap': gaps[1], 'C_01': round(C01,4), 
                                         'C_2': round(C2,4)})
                    computed = True
    
    results['chern'] = chern_results
    
    # ═══ 3. Best topological candidate — convergence + physics ═══
    topo = [c for c in chern_results if 'C_01' in c and abs(round(c.get('C_01',0))) > 0]
    if not topo:
        topo = [c for c in chern_results if 'C' in c and abs(round(c.get('C',0))) > 0]
    
    if topo:
        best = max(topo, key=lambda e: e['gap'])
        Phi_best = best['Phi_pi'] * np.pi
        gi = best.get('gap_idx', 0)
        
        print(f"\n{'='*60}")
        print(f"3. BEST TOPOLOGICAL: Φ/π={best['Phi_pi']}")
        print("="*60)
        
        H_b = lambda kx,ky: H_kagome_flux(kx,ky,Phi=Phi_best)
        
        # Convergence
        for nk in [40, 60, 80, 100]:
            if gi == 1:
                C, g = composite_chern(H_b, nk=nk, n_occ=2)
            else:
                C = single_chern(H_b, nk=nk, band_idx=0)
                _, gaps, _, _ = band_info(H_b, nk=nk)
                g = gaps[gi]
            print(f"  nk={nk:3d}: C={C:+.6f}  gap={g:.4f}")
        
        # Physical scales
        t_phys = 100.0  # meV
        gap_meV = best['gap'] * t_phys
        print(f"\n  Physical (t={t_phys} meV):")
        print(f"    Gap: {gap_meV:.1f} meV")
        print(f"    {'≥ 25 meV ✅ WIN' if gap_meV >= 25 else '< 25 meV — need larger flux or t'}")
        
        results['best_topo'] = best
        
        # Disorder
        print(f"\n{'='*60}")
        print(f"4. DISORDER (C5)")
        print("="*60)
        
        W_d_vals = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30]
        dis = disorder_test(H_b, nk=30, W_d_values=W_d_vals, n_samples=30, gap_idx=gi)
        for d in dis:
            meV = d['W_d'] * t_phys
            gmeV = d['gap_mean'] * t_phys
            ok = "✅" if d['gap_mean'] > 0.05 else "❌"
            print(f"  W_d={meV:5.1f} meV  gap={gmeV:6.1f} ± {d['gap_std']*t_phys:.1f} meV  "
                  f"min={d['gap_min']*t_phys:.1f}  {ok}")
        results['disorder'] = dis
    else:
        print(f"\n  No topological candidate found — checking if ANY gap exists with structure...")
    
    elapsed = time.time() - t0
    results['elapsed_s'] = round(elapsed, 1)
    print(f"\n{'='*60}")
    print(f"COMPLETE in {elapsed:.1f}s")
    print("="*60)
    
    with open('kagome_uniform_flux.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("→ kagome_uniform_flux.json")


if __name__ == '__main__':
    main()
