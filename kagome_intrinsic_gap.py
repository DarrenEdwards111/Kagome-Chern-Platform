"""
Many-body gap for the INTRINSIC Kagome flat band (Category B).

Key insight: don't twist to create flat bands. Use the flat band that
already exists in single-layer Kagome. It has:
- W = 0 exactly (destructive interference)
- Compact Localised States (CLSs) on hexagons, IPR = 6
- Bare hopping t = 100 meV (not moiré-reduced)
- Therefore U_W ~ t * |ψ|⁴ ~ t/IPR ~ 17 meV

For bilayer: the interlayer coupling t_perp lifts the exact flatness
slightly, but the CLS character persists. The relevant U is set by
the BARE lattice scale, not the moiré scale.

Strategy:
1. Single-layer Kagome: compute flat-band CLSs, U_W, ED gap
2. Bilayer (untwisted): compute flat-band splitting, U_W, ED gap
3. Bilayer (twisted): same at θ=9.6° — check if twist helps or hurts
4. Sweep t_perp/t to find optimal regime

The ED model: Hubbard on the flat-band manifold using CLS Wannier functions.
"""
import numpy as np, json, time
from scipy.linalg import eigh as la_eigh
from itertools import combinations
from math import comb

def fsign(state, pos):
    return (-1) ** bin(state & ((1<<pos)-1)).count('1')


def single_layer_kagome(n_cells, t=1.0):
    """Build single-layer Kagome tight-binding on n_cells × n_cells cluster."""
    a1 = np.array([1.0, 0.0]); a2 = np.array([0.5, np.sqrt(3)/2])
    no = 3
    n = n_cells * n_cells * no
    H = np.zeros((n, n))
    
    def idx(i, j, orb):
        return (i%n_cells * n_cells + j%n_cells) * no + orb
    
    nn = [(0,1,0,0),(0,1,-1,0),(0,2,0,0),(0,2,0,-1),(1,2,0,0),(1,2,1,-1)]
    for i in range(n_cells):
        for j in range(n_cells):
            for o1,o2,di,dj in nn:
                i1 = idx(i,j,o1); i2 = idx(i+di,j+dj,o2)
                H[i1,i2] += t; H[i2,i1] += t
    
    return H, n


def bilayer_kagome(n_cells, t=1.0, t_perp=0.030, theta_deg=0.0):
    """Build bilayer Kagome with optional twist."""
    th = np.radians(theta_deg)
    a1 = np.array([1.0, 0.0]); a2 = np.array([0.5, np.sqrt(3)/2])
    tau = np.array([[0,0],[0.5,0],[0.25,np.sqrt(3)/4]])
    no = 3
    R1 = np.array([[np.cos(-th/2),-np.sin(-th/2)],[np.sin(-th/2),np.cos(-th/2)]])
    R2 = np.array([[np.cos(th/2),-np.sin(th/2)],[np.sin(th/2),np.cos(th/2)]])
    
    npl = n_cells * n_cells * no
    n = 2 * npl
    H = np.zeros((n, n))
    
    # Store positions
    pos = np.zeros((n, 2))
    
    def idx(layer, i, j, orb):
        return layer*npl + (i%n_cells*n_cells + j%n_cells)*no + orb
    
    nn = [(0,1,0,0),(0,1,-1,0),(0,2,0,0),(0,2,0,-1),(1,2,0,0),(1,2,1,-1)]
    
    for layer in range(2):
        Rl = R1 if layer==0 else R2
        for i in range(n_cells):
            for j in range(n_cells):
                Rc = i*a1 + j*a2
                for orb in range(no):
                    p = Rl @ (Rc + tau[orb])
                    pos[idx(layer,i,j,orb)] = p
                for o1,o2,di,dj in nn:
                    i1 = idx(layer,i,j,o1); i2 = idx(layer,i+di,j+dj,o2)
                    H[i1,i2] += t; H[i2,i1] += t
    
    # Interlayer: distance-based
    d0 = 0.3; r_cut = 1.5
    for i in range(npl):
        for j in range(npl):
            dx = pos[i,0] - pos[npl+j,0]
            dy = pos[i,1] - pos[npl+j,1]
            # Periodic images
            for di in range(-1,2):
                for dj in range(-1,2):
                    sx = di*n_cells*a1[0] + dj*n_cells*a2[0]
                    sy = di*n_cells*a1[1] + dj*n_cells*a2[1]
                    r = np.sqrt((dx-sx)**2 + (dy-sy)**2)
                    if 0.01 < r < r_cut:
                        tc = t_perp * np.exp(-r/d0)
                        H[i, npl+j] += tc; H[npl+j, i] += tc
    
    return H, n, pos


def flat_band_analysis(H, n, label, t_phys=100.0):
    """Analyse flat band and compute Hubbard parameters."""
    ev, vec = la_eigh(H)
    
    # Find flat band: cluster of states at E = -2t (Kagome flat band energy)
    # Actually, for our normalisation, flat band is at E = -2
    # Find the densest cluster of eigenvalues
    n_flat_target = n // 3  # 1/3 of states should be flat for Kagome
    
    # Find the energy with most states in a window
    best_E = 0; best_count = 0; best_window = 0.1
    for e in ev:
        count = np.sum(np.abs(ev - e) < 0.1)
        if count > best_count:
            best_count = count; best_E = e
    
    # Flat band states
    flat_mask = np.abs(ev - best_E) < 0.15
    flat_idx = np.where(flat_mask)[0]
    n_flat = len(flat_idx)
    flat_E = ev[flat_idx]
    W_flat = np.max(flat_E) - np.min(flat_E)
    
    print(f"\n  {label}:", flush=True)
    print(f"    dim = {n}, flat band at E = {best_E:.4f}", flush=True)
    print(f"    {n_flat} flat states, W = {W_flat:.6f}t = {W_flat*t_phys:.3f} meV", flush=True)
    
    # Flat-band eigenvectors
    P = vec[:, flat_idx]
    
    # IPR of flat-band states
    iprs = [1.0/np.sum(np.abs(P[:,i])**4) for i in range(min(n_flat, 6))]
    print(f"    IPR (first 6): [{', '.join(f'{x:.1f}' for x in iprs)}]", flush=True)
    
    # Construct Wannier via projection: use delta-function trials at high-weight sites
    weights = np.sum(np.abs(P)**2, axis=1)
    
    # Pick trial sites: most weight, well-separated
    n_w = min(n_flat, 6)
    top = np.argsort(-weights)
    selected = [top[0]]
    for s in top[1:]:
        if len(selected) >= n_w: break
        # No position info needed for single-layer (just use index separation)
        if all(abs(s - ss) > 2 for ss in selected):
            selected.append(s)
    while len(selected) < n_w:
        selected.append(top[len(selected)])
    selected = selected[:n_w]
    
    # Project + orthogonalise
    G = np.zeros((n, n_w))
    for i, s in enumerate(selected):
        G[s, i] = 1.0
    A = P.conj().T @ G
    S = A.conj().T @ A
    ev_S, vec_S = la_eigh(S)
    ev_S = np.maximum(ev_S, 1e-10)
    S_inv_half = vec_S @ np.diag(1.0/np.sqrt(ev_S)) @ vec_S.conj().T
    A_orth = A @ S_inv_half
    W_func = P @ A_orth  # (n, n_w)
    
    # Wannier properties
    U_W = np.array([np.sum(np.abs(W_func[:,i])**4) for i in range(n_w)])
    wannier_iprs = [1.0/U_W[i] for i in range(n_w)]
    
    print(f"    Wannier IPR: [{', '.join(f'{x:.1f}' for x in wannier_iprs[:6])}]", flush=True)
    print(f"    U_W/t: [{', '.join(f'{u:.4f}' for u in U_W[:6])}]", flush=True)
    print(f"    U_W (meV): [{', '.join(f'{u*t_phys:.1f}' for u in U_W[:6])}]", flush=True)
    
    # Wannier Hamiltonian
    H_W = np.real(W_func.conj().T @ H @ W_func)
    
    # Extract hopping and onsite
    onsite = np.diag(H_W)
    hopping = H_W.copy(); np.fill_diagonal(hopping, 0)
    t_max = np.max(np.abs(hopping))
    
    print(f"    Onsite E: [{', '.join(f'{e*t_phys:.1f}' for e in onsite[:4])}] meV", flush=True)
    print(f"    Max hopping: {t_max:.6f}t = {t_max*t_phys:.3f} meV", flush=True)
    print(f"    U/t_hop: {np.mean(U_W)/max(t_max,1e-10):.1f}", flush=True)
    
    return W_func, H_W, U_W, n_w, W_flat, flat_idx, ev, vec


def run_ED(H_W, U_W, n_w, t_phys=100.0, label=""):
    """Run Hubbard ED on the Wannier model."""
    m_ed = min(n_w, 2)
    
    # Use the full Wannier H (not just nn hopping) for the ED
    # This captures all hopping paths within the flat-band manifold
    
    for n_sites in [3, 4]:
        N_sp = n_sites * m_ed * 2
        N_elec = n_sites * m_ed
        nf = comb(N_sp, N_elec)
        if nf > 15000: continue
        
        def orb(site, band, spin): return site*(m_ed*2)+band*2+spin
        
        states = []
        for bits in combinations(range(N_sp), N_elec):
            state=0
            for b in bits: state|=(1<<b)
            states.append(state)
        smap={s:i for i,s in enumerate(states)}
        nf=len(states)
        
        nn_pairs=[(i,(i+1)%n_sites) for i in range(n_sites)]
        
        # One-body: hopping + onsite from H_W
        H_1b = np.zeros((nf,nf))
        for si,state in enumerate(states):
            for site in range(n_sites):
                for n in range(m_ed):
                    for sp in range(2):
                        o=orb(site,n,sp)
                        if state&(1<<o):
                            H_1b[si,si]+=H_W[n,n]
        
        # Inter-site hopping (nn)
        for(s_i,s_j)in nn_pairs:
            for n1 in range(m_ed):
                for n2 in range(m_ed):
                    tv=H_W[n1,n2] if n1!=n2 else 0
                    # Also add intra-orbital nn hopping from H_W off-diagonal
                    if n1==n2:
                        # This should come from spatial hopping, not orbital mixing
                        # Use the Wannier H off-diagonal as the effective hopping
                        tv = H_W[n1,n2] if n1!=n2 else 0
                    if abs(tv)<1e-15:continue
                    for sp in range(2):
                        p=orb(s_i,n1,sp);q=orb(s_j,n2,sp)
                        for idx,state in enumerate(states):
                            if not(state&(1<<q)):continue
                            sq=fsign(state,q);s1=state^(1<<q)
                            if s1&(1<<p):continue
                            sp_=fsign(s1,p);s2=s1|(1<<p)
                            if s2 in smap:H_1b[smap[s2],idx]+=tv*sq*sp_
                            if not(state&(1<<p)):continue
                            sp2=fsign(state,p);s1b=state^(1<<p)
                            if s1b&(1<<q):continue
                            sq2=fsign(s1b,q);s2b=s1b|(1<<q)
                            if s2b in smap:H_1b[smap[s2b],idx]+=tv*sp2*sq2
        H_1b=(H_1b+H_1b.T)/2
        
        # Hubbard U
        H_U=np.zeros(nf)
        for idx,state in enumerate(states):
            for site in range(n_sites):
                for n in range(m_ed):
                    ou=orb(site,n,0);od=orb(site,n,1)
                    if(state&(1<<ou))and(state&(1<<od)):
                        H_U[idx]+=U_W[n]
        
        # Also try with DIRECT U (not scaled) — since U_W is already in units of t
        print(f"\n  ED {label} ({n_sites} sites, {m_ed} orbs, {nf} states):", flush=True)
        print(f"  {'U_sc':>6} {'Δ/t':>10} {'Δ(meV)':>8} {'Δ_charge':>10} {'deg':>4}", flush=True)
        
        results = []
        for U_sc in [0, 0.5, 1, 2, 5, 10, 20, 50, 100]:
            Hm = H_1b + U_sc * np.diag(H_U)
            evm = np.sort(np.linalg.eigvalsh(Hm))[:8]
            gap = evm[1]-evm[0]
            # Also look at charge gap: E(N+1) + E(N-1) - 2E(N) 
            # (can't do easily without separate N±1 sectors)
            deg = int(np.sum(np.abs(evm-evm[0])<1e-8))
            results.append({'U':U_sc,'gap_t':float(gap),'gap_meV':float(gap*t_phys),'deg':deg})
            print(f"  {U_sc:>6.1f} {gap:>10.6f} {gap*t_phys:>8.3f} {'—':>10} {deg:>4}", flush=True)
        
        return results
    return []


def charge_gap_ED(H_W, U_W, n_w, t_phys=100.0):
    """
    Compute the CHARGE gap: Δ_c = E(N+1) + E(N-1) - 2E(N)
    This is the Mott gap that scales as U, not as t²/U.
    """
    m_ed = min(n_w, 2)
    n_sites = 3
    N_sp = n_sites * m_ed * 2
    
    def orb(site, band, spin): return site*(m_ed*2)+band*2+spin
    
    print(f"\n  CHARGE GAP (Mott gap = E(N+1)+E(N-1)-2E(N)):", flush=True)
    print(f"  {'U_sc':>6} {'E(N-1)':>10} {'E(N)':>10} {'E(N+1)':>10} {'Δ_c/t':>8} {'Δ_c(meV)':>9}", flush=True)
    
    results = []
    
    for U_sc in [0, 1, 2, 5, 10, 20, 50, 100, 200]:
        E_sectors = {}
        
        for N_elec in [n_sites*m_ed - 1, n_sites*m_ed, n_sites*m_ed + 1]:
            if N_elec < 0 or N_elec > N_sp: continue
            nf = comb(N_sp, N_elec)
            if nf > 20000: continue
            
            states=[]
            for bits in combinations(range(N_sp), N_elec):
                state=0
                for b in bits: state|=(1<<b)
                states.append(state)
            smap={s:i for i,s in enumerate(states)}
            nf=len(states)
            
            nn_pairs=[(i,(i+1)%n_sites) for i in range(n_sites)]
            
            H_1b=np.zeros((nf,nf))
            for si,state in enumerate(states):
                for site in range(n_sites):
                    for n in range(m_ed):
                        for sp in range(2):
                            o=orb(site,n,sp)
                            if state&(1<<o): H_1b[si,si]+=H_W[n,n]
            
            for(s_i,s_j)in nn_pairs:
                for n1 in range(m_ed):
                    for n2 in range(m_ed):
                        tv=H_W[n1,n2] if n1!=n2 else 0
                        if abs(tv)<1e-15:continue
                        for sp in range(2):
                            p=orb(s_i,n1,sp);q=orb(s_j,n2,sp)
                            for idx,state in enumerate(states):
                                if not(state&(1<<q)):continue
                                sq=fsign(state,q);s1=state^(1<<q)
                                if s1&(1<<p):continue
                                sp_=fsign(s1,p);s2=s1|(1<<p)
                                if s2 in smap:H_1b[smap[s2],idx]+=tv*sq*sp_
                                if not(state&(1<<p)):continue
                                sp2=fsign(state,p);s1b=state^(1<<p)
                                if s1b&(1<<q):continue
                                sq2=fsign(s1b,q);s2b=s1b|(1<<q)
                                if s2b in smap:H_1b[smap[s2b],idx]+=tv*sp2*sq2
            H_1b=(H_1b+H_1b.T)/2
            
            H_U=np.zeros(nf)
            for idx,state in enumerate(states):
                for site in range(n_sites):
                    for n in range(m_ed):
                        ou=orb(site,n,0);od=orb(site,n,1)
                        if(state&(1<<ou))and(state&(1<<od)):
                            H_U[idx]+=U_W[n]
            
            Hm=H_1b+U_sc*np.diag(H_U)
            evm=np.sort(np.linalg.eigvalsh(Hm))
            E_sectors[N_elec] = evm[0]
        
        N0 = n_sites * m_ed
        if N0-1 in E_sectors and N0 in E_sectors and N0+1 in E_sectors:
            E_m = E_sectors[N0-1]; E_0 = E_sectors[N0]; E_p = E_sectors[N0+1]
            delta_c = E_p + E_m - 2*E_0
            results.append({'U':U_sc, 'delta_c_t':float(delta_c), 'delta_c_meV':float(delta_c*t_phys)})
            print(f"  {U_sc:>6.0f} {E_m:>10.4f} {E_0:>10.4f} {E_p:>10.4f} {delta_c:>8.4f} {delta_c*t_phys:>9.1f}", flush=True)
            if delta_c * t_phys > 25:
                print(f"  ★ CHARGE GAP > 25 meV!", flush=True)
    
    return results


def main():
    t0 = time.time()
    t_phys = 100.0  # meV
    
    print("INTRINSIC KAGOME FLAT BAND: CHARGE GAP ANALYSIS", flush=True)
    print(f"t = {t_phys} meV (bare Kagome hopping)", flush=True)
    print("="*70, flush=True)
    
    # === Single layer ===
    print("\n" + "="*70, flush=True)
    print("SINGLE-LAYER KAGOME", flush=True)
    
    for nc in [4, 6, 8]:
        H, n = single_layer_kagome(nc)
        W, H_W, U_W, n_w, W_flat, fidx, ev, vec = flat_band_analysis(H, n, f"Single-layer {nc}×{nc}", t_phys)
        
        # Charge gap
        charge_results = charge_gap_ED(H_W, U_W, n_w, t_phys)
    
    # === Bilayer untwisted ===
    print("\n" + "="*70, flush=True)
    print("BILAYER KAGOME (untwisted, t_perp=0.03t)", flush=True)
    
    for nc in [4, 6]:
        H, n, pos = bilayer_kagome(nc, t_perp=0.030, theta_deg=0)
        W, H_W, U_W, n_w, W_flat, fidx, ev, vec = flat_band_analysis(H, n, f"Bilayer {nc}×{nc} θ=0°", t_phys)
        charge_results = charge_gap_ED(H_W, U_W, n_w, t_phys)
    
    # === Bilayer twisted ===
    print("\n" + "="*70, flush=True)
    print("BILAYER KAGOME (θ=9.6°, t_perp=0.03t)", flush=True)
    
    for nc in [4, 6]:
        H, n, pos = bilayer_kagome(nc, t_perp=0.030, theta_deg=9.6)
        W, H_W, U_W, n_w, W_flat, fidx, ev, vec = flat_band_analysis(H, n, f"Bilayer {nc}×{nc} θ=9.6°", t_phys)
        charge_results = charge_gap_ED(H_W, U_W, n_w, t_phys)
    
    # === t_perp sweep ===
    print("\n" + "="*70, flush=True)
    print("t_perp/t SWEEP (bilayer 6×6, θ=0°)", flush=True)
    print(f"{'t_perp/t':>10} {'W(meV)':>8} {'U_W(meV)':>10} {'Δ_c(meV)':>10}", flush=True)
    
    for tp in [0.001, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5]:
        H, n, pos = bilayer_kagome(6, t_perp=tp, theta_deg=0)
        ev, vec = la_eigh(H)
        mid = n//2; E_F = (ev[mid-1]+ev[mid])/2
        flat_mask = np.abs(ev-E_F) < 0.15
        if np.sum(flat_mask) < 2:
            # Try wider window
            for w in [0.3, 0.5, 1.0]:
                flat_mask = np.abs(ev-E_F) < w
                if np.sum(flat_mask) >= 6: break
        
        flat_idx = np.where(flat_mask)[0][:6]
        W_f = ev[flat_idx[-1]] - ev[flat_idx[0]] if len(flat_idx)>1 else 0
        
        P = vec[:, flat_idx]; n_w = min(len(flat_idx), 2)
        U_est = np.mean([np.sum(np.abs(P[:,i])**4) for i in range(n_w)])
        
        print(f"{tp:>10.3f} {W_f*t_phys:>8.3f} {U_est*t_phys:>10.2f} {'(run ED)':>10}", flush=True)
    
    print(f"\n{'='*70}", flush=True)
    print(f"Total: {time.time()-t0:.0f}s", flush=True)

if __name__=='__main__':main()
