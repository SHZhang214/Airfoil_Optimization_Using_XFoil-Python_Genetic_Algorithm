import subprocess
import os
import numpy as np
import matplotlib.pyplot as plt

def deform_airfoil(params, output_file='deformed.dat', base_file='naca64210.dat', xfoil_dir=r"D:\XFOIL6.99"):
    """Deform NACA 64-210 using Bernstein polynomials"""
    os.chdir(xfoil_dir)
    
    base = np.loadtxt(base_file, skiprows=1)
    x = base[:, 0]
    y_base = base[:, 1]
    
    u = (x - x.min()) / (x.max() - x.min())
    
    deformation = np.zeros_like(u)
    coeffs = [1, 9, 36, 84, 126, 126, 84, 36, 9, 1]
    for i, p in enumerate(params):
        deformation += p * coeffs[i] * (u**i) * ((1-u)**(9-i))
    
    # Increased deformation amplitude for visible changes
    y_new = y_base + deformation * 0.04
    
    coords = np.column_stack([x, y_new])
    np.savetxt(output_file, coords, fmt='%.6f', header='Deformed', comments='')
    return output_file

counter = 0  # Global counter

def objective(params, mach=0.6):
    global counter
    counter += 1
    os.chdir(r"D:\XFOIL6.99")
    
    airfoil_file = f"deformed_{counter}.dat"
    polar_file = f"polar_{counter}.txt"
    
    deform_airfoil(params, output_file=airfoil_file)
    
    if not os.path.exists(airfoil_file):
        return 1e6
    
    alpha = 3.0
    script = f"""
LOAD {airfoil_file}
OPER
VISC 3000000
MACH {mach}
PACC
{polar_file}
polar3
ALFA {alpha}
QUIT
"""
    with open('script_temp.txt', 'w') as f:
        f.write(script)
    
    subprocess.run('xfoil.exe < script_temp.txt', shell=True, capture_output=True)
    
    cl, cd = None, None
    if os.path.exists(polar_file):
        with open(polar_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 7:
                    try:
                        float(parts[0])
                        cl = float(parts[1])
                        cd = float(parts[2])
                        break
                    except:
                        pass
    
    # Cleanup
    for f in [airfoil_file, polar_file, 'script_temp.txt']:
        try: os.remove(f)
        except: pass
    
    if cl is None or cd is None or cd <= 0:
        return 1e6

    baseline_cd = 0.00822
    baseline_ld = 0.6342 / 0.00822
    
    # CD must be lower than baseline
    if cd >= baseline_cd:
        return 1e6 + (cd - baseline_cd) * 10000  # CD penalty
    
    ld_ratio = cl / cd
    # LD must be higher than baseline
    if ld_ratio <= baseline_ld:
        return 1e6 + (baseline_ld - ld_ratio) * 10000  # LD penalty
    return -ld_ratio

def genetic_algorithm(pop_size=20, generations=30):
    np.random.seed(42)  # Fixed seed for reproducibility
    dim = 10
    bounds = (-0.05, 0.05)
    
    pop = np.random.uniform(bounds[0], bounds[1], (pop_size, dim))
    best_history = []
    best_params = None
    best_score = float('inf')
    
    for gen in range(generations):
        scores = np.array([objective(ind) for ind in pop], dtype=float)
        
        min_idx = np.argmin(scores)
        if scores[min_idx] < best_score:
            best_score = scores[min_idx]
            best_params = pop[min_idx].copy()
        best_history.append(best_score)
        
        print(f"Gen {gen+1}: Best = {best_score:.6f}")
        
        # Selection
        new_pop = []
        for _ in range(pop_size):
            i, j = np.random.choice(pop_size, 2, replace=False)
            parent = pop[i] if scores[i] < scores[j] else pop[j]
            new_pop.append(parent.copy())
        new_pop = np.array(new_pop)
        
        # Crossover
        for i in range(0, pop_size, 2):
            if i+1 < pop_size:
                mask = np.random.rand(dim) < 0.5
                child1, child2 = new_pop[i].copy(), new_pop[i+1].copy()
                child1[mask], child2[mask] = new_pop[i+1][mask], new_pop[i][mask]
                new_pop[i], new_pop[i+1] = child1, child2
        
        # Mutation
        for i in range(pop_size):
            if np.random.rand() < 0.1:
                new_pop[i] += np.random.normal(0, 0.02, dim)
                new_pop[i] = np.clip(new_pop[i], bounds[0], bounds[1])
        
        pop = new_pop
    
    return best_params, best_score, best_history


def plot_optimized_airfoil(best_params, base_file='naca64210.dat', xfoil_dir=r"D:\XFOIL6.99"):
    os.chdir(xfoil_dir)
    
    # Generate optimized airfoil
    deform_airfoil(best_params, output_file='optimized.dat')
    
    # Load both airfoils
    base = np.loadtxt(base_file, skiprows=1)
    opt = np.loadtxt('optimized.dat', skiprows=1)
    
    plt.figure(figsize=(14, 5))
    
    # Full view
    plt.subplot(1, 3, 1)
    plt.plot(base[:, 0], base[:, 1], 'b--', label='Baseline', linewidth=2)
    plt.plot(opt[:, 0], opt[:, 1], 'r-', label='Optimized', linewidth=2)
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.xlabel('x/c')
    plt.ylabel('y/c')
    plt.legend()
    plt.title('Full View')
    
    # Zoomed upper surface
    plt.subplot(1, 3, 2)
    plt.plot(base[:, 0], base[:, 1], 'b--', label='Baseline', linewidth=2)
    plt.plot(opt[:, 0], opt[:, 1], 'r-', label='Optimized', linewidth=2)
    plt.ylim(-0.01, 0.07)
    plt.xlim(0, 0.5)
    plt.grid(True, alpha=0.3)
    plt.xlabel('x/c')
    plt.ylabel('y/c')
    plt.title('Zoomed (Upper Surface)')
    
    # Difference plot
    plt.subplot(1, 3, 3)
    diff = opt[:, 1] - base[:, 1]
    plt.plot(base[:, 0], diff * 1000, 'g-', linewidth=2)  # mm
    plt.grid(True, alpha=0.3)
    plt.xlabel('x/c')
    plt.ylabel('Δy (mm)')
    plt.title('Shape Difference')
    
    plt.tight_layout()
    plt.show()
    
    # Print stats
    print(f"\nShape changes:")
    print(f"Max thickness change: {(diff.max() - diff.min()) * 1000:.2f} mm")
    print(f"Max camber change: {diff.max() * 1000:.2f} mm")

if __name__ == "__main__":
    best_params, best_score, history = genetic_algorithm(pop_size=15, generations=20)
    print(f"Best lift-drag ratio: {best_score:.6f}")
    plot_optimized_airfoil(best_params)
    deform_airfoil(best_params, output_file='optimized_final.dat')