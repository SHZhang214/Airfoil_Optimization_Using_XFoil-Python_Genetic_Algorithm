import subprocess
import os
import numpy as np
import matplotlib.pyplot as plt

def cst_airfoil(params, n=60):
    # Cluster points near leading edge (more density at front)
    beta = 0.5
    u = np.linspace(0, 1, n)
    u = u**(1/beta)
    
    C = u**0.5 * (1-u)
    S = np.zeros_like(u)
    coeffs = [1, 9, 36, 84, 126, 126, 84, 36, 9, 1]
    for i, p in enumerate(params[:10]):
        S += p * coeffs[i] * (u**i) * ((1-u)**(9-i))
    
    thickness = 0.08 + params[10] * 0.06
    camber = 0.01 + params[11] * 0.03
    
    # Thickness with rounded leading edge
    thickness_dist = thickness * (C + u * 0.005)
    camber_line = camber * (1 - np.cos(2 * np.pi * u)) / 2
    
    zu = thickness_dist + camber_line
    zl = -thickness_dist + camber_line
    
    # Smooth
    from scipy.ndimage import uniform_filter1d
    zu = uniform_filter1d(zu, size=3, mode='nearest')
    zl = uniform_filter1d(zl, size=3, mode='nearest')
    
    x = np.concatenate([np.flip(u[:-1]), u[1:]])
    y = np.concatenate([np.flip(zu[:-1]), zl[1:]])
    coords = np.column_stack([x, y])
    return coords

def save_airfoil(coords, filename='cst_airfoil.dat', xfoil_dir=r"D:\XFOIL6.99"):
    """Save coordinates to DAT file"""
    os.chdir(xfoil_dir)
    np.savetxt(filename, coords, fmt='%.6f', header='CST Airfoil', comments='')
    return filename

counter = 0

def objective(params, mach=0.6):
    """
    params: 12 variables
    - params[0:10]: CST shape coefficients
    - params[10]: thickness (0.05~0.15)
    - params[11]: camber (0~0.04)
    """
    global counter
    counter += 1
    os.chdir(r"D:\XFOIL6.99")
    
    # Generate airfoil from CST
    coords = cst_airfoil(params)
    airfoil_file = f"cst_{counter}.dat"
    polar_file = f"polar_{counter}.txt"
    
    np.savetxt(airfoil_file, coords, fmt='%.6f', header='CST Airfoil', comments='')
    
    # Run XFoil at alpha=3°
    alpha = 3.0
    script = f"""
LOAD {airfoil_file}
OPER
VISC 3000000
MACH {mach}
PACC
{polar_file}
polar4
ALFA {alpha}
QUIT
"""
    with open('script_temp.txt', 'w') as f:
        f.write(script)
    
    subprocess.run('xfoil.exe < script_temp.txt', shell=True, capture_output=True)
    
    # Parse results
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

    # Realistic for real engineering
    thickness = coords[:,1].max() - coords[:,1].min()
    if thickness < 0.06 or thickness > 0.15:
        return 1e6
    
    # Objective: maximize L/D
    ld_ratio = cl / cd
    return -ld_ratio

def genetic_algorithm(pop_size=40, generations=50):
    np.random.seed(42)
    
    dim = 12
    bounds = [
        (-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2),
        (-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2),
        (-0.2, 0.2), (-0.2, 0.2), (0.0, 1.0), (0.0, 1.0)
    ]
    
    # Initialize population correctly: (pop_size, dim)
    pop = np.array([
        np.random.uniform(bounds[i][0], bounds[i][1], pop_size) 
        for i in range(dim)
    ]).T  # Transpose to get (pop_size, dim)
    
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
        
        print(f"Gen {gen+1}: Best L/D = {-best_score:.2f}")
        
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
            if np.random.rand() < 0.15:
                idx = np.random.randint(dim)
                new_pop[i][idx] += np.random.normal(0, 0.02)
                new_pop[i][idx] = np.clip(new_pop[i][idx], bounds[idx][0], bounds[idx][1])
        
        pop = new_pop
    
    return best_params, -best_score, best_history

if __name__ == "__main__":
    print("Starting CST-based optimization")
    
    best_params, best_ld, history = genetic_algorithm(pop_size=30, generations=50)
    
    print("\n" + "=" * 50)
    print(f"Best L/D: {best_ld:.2f}")
    print(f"Best params: {best_params}")
    
    # Generate final airfoil
    coords = cst_airfoil(best_params)
    save_airfoil(coords, 'cst_optimized.dat')
    print("Saved to cst_optimized.dat")
    
    # Plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(coords[:, 0], coords[:, 1], 'b-', linewidth=2)
    plt.axis('equal')
    plt.grid(True)
    plt.xlabel('x/c')
    plt.ylabel('y/c')
    plt.title('Optimized CST Airfoil')
    
    plt.subplot(1, 2, 2)
    plt.plot(history, 'r-', linewidth=2)
    plt.xlabel('Generation')
    plt.ylabel('Best L/D')
    plt.grid(True)
    plt.title('Convergence History')
    
    plt.tight_layout()
    plt.show()