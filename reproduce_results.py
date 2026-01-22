import numpy as np
import pandas as pd
from dataclasses import dataclass

# --- 1. Definicja Solvera (Model Fizyczny) ---
@dataclass
class PhysicsParams:
    m: float = 1.0      # Masa
    c: float = 0.8      # Tłumienie
    k: float = 1.2      # Sztywność
    
    def damping_ratio(self) -> float:
        return self.c / (2 * np.sqrt(self.m * self.k))

class VectorizedLigandSolver:
    DTYPE = np.float64
    
    def __init__(self, target_vector, weights, params=None, guidance_alpha=1.0):
        self.target_vector = np.asarray(target_vector, dtype=self.DTYPE)
        self.weights = np.asarray(weights, dtype=self.DTYPE)
        self.guidance_alpha = float(guidance_alpha)
        self.params = params if params is not None else PhysicsParams()
    
    def normalize_library(self, ligand_matrix):
        ligand_matrix = np.asarray(ligand_matrix, dtype=self.DTYPE)
        # Zabezpieczenie przed dzieleniem przez zero
        target_safe = np.maximum(np.abs(self.target_vector), 1e-12)
        normalized_ratios = ligand_matrix / target_safe
        valid_mask = np.ones(ligand_matrix.shape[0], dtype=bool)
        return normalized_ratios, valid_mask
    
    def rank_ligands(self, ligand_matrix, **solver_kwargs):
        normalized_ratios, valid_mask = self.normalize_library(ligand_matrix)
        valid_indices = np.where(valid_mask)[0]
        valid_ratios = normalized_ratios[valid_mask]
        
        # Parametry czasu
        max_time = solver_kwargs.get('max_time', 10.0)
        num_points = solver_kwargs.get('num_points', 200)
        time_points = np.linspace(0, max_time, num_points, dtype=self.DTYPE)
        
        # Parametry fizyczne
        zeta = self.params.damping_ratio()
        ω_n = np.sqrt(self.params.k / self.params.m)
        
        # Obliczenie obwiedni f(t) - analityczne rozwiązanie ODE
        if zeta < 1.0:
            σ = zeta * ω_n
            ω_d = ω_n * np.sqrt(1 - zeta**2)
            f_t = np.exp(-σ * time_points) * (np.cos(ω_d * time_points) + (σ/ω_d) * np.sin(ω_d * time_points))
        elif np.isclose(zeta, 1.0):
            f_t = (1 + ω_n * time_points) * np.exp(-ω_n * time_points)
        else:
            term = np.sqrt(zeta**2 - 1.0)
            r1 = -ω_n * (zeta - term)
            r2 = -ω_n * (zeta + term)
            f_t = (r2 * np.exp(r1 * time_points) - r1 * np.exp(r2 * time_points)) / (r2 - r1)
            
        # Bezpieczna amplituda
        F_t = np.maximum(f_t ** 2, 0.0)
        f_t_safe = np.sqrt(F_t)
        
        # Obliczenie metryk
        diff = (valid_ratios - 1.0)
        A_vals = self.guidance_alpha ** 2 * np.sum(self.weights * (diff ** 2), axis=1)
        sqrt_A = np.sqrt(A_vals)
        
        final_dist_vals = sqrt_A * f_t_safe[-1]
        
        # Tworzenie DataFrame z wynikami
        metrics = pd.DataFrame({
            "Final_Distance": final_dist_vals,
            "Original_Index": valid_indices
        })
        
        # Sortowanie i Ranking
        metrics_sorted = metrics.sort_values("Final_Distance", ascending=True).reset_index(drop=True)
        metrics_sorted["Rank"] = np.arange(1, len(metrics_sorted) + 1)
        
        return {
            "metrics": metrics_sorted,
            "ligand_matrix": ligand_matrix[valid_mask]
        }

# --- 2. Generowanie Danych (N=10,000) ---
print("Generowanie danych syntetycznych...")
np.random.seed(42) # Klucz do powtarzalności wyników
N_LIGANDS = 10000

pIC50 = np.random.uniform(4.0, 8.0, N_LIGANDS)
Selectivity = np.random.uniform(20, 200, N_LIGANDS)
LogP = np.random.uniform(1.0, 4.0, N_LIGANDS)
MW = np.random.uniform(300, 400, N_LIGANDS)
TPSA = np.random.uniform(50, 120, N_LIGANDS)
CLint = np.random.uniform(0.005, 0.1, N_LIGANDS)

ligand_ids = [f"L{i:05d}" for i in range(1, N_LIGANDS + 1)]

# DataFrame bazowy
df_data = pd.DataFrame({
    'Ligand_ID': ligand_ids,
    'pIC50': pIC50,
    'Selectivity': Selectivity,
    'logP': LogP,
    'MW': MW,
    'TPSA': TPSA,
    'CLint': CLint
})

# --- 3. Uruchomienie Obliczeń ---
print("Uruchamianie solvera Information Force...")
ligand_matrix = np.column_stack([
    pIC50, Selectivity, LogP, MW/100.0, TPSA/100.0, CLint*10.0
])

target_vector = np.array([7.0, 150.0, 2.8, 3.5, 0.9, 0.1])
weights = np.array([1.0, 1.0, 0.5, 0.3, 0.5, 0.4])

solver = VectorizedLigandSolver(target_vector, weights)
results = solver.rank_ligands(ligand_matrix)
metrics = results['metrics']

# --- 4. Obliczenie Heurystyki (MRD) ---
normalized, _ = solver.normalize_library(ligand_matrix)
mrd_all = np.mean(np.abs(normalized - 1.0), axis=1)

# --- 5. Scalanie Wyników ---
# Dopasowanie wyników z powrotem do oryginalnych ID ligandów
metrics_aligned = metrics.set_index('Original_Index').sort_index()

df_final = df_data.copy()
df_final['MRD'] = mrd_all
df_final['Final_Distance'] = metrics_aligned['Final_Distance']
df_final['Rank'] = metrics_aligned['Rank']

# --- 6. Zapis do CSV ---
output_filename = 'ligands_10k_supplementary.csv'
df_final.to_csv(output_filename, index=False, sep=',')

print(f"Sukces! Plik '{output_filename}' został wygenerowany.")
print(f"Zawiera {len(df_final)} wierszy.")
print("Pierwsze 5 wierszy:")
print(df_final.head())
