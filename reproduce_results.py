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
import matplotlib.pyplot as plt
import seaborn as sns

print("\nGenerowanie wykresów...")

# --- 7. Figure 1: Heatmap (Parameter Sensitivity) ---
# Analiza dla najlepszego liganda (Rank 1)
best_idx = metrics_aligned[metrics_aligned['Rank'] == 1].index[0]
ratios, _ = solver.normalize_library(ligand_matrix)
best_ligand_ratios = ratios[best_idx:best_idx+1]

# Siatka parametrów
alphas = np.linspace(0.1, 5.0, 20)
ks = np.logspace(-1, 1, 20)
dist_grid = np.zeros((len(alphas), len(ks)))

# Prekalkulacja stałej części energii
diff = best_ligand_ratios - 1.0
A_base = np.sum(weights * (diff**2))

for i, a in enumerate(alphas):
    for j, k_val in enumerate(ks):
        # Parametry fizyczne dla danego punktu siatki
        p = PhysicsParams(k=k_val)
        zeta = p.damping_ratio()
        wn = np.sqrt(k_val/1.0)
        t = 10.0
        
        # Obliczenie f(t) w punkcie t=10
        if zeta < 1.0:
            sigma = zeta * wn
            wd = wn * np.sqrt(1 - zeta**2)
            ft = np.exp(-sigma * t) * (np.cos(wd * t) + (sigma/wd) * np.sin(wd * t))
        elif np.isclose(zeta, 1.0):
            ft = (1 + wn * t) * np.exp(-wn * t)
        else:
            term = np.sqrt(zeta**2 - 1.0)
            r1 = -wn * (zeta - term)
            r2 = -wn * (zeta + term)
            ft = (r2 * np.exp(r1 * t) - r1 * np.exp(r2 * t)) / (r2 - r1)
        
        f_safe = np.sqrt(max(ft**2, 0))
        dist_grid[i, j] = a * np.sqrt(A_base) * f_safe

plt.figure(figsize=(8, 6))
sns.heatmap(dist_grid, xticklabels=np.round(ks, 1), yticklabels=np.round(alphas, 1), cmap='viridis')
plt.xlabel('Stiffness k')
plt.ylabel('Guidance Alpha')
plt.title(f'Grid Search: Final Distance (Ligand {df_data.iloc[best_idx]["Ligand_ID"]})')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('fig1_heatmap.png', dpi=300)
plt.close()
print(" - Wygenerowano fig1_heatmap.png")

# --- 8. Figure 2: Scatter Plot (Top 5% Analysis) ---
# Definicja Top 5%
n_top = int(N_LIGANDS * 0.05)
metrics_sorted = metrics.sort_values("Final_Distance")
top_5_mask = metrics_sorted.index[:n_top]

# Pobranie danych do wykresu
mrd_sorted = df_final.loc[metrics_sorted['Original_Index'], 'MRD'].values
dist_sorted = metrics_sorted['Final_Distance'].values

plt.figure(figsize=(8, 6))
# Rysuj wszystkie (szare)
plt.scatter(mrd_sorted[n_top:], dist_sorted[n_top:], alpha=0.3, c='gray', s=10, label='Bottom 95%')
# Rysuj Top 5% (czerwone)
plt.scatter(mrd_sorted[:n_top], dist_sorted[:n_top], alpha=0.8, c='red', s=15, label='Top 5% (Lead Opt.)')

# Linia trendu dla całości
m, b = np.polyfit(mrd_sorted, dist_sorted, 1)
plt.plot(mrd_sorted, m*mrd_sorted + b, 'k--', lw=1, label=f'Global Trend')

plt.xlabel('Mean Ratio Deviation (MRD)')
plt.ylabel('Model Final Distance')
plt.title(f'Lead Optimization Divergence (N={N_LIGANDS})')
plt.legend()
plt.tight_layout()
plt.savefig('top5_analysis.png', dpi=300)
plt.close()
print(" - Wygenerowano top5_analysis.png")

# --- 9. Figure 3: Histogram ---
plt.figure(figsize=(8, 6))
plt.hist(df_final['Final_Distance'], bins=40, color='green', alpha=0.7, edgecolor='black')
plt.xlabel('Final Weighted Distance')
plt.ylabel('Frequency')
plt.title('Distribution of Scores')
plt.tight_layout()
plt.savefig('fig3_hist.png', dpi=300)
plt.close()
print(" - Wygenerowano fig3_hist.png")

# --- 10. Figure 4: Weight Sensitivity ---
# Wybór reprezentatywnych ligandów (Top 1, Rank 2500, 5000, 7500)
indices_to_test = [0, 2499, 4999, 7499]
sel_orig_indices = metrics_sorted.iloc[indices_to_test]['Original_Index'].values
sel_ligand_ids = df_data.iloc[sel_orig_indices]['Ligand_ID'].values

w_schemes = {
    'Default': np.array([1.0, 1.0, 0.5, 0.3, 0.5, 0.4]),
    'Equal': np.ones(6),
    'Potency': np.array([2.0, 1.0, 0.5, 0.3, 0.5, 0.4])
}

res_sensitivity = []
sub_matrix = ligand_matrix[sel_orig_indices]

# Pętla po schematach wag
for name, w in w_schemes.items():
    s_temp = VectorizedLigandSolver(target_vector, w)
    norm_temp, _ = s_temp.normalize_library(sub_matrix)
    diff_temp = norm_temp - 1.0
    A_temp = np.sum(w * diff_temp**2, axis=1) # alpha=1
    
    # Obliczenie f(t) dla domyślnych parametrów (k=1.2, m=1, c=0.8)
    # (Używamy wartości z pętli głównej lub przeliczamy raz jeszcze dla pewności)
    wn = np.sqrt(1.2); z = 0.8/(2*wn); sigma=z*wn; wd=wn*np.sqrt(1-z**2)
    ft_val = np.exp(-sigma*10)*(np.cos(wd*10)+(sigma/wd)*np.sin(wd*10))
    ft_safe = np.sqrt(max(ft_val**2, 0))
    
    dists = np.sqrt(A_temp) * ft_safe
    
    for i, dist in enumerate(dists):
        res_sensitivity.append({
            'Ligand': sel_ligand_ids[i],
            'Scheme': name,
            'Distance': dist
        })

df_sens = pd.DataFrame(res_sensitivity)

plt.figure(figsize=(10, 6))
sns.barplot(data=df_sens, x='Ligand', y='Distance', hue='Scheme')
plt.title('Sensitivity to Weight Schemes')
plt.tight_layout()
plt.savefig('fig4_sensitivity.png', dpi=300)
plt.close()
print(" - Wygenerowano fig4_sensitivity.png")
print("\nGotowe! Wszystkie pliki zostały utworzone.")
