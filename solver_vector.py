@dataclass
class PhysicsParams:
    """ODE physics parameters with validation."""
    m: float = 1.0      # Inertia (mass)
    c: float = 0.8      # Damping coefficient
    k: float = 1.2      # Stiffness
    
    def __post_init__(self):
        # Validate physics parameters
        if self.m <= 0:
            raise ValueError(f"Mass m must be positive, got {self.m}")
        if self.c < 0:
            raise ValueError(f"Damping c must be non-negative, got {self.c}")
        if self.k <= 0:
            raise ValueError(f"Stiffness k must be positive, got {self.k}")
    
    def damping_ratio(self) -> float:
        """Compute damping ratio ζ = c / (2√(m k))."""
        return self.c / (2 * np.sqrt(self.m * self.k))

class VectorizedLigandSolver:
    """
    Production-grade vectorized solver for Information Force Framework.
    Implements:
    - GUARD 0: Input normalization (divide by zero protection)
    - GUARD 1: Trajectory calculation (overflow/underflow detection)
    - GUARD 2a/2b/2c: Metrics computation (trajectory validation + output checks)
    """
    DTYPE = np.float64
    EPSILON_DIV_ZERO = 1e-12
    
    def __init__(self, target_vector: np.ndarray, weights: np.ndarray,
                 params: PhysicsParams = None, guidance_alpha: float = 1.0,
                 nan_policy: str = 'raise', memory_safe: bool = True):
        """
        Initialize the solver with target properties and ODE parameters.
        """
        self.target_vector = np.asarray(target_vector, dtype=self.DTYPE)
        if not np.all(np.isfinite(self.target_vector)):
            raise ValueError("target_vector contains NaN or Inf")
        self.weights = np.asarray(weights, dtype=self.DTYPE)
        if not np.all(np.isfinite(self.weights)):
            raise ValueError("weights contains NaN or Inf")
        if self.target_vector.shape != self.weights.shape:
            raise ValueError("target_vector and weights must have same shape")
        if not (0.01 <= guidance_alpha <= 100) or not np.isfinite(guidance_alpha):
            raise ValueError(f"guidance_alpha must be in [0.01, 100], got {guidance_alpha}")
        self.guidance_alpha = float(guidance_alpha)
        self.params = params if params is not None else PhysicsParams()
        if nan_policy not in ['raise', 'drop']:
            raise ValueError("nan_policy must be 'raise' or 'drop'")
        self.nan_policy = nan_policy
        self.memory_safe = memory_safe  # default memory-safe mode
    
    def normalize_library(self, ligand_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        GUARD 0: Normalize ligand properties to ratio vectors.
        Converts: r_i = x_i / x_target_i (with epsilon-clamping to prevent /0).
        Returns normalized_ratios and boolean mask of valid ligands.
        """
        ligand_matrix = np.asarray(ligand_matrix, dtype=self.DTYPE)
        # GUARD 0 Pre-check: input finiteness
        if not np.all(np.isfinite(ligand_matrix)):
            if self.nan_policy == 'raise':
                raise ValueError("GUARD 0 PRE-CHECK FAILED: ligand_matrix contains NaN or Inf")
            else:
                valid_mask = np.all(np.isfinite(ligand_matrix), axis=1)
        else:
            valid_mask = np.ones(ligand_matrix.shape[0], dtype=bool)
        # Epsilon clamping for target_vector to avoid division by zero
        target_safe = np.maximum(np.abs(self.target_vector), self.EPSILON_DIV_ZERO)
        normalized_ratios = ligand_matrix / target_safe
        # GUARD 0 Post-check: finiteness of ratios
        is_finite = np.all(np.isfinite(normalized_ratios), axis=1)
        valid_mask = valid_mask & is_finite
        if not np.any(valid_mask):
            if self.nan_policy == 'raise':
                raise ValueError("GUARD 0 POST-CHECK FAILED: All normalized ratios contain NaN/Inf")
            else:
                warnings.warn("GUARD 0 POST-CHECK: All ligands filtered due to NaN/Inf")
        return normalized_ratios, valid_mask
    
    def solve_analytic_trajectories(self, initial_ratios: np.ndarray,
                                    time_points: np.ndarray = None,
                                    max_time: float = 10.0, num_points: int = 200
                                   ) -> Tuple[np.ndarray, np.ndarray]:
        """
        GUARD 1: Solve ODE trajectories analytically for given initial ratios.
        Returns trajectories (T,N,P) and time_points (T,).
        """
        initial_ratios = np.asarray(initial_ratios, dtype=self.DTYPE)
        if time_points is None:
            time_points = np.linspace(0, max_time, num_points, dtype=self.DTYPE)
        else:
            time_points = np.asarray(time_points, dtype=self.DTYPE)
        N, P = initial_ratios.shape
        m, c, k = self.params.m, self.params.c, self.params.k
        zeta = self.params.damping_ratio()
        ω_n = np.sqrt(k / m)
        # Compute force direction (gradient) for each ligand and feature
        force_direction = self.weights * (initial_ratios - 1.0)  # shape (N,P)
        trajectories = np.zeros((len(time_points), N, P), dtype=self.DTYPE)
        # Solve trajectories depending on damping regime
        if zeta < 1.0:
            # Underdamped
            ω_d = ω_n * np.sqrt(1 - zeta**2)
            σ = zeta * ω_n
            for t_idx, t in enumerate(time_points):
                exp_term = np.exp(-σ * t)
                if exp_term < np.finfo(self.DTYPE).tiny:
                    exp_term = 0.0
                cos_term = np.cos(ω_d * t)
                sin_term = np.sin(ω_d * t)
                steady_state_shift = (self.guidance_alpha / k) * force_direction  # (N,P)
                oscillation = exp_term * (cos_term + (zeta/np.sqrt(1 - zeta**2)) * sin_term)  # scalar
                if not np.isfinite(oscillation):
                    oscillation = 0.0
                trajectories[t_idx] = initial_ratios + steady_state_shift * oscillation
                # GUARD 1: Validate trajectory values
                if not np.all(np.isfinite(trajectories[t_idx])):
                    if self.nan_policy == 'raise':
                        raise ValueError(f"GUARD 1 FAILED: Trajectory contains NaN/Inf at t={t}")
                    trajectories[t_idx] = np.nan_to_num(trajectories[t_idx], nan=initial_ratios,
                                                        posinf=1e10, neginf=-1e10)
        elif zeta > 1.0:
            # Overdamped
            term = np.sqrt(zeta**2 - 1.0)
            r1 = -ω_n * (zeta - term)
            r2 = -ω_n * (zeta + term)
            for t_idx, t in enumerate(time_points):
                exp1 = np.exp(r1 * t)
                exp2 = np.exp(r2 * t)
                if exp1 < np.finfo(self.DTYPE).tiny:
                    exp1 = 0.0
                if exp2 < np.finfo(self.DTYPE).tiny:
                    exp2 = 0.0
                steady_state_shift = (self.guidance_alpha / k) * force_direction  # (N,P)
                response = (exp1 - exp2) / (2 * ω_n * np.sqrt(zeta**2 - 1))
                if not np.isfinite(response):
                    response = 0.0
                trajectories[t_idx] = initial_ratios + steady_state_shift * response
                if not np.all(np.isfinite(trajectories[t_idx])):
                    if self.nan_policy == 'raise':
                        raise ValueError(f"GUARD 1 FAILED: Trajectory contains NaN/Inf at t={t}")
                    trajectories[t_idx] = np.nan_to_num(trajectories[t_idx], nan=initial_ratios,
                                                        posinf=1e10, neginf=-1e10)
        else:
            # Critically damped
            for t_idx, t in enumerate(time_points):
                exp_term = np.exp(-ω_n * t)
                if exp_term < np.finfo(self.DTYPE).tiny:
                    exp_term = 0.0
                steady_state_shift = (self.guidance_alpha / k) * force_direction  # (N,P)
                response = (1 + ω_n * t)
                if not np.isfinite(response):
                    response = 1.0
                trajectories[t_idx] = initial_ratios + steady_state_shift * exp_term * response
                if not np.all(np.isfinite(trajectories[t_idx])):
                    if self.nan_policy == 'raise':
                        raise ValueError(f"GUARD 1 FAILED: Trajectory contains NaN/Inf at t={t}")
                    trajectories[t_idx] = np.nan_to_num(trajectories[t_idx], nan=initial_ratios,
                                                        posinf=1e10, neginf=-1e10)
        return trajectories, time_points
    
    def calculate_metrics(self, trajectories: np.ndarray, time_points: np.ndarray) -> pd.DataFrame:
        """
        GUARD 2a/2b/2c: Compute ranking metrics (VCF_AUC, VCF_avg, Final_Distance).
        """
        trajectories = np.asarray(trajectories, dtype=self.DTYPE)
        # GUARD 2a: Validate trajectories
        if not np.all(np.isfinite(trajectories)):
            if self.nan_policy == 'raise':
                raise ValueError("GUARD 2a FAILED: Input trajectories contain NaN or Inf")
            else:
                n_invalid = np.sum(~np.isfinite(trajectories))
                warnings.warn(f"GUARD 2a: {n_invalid} NaN/Inf values in trajectories detected")
                trajectories = np.nan_to_num(trajectories, nan=1.0, posinf=1e10, neginf=-1e10)
        T, N, P = trajectories.shape
        time_points = np.asarray(time_points, dtype=self.DTYPE)
        # Compute weighted distance at each time step
        deviation = trajectories - 1.0  # shape (T, N, P)
        weighted_sq = self.weights[np.newaxis, np.newaxis, :] * (deviation ** 2)  # shape (T, N, P)
        distance_sq = np.sum(weighted_sq, axis=2)  # shape (T, N)
        # GUARD 2b: Validate distance computation
        if not np.all(np.isfinite(distance_sq)):
            if self.nan_policy == 'raise':
                raise ValueError("GUARD 2b FAILED: Distance squared contains NaN or Inf")
            else:
                n_invalid = np.sum(~np.isfinite(distance_sq))
                warnings.warn(f"GUARD 2b: {n_invalid} NaN/Inf values in distance_sq detected")
                distance_sq = np.nan_to_num(distance_sq, nan=0.0, posinf=1e10, neginf=0.0)
        # Clamp negatives to 0 and sqrt
        distance_sq = np.maximum(distance_sq, 0.0)
        distance_t = np.sqrt(distance_sq)  # shape (T, N)
        if not np.all(np.isfinite(distance_t)):
            if self.nan_policy == 'raise':
                raise ValueError("GUARD 2b FAILED: Distance after sqrt contains NaN or Inf")
            else:
                distance_t = np.nan_to_num(distance_t, nan=0.0, posinf=1e10, neginf=0.0)
        # Integrate distance over time to get VCF metrics
        vcf_auc = np.zeros(N, dtype=self.DTYPE)
        for i in range(N):
            vcf_auc[i] = np.trapz(distance_t[:, i], time_points)
        total_time = time_points[-1] - time_points[0]
        vcf_avg = vcf_auc / total_time if total_time > 0 else np.zeros_like(vcf_auc)
        final_distance = distance_t[-1, :]
        # GUARD 2c: Validate outputs
        for name, values in {"VCF_AUC": vcf_auc, "VCF_avg": vcf_avg, "Final_Distance": final_distance}.items():
            if not np.all(np.isfinite(values)):
                if self.nan_policy == 'raise':
                    raise ValueError(f"GUARD 2c FAILED: {name} contains NaN or Inf")
                else:
                    n_invalid = np.sum(~np.isfinite(values))
                    warnings.warn(f"GUARD 2c: {name} has {n_invalid} NaN/Inf values")
                    # Replace invalid values with safe defaults
                    values[...] = np.nan_to_num(values, nan=0.0, posinf=1e10, neginf=0.0)
        # Prepare DataFrame of metrics
        metrics_df = pd.DataFrame({
            "VCF_AUC": vcf_auc,
            "VCF_avg": vcf_avg,
            "Final_Distance": final_distance,
            "Rank": np.argsort(final_distance) + 1
        })
        return metrics_df
    
    def rank_ligands(self, ligand_matrix: np.ndarray, return_trajectories: bool = False,
                     **solver_kwargs) -> Dict[str, Any]:
        """
        Complete pipeline: normalize -> (solve trajectories) -> compute metrics -> rank.
        Returns a dict with keys: metrics, ligand_matrix, valid_indices, n_valid, n_total, n_filtered,
        and optionally trajectories and time_points if return_trajectories=True.
        """
        # GUARD 0: Normalize input library
        normalized_ratios, valid_mask = self.normalize_library(ligand_matrix)
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            raise ValueError("No valid ligands after normalization")
        valid_ratios = normalized_ratios[valid_mask]  # shape (N_valid, P)
        
        if self.memory_safe and not return_trajectories:
            # Memory-safe metrics computation without storing full trajectories
            # Generate time grid
            max_time = solver_kwargs.get('max_time', 10.0)
            num_points = solver_kwargs.get('num_points', 200)
            time_points = solver_kwargs.get('time_points', None)
            if time_points is None:
                time_points = np.linspace(0, max_time, num_points, dtype=self.DTYPE)
            else:
                time_points = np.asarray(time_points, dtype=self.DTYPE)
            # Compute A_n constants for each ligand (weighted squared initial deviation)
            diff = (valid_ratios - 1.0)  # shape (N_valid, P)
            A_vals = self.guidance_alpha ** 2 * np.sum(self.weights * (diff ** 2), axis=1)  # shape (N_valid,)
            # Compute common F(t) depending on damping regime
            zeta = self.params.damping_ratio()
            ω_n = np.sqrt(self.params.k / self.params.m)
            if zeta < 1.0:
                # underdamped: f(t) = e^{-σ t}[cos(ω_d t) + (σ/ω_d) sin(ω_d t)]
                σ = zeta * ω_n
                ω_d = ω_n * np.sqrt(1 - zeta**2)
                f_t = np.exp(-σ * time_points) * (np.cos(ω_d * time_points) + (σ/ω_d) * np.sin(ω_d * time_points))
            elif zeta > 1.0:
                # overdamped: combination of two exponentials
                term = np.sqrt(zeta**2 - 1.0)
                r1 = -ω_n * (zeta - term)
                r2 = -ω_n * (zeta + term)
                exp1 = np.exp(r1 * time_points)
                exp2 = np.exp(r2 * time_points)
                # Using derived formula: F(t) = ((r2-r1)^{-1} [r2 e^{r1 t} - r1 e^{r2 t}])^2. 
                # (This comes from the c1, c2 solution; ensures F(t) >= 0.)
                denom = (r1 - r2)
                # Compute F(t) in a numerically safe way
                F_t = ((exp1 * (r2/denom) - exp2 * (r1/denom))) ** 2
                # If any tiny negative values arise from floating error, clamp them:
                F_t = np.maximum(F_t, 0.0)
                # Now sqrt for integration
                f_t = np.sqrt(F_t)
            else:
                # critically damped: f(t) = (1 + ω_n t) e^{-ω_n t}
                f_t = (1.0 + ω_n * time_points) * np.exp(-ω_n * time_points)
            # For underdamped and critical, F(t) = [f(t)]^2 so we can get f_t for integration directly.
            if zeta < 1.0 or zeta == 1.0:
                # ensure no negative due to roundoff (shouldn't happen for squared formula)
                F_t = f_t ** 2
                F_t = np.maximum(F_t, 0.0)
                f_t = np.sqrt(F_t)
            # Integrate sqrt(F(t)) over time once (common for all ligands)
            I = np.trapz(f_t, time_points)  # scalar
            # Compute metrics for each ligand
            sqrt_A = np.sqrt(A_vals)  # shape (N_valid,)
            vcf_auc_vals = sqrt_A * I
            total_time = float(time_points[-1] - time_points[0])
            vcf_avg_vals = vcf_auc_vals / total_time if total_time > 0 else np.zeros_like(vcf_auc_vals)
            final_dist_vals = sqrt_A * f_t[-1]
            # GUARD 2c: validate outputs
            for name, arr in {"VCF_AUC": vcf_auc_vals, "VCF_avg": vcf_avg_vals, "Final_Distance": final_dist_vals}.items():
                if not np.all(np.isfinite(arr)):
                    if self.nan_policy == 'raise':
                        raise ValueError(f"GUARD 2c FAILED: {name} contains NaN or Inf")
                    else:
                        n_invalid = np.sum(~np.isfinite(arr))
                        warnings.warn(f"GUARD 2c: {name} has {n_invalid} NaN/Inf values")
                        arr[...] = np.nan_to_num(arr, nan=0.0, posinf=1e10, neginf=0.0)
            # Build metrics DataFrame and sort
            metrics = pd.DataFrame({
                "VCF_AUC": vcf_auc_vals,
                "VCF_avg": vcf_avg_vals,
                "Final_Distance": final_dist_vals
            })
            # New ranking: by Final_Distance ascending (consistent with prior version)
            metrics_sorted = metrics.sort_values("Final_Distance", ascending=True).reset_index(drop=True)
            metrics_sorted["Rank"] = np.arange(1, len(metrics_sorted) + 1)
            metrics_sorted["Original_Index"] = valid_indices
            result = {
                "metrics": metrics_sorted,
                "ligand_matrix": ligand_matrix[valid_mask],
                "valid_indices": valid_indices,
                "n_valid": len(valid_indices),
                "n_total": len(ligand_matrix),
                "n_filtered": len(ligand_matrix) - len(valid_indices)
            }
            # (No trajectories computed in memory-safe mode unless explicitly requested)
        else:
            # Standard mode (compute full trajectories and metrics)
            trajectories, time_points = self.solve_analytic_trajectories(valid_ratios, **solver_kwargs)
            metrics = self.calculate_metrics(trajectories, time_points)
            metrics["Original_Index"] = valid_indices
            metrics_sorted = metrics.sort_values("Final_Distance", ascending=True).reset_index(drop=True)
            result = {
                "metrics": metrics_sorted,
                "ligand_matrix": ligand_matrix[valid_mask],
                "valid_indices": valid_indices,
                "n_valid": len(valid_indices),
                "n_total": len(ligand_matrix),
                "n_filtered": len(ligand_matrix) - len(valid_indices)
            }
            if return_trajectories:
                result["trajectories"] = trajectories
                result["time_points"] = time_points
        return result
