#!/usr/bin/env python3
"""
Complete implementation of quantum trajectory generation and SDE model training.
Based on: "Quantum-tailored machine-learning characterization of a superconducting qubit"

This script generates artificial quantum trajectory data and trains an SDE model
to learn system parameters from weak measurement records.
"""

import numpy as np
import qutip as qt
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# PART 1: SYSTEM CONFIGURATION
# ============================================================================

@dataclass
class SystemConfig:
    """Configuration for transmon qubit in dispersive measurement."""
    
    # Time parameters
    dt_sim: float = 0.001      # Simulation time step (microseconds)
    dt_exp: float = 0.04       # Experimental time step (microseconds)
    T_max: float = 8.0         # Maximum evolution time (microseconds)
    
    # Qubit parameters (from experimental calibration)
    Omega_R: float = 2 * np.pi * 0.222e-6  # Rabi frequency (rad/s)
    Gamma_d: float = 2 * np.pi * 0.47e-6   # Measurement dephasing rate
    eta: float = 0.147                      # Quantum efficiency
    
    # Preparation and measurement
    prep_states: list = None
    meas_axes: list = None
    
    # Dataset parameters
    n_trajectories: int = 10000  # Reduced for demo (paper uses 1.75M)
    train_ratio: float = 0.75
    val_ratio: float = 0.16
    test_ratio: float = 0.09
    rtol: float = 1e-5
    atol: float = 1e-7
    
    def __post_init__(self):
        if self.prep_states is None:
            self.prep_states = ['0', '1', '+', '-', '+i', '-i']
        if self.meas_axes is None:
            self.meas_axes = ['X', 'Y', 'Z']


# ============================================================================
# PART 2: QUANTUM OPERATORS
# ============================================================================

class QuantumOps:
    """Create and cache quantum operators."""
    
    _cache = {}
    
    @classmethod
    def get_operators(cls):
        """Get Pauli operators for a 2-level qubit."""
        if 'paulis' not in cls._cache:
            cls._cache['paulis'] = {
                'X': qt.sigmax(),
                'Y': qt.sigmay(),
                'Z': qt.sigmaz(),
                'I': qt.qeye(2)
            }
        return cls._cache['paulis']
    
    @classmethod
    def get_basis_states(cls):
        """Get basis states on the Bloch sphere."""
        if 'states' not in cls._cache:
            paulis = cls.get_operators()
            sx, sy, sz = paulis['X'], paulis['Y'], paulis['Z']
            
            # Create cardinal states as density matrices
            state_0 = qt.basis(2, 0) * qt.basis(2, 0).dag()
            state_1 = qt.basis(2, 1) * qt.basis(2, 1).dag()
            
            # Superposition states
            state_plus = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
            state_plus = state_plus * state_plus.dag()
            
            state_minus = (qt.basis(2, 0) - qt.basis(2, 1)).unit()
            state_minus = state_minus * state_minus.dag()
            
            state_plus_i = (qt.basis(2, 0) + 1j*qt.basis(2, 1)).unit()
            state_plus_i = state_plus_i * state_plus_i.dag()
            
            state_minus_i = (qt.basis(2, 0) - 1j*qt.basis(2, 1)).unit()
            state_minus_i = state_minus_i * state_minus_i.dag()
            
            cls._cache['states'] = {
                '0': state_0,
                '1': state_1,
                '+': state_plus,
                '-': state_minus,
                '+i': state_plus_i,
                '-i': state_minus_i
            }
        
        return cls._cache['states']


# ============================================================================
# PART 3: TRAJECTORY GENERATION
# ============================================================================

def generate_single_trajectory(rho0, config):
    """
    Generate a single quantum trajectory using QuTiP's SME solver.
    
    Integrates: dρ = -i[H_R, ρ]dt + D[L]ρdt + √η H[L]ρ dW_I + √η H[iL]ρ dW_Q
    
    Parameters
    ----------
    rho0 : Qobj
        Initial density matrix
    config : SystemConfig
        System configuration
    
    Returns
    -------
    result : dict
        Contains trajectory data, measurement records, and final outcome
    """
    
    # Define operators
    paulis = QuantumOps.get_operators()
    H_R = (config.Omega_R / 2.0) * paulis['X']
    L = np.sqrt(config.Gamma_d / 2.0) * paulis['Z']
    
    # Time grid
    times = np.arange(0, config.T_max, config.dt_sim)
    
    # Solve SME using Milstein method
    result = qt.smesolve(
        H_R,
        rho0,
        times,
        c_ops=[],
        sc_ops=[L],
        e_ops=[paulis['X'], paulis['Y'], paulis['Z']],
        ntraj=config.n_trajectories,
        target_tol=(config.atol, config.rtol),
        options=qt.Options(method='milstein', store_states=True)
    )
    
    # Extract expectation values
    expect_X = np.array(result.expect[0])
    expect_Y = np.array(result.expect[1])
    expect_Z = np.array(result.expect[2])
    
    # Coarse-grain measurement to experimental time step
    n_coarse = int(config.dt_exp / config.dt_sim)
    M_record = []
    
    for i in range(0, len(expect_Z), n_coarse):
        # Average over coarse-grain interval
        M_chunk = expect_Z[i:i+n_coarse]
        # Add realistic noise
        M_noisy = np.mean(M_chunk) + np.random.randn() * np.std(M_chunk) * 0.1
        M_record.append(M_noisy)
    
    # Generate final measurement outcome
    rho_final = result.states[-1]
    p_1 = (qt.basis(2, 1) * qt.basis(2, 1).dag() * rho_final).tr().real
    final_outcome = 1 if np.random.rand() < p_1 else -1
    
    return {
        'M': np.array(M_record),
        'final_outcome': final_outcome,
        'p_1': p_1,
        'rho_final': rho_final,
        'expects': (expect_X, expect_Y, expect_Z)
    }


def generate_dataset(config):
    """
    Generate complete training dataset.
    
    Parameters
    ----------
    config : SystemConfig
        System configuration
    
    Returns
    -------
    dataset : dict
        Training dataset with measurements, outcomes, and metadata
    """
    
    states = QuantumOps.get_basis_states()
    
    measurements = []
    outcomes = []
    prep_states = []
    probs = []
    
    n_per_prep = config.n_trajectories // len(config.prep_states)
    
    print(f"Generating {config.n_trajectories} trajectories...")
    
    for prep_label in config.prep_states:
        for traj_idx in range(n_per_prep):
            rho0 = states[prep_label]
            traj = generate_single_trajectory(rho0, config)
            
            measurements.append(traj['M'])
            outcomes.append(traj['final_outcome'])
            prep_states.append(prep_label)
            probs.append(traj['p_1'])
            
            if (traj_idx + 1) % max(1, n_per_prep // 5) == 0:
                print(f"  {prep_label}: {traj_idx + 1}/{n_per_prep} trajectories")
    
    # Convert outcomes from ±1 to {0, 1}
    outcomes_binary = np.array([(o + 1) // 2 for o in outcomes])
    
    # Shuffle
    perm = np.random.permutation(len(measurements))
    
    dataset = {
        'measurements': np.array(measurements)[perm],
        'outcomes': outcomes_binary[perm],
        'prep_states': np.array(prep_states)[perm],
        'probs': np.array(probs)[perm]
    }
    
    return dataset


def split_dataset(dataset, train_ratio=0.75, val_ratio=0.16):
    """Split dataset into train, validation, and test sets."""
    
    n = len(dataset['measurements'])
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_set = {k: v[:n_train] for k, v in dataset.items()}
    val_set = {k: v[n_train:n_train+n_val] for k, v in dataset.items()}
    test_set = {k: v[n_train+n_val:] for k, v in dataset.items()}
    
    return train_set, val_set, test_set


# ============================================================================
# PART 4: SDE MODEL
# ============================================================================

class SDEModel:
    """
    Differentiable SDE model for learning quantum dynamics.
    
    Learns parameters: Ω_R, Γ_d, η from weak measurement data.
    """
    
    def __init__(self, config):
        """Initialize with system configuration."""
        
        self.config = config
        self.paulis = QuantumOps.get_operators()
        
        # Parameters stored as log for numerical stability
        self.params = np.array([
            np.log(config.Omega_R),
            np.log(config.Gamma_d),
            config.eta  # Stored directly as it's bounded [0, 1]
        ])
    
    def get_params(self):
        """Extract physical parameters."""
        Omega_R = np.exp(self.params[0])
        Gamma_d = np.exp(self.params[1])
        eta = np.clip(self.params[2], 0.0, 1.0)
        return Omega_R, Gamma_d, eta
    
    def milstein_step(self, rho, M_t, dt, Omega_R, Gamma_d, eta):
        """
        Single Milstein integration step for the SME.
        
        dρ = -i[H_R, ρ]dt + D[L]ρdt + √η H[L]ρ dW
        """
        
        # Operators
        H_R = (Omega_R / 2.0) * self.paulis['X']
        L = np.sqrt(Gamma_d / 2.0) * self.paulis['Z']
        L_dag = L.dag()
        
        # Deterministic part
        comm = -1j * (H_R * rho - rho * H_R)
        dissip = L * rho * L_dag - 0.5 * (L_dag * L * rho + rho * L_dag * L)
        
        # Stochastic part (measurement superoperator)
        trace_term = (L * rho + rho * L_dag).tr().real
        H_L = L * rho + rho * L_dag - trace_term * rho
        
        # Update (Milstein scheme)
        d_rho = (comm + dissip + np.sqrt(eta) * H_L * M_t) * dt
        rho_new = rho + d_rho
        
        # Enforce hermiticity and positivity
        rho_new = (rho_new + rho_new.dag()) / 2.0
        
        return rho_new
    
    def predict(self, measurement_record, rho0):
        """
        Predict final measurement probability given measurement record.
        
        Parameters
        ----------
        measurement_record : array
            Weak measurement time series
        rho0 : Qobj
            Initial quantum state
        
        Returns
        -------
        p_1 : float
            Probability of measuring |1⟩
        """
        
        Omega_R, Gamma_d, eta = self.get_params()
        
        rho = rho0
        
        # Evolve through measurement record
        for M_t in measurement_record:
            rho = self.milstein_step(rho, M_t, self.config.dt_exp,
                                    Omega_R, Gamma_d, eta)
        
        # Measurement probability
        proj_1 = qt.basis(2, 1) * qt.basis(2, 1).dag()
        p_1 = (proj_1 * rho).tr().real
        
        return np.clip(p_1, 1e-8, 1 - 1e-8)


# ============================================================================
# PART 5: LOSS AND OPTIMIZATION
# ============================================================================

def cross_entropy_loss(model, measurements, outcomes, initial_states):
    """
    Binary cross-entropy loss.
    
    L = -1/N Σ[y*log(p) + (1-y)*log(1-p)]
    """
    
    loss = 0.0
    
    for i in range(len(measurements)):
        y = outcomes[i]
        p = model.predict(measurements[i], initial_states[i])
        
        loss += -(y * np.log(p) + (1 - y) * np.log(1 - p))
    
    return loss / len(measurements)


def objective(params, model, measurements, outcomes, initial_states):
    """Wrapper for scipy optimizer."""
    print(f"Objective called with params: {params}")
    model.params = params.copy()
    return cross_entropy_loss(model, measurements, outcomes, initial_states)


def train_model(train_set, val_set, config, method='L-BFGS-B', maxiter=100):
    """
    Train SDE model using scipy optimization.
    
    Parameters
    ----------
    train_set, val_set : dict
        Training and validation data
    config : SystemConfig
        System configuration
    method : str
        Optimization method
    maxiter : int
        Maximum iterations
    
    Returns
    -------
    model : SDEModel
        Trained model
    history : dict
        Training history
    """
    
    # Initialize model and state converters
    model = SDEModel(config)
    states = QuantumOps.get_basis_states()
    
    # Convert prep_states to initial states
    train_init = [states[ps] for ps in train_set['prep_states']]
    val_init = [states[ps] for ps in val_set['prep_states']]
    
    history = {'train_loss': [], 'val_loss': [], 'params': []}
    
    # Training function
    def callback(xk):
        """Called after each iteration."""
        model.params = xk.copy()
        train_loss = cross_entropy_loss(model, train_set['measurements'],
                                       train_set['outcomes'], train_init)
        val_loss = cross_entropy_loss(model, val_set['measurements'],
                                     val_set['outcomes'], val_init)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['params'].append(model.get_params())

        print(f"Iter {len(history['train_loss'])}: Train={train_loss:.6f}, Val={val_loss:.6f}")
        
        if len(history['train_loss']) % 10 == 0:
            print(f"Iter {len(history['train_loss'])}: Train={train_loss:.6f}, Val={val_loss:.6f}")
    
    # Optimize
    print("Training model...")
    result = minimize(
        objective,
        model.params,
        args=(model, train_set['measurements'], train_set['outcomes'], train_init),
        method=method,
        callback=callback,
        options={'maxiter': maxiter, 'ftol': 1e-8}
    )
    
    model.params = result.x
    print(f"\nOptimization completed: {result.message}")
    
    return model, history


# ============================================================================
# PART 6: EVALUATION
# ============================================================================

def evaluate(test_set, model, config):
    """
    Evaluate model on test set.
    
    Parameters
    ----------
    test_set : dict
        Test data
    model : SDEModel
        Trained model
    config : SystemConfig
        System configuration
    
    Returns
    -------
    metrics : dict
        Performance metrics
    """
    
    states = QuantumOps.get_basis_states()
    test_init = [states[ps] for ps in test_set['prep_states']]
    
    # Predictions
    preds = []
    for i in range(len(test_set['measurements'])):
        p = model.predict(test_set['measurements'][i], test_init[i])
        preds.append(p)
    
    preds = np.array(preds)
    
    # Loss
    ce_loss = cross_entropy_loss(model, test_set['measurements'],
                                test_set['outcomes'], test_init)
    
    # Accuracy
    pred_binary = (preds > 0.5).astype(int)
    accuracy = np.mean(pred_binary == test_set['outcomes'])
    
    return {
        'cross_entropy': ce_loss,
        'accuracy': accuracy,
        'predictions': preds
    }


# ============================================================================
# PART 7: VISUALIZATION
# ============================================================================

def plot_training_history(history):
    """Plot training curves."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    iterations = np.arange(len(history['train_loss']))
    ax1.plot(iterations, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(iterations, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Cross-Entropy Loss', fontsize=12)
    ax1.set_title('Training History', fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Parameter evolution
    params = np.array(history['params'])
    Omega_R_vals = params[:, 0] / (2 * np.pi) * 1e6
    Gamma_d_vals = params[:, 1] / (2 * np.pi) * 1e6
    eta_vals = params[:, 2]
    
    ax2.plot(iterations, Omega_R_vals, 'b-', label='Ω_R / 2π (MHz)', linewidth=2)
    ax2.plot(iterations, Gamma_d_vals, 'r-', label='Γ_d / 2π (MHz)', linewidth=2)
    ax2.plot(iterations, eta_vals * 10, 'g-', label='η × 10', linewidth=2)  # Scale for visibility
    
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Parameter Value', fontsize=12)
    ax2.set_title('Parameter Evolution', fontsize=13)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sde_training.png', dpi=150)
    print("Saved: sde_training.png")
    plt.show()


def plot_predictions(test_set, predictions):
    """Plot prediction accuracy."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram of predictions
    bins = np.linspace(0, 1, 51)
    
    # True outcomes
    true_0 = predictions[test_set['outcomes'] == 0]
    true_1 = predictions[test_set['outcomes'] == 1]
    
    ax.hist(true_0, bins=bins, alpha=0.6, label='True outcome = 0', density=True)
    ax.hist(true_1, bins=bins, alpha=0.6, label='True outcome = 1', density=True)
    
    ax.set_xlabel('Predicted P(1)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Prediction Distribution', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sde_predictions.png', dpi=150)
    print("Saved: sde_predictions.png")
    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    
    # Configuration
    config = SystemConfig(n_trajectories=5000)  # Reduced for faster demo
    
    print("="*70)
    print("Quantum Trajectory Generation and SDE Model Training")
    print("="*70)
    print(f"\nSystem parameters:")
    print(f"  Ω_R / 2π = {config.Omega_R / (2*np.pi) * 1e6:.3f} MHz")
    print(f"  Γ_d / 2π = {config.Gamma_d / (2*np.pi) * 1e6:.3f} MHz")
    print(f"  η = {config.eta:.4f}")
    
    # Generate dataset
    print("\n[1] Generating dataset...")
    dataset = generate_dataset(config)
    # n_per_prep = config.n_trajectories // len(config.prep_states)
    # length = n_per_prep * len(config.prep_states)
    # dataset = {
    #     'measurements': [np.random.random(config.n_trajectories).tolist() for _ in range(len(range(length)))],
    #     'outcomes': np.random.choice([0, 1], length).tolist(),
    #     'prep_states': np.random.choice(config.prep_states, length).tolist(),
    #     'probs': np.random.random(length).tolist()
    # }
    print(np.shape(dataset['measurements']))
    train_set, val_set, test_set = split_dataset(dataset)
    print(f"    Train: {len(train_set['measurements'])}")
    print(f"    Val: {len(val_set['measurements'])}")
    print(f"    Test: {len(test_set['measurements'])}")
    
    # Train model
    print("\n[2] Training SDE model...")
    model, history = train_model(train_set, val_set, config, maxiter=50)
    
    # Report learned parameters
    Omega_R_learned, Gamma_d_learned, eta_learned = model.get_params()
    print(f"\nLearned parameters:")
    print(f"  Ω_R / 2π = {Omega_R_learned / (2*np.pi) * 1e6:.3f} MHz (true: {config.Omega_R / (2*np.pi) * 1e6:.3f} MHz)")
    print(f"  Γ_d / 2π = {Gamma_d_learned / (2*np.pi) * 1e6:.3f} MHz (true: {config.Gamma_d / (2*np.pi) * 1e6:.3f} MHz)")
    print(f"  η = {eta_learned:.4f} (true: {config.eta:.4f})")
    
    # Evaluate
    print("\n[3] Evaluating on test set...")
    metrics = evaluate(test_set, model, config)
    print(f"    Cross-entropy loss: {metrics['cross_entropy']:.6f}")
    print(f"    Accuracy: {metrics['accuracy']:.4f}")
    
    # Visualize
    print("\n[4] Generating plots...")
    plot_training_history(history)
    plot_predictions(test_set, metrics['predictions'])
    
    print("\nDone!")
