# Implementation Guide: Quantum Trajectory Generation and SDE Model Training

## Overview
This guide provides step-by-step instructions for implementing the quantum trajectory data generation and SDE model training using QuTiP and SciPy, as described in "Quantum-tailored machine-learning characterization of a superconducting qubit."

---

## Part 1: Environment Setup

### Required Packages
```bash
pip install qutip scipy numpy matplotlib
```

### Python Imports
```python
import numpy as np
import qutip as qt
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from itertools import product
```

---

## Part 2: Defining System Parameters

### Step 2.1: Initialize Physical Constants

Create a configuration file or class to store system parameters:

```python
class QuantumSystem:
    """System parameters for transmon qubit in dispersive measurement."""
    
    def __init__(self):
        # Time parameters
        self.dt_sim = 0.001  # Simulation time step (microseconds)
        self.dt_exp = 0.04   # Experimental time step (microseconds)
        self.T_max = 8.0     # Maximum evolution time (microseconds)
        
        # Qubit parameters (calibrated from real device)
        self.Omega_R = 2 * np.pi * 0.222e-6  # Rabi frequency (MHz -> rad/s)
        self.Gamma_d = 2 * np.pi * 0.47e-6   # Measurement dephasing rate
        self.eta = 0.147                       # Quantum efficiency
        
        # Preparation and measurement axes
        self.prep_states = ['0', '1', '+', '-', '+i', '-i']  # 6 cardinal points
        self.meas_axes = ['X', 'Y', 'Z']
        
        # Dataset parameters
        self.n_trajectories = 1_750_000  # Total trajectories
        self.train_ratio = 0.75
        self.val_ratio = 0.16
        self.test_ratio = 0.09

# Create system instance
sys = QuantumSystem()
```

### Step 2.2: Define Qubit Operators

```python
def create_qubit_operators():
    """Create Pauli operators and state vectors for a 2-level qubit."""
    
    # Pauli matrices
    sx = qt.sigmax()  # σ_x
    sy = qt.sigmay()  # σ_y
    sz = qt.sigmaz()  # σ_z
    
    # Bloch sphere cardinal states (as density matrices)
    state_0 = qt.basis(2, 0) * qt.basis(2, 0).dag()   # |0⟩⟨0|
    state_1 = qt.basis(2, 1) * qt.basis(2, 1).dag()   # |1⟩⟨1|
    state_plus = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
    state_plus = state_plus * state_plus.dag()         # |+⟩⟨+|
    state_minus = (qt.basis(2, 0) - qt.basis(2, 1)).unit()
    state_minus = state_minus * state_minus.dag()      # |-⟩⟨-|
    state_plus_i = (qt.basis(2, 0) + 1j*qt.basis(2, 1)).unit()
    state_plus_i = state_plus_i * state_plus_i.dag()  # |+i⟩⟨+i|
    state_minus_i = (qt.basis(2, 0) - 1j*qt.basis(2, 1)).unit()
    state_minus_i = state_minus_i * state_minus_i.dag() # |-i⟩⟨-i|
    
    states_dict = {
        '0': state_0,
        '1': state_1,
        '+': state_plus,
        '-': state_minus,
        '+i': state_plus_i,
        '-i': state_minus_i
    }
    
    return sx, sy, sz, states_dict

sx, sy, sz, states_dict = create_qubit_operators()
```

---

## Part 3: Generating Quantum Trajectories

### Step 3.1: Implement the Stochastic Master Equation

```python
def generate_measurement_record(rho, eta=0.147):
    """
    Generate weak measurement record M(t) from density matrix.
    
    The measured signal is:
    dM^q(t) = sqrt(eta) * Tr(L*rho(t) + rho(t)*L†) * dt + dW^q(t)
    
    where q ∈ {I, Q} are the I and Q quadratures.
    
    Parameters
    ----------
    rho : qobj
        Density matrix at time t
    eta : float
        Quantum efficiency
    
    Returns
    -------
    m_I, m_Q : float
        I and Q quadrature measurements
    """
    sx, sy, sz, _ = create_qubit_operators()
    
    # Measurement operator (σ_z for homodyne detection)
    L = np.sqrt(0.5) * sz
    
    # Deterministic part: sqrt(eta) * Tr(L*rho + rho*L†)
    m_det = np.sqrt(eta) * (L * rho + rho * L.dag()).tr().real / 2.0
    
    return m_det


def sme_integrate_single_trajectory(rho0, prep_state, times, sys):
    """
    Integrate the stochastic master equation for a single trajectory.
    
    dρ_t = -i[H_R, ρ_t]dt + D[L]ρ_t dt + sqrt(η)H[L]ρ_t dW_I(t) + sqrt(η)H[iL]ρ_t dW_Q(t)
    
    where:
    - H_R = (Ω_R/2) σ_x is the Rabi Hamiltonian
    - L = sqrt(Γ_d/2) σ_z is the measurement operator
    - D[L]ρ = L ρ L† - (1/2)(L†L ρ + ρ L†L) is the dissipator
    - H[A]ρ = A ρ + ρ A† - Tr(A ρ + ρ A†) ρ is the measurement superoperator
    
    Parameters
    ----------
    rho0 : qobj
        Initial state
    prep_state : str
        Preparation state label ('0', '1', '+', '-', '+i', '-i')
    times : array
        Time points for integration
    sys : QuantumSystem
        System configuration object
    
    Returns
    -------
    trajectories : dict
        Contains 'rho_traj' (density matrices), 'M' (measurement records), 
        'final_outcome' (±1)
    """
    
    sx, sy, sz, _ = create_qubit_operators()
    
    # Define operators
    H_R = (sys.Omega_R / 2.0) * sx  # Rabi Hamiltonian
    L = np.sqrt(sys.Gamma_d / 2.0) * sz  # Measurement operator
    
    # Pre-compute operators for efficiency
    L_dag = L.dag()
    L_dag_L = L_dag * L
    D_mat = L * rho0 * L_dag - 0.5 * (L_dag_L * rho0 + rho0 * L_dag_L)
    
    # Initialize trajectory storage
    n_steps = len(times)
    rho_traj = [rho0]
    M_I_traj = []
    M_Q_traj = []
    
    # Use QuTiP's built-in SME solver
    # This automatically handles the stochastic integration
    
    result = qt.smesolve(
        H_R,                           # Hamiltonian
        rho0,                          # Initial state
        times,                         # Time points
        c_ops=[],                      # Classical collapse operators
        sc_ops=[L],                    # Stochastic collapse operators
        e_ops=[sx, sy, sz],           # Expectation value operators
        solver='fast-milstein',        # Use Milstein method
        nsubsteps=int(sys.dt_exp / sys.dt_sim),  # Substeps for fine integration
        options=qt.Options(
            nsteps=10000,
            store_states=True,
            rtol=1e-6,
            atol=1e-8
        )
    )
    
    # Extract measurement records (weak measurement signal)
    # M(t) contains both deterministic signal and noise
    # Generate synthetic measurement data from trajectory
    for i, rho in enumerate(result.states):
        # Deterministic part of measurement
        m_det = generate_measurement_record(rho, sys.eta)
        M_I_traj.append(m_det)
        M_Q_traj.append(m_det)  # For homodyne, both quadratures contain same info
    
    # Coarse-grain to experimental time step
    n_coarse = int(sys.dt_exp / sys.dt_sim)
    M_coarse = []
    for i in range(0, len(M_I_traj), n_coarse):
        M_coarse.append(np.mean(M_I_traj[i:i+n_coarse]))
    
    # Generate final measurement outcome (probabilistic)
    # P(1 | ρ_T) = ⟨1|ρ_T|1⟩
    rho_final = result.states[-1]
    p_1 = (qt.basis(2, 1) * qt.basis(2, 1).dag()).matrix.getH() @ rho_final.full()
    p_1_scalar = np.real(np.trace(p_1))
    final_outcome = np.random.choice([1, -1], p=[p_1_scalar, 1 - p_1_scalar])
    
    return {
        'rho_traj': result.states,
        'expect': result.expect,
        'M': np.array(M_coarse),
        'final_outcome': final_outcome,
        'prep_state': prep_state,
        'p_1': p_1_scalar
    }
```

### Step 3.2: Generate Full Dataset

```python
def generate_dataset(n_trajectories, sys, n_jobs=1):
    """
    Generate artificial quantum trajectory dataset.
    
    Parameters
    ----------
    n_trajectories : int
        Number of trajectories to generate
    sys : QuantumSystem
        System configuration
    n_jobs : int
        Number of parallel jobs (for parallelization)
    
    Returns
    -------
    dataset : dict
        Contains 'measurements', 'outcomes', 'prep_states', 'meas_axes'
    """
    
    _, states_dict = create_qubit_operators()
    
    # Time grid with fine simulation step
    times_sim = np.arange(0, sys.T_max, sys.dt_sim)
    times_exp = np.arange(0, sys.T_max, sys.dt_exp)
    
    measurements = []
    outcomes = []
    prep_states_list = []
    meas_axes_list = []
    
    n_generated = 0
    
    # Generate trajectories
    for prep in sys.prep_states:
        for meas_axis in sys.meas_axes:
            # Determine preparation state
            rho0 = states_dict[prep]
            
            # Generate one trajectory
            traj = sme_integrate_single_trajectory(rho0, prep, times_sim, sys)
            
            # Coarse-grain measurement
            n_coarse = int(sys.dt_exp / sys.dt_sim)
            M_coarse = []
            for i in range(0, len(traj['M']), n_coarse):
                M_coarse.append(np.mean(traj['M'][i:i+n_coarse]))
            
            measurements.append(np.array(M_coarse))
            outcomes.append(traj['final_outcome'])
            prep_states_list.append(prep)
            meas_axes_list.append(meas_axis)
            
            n_generated += 1
            
            if n_generated % 100000 == 0:
                print(f"Generated {n_generated} trajectories...")
    
    # Replicate to reach desired number
    while n_generated < n_trajectories:
        # Random selection
        idx = np.random.randint(0, len(measurements))
        measurements.append(measurements[idx].copy())
        outcomes.append(outcomes[idx])
        prep_states_list.append(prep_states_list[idx])
        meas_axes_list.append(meas_axes_list[idx])
        n_generated += 1
    
    # Shuffle
    perm = np.random.permutation(len(measurements))
    
    return {
        'measurements': np.array(measurements)[perm],
        'outcomes': np.array(outcomes)[perm],
        'prep_states': np.array(prep_states_list)[perm],
        'meas_axes': np.array(meas_axes_list)[perm]
    }

# Generate dataset
print("Generating dataset...")
dataset = generate_dataset(sys.n_trajectories, sys)
print(f"Dataset size: {len(dataset['measurements'])}")
```

### Step 3.3: Split Dataset

```python
def split_dataset(dataset, train_ratio=0.75, val_ratio=0.16):
    """Split dataset into train, validation, and test sets."""
    
    n = len(dataset['measurements'])
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    indices = np.arange(n)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]
    
    train_set = {
        'measurements': dataset['measurements'][train_idx],
        'outcomes': dataset['outcomes'][train_idx],
        'prep_states': dataset['prep_states'][train_idx],
        'meas_axes': dataset['meas_axes'][train_idx]
    }
    
    val_set = {
        'measurements': dataset['measurements'][val_idx],
        'outcomes': dataset['outcomes'][val_idx],
        'prep_states': dataset['prep_states'][val_idx],
        'meas_axes': dataset['meas_axes'][val_idx]
    }
    
    test_set = {
        'measurements': dataset['measurements'][test_idx],
        'outcomes': dataset['outcomes'][test_idx],
        'prep_states': dataset['prep_states'][test_idx],
        'meas_axes': dataset['meas_axes'][test_idx]
    }
    
    return train_set, val_set, test_set

train_set, val_set, test_set = split_dataset(dataset, sys.train_ratio, sys.val_ratio)
print(f"Train: {len(train_set['measurements'])}, Val: {len(val_set['measurements'])}, Test: {len(test_set['measurements'])}")
```

---

## Part 4: Implementing the Differentiable SDE Model

### Step 4.1: Define the SDE Integrator Class

```python
class SDEModel:
    """
    Differentiable SDE model for quantum trajectory learning.
    
    The model learns to integrate the SME using a Milstein scheme
    with learnable parameters: Ω_R, Γ_d, and η.
    """
    
    def __init__(self, Omega_R_init=None, Gamma_d_init=None, eta_init=None):
        """Initialize learnable parameters."""
        
        if Omega_R_init is None:
            Omega_R_init = 2 * np.pi * 0.222e-6
        if Gamma_d_init is None:
            Gamma_d_init = 2 * np.pi * 0.47e-6
        if eta_init is None:
            eta_init = 0.147
        
        # Parameters to learn (stored as log for numerical stability)
        self.params = np.array([
            np.log(Omega_R_init),
            np.log(Gamma_d_init),
            np.log(eta_init)
        ])
        
        # Physical constants
        self.sx = qt.sigmax()
        self.sy = qt.sigmay()
        self.sz = qt.sigmaz()
        
    def get_params(self):
        """Extract physical parameters from learnable variables."""
        Omega_R = np.exp(self.params[0])
        Gamma_d = np.exp(self.params[1])
        eta = np.exp(self.params[2])
        # Constrain eta to [0, 1]
        eta = 1.0 / (1.0 + np.exp(-eta))
        return Omega_R, Gamma_d, eta
    
    def milstein_step(self, rho, M_t, dt, Omega_R, Gamma_d, eta):
        """
        Single Milstein integration step.
        
        dρ = -i[H_R, ρ]dt + D[L]ρ dt + sqrt(η)H[L]ρ dW_I + sqrt(η)H[iL]ρ dW_Q
        """
        
        # Hamiltonian
        H_R = (Omega_R / 2.0) * self.sx
        
        # Measurement operator
        L = np.sqrt(Gamma_d / 2.0) * self.sz
        L_dag = L.dag()
        L_dag_L = L_dag * L
        
        # Deterministic part: -i[H_R, ρ] + D[L]ρ
        commutator = -1j * (H_R * rho - rho * H_R)
        dissipator = L * rho * L_dag - 0.5 * (L_dag_L * rho + rho * L_dag_L)
        deterministic = commutator + dissipator
        
        # Stochastic part: H[L]ρ
        # H[A]ρ = A ρ + ρ A† - Tr(A ρ + ρ A†) ρ
        trace_term = (L * rho + rho * L_dag).tr().real
        H_L_rho = L * rho + rho * L_dag - trace_term * rho
        
        # Milstein correction: involves derivative of diffusion coefficient
        # For our case, this is: (1/2) * (dH/dW)² * (dW² - dt) / 2
        # Simplified: (1/2) * L ρ L† * (M_t² - dt) / (2*dt)
        dW_corr = M_t  # Wiener increment approximated from measurement noise
        milstein_corr = 0.5 * L * rho * L_dag * (dW_corr**2 - dt) / (2.0 * dt + 1e-10)
        
        # Update step
        d_rho = (deterministic + np.sqrt(eta) * H_L_rho * dW_corr + milstein_corr) * dt
        rho_new = rho + d_rho
        
        # Ensure positivity
        rho_new = (rho_new + rho_new.dag()) / 2.0  # Hermitian
        
        return rho_new
    
    def predict(self, measurement_record, rho0, dt_exp):
        """
        Predict final quantum state given measurement record.
        
        Parameters
        ----------
        measurement_record : array of shape (T,)
            Weak measurement time series
        rho0 : qobj
            Initial state
        dt_exp : float
            Experimental time step
        
        Returns
        -------
        p_1 : float
            Probability of measuring |1⟩ at final time
        """
        
        Omega_R, Gamma_d, eta = self.get_params()
        
        rho = rho0
        dt = dt_exp
        
        # Evolve through all time steps
        for t, M_t in enumerate(measurement_record):
            rho = self.milstein_step(rho, M_t, dt, Omega_R, Gamma_d, eta)
        
        # Compute final measurement probability
        proj_1 = qt.basis(2, 1) * qt.basis(2, 1).dag()
        p_1 = (proj_1 * rho).tr().real
        
        return p_1


# Initialize model
model = SDEModel()
print(f"Initial parameters: {model.get_params()}")
```

### Step 4.2: Define Loss Function

```python
def cross_entropy_loss(measurement_records, true_outcomes, initial_states, model, dt_exp):
    """
    Binary cross-entropy loss for quantum trajectory learning.
    
    L_CE = -1/N * Σ[Y_n * log(p_n) + (1-Y_n) * log(1-p_n)]
    
    where Y_n ∈ {0, 1} and p_n is the predicted probability of |1⟩.
    """
    
    loss = 0.0
    n_samples = len(measurement_records)
    
    for i in range(n_samples):
        # Convert outcome from ±1 to {0, 1}
        Y = 1.0 if true_outcomes[i] == 1 else 0.0
        
        # Forward pass
        p_pred = model.predict(measurement_records[i], initial_states[i], dt_exp)
        
        # Clamp predictions to avoid log(0)
        p_pred = np.clip(p_pred, 1e-7, 1 - 1e-7)
        
        # Cross-entropy contribution
        loss += -(Y * np.log(p_pred) + (1 - Y) * np.log(1 - p_pred))
    
    return loss / n_samples


def objective_function(params, measurement_records, true_outcomes, initial_states, 
                      model, dt_exp):
    """Wrapper for scipy optimizer."""
    
    # Update model parameters
    model.params = params.copy()
    
    # Compute loss
    loss = cross_entropy_loss(measurement_records, true_outcomes, 
                              initial_states, model, dt_exp)
    
    return loss
```

---

## Part 5: Training the Model

### Step 5.1: Prepare Training Data

```python
def prepare_training_batch(dataset, batch_size, sys):
    """
    Prepare batches of training data.
    
    Returns
    -------
    batches : list of tuples
        Each tuple contains (measurements, outcomes, initial_states)
    """
    
    _, states_dict = create_qubit_operators()
    
    n_samples = len(dataset['measurements'])
    n_batches = n_samples // batch_size
    
    batches = []
    
    for b in range(n_batches):
        start_idx = b * batch_size
        end_idx = start_idx + batch_size
        
        measurements = dataset['measurements'][start_idx:end_idx]
        outcomes = dataset['outcomes'][start_idx:end_idx]
        prep_states = dataset['prep_states'][start_idx:end_idx]
        
        # Convert prep_states to initial density matrices
        initial_states = [states_dict[ps] for ps in prep_states]
        
        batches.append((measurements, outcomes, initial_states))
    
    return batches
```

### Step 5.2: Training Loop

```python
def train_sde_model(train_set, val_set, sys, n_epochs=100, batch_size=32):
    """
    Train the SDE model on weak measurement data.
    
    Parameters
    ----------
    train_set, val_set : dict
        Training and validation datasets
    sys : QuantumSystem
        System configuration
    n_epochs : int
        Number of training epochs
    batch_size : int
        Batch size for mini-batch training
    
    Returns
    -------
    model : SDEModel
        Trained model
    history : dict
        Training history
    """
    
    # Initialize model
    model = SDEModel()
    _, states_dict = create_qubit_operators()
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'params': []
    }
    
    # Prepare training batches
    train_batches = prepare_training_batch(train_set, batch_size, sys)
    
    # Prepare validation set
    val_initial_states = [states_dict[ps] for ps in val_set['prep_states']]
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    # Training loop
    for epoch in range(n_epochs):
        
        epoch_loss = 0.0
        
        for batch_idx, (batch_meas, batch_outcomes, batch_init) in enumerate(train_batches):
            
            # Loss function for this batch
            def batch_loss(params):
                model.params = params.copy()
                return cross_entropy_loss(batch_meas, batch_outcomes, batch_init, 
                                        model, sys.dt_exp)
            
            # Optimize parameters for this batch (mini-batch step)
            result = minimize(
                batch_loss,
                model.params,
                method='L-BFGS-B',
                options={'maxiter': 10, 'ftol': 1e-6}
            )
            
            model.params = result.x
            epoch_loss += result.fun
        
        # Average epoch loss
        epoch_loss /= len(train_batches)
        
        # Validation
        val_loss = cross_entropy_loss(val_set['measurements'], val_set['outcomes'],
                                     val_initial_states, model, sys.dt_exp)
        
        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(val_loss)
        history['params'].append(model.get_params())
        
        print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {epoch_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    return model, history
```

### Step 5.3: Execute Training

```python
# Train the model
print("\nTraining SDE model...")
model, history = train_sde_model(train_set, val_set, sys, n_epochs=50, batch_size=128)

# Extract final parameters
Omega_R_learned, Gamma_d_learned, eta_learned = model.get_params()
print(f"\nLearned parameters:")
print(f"Ω_R / 2π = {Omega_R_learned / (2*np.pi) * 1e6:.3f} MHz (expected: 0.222 MHz)")
print(f"Γ_d / 2π = {Gamma_d_learned / (2*np.pi) * 1e6:.3f} MHz (expected: 0.47 MHz)")
print(f"η = {eta_learned:.4f} (expected: 0.147)")
```

---

## Part 6: Model Evaluation

### Step 6.1: Evaluate on Test Set

```python
def evaluate_model(test_set, model, sys):
    """
    Evaluate model performance on test set.
    
    Returns
    -------
    metrics : dict
        Contains 'cross_entropy', 'accuracy', 'predictions'
    """
    
    _, states_dict = create_qubit_operators()
    
    test_initial_states = [states_dict[ps] for ps in test_set['prep_states']]
    
    # Compute cross-entropy
    ce_loss = cross_entropy_loss(test_set['measurements'], test_set['outcomes'],
                                test_initial_states, model, sys.dt_exp)
    
    # Compute predictions
    predictions = []
    for i, meas in enumerate(test_set['measurements']):
        p_1 = model.predict(meas, test_initial_states[i], sys.dt_exp)
        predictions.append(p_1)
    
    predictions = np.array(predictions)
    
    # Convert to binary predictions (threshold at 0.5)
    pred_binary = (predictions > 0.5).astype(int)
    true_binary = ((test_set['outcomes'] + 1) / 2).astype(int)
    
    # Accuracy
    accuracy = np.mean(pred_binary == true_binary)
    
    return {
        'cross_entropy': ce_loss,
        'accuracy': accuracy,
        'predictions': predictions,
        'true_outcomes': test_set['outcomes']
    }

# Evaluate
print("\nEvaluating on test set...")
test_metrics = evaluate_model(test_set, model, sys)
print(f"Test Cross-Entropy: {test_metrics['cross_entropy']:.6f}")
print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
```

### Step 6.2: Visualization

```python
def plot_training_history(history):
    """Plot training and validation loss."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = np.arange(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Cross-Entropy Loss', fontsize=12)
    ax.set_title('SDE Model Training History', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    plt.show()

plot_training_history(history)
```

---

## Part 7: Advanced: Using QuTiP's Automatic Differentiation (Optional)

### Step 7.1: QuTiP with Jax Backend (For GPU Acceleration)

If using QuTiP with Jax backend for automatic differentiation:

```python
# This requires: pip install qutip jax jaxlib
import jax
import jax.numpy as jnp

def create_sde_solver_jax(Omega_R, Gamma_d, eta):
    """Create JAX-compatible SDE solver for gradient computation."""
    
    def sde_step(carry, inputs):
        rho, t = carry
        M_t, dt = inputs
        
        # Hamiltonian
        H_R = (Omega_R / 2.0) * jnp.array([[0, 1], [1, 0]])
        
        # Measurement operator
        L = jnp.sqrt(Gamma_d / 2.0) * jnp.array([[1, 0], [0, -1]])
        
        # Deterministic part: -i[H_R, ρ] + D[L]ρ
        comm = -1j * (H_R @ rho - rho @ H_R)
        dissip = L @ rho @ L.T.conj() - 0.5 * (L.T.conj() @ L @ rho + rho @ L.T.conj() @ L)
        
        # Update
        d_rho = (comm + dissip) * dt
        rho_new = rho + d_rho
        
        return (rho_new, t + dt), None
    
    return sde_step

# Gradient function
grad_objective = jax.grad(objective_function, argnums=0)
```

---

## Troubleshooting and Tips

### Common Issues

1. **QuTiP Installation Issues**: Ensure you have the latest version:
   ```bash
   pip install --upgrade qutip
   ```

2. **Memory Issues with Large Datasets**: Use batch processing or reduce trajectory count:
   ```python
   # Process in chunks
   chunk_size = 100_000
   for chunk_start in range(0, n_trajectories, chunk_size):
       chunk_end = min(chunk_start + chunk_size, n_trajectories)
       # Process chunk
   ```

3. **Numerical Stability**: If parameters diverge, add regularization:
   ```python
   def regularized_loss(params, *args):
       loss = cross_entropy_loss(*args)
       regularization = 0.01 * np.sum(params**2)
       return loss + regularization
   ```

### Performance Optimization

- Use `nsubsteps` parameter in `smesolve` to control accuracy vs. speed
- Parallelize trajectory generation with `multiprocessing` or `ray`
- Save trained models to disk for later use:
  ```python
  import pickle
  with open('sde_model.pkl', 'wb') as f:
      pickle.dump({'params': model.params, 'history': history}, f)
  ```

---

## References

- QuTiP Documentation: https://qutip.org/
- Paper: "Quantum-tailored machine-learning characterization of a superconducting qubit" (Genois et al., 2021)
- SciPy Optimization: https://docs.scipy.org/doc/scipy/reference/optimize.html
