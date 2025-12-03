# QuTiP-Specific Implementation Details

## Overview
This document provides detailed QuTiP-specific implementation patterns for quantum trajectory generation and SDE integration, tailored to the paper's methodology.

---

## 1. QuTiP SME Solver Configuration

### 1.1 Basic SME Setup

The stochastic master equation (SME) is solved using `qutip.stochastic.smesolve`:

```python
import qutip as qt
import numpy as np

# Define system
H = qt.sigmax() * omega  # Hamiltonian
rho0 = qt.basis(2, 0) * qt.basis(2, 0).dag()  # Initial state
L = qt.sigmaz()  # Stochastic collapse operator (measurement operator)

# Time points
times = np.linspace(0, 10, 1000)

# Solve SME
result = qt.smesolve(
    H,                          # Hamiltonian
    rho0,                       # Initial state
    times,                      # Time points
    c_ops=[],                   # Deterministic collapse operators (empty)
    sc_ops=[L],                 # Stochastic collapse operators
    e_ops=[qt.sigmax(), qt.sigmay(), qt.sigmaz()],  # Expectation values
    solver='fast-milstein',     # Use fast Milstein integrator
    nsubsteps=1,                # Number of substeps per output time
    options=qt.Options(
        nsteps=10000,           # Max steps
        rtol=1e-6,              # Relative tolerance
        atol=1e-8,              # Absolute tolerance
        store_states=True       # Store all states
    )
)
```

### 1.2 Solver Options

Available SME solvers in QuTiP:

```python
# Solvers (from fastest to most accurate)
solver='euler'              # Order 0.5 (fast but less accurate)
solver='milstein'           # Order 1.0 (good balance)
solver='fast-milstein'      # Order 1.0 optimized (recommended)
solver='taylor15'           # Order 1.5 (most accurate, slowest)

# Methods
method='homodyne'           # Homodyne detection (standard)
method='heterodyne'         # Heterodyne detection
```

### 1.3 Understanding the SME Form in QuTiP

QuTiP implements the SME as:

$$d\rho = -i[H, \rho]dt + \mathcal{D}[L_c]\rho dt + \sum_i \mathcal{H}[S_i]\rho dW_i$$

Where:
- `c_ops`: List of classical collapse operators (L_c) for deterministic dissipation
- `sc_ops`: List of stochastic collapse operators for measurement backaction
- `dW_i`: Independent Wiener increments

---

## 2. Measurement Record Extraction

### 2.1 Getting the Measurement Signal

QuTiP doesn't directly return measurement records; they must be computed from the solution:

```python
def extract_measurement_records(result, L, eta, method='homodyne'):
    """
    Extract weak measurement records from SME solution.
    
    For measurement operator L, the measurement signal is:
    dM^I = sqrt(eta) * Tr(L*rho + rho*L†) dt + dW^I
    
    Parameters
    ----------
    result : Result object
        SME solution from smesolve
    L : Qobj
        Measurement operator
    eta : float
        Quantum efficiency (0 to 1)
    method : str
        'homodyne' or 'heterodyne'
    
    Returns
    -------
    M_I, M_Q : ndarray
        I and Q quadrature measurement signals
    """
    
    M_I = []
    M_Q = []
    
    L_dag = L.dag()
    
    for t_idx, rho in enumerate(result.states):
        # Deterministic signal component
        signal = (L * rho + rho * L_dag).tr().real
        
        # Add quantum noise
        dW_I = np.random.randn() * np.sqrt(result.times[1] - result.times[0])
        dW_Q = np.random.randn() * np.sqrt(result.times[1] - result.times[0])
        
        # Measurement record
        m_I = np.sqrt(eta) * signal + dW_I
        m_Q = np.sqrt(eta) * signal + dW_Q if method == 'heterodyne' else m_I
        
        M_I.append(m_I)
        M_Q.append(m_Q)
    
    return np.array(M_I), np.array(M_Q)
```

### 2.2 Coarse-Graining Measurements

QuTiP simulations use fine time steps; experimental data uses coarser steps:

```python
def coarse_grain_measurement(M_fine, dt_fine, dt_target):
    """
    Downsample measurement records to experimental time scale.
    
    Parameters
    ----------
    M_fine : ndarray
        Fine-grained measurement record
    dt_fine : float
        Fine time step (from QuTiP simulation)
    dt_target : float
        Target (experimental) time step
    
    Returns
    -------
    M_coarse : ndarray
        Coarse-grained measurement
    """
    
    n_coarse = int(np.round(dt_target / dt_fine))
    n_total = len(M_fine)
    n_output = n_total // n_coarse
    
    M_coarse = np.zeros(n_output)
    
    for i in range(n_output):
        start = i * n_coarse
        end = start + n_coarse
        # Average over coarse interval
        M_coarse[i] = np.mean(M_fine[start:end])
    
    return M_coarse
```

---

## 3. Advanced: Custom Stochastic Operators

### 3.1 Heterodyne Detection

For two-quadrature measurement (I and Q):

```python
def heterodyne_sme_solve(H, rho0, times, L_measurement, config):
    """
    Solve SME with heterodyne detection (both quadratures).
    
    In heterodyne detection:
    dM^I = sqrt(eta) * Tr(L*rho + rho*L†) dt + dW^I
    dM^Q = sqrt(eta) * i*Tr(L*rho - rho*L†) dt + dW^Q
    """
    
    L = L_measurement
    L_dag = L.dag()
    
    # For heterodyne, we need two independent stochastic operators
    # representing the two quadratures
    sc_ops = [L, 1j * L]  # Two quadratures
    
    result = qt.smesolve(
        H,
        rho0,
        times,
        c_ops=[],
        sc_ops=sc_ops,
        e_ops=[qt.sigmax(), qt.sigmay(), qt.sigmaz()],
        solver='fast-milstein',
        options=qt.Options(store_states=True, rtol=1e-6, atol=1e-8)
    )
    
    return result
```

### 3.2 Multiple Stochastic Operators

For systems with multiple measurement channels:

```python
# Example: Simultaneous measurement of two qubits
L1 = qt.tensor(qt.sigmaz(), qt.qeye(2))  # Measure qubit 1
L2 = qt.tensor(qt.qeye(2), qt.sigmaz())  # Measure qubit 2

result = qt.smesolve(
    H,
    rho0,
    times,
    c_ops=[],
    sc_ops=[L1, L2],  # Multiple independent measurement channels
    solver='fast-milstein'
)
```

---

## 4. Parameter Estimation with Numerical Integration

### 4.1 Creating a Differentiable Wrapper

For parameter optimization, wrap the SME solver to be differentiable:

```python
from scipy.optimize import minimize

class ParameterizedSDE:
    """Wrapper for parametrized SDE integration."""
    
    def __init__(self, config):
        self.config = config
    
    def create_hamiltonian(self, params):
        """Create Hamiltonian from parameters."""
        Omega_R = params[0]
        return (Omega_R / 2.0) * qt.sigmax()
    
    def create_measurement_ops(self, params):
        """Create measurement operators from parameters."""
        Gamma_d = params[1]
        return [np.sqrt(Gamma_d / 2.0) * qt.sigmaz()]
    
    def forward(self, params, rho0, times, measurement_record):
        """
        Forward pass: integrate SME with given parameters.
        
        Parameters
        ----------
        params : array
            [Omega_R, Gamma_d, eta]
        rho0 : Qobj
            Initial state
        times : array
            Time points
        measurement_record : array
            Weak measurement data (driver for stochastic evolution)
        
        Returns
        -------
        final_state : Qobj
            Final density matrix
        """
        
        H = self.create_hamiltonian(params)
        sc_ops = self.create_measurement_ops(params)
        
        result = qt.smesolve(
            H,
            rho0,
            times,
            c_ops=[],
            sc_ops=sc_ops,
            solver='fast-milstein',
            nsubsteps=1,
            options=qt.Options(rtol=1e-6, atol=1e-8, store_states=True)
        )
        
        return result.states[-1]
    
    def loss(self, params, rho0, times, measurement_record, true_outcome):
        """Compute loss for parameter optimization."""
        
        rho_final = self.forward(params, rho0, times, measurement_record)
        
        # Measurement probability
        proj_1 = qt.basis(2, 1) * qt.basis(2, 1).dag()
        p_1 = (proj_1 * rho_final).tr().real
        
        # Convert true_outcome from ±1 to {0, 1}
        y = (true_outcome + 1) / 2
        
        # Binary cross-entropy
        p_1 = np.clip(p_1, 1e-8, 1 - 1e-8)
        loss_val = -(y * np.log(p_1) + (1 - y) * np.log(1 - p_1))
        
        return loss_val
```

### 4.2 Batch Optimization

```python
def optimize_parameters_batch(param_sde, batch_data, initial_params):
    """
    Optimize parameters on a batch of trajectories.
    
    Parameters
    ----------
    param_sde : ParameterizedSDE
        Parametrized SDE model
    batch_data : list of tuples
        [(rho0, times, M_record, outcome), ...]
    initial_params : array
        Initial parameter guess
    
    Returns
    -------
    result : OptimizeResult
        Optimization result
    """
    
    def batch_loss(params):
        """Loss summed over batch."""
        total_loss = 0.0
        for rho0, times, M_record, outcome in batch_data:
            total_loss += param_sde.loss(params, rho0, times, M_record, outcome)
        return total_loss / len(batch_data)
    
    result = minimize(
        batch_loss,
        initial_params,
        method='L-BFGS-B',
        options={'maxiter': 100, 'ftol': 1e-8}
    )
    
    return result
```

---

## 5. Performance Optimization Tips

### 5.1 Reducing Computation Time

```python
# 1. Use fast-milstein solver
solver = 'fast-milstein'

# 2. Minimize output time points
times = np.linspace(0, T, 200)  # Only 200 output times
# SME solves internally with fine step

# 3. Disable state storage if not needed
options=qt.Options(store_states=False)  # Much faster!

# 4. Use smaller tolerances for faster (less accurate) solutions
options=qt.Options(rtol=1e-5, atol=1e-7)  # vs default 1e-6, 1e-8

# 5. Reduce number of substeps
nsubsteps=1  # Minimum for Milstein

# Combined example for speed
result = qt.smesolve(
    H, rho0, times,
    c_ops=[], sc_ops=[L],
    solver='fast-milstein',
    nsubsteps=1,
    options=qt.Options(
        store_states=False,
        rtol=1e-5,
        atol=1e-7,
        nsteps=5000
    )
)
```

### 5.2 Parallelization

```python
from multiprocessing import Pool
import functools

def generate_single_trajectory_wrapper(i, config):
    """Wrapper for parallelization."""
    rho0 = random_preparation()
    result = qt.smesolve(...)
    return result

# Parallel trajectory generation
n_trajectories = 1000
n_jobs = 8

with Pool(n_jobs) as pool:
    results = pool.map(
        functools.partial(generate_single_trajectory_wrapper, config=config),
        range(n_trajectories)
    )
```

### 5.3 GPU Acceleration (with QuTiP + CuPy)

```python
# Requires: pip install cupy-cuda11x (adjust 11x for your CUDA version)

import qutip as qt

# Enable GPU backend
qt.settings.core['SPARSE_SOLVER'] = 'GPU'

# QuTiP will automatically use GPU for matrix operations
result = qt.smesolve(...)
```

---

## 6. Verification and Testing

### 6.1 Verify Positivity

```python
def check_positivity(rho):
    """Verify density matrix is positive."""
    eigenvalues = rho.eigenenergies()
    return np.all(eigenvalues >= -1e-10)

# Check all states in trajectory
for state in result.states:
    assert check_positivity(state), "Non-positive state encountered!"
```

### 6.2 Verify Normalization

```python
def check_normalization(rho):
    """Verify density matrix is normalized."""
    trace = rho.tr().real
    return np.isclose(trace, 1.0, atol=1e-6)

# Check
for state in result.states:
    assert check_normalization(state), "Non-normalized state!"
```

### 6.3 Verify Conservation Laws

For a system with H and Lindbladian L = L†L:

```python
def verify_conservation(result, H, L):
    """Verify conservation of expected properties."""
    
    for i, state in enumerate(result.states):
        # Energy should vary smoothly
        E = (H * state).tr().real
        
        # Purity (for pure initial state) should decay
        purity = (state * state).tr().real
        
        if i > 0:
            assert 0 <= purity <= 1.01, f"Invalid purity: {purity}"

verify_conservation(result, H, L)
```

---

## 7. Debugging Common Issues

### Issue 1: NaN or Inf in Solution

**Cause**: Time step too large, numerical instability

**Solution**:
```python
# Reduce time step
dt_sim = 0.0001  # Smaller than default 0.001

# Increase accuracy
options=qt.Options(rtol=1e-8, atol=1e-10)

# Use more robust integrator
solver='taylor15'
```

### Issue 2: Memory Issues with Large Simulations

**Cause**: Storing all intermediate states

**Solution**:
```python
# Don't store states if not needed
options=qt.Options(store_states=False)

# Or process in chunks
n_chunks = 10
for chunk in range(n_chunks):
    times_chunk = np.linspace(chunk*T/n_chunks, (chunk+1)*T/n_chunks, 100)
    result = qt.smesolve(..., times=times_chunk)
    # Process and discard result
```

### Issue 3: Slow Convergence

**Cause**: Poor parameter initialization

**Solution**:
```python
# Use multiple random initializations
best_result = None
best_loss = float('inf')

for trial in range(10):
    params_init = np.random.randn(3) * 0.1 + params_true
    result = minimize(loss_fn, params_init, ...)
    
    if result.fun < best_loss:
        best_loss = result.fun
        best_result = result
```

---

## 8. Example: Complete Workflow

```python
import qutip as qt
import numpy as np
from scipy.optimize import minimize

# 1. Setup
config = SystemConfig()
times = np.linspace(0, 8, 200)
dt_fine = 0.001

# 2. Generate trajectory
H = (0.222e-6 * 2 * np.pi / 2) * qt.sigmax()
L = np.sqrt(0.47e-6 * 2 * np.pi / 2) * qt.sigmaz()
rho0 = qt.basis(2, 0) * qt.basis(2, 0).dag()

result = qt.smesolve(
    H, rho0, times,
    c_ops=[], sc_ops=[L],
    solver='fast-milstein',
    options=qt.Options(store_states=True, rtol=1e-6, atol=1e-8)
)

# 3. Extract measurement record
M_fine = extract_measurement_records(result, L, eta=0.147)[0]
M_coarse = coarse_grain_measurement(M_fine, dt_fine, config.dt_exp)

# 4. Get true outcome
rho_final = result.states[-1]
p_1 = (qt.basis(2, 1) * qt.basis(2, 1).dag() * rho_final).tr().real
true_outcome = 1 if np.random.rand() < p_1 else -1

# 5. Train model (see earlier sections)
# ...
```

---

## References

- QuTiP Documentation: https://qutip.org/docs/latest/
- SME Solver: https://qutip.org/docs/latest/guide/dynamics/dynamics-stochastic.html
- Milstein Method: Kloeden & Platen, "Numerical Solution of Stochastic Differential Equations"
