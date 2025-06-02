import numpy as np
import dimod
from dwave.system import EmbeddingComposite, DWaveSampler
from scipy.sparse import csc_matrix, issparse

def axb_to_qubo(A, b, num_bits=4, x_range=(-10, 10)):
    """
    Converts Ax = b into a QUBO formulation.
    Each x_i is encoded with `num_bits` bits, scaled to `x_range`.
    """
    n = A.shape[1]
    scale = (x_range[1] - x_range[0]) / (2**num_bits - 1)
    offset = x_range[0]
    
    # Binary variable names: x_i_j = bit j of x_i
    qubo = dimod.BinaryQuadraticModel('BINARY')
    
    for i in range(n):
        for j in range(num_bits):
            var = f'x_{i}_{j}'
            qubo.add_variable(var)
    
    # Build QUBO for objective ||Ax - b||²
    for row in range(A.shape[0]):
        coeffs = []
        vars = []
        for i in range(n):
            a = A[row, i]
            for j in range(num_bits):
                bit_val = 2**j * scale
                coeffs.append(a * bit_val)
                vars.append(f'x_{i}_{j}')
        
        # Now form the squared term: (sum(coeffs * vars) - b[row])²
        for k in range(len(vars)):
            for l in range(k, len(vars)):
                term = coeffs[k] * coeffs[l]
                if k == l:
                    qubo.add_variable(vars[k], qubo.get_linear(vars[k], 0.0) + term)
                else:
                    qubo.add_interaction(vars[k], vars[l],
                                         qubo.get_quadratic(vars[k], vars[l], 0.0) + 2 * term)
            qubo.offset += b[row]**2  # constant term
            for m in range(len(vars)):
                qubo.add_linear(vars[m], qubo.get_linear(vars[m], 0.0) - 2 * coeffs[m] * b[row])
    
    return qubo, scale, offset

def decode_solution(sample, n, num_bits, scale, offset):
    """
    Convert binary sample to real-valued x vector.
    """
    x = np.zeros(n)
    for i in range(n):
        value = 0
        for j in range(num_bits):
            bit = sample[f'x_{i}_{j}']
            value += bit * (2 ** j)
        x[i] = value * scale + offset
    return x

def solve_sparse_linear_dwave(A, b, num_bits=4, x_range=(-10, 10), use_qpu=False):
    """
    Solve Ax = b using D-Wave by converting to QUBO.
    """
    if issparse(A):
        A = A.toarray()

    qubo, scale, offset = axb_to_qubo(A, b, num_bits, x_range)

    if use_qpu:
        sampler = EmbeddingComposite(DWaveSampler())
    else:
        import neal
        sampler = neal.SimulatedAnnealingSampler()

    sampleset = sampler.sample(qubo, num_reads=100)
    best_sample = sampleset.first.sample
    x = decode_solution(best_sample, A.shape[1], num_bits, scale, offset)
    
    return x
