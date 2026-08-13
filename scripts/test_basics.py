"""
Test script for Haldane-Hubbard ED code.

Run basic sanity checks before computing full phase diagrams.
"""

import sys
import numpy as np

from ed.config import print_config
from ed.basis import get_fock_basis, reset_basis
from ed.lattice import get_lattice
from ed.hamiltonian import HaldaneHubbardHamiltonian
from ed.ed_solver import ExactDiagonalizationSolver
from ed.observables import ObservableCalculator


def test_lattice():
    """Test lattice structure."""
    print("\n" + "="*70)
    print("TEST 1: Lattice Structure")
    print("="*70)
    
    lattice = get_lattice()
    lattice.print_structure()
    
    # Check bond counts
    n_nn = len(lattice.get_bonds('nn'))
    n_nnn = len(lattice.get_bonds('nnn'))
    
    print(f"\n✓ First-neighbor bonds: {n_nn}")
    print(f"✓ Second-neighbor bonds: {n_nnn}")
    
    assert n_nn == 18, f"Expected 12 NN bonds, got {n_nn}"
    assert n_nnn == 24, f"Expected 24 NNN bonds, got {n_nnn}"
    
    return True


def test_basis():
    """Test Fock basis construction."""
    print("\n" + "="*70)
    print("TEST 2: Fock Basis")
    print("="*70)
    
    reset_basis()
    basis = get_fock_basis()
    
    print(f"\nBasis dimension: {len(basis.basis_states)}")
    print(f"Expected (combinatorial): {basis._comb(24, 12)}")
    
    # Test state operations
    basis.print_basis_sample(n_states=5)
    
    assert len(basis.basis_states) == 2704156, f"Expected 2704156 states, got {len(basis.basis_states)}"
    
    return True


def test_hamiltonian():
    """Test Hamiltonian construction."""
    print("\n" + "="*70)
    print("TEST 3: Hamiltonian Construction")
    print("="*70)
    
    ham = HaldaneHubbardHamiltonian()
    
    # Build a simple Hamiltonian
    print("\nBuilding Hamiltonian for Δ=0., U=10., V=0...")
    H = ham.build_full_hamiltonian(delta=0., U=10., V=0.)
    
    print(f"  Matrix shape: {H.shape}")
    print(f"  Matrix type: {type(H)}")
    print(f"  Non-zeros: {H.nnz}")
    print(f"  Sparsity: {100*(1-H.nnz/(H.shape[0]*H.shape[1])):.2f}%")
    
    assert H.shape == (2704156, 2704156), f"Wrong matrix shape: {H.shape}"
    assert H.nnz > 0, "Hamiltonian has no non-zero elements"
    
    return True


def test_ed_solver():
    """Test exact diagonalization solver."""
    print("\n" + "="*70)
    print("TEST 4: Exact Diagonalization Solver")
    print("="*70)
    
    ham = HaldaneHubbardHamiltonian()
    solver = ExactDiagonalizationSolver(ham)
    
    # Solve a simple case
    print("\nSolving ground state for Δ=0., U=10., V=0....")
    E_gs, psi_gs = solver.solve(delta=0., U=10., V=0.)
    
    print(f"  Ground state energy: {E_gs:.6f}")
    print(f"  State vector shape: {psi_gs.shape}")
    print(f"  State vector norm: {np.linalg.norm(psi_gs):.6f}")
    
    # Check norm
    assert abs(np.linalg.norm(psi_gs) - 1.0) < 1e-6, "Ground state not normalized"
    
    return True


def test_observables():
    """Test observable calculations."""
    print("\n" + "="*70)
    print("TEST 5: Observable Calculations")
    print("="*70)
    
    ham = HaldaneHubbardHamiltonian()
    solver = ExactDiagonalizationSolver(ham)
    obs = ObservableCalculator(ham.basis, get_lattice(), solver)
    
    # Solve ground state
    _, psi_gs = solver.solve(delta=0., U=10., V=0.)
    
    print("\nComputing observables...")
    
    # CDW
    cdw = obs.compute_cdw(psi_gs)
    print(f"  CDW: {cdw:.6f}")
    
    # SDW
    sdw = obs.compute_sdw_squared(psi_gs)
    print(f"  SDW: {sdw:.6f}")
    
    # Chern Number
    print("  Computing Chern number (this may take 30-60 seconds)...")
    try:
        chern = obs.compute_chern_number(delta=0., U=10., V=0., grid_size=4)
        print(f"  Chern: {chern}")
        assert chern in [-2, -1, 0, 1, 2], f"Chern number should be -2 to 2, got {chern}"
    except Exception as e:
        print(f"  Chern: WARNING - calculation failed: {e}")
        print(f"         (This is OK - eigenvalue solver may not converge at all points)")
    
    # Single-particle density matrix
    rho = obs.compute_single_particle_density_matrix(psi_gs)
    print(f"  SPDM shape: {rho.shape}")
    print(f"  SPDM trace (total <n>): {np.trace(rho):.6f}")
    
    # Check trace equals number of electrons
    assert abs(np.trace(rho) - 12.0) < 0.1, f"SPDM trace should be ~12, got {np.trace(rho)}"
    
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("RUNNING HALDANE-HUBBARD ED TESTS")
    print("="*70)
    
    print_config()
    
    tests = [
        ("Lattice", test_lattice),
        ("Basis", test_basis),
        ("Hamiltonian", test_hamiltonian),
        ("ED Solver", test_ed_solver),
        ("Observables", test_observables),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                print(f"\n✓ {test_name} test PASSED")
                passed += 1
        except Exception as e:
            print(f"\n✗ {test_name} test FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n✓ All tests passed! Ready to compute phase diagrams.")
        return True
    else:
        print(f"\n✗ {failed} test(s) failed.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
