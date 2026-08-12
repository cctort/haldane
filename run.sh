#!/bin/bash
# Convenience script for running tests and computations

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

print_header() {
    echo -e "\n${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Default action
ACTION="${1:-test}"

case "$ACTION" in
    test)
        print_header "Running unit tests"
        python scripts/test_basics.py
        if [ $? -eq 0 ]; then
            print_success "All tests passed!"
        else
            print_error "Tests failed!"
            exit 1
        fi
        ;;
    
    phase_diagram)
        print_header "Computing phase diagrams"
        print_header "This will take 2-4 hours..."
        python scripts/phase_diagram.py
        print_success "Phase diagrams computed!"
        ;;
    
    plot)
        print_header "Plotting results"
        python scripts/plot_results.py
        ;;
    
    quick)
        print_header "Quick test (single point)"
        python -c "
import sys
sys.path.insert(0, '.')
from src.config import print_config
from src.hamiltonian import HaldaneHubbardHamiltonian
from src.ed_solver import ExactDiagonalizationSolver
from src.observables import Observables
from src.lattice import get_lattice

print_config()

print('\nBuilding Hamiltonian...')
ham = HaldaneHubbardHamiltonian()

print('Solving ground state...')
solver = ExactDiagonalizationSolver(ham)
E_gs, psi_gs = solver.solve(delta=0.5, U=1.0, V=0.0)

print(f'Ground state energy: {E_gs:.6f}')

print('Computing observables...')
obs = Observables(ham.basis, get_lattice())
cdw = obs.compute_cdw_structure_factor(psi_gs)
sdw = obs.compute_sdw_structure_factor(psi_gs)

print(f'CDW: {cdw:.6f}')
print(f'SDW: {sdw:.6f}')
print('\n✓ Quick test completed successfully!')
"
        ;;
    
    install)
        print_header "Installing dependencies"
        pip install -q -r requirements.txt
        print_success "Dependencies installed!"
        ;;
    
    clean)
        print_header "Cleaning up"
        find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
        find . -type f -name "*.pyc" -delete 2>/dev/null || true
        rm -rf .pytest_cache 2>/dev/null || true
        print_success "Cleaned!"
        ;;
    
    help|--help|-h)
        cat << EOF
Haldane-Hubbard ED Code - Convenience Script

Usage: bash run.sh [ACTION]

Actions:
  test              Run unit tests (default)
  quick             Quick test with single point
  phase_diagram     Compute full phase diagrams (2-4 hours)
  plot              Plot phase diagram results
  install           Install Python dependencies
  clean             Clean up temporary files
  help              Show this help message

Examples:
  bash run.sh test          # Run all tests
  bash run.sh quick         # Quick sanity check
  bash run.sh phase_diagram # Compute phase diagrams
  bash run.sh plot          # Visualize results

For detailed documentation, see README.md
EOF
        ;;
    
    *)
        print_error "Unknown action: $ACTION"
        echo "Run 'bash run.sh help' for usage information"
        exit 1
        ;;
esac

exit 0
