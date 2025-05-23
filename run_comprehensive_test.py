#!/usr/bin/env python3
"""
Comprehensive test script to verify all H2 optimization components are working.
"""

import subprocess
import sys

def run_test(test_name, script_name):
    """Run a test script and return success status."""
    try:
        result = subprocess.run(['python', script_name], capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(f'✅ {test_name} - PASSED')
            return True
        else:
            print(f'❌ {test_name} - FAILED')
            print(f'   Error: {result.stderr.strip()[:100]}...')
            return False
    except subprocess.TimeoutExpired:
        print(f'⏰ {test_name} - TIMEOUT')
        return False
    except Exception as e:
        print(f'💥 {test_name} - ERROR: {e}')
        return False

def main():
    print('🔧 Running Comprehensive H2 Optimization Test')
    print('=' * 50)
    
    tests = [
        ('Test 1: Pyomo Component Fix', 'test_pyomo_component_fix.py'),
        ('Test 2: Variable Scoping Fix', 'test_variable_scoping_fix.py'),
        ('Test 3: H2 Revenue Calculation', 'test_h2_revenue_calculation.py'),
        ('Test 4: Overall H2 Optimization', 'test_h2_optimization.py'),
    ]
    
    results = []
    for test_name, script_name in tests:
        success = run_test(test_name, script_name)
        results.append(success)
    
    print('\n' + '=' * 50)
    if all(results):
        print('🎉 ALL TESTS PASSED! H2 optimization system is ready.')
        print('\n✅ Summary of fixes verified:')
        print('  • Variable scoping issue resolved')
        print('  • Pyomo component assignment fixed')
        print('  • H2 revenue calculation corrected')
        print('  • Complete H2 optimization workflow functional')
        return True
    else:
        failed_count = len([r for r in results if not r])
        print(f'❌ {failed_count} out of {len(tests)} tests failed.')
        print('   Please check the implementation.')
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 