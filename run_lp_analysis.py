"""
Quick Run Script for Local Projections Analysis
================================================
Simple script to run the Local Projections analysis with one command.

Usage:
    python run_lp_analysis.py
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from local_projections_thailand import main

if __name__ == '__main__':
    print("\n" + "🇹🇭 "*20)
    print("Thailand DVA Local Projections Analysis")
    print("🇹🇭 "*20 + "\n")
    
    try:
        # Run main analysis
        lp_analysis = main()
        
        print("\n✅ Analysis completed successfully!")
        print("\nYou can access the results object as: lp_analysis")
        print("  - lp_analysis.data: The merged dataset")
        print("  - lp_analysis.results: Local Projections results")
        print("  - lp_analysis.adf_results: ADF test results")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
