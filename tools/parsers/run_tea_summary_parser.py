#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runner script for TEA Summary Parser
Executes the parsing and analysis of all 42 reactor TEA Summary Reports
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.tea_summary_parser import TEASummaryParser
import logging

def main():
    """Main execution function"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('tea_parsing.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting TEA Summary Report parsing for 42 reactors...")
    
    try:
        # Initialize parser
        parser = TEASummaryParser(tea_output_dir="output/tea/cs1")
        
        # Parse all reactors
        logger.info("Step 1: Parsing all reactor TEA Summary Reports...")
        reactor_data = parser.parse_all_reactors()
        
        if not reactor_data:
            logger.error("No reactor data was successfully parsed. Please check:")
            logger.error("1. TEA Summary Report files exist in output/tea/cs1/")
            logger.error("2. Files follow the expected naming pattern (*TEA_Summary_Report.txt)")
            logger.error("3. File content matches the expected report format")
            return
        
        logger.info(f"Successfully parsed {len(reactor_data)} reactors")
        
        # Create Excel output
        logger.info("Step 2: Creating Excel analysis file...")
        parser.create_excel_output("output/tea/tea_summary_analysis.xlsx")
        
        # Generate summary report
        logger.info("Step 3: Generating summary report...")
        parser.generate_summary_report("output/tea/reactor_summary.txt")
        
        logger.info("=" * 60)
        logger.info("TEA Summary parsing completed successfully!")
        logger.info("=" * 60)
        logger.info("Output files created:")
        logger.info("- Excel analysis: output/tea/tea_summary_analysis.xlsx")
        logger.info("- Summary report: output/tea/reactor_summary.txt")
        logger.info("- Log file: tea_parsing.log")
        logger.info("=" * 60)
        
        # Print quick summary
        logger.info("\nQuick Summary:")
        logger.info(f"Total reactors processed: {len(reactor_data)}")
        
        # Show sample of parsed data
        sample_reactor = list(reactor_data.keys())[0]
        sample_data = reactor_data[sample_reactor]
        logger.info(f"Sample reactor: {sample_reactor}")
        logger.info(f"Data fields extracted: {len(sample_data)} fields")
        
        # Show key metrics if available
        if 'npv_usd' in sample_data:
            logger.info(f"Sample NPV: ${sample_data['npv_usd']:,.0f}")
        if 'irr_percent' in sample_data:
            logger.info(f"Sample IRR: {sample_data['irr_percent']:.2f}%")
        if 'lcoh_usd_per_kg' in sample_data:
            logger.info(f"Sample LCOH: ${sample_data['lcoh_usd_per_kg']:.3f}/kg")
            
    except Exception as e:
        logger.error(f"Error during TEA parsing: {str(e)}", exc_info=True)
        logger.error("Please check the log file for detailed error information")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 