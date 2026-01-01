#!/usr/bin/env python3
"""
Notebook validation script for SynaDB v1.0.0 Showcase.

This script validates all notebooks in the showcase:
1. Checks JSON structure validity
2. Verifies required sections exist
3. Validates Python syntax in code cells
4. Reports any issues found

Usage:
    python validate_notebooks.py
    python validate_notebooks.py --verbose
    python validate_notebooks.py --fix  # Add missing comments to cells
"""

import os
import sys
import json
import ast
import glob
import argparse
from typing import List, Dict, Any, Tuple


def get_all_notebooks() -> List[str]:
    """Get all notebook paths in the v1.0.0 showcase."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    notebooks = []
    
    for subdir in ['vector_stores', 'experiment_tracking', 'data_loading', 
                   'model_registry', 'llm_frameworks', 'specialized']:
        pattern = os.path.join(base_dir, subdir, '*.ipynb')
        notebooks.extend(glob.glob(pattern))
    
    return sorted(notebooks)


def load_notebook(path: str) -> Dict[str, Any]:
    """Load a Jupyter notebook as JSON."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_cell_source(cell: Dict[str, Any]) -> str:
    """Get the source content of a cell as a string."""
    source = cell.get('source', [])
    if isinstance(source, list):
        return ''.join(source)
    return source


def validate_json_structure(path: str) -> List[str]:
    """Validate notebook JSON structure."""
    errors = []
    try:
        nb = load_notebook(path)
        
        if 'nbformat' not in nb:
            errors.append("Missing nbformat field")
        elif nb['nbformat'] < 4:
            errors.append(f"Notebook format too old: {nb['nbformat']}")
        
        if 'cells' not in nb:
            errors.append("Missing cells field")
        elif len(nb['cells']) == 0:
            errors.append("No cells in notebook")
            
    except json.JSONDecodeError as e:
        errors.append(f"Invalid JSON: {e}")
    except Exception as e:
        errors.append(f"Error loading notebook: {e}")
    
    return errors


def validate_python_syntax(path: str) -> List[str]:
    """Validate Python syntax in code cells."""
    errors = []
    try:
        nb = load_notebook(path)
        
        for i, cell in enumerate(nb.get('cells', [])):
            if cell.get('cell_type') != 'code':
                continue
            
            source = get_cell_source(cell)
            if not source.strip():
                continue
            
            try:
                ast.parse(source)
            except SyntaxError as e:
                errors.append(f"Cell {i+1}: Syntax error at line {e.lineno}: {e.msg}")
                
    except Exception as e:
        errors.append(f"Error validating syntax: {e}")
    
    return errors


def validate_required_sections(path: str) -> List[str]:
    """Validate required notebook sections exist."""
    errors = []
    try:
        nb = load_notebook(path)
        all_source = ' '.join(get_cell_source(c) for c in nb.get('cells', []))
        
        # Check for header
        if 'display_header' not in all_source:
            errors.append("Missing display_header call")
        
        # Check for TOC
        if 'sections' not in all_source and 'Table of Contents' not in all_source:
            errors.append("Missing Table of Contents")
        
        # Check for conclusion
        if 'Conclusion' not in all_source and 'conclusion_box' not in all_source:
            errors.append("Missing conclusions section")
        
        # Check for system info
        if 'display_system_info' not in all_source and 'system_info' not in all_source:
            errors.append("Missing system info display")
            
    except Exception as e:
        errors.append(f"Error validating sections: {e}")
    
    return errors


def validate_imports(path: str) -> List[str]:
    """Validate required imports exist."""
    errors = []
    try:
        nb = load_notebook(path)
        code_cells = [c for c in nb.get('cells', []) if c.get('cell_type') == 'code']
        
        if not code_cells:
            errors.append("No code cells found")
            return errors
        
        first_source = get_cell_source(code_cells[0])
        
        if 'from utils' not in first_source and 'import utils' not in first_source:
            errors.append("First cell should import from utils module")
            
    except Exception as e:
        errors.append(f"Error validating imports: {e}")
    
    return errors


def validate_notebook(path: str, verbose: bool = False) -> Tuple[bool, List[str]]:
    """Validate a single notebook."""
    all_errors = []
    
    # JSON structure
    errors = validate_json_structure(path)
    if errors:
        all_errors.extend([f"[Structure] {e}" for e in errors])
    
    # Python syntax
    errors = validate_python_syntax(path)
    if errors:
        all_errors.extend([f"[Syntax] {e}" for e in errors])
    
    # Required sections
    errors = validate_required_sections(path)
    if errors:
        all_errors.extend([f"[Sections] {e}" for e in errors])
    
    # Imports
    errors = validate_imports(path)
    if errors:
        all_errors.extend([f"[Imports] {e}" for e in errors])
    
    return len(all_errors) == 0, all_errors


def main():
    parser = argparse.ArgumentParser(description='Validate SynaDB v1.0.0 Showcase notebooks')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()
    
    notebooks = get_all_notebooks()
    
    print(f"Validating {len(notebooks)} notebooks...\n")
    
    passed = 0
    failed = 0
    all_issues = []
    
    for path in notebooks:
        name = os.path.basename(path)
        valid, errors = validate_notebook(path, args.verbose)
        
        if valid:
            print(f"  [PASS] {name}")
            passed += 1
        else:
            print(f"  [FAIL] {name}")
            failed += 1
            all_issues.append((name, errors))
            if args.verbose:
                for error in errors:
                    print(f"      {error}")
    
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    
    if all_issues:
        print(f"\nIssues found in {len(all_issues)} notebooks:")
        for name, errors in all_issues:
            print(f"\n  {name}:")
            for error in errors:
                print(f"    - {error}")
    else:
        print("\nAll notebooks validated successfully!")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
