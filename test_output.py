#!/usr/bin/env python3
"""
Script to run main.test_schema_tools() and capture the output.
"""
import sys
import io
import traceback

# Redirect stdout to capture output
original_stdout = sys.stdout
captured_output = io.StringIO()
sys.stdout = captured_output

try:
    import main
    main.test_schema_tools()
except Exception as e:
    print(f"\nError: {type(e).__name__}: {e}")
    traceback.print_exc()
finally:
    # Restore stdout
    sys.stdout = original_stdout
    
    # Print captured output
    print("===== CAPTURED OUTPUT =====")
    print(captured_output.getvalue()) 