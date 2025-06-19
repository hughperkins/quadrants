#!/usr/bin/env python3
"""
Test script to demonstrate separate struct compilation in Taichi.
This shows how struct PTX can be compiled separately from kernels.
"""

import taichi as ti
import time

def test_struct_compilation_separation():
    """Test that struct compilation is separate from kernel compilation."""
    
    print("=== Testing Separate Struct Compilation ===")
    
    # Initialize Taichi with CUDA
    ti.init(arch=ti.cuda, print_ir=True)
    
    # Create a simple field
    x = ti.field(ti.i32, shape=(5, 5))
    
    print("\n--- Step 1: Creating SNode Tree ---")
    print("This should trigger struct compilation to PTX")
    
    # This should trigger struct compilation
    x.fill(0)
    
    print("\n--- Step 2: Compiling First Kernel ---")
    @ti.kernel
    def kernel1():
        for i, j in x:
            x[i, j] = i * 10 + j
    
    # This should use the pre-compiled struct PTX
    kernel1()
    
    print("\n--- Step 3: Compiling Second Kernel ---")
    @ti.kernel
    def kernel2():
        for i, j in x:
            x[i, j] = x[i, j] * 2
    
    # This should reuse the same struct PTX without recompilation
    kernel2()
    
    print("\n--- Step 4: Verifying Results ---")
    print("Expected: Each element should be (i*10 + j) * 2")
    print("Actual values:")
    for i in range(5):
        for j in range(5):
            print(f"x[{i},{j}] = {x[i, j]}", end=" ")
        print()
    
    print("\n=== Test Complete ===")

def test_different_field_sizes():
    """Test that changing field dimensions doesn't trigger kernel recompilation."""
    
    print("\n=== Testing Field Size Independence ===")
    
    ti.init(arch=ti.cuda, print_ir=True)
    
    # Test with different field sizes
    for size in [3, 5, 7]:
        print(f"\n--- Testing with field size {size}x{size} ---")
        
        # Create field with different size
        x = ti.field(ti.i32, shape=(size, size))
        
        @ti.kernel
        def test_kernel():
            for i, j in x:
                x[i, j] = i * size + j
        
        # This should reuse cached struct PTX for the same field type
        test_kernel()
        
        print(f"Field size {size}x{size} completed successfully")

if __name__ == "__main__":
    test_struct_compilation_separation()
    test_different_field_sizes() 