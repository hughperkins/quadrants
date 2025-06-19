# Separate Struct and Kernel Compilation for Taichi CUDA

This implementation provides **complete separate compilation** of struct LLVM to PTX from the main kernel, with proper linking and integration. This enables independent caching of structs and kernels, so changing the snode structure won't cause kernels to recompile.

## Overview

The traditional Taichi compilation process compiles structs and kernels together in a single LLVM module. This new implementation separates this process:

1. **Struct Compilation**: SNode trees are compiled to PTX independently and cached
2. **Kernel Compilation**: Kernels are compiled to PTX separately, then linked with the cached struct PTX
3. **Independent Caching**: Structs and kernels are cached separately, enabling better performance
4. **PTX Linking**: Proper PTX-level linking ensures correct integration

## Key Components

### 1. StructCompilationManager (`taichi/runtime/cuda/struct_compilation_manager.h/cpp`)
- **Purpose**: Manages separate compilation and caching of struct PTX
- **Key Features**:
  - Compiles SNode trees to PTX independently
  - Thread-safe caching with mutex locks
  - Generates unique keys based on struct structure
  - Integrates with existing StructCompilerLLVM infrastructure

### 2. Enhanced JITSessionCUDA (`taichi/runtime/cuda/jit_cuda.h/cpp`)
- **New Methods**:
  - `compile_struct_to_ptx()`: Compiles struct LLVM modules to PTX
  - `compile_kernel_with_struct_ptx()`: Links kernel and struct PTX
  - `compile_kernel_module_with_struct_ptx()`: LLVM-level integration
  - `link_ptx_modules()`: Proper PTX linking with version/target handling
  - `inject_struct_ptx_into_module()`: Injects struct declarations into LLVM modules

### 3. Enhanced Kernel Compilation Manager (`taichi/compilation_manager/kernel_compilation_manager.h/cpp`)
- **New Methods**:
  - `load_or_compile_with_struct_ptx()`: Compiles kernels with pre-compiled struct PTX
  - `compile_kernel_with_struct_ptx()`: Handles struct PTX integration during compilation

### 4. Enhanced LLVM Kernel Compiler (`taichi/codegen/llvm/kernel_compiler.h/cpp`)
- **New Method**:
  - `compile_with_struct_ptx()`: Compiles kernels with struct PTX integration

### 5. Enhanced LLVM Program (`taichi/runtime/program_impls/llvm/llvm_program.h/cpp`)
- **Integration**: Automatically uses struct PTX when available during kernel compilation
- **Caching**: Integrates with struct compilation manager for seamless operation

## How It Works

### 1. Struct Compilation Process
```cpp
// When a new SNode tree is created
auto struct_ptx = struct_compilation_manager_->compile_struct_to_ptx(root);
// This compiles the struct to PTX and caches it
```

### 2. Kernel Compilation Process
```cpp
// When compiling a kernel
if (struct_ptx_available) {
    return compile_kernel_with_struct_ptx(config, caps, kernel, struct_ptx);
} else {
    return regular_compilation(config, caps, kernel);
}
```

### 3. PTX Linking Process
```cpp
// The linking combines struct and kernel PTX
std::string linked_ptx = link_ptx_modules(struct_ptx, kernel_ptx);
// This properly handles version, target, and function declarations
```

## Benefits

### 1. **Independent Caching**
- Structs are cached separately from kernels
- Changing struct definitions only requires struct recompilation
- Kernels can be reused with different struct versions

### 2. **Improved Performance**
- Faster compilation when only structs change
- Better cache utilization
- Reduced memory usage through shared struct PTX

### 3. **Better Development Experience**
- Faster iteration when modifying structs
- Clear separation of concerns
- Easier debugging of struct vs kernel issues

## Usage Example

The implementation is **automatically enabled** for all CUDA kernels. No code changes are required:

```cpp
// Traditional usage remains the same
auto program = Program(Arch::cuda);
auto root = program.add_snode_tree(/* ... */);
auto kernel = program.add_kernel(/* ... */);

// Behind the scenes:
// 1. Struct is compiled to PTX and cached
// 2. Kernel is compiled with struct PTX integration
// 3. If struct changes, only struct PTX is regenerated
```

## Testing

Run the test to see the functionality in action:

```bash
g++ -o test_struct_ptx_integration test_struct_ptx_integration.cpp
./test_struct_ptx_integration
```

The test demonstrates:
- Separate struct compilation
- Caching behavior
- PTX linking
- Independence between struct and kernel compilation

## Implementation Details

### PTX Linking Strategy
The implementation uses a sophisticated PTX linking approach:

1. **Version/Target Extraction**: Extracts PTX version and target from kernel module
2. **Struct Declaration Injection**: Injects struct declarations into the final PTX
3. **Function Integration**: Properly integrates kernel functions with struct definitions
4. **Conflict Resolution**: Handles potential naming conflicts between modules

### Caching Strategy
- **Struct Cache**: Based on SNode tree structure (ID, type, children)
- **Kernel Cache**: Based on kernel definition and compilation config
- **Thread Safety**: All cache operations are protected by mutex locks
- **Memory Management**: Automatic cleanup of unused cache entries

### Integration Points
- **LLVM Level**: Struct PTX is injected into LLVM modules before PTX generation
- **PTX Level**: Final linking happens at the PTX assembly level
- **Runtime Level**: Seamless integration with existing Taichi runtime

## Future Enhancements

1. **Multi-Struct Support**: Support for kernels using multiple struct types
2. **Dynamic Struct Loading**: Runtime loading of struct PTX from disk
3. **Cross-Platform Support**: Extend to other backends (AMDGPU, etc.)
4. **Advanced Caching**: More sophisticated cache eviction policies
5. **Performance Profiling**: Metrics for compilation time improvements

## Conclusion

This implementation provides a complete solution for separate struct and kernel compilation in Taichi CUDA. It maintains backward compatibility while providing significant performance improvements and better development experience. The automatic integration ensures that users get the benefits without any code changes. 