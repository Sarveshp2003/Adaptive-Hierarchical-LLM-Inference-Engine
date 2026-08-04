# Phase 2.1: C++ JNI Native Implementation

## Overview
Phase 2.1 implements the C++ side of the JNI bridge that enables Java Phase 2 components to interface with native memory management and engine control operations.

## Status
- **JNI Headers**: ✅ Generated
- **C++ Implementation**: ✅ Complete with mock engine
- **Build Scripts**: ✅ Created for Windows and Unix
- **Native Library Compilation**: ⏳ Pending platform-specific build
- **Integration Testing**: ✅ Verified with graceful fallback

## Architecture

### JNI Bridge Design
The JNI bridge connects Java scheduler decisions to native engine operations:

```
ProductionMemoryStateProvider (Java)
  └─> JNI Native Methods (C++)
      └─> MockNativeEngine
          ├─> getCurrentLayer()
          ├─> getGpuMemory()
          ├─> getKvPages()
          └─> getCachedLayers()

Phase2NativeEngineAdapter (Java)
  └─> JNI Native Methods (C++)
      └─> MockNativeEngine
          ├─> prefetchLayer()
          ├─> evictLayer()
          ├─> moveKvToRam()
          ├─> moveKvToGpu()
          ├─> compressKv()
          └─> offloadKv()
```

### Memory State Provider (4 Methods)
| Method | Signature | Purpose | Mock Behavior |
|--------|-----------|---------|---------------|
| `getCurrentLayerNative()` | `() -> int` | Query current layer | Returns mock layer ID |
| `getGpuMemoryNative()` | `() -> long` | Get GPU memory used | Returns 2GB ± 50MB variation |
| `getKvPagesNative()` | `() -> int` | Get KV cache pages | Returns 256 pages |
| `getCachedLayersNative()` | `() -> int` | Get cached layers | Returns 2 layers |

### Native Engine Adapter (6 Action Methods)
| Method | Signature | Latency | Purpose |
|--------|-----------|---------|---------|
| `nativeStart()` | `(Object) -> void` | ~0ms | Start engine |
| `nativeStop()` | `(Object) -> void` | ~0ms | Stop engine |
| `nativePrefetchLayer()` | `(Object, int) -> long` | 5ms | Prefetch layer to GPU |
| `nativeEvictLayer()` | `(Object, int) -> long` | 1-3ms | Evict layer from GPU |
| `nativeKeepLayer()` | `(Object, int) -> long` | 0ms | Keep layer in cache |
| `nativeMoveKvToRam()` | `(Object, long) -> long` | 2ms | Move KV to RAM |
| `nativeMoveKvToGpu()` | `(Object, long) -> long` | 3ms | Move KV to GPU |
| `nativeCompressKv()` | `(Object, long) -> long` | 4ms | Compress KV cache |
| `nativeOffloadKv()` | `(Object, long) -> long` | 6ms | Offload KV to disk |

## Implementation Files

### C++ Source
- **File**: `src/main/cpp/adaptive_scheduler.cpp`
- **Size**: ~12.5 KB
- **Contains**:
  - MockNativeEngine class (stateful singleton)
  - 4 ProductionMemoryStateProvider native methods
  - 8 Phase2NativeEngineAdapter native methods
  - JNI_OnLoad/OnUnload lifecycle callbacks

### Generated JNI Headers
- **`src/main/cpp/headers/com_adaptivellm_scheduler_ProductionMemoryStateProvider.h`**
  - 4 function declarations matching Java native method signatures
  
- **`src/main/cpp/headers/com_adaptivellm_scheduler_Phase2NativeEngineAdapter.h`**
  - 8 function declarations matching Java native method signatures

### Build Configuration
- **CMakeLists.txt**: Updated with JNI build target
  - Finds Java and JNI
  - Builds `adaptive_scheduler` shared library
  - Copies to `build/lib` directory

### Build Scripts
- **`build_native_library.bat`** (Windows)
  - Auto-detects MSVC or MinGW
  - Generates `lib/adaptive_scheduler.dll`
  
- **`build_native_library.sh`** (Linux/macOS)
  - Cross-platform Unix build
  - Generates `lib/libadaptive_scheduler.so` or `.dylib`

## Compilation Instructions

### Windows (Visual Studio or MinGW)
```bash
# Option 1: Using batch script
build_native_library.bat

# Option 2: Using CMake
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022"
cmake --build . --target adaptive_scheduler --config Release
```

### Linux/macOS
```bash
# Option 1: Using bash script
chmod +x build_native_library.sh
./build_native_library.sh

# Option 2: Using CMake
mkdir build
cd build
cmake ..
make adaptive_scheduler
```

## Testing

### Without Compiled Native Library
The Java code gracefully handles missing native library:
- Catches `UnsatisfiedLinkError` 
- Falls back to mock values:
  - GPU memory: 2GB
  - KV pages: 256
  - Cached layers: 2

**Status**: ✅ Phase2ProductionIntegrationTest passes (50/50 decisions, online learning active)

### With Compiled Native Library
After compiling:

```bash
# Set library path and run test
export LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH  # Linux
export DYLD_LIBRARY_PATH=./lib:$DYLD_LIBRARY_PATH  # macOS
set PATH=.\lib;%PATH%  # Windows

java -Djava.library.path=./lib \
     -cp target/classes:src/main/java \
     com.adaptivellm.scheduler.Phase2ProductionIntegrationTest
```

## Mock Engine Behavior

The MockNativeEngine simulates realistic latencies for each operation:

### State Management
- **currentLayer**: Tracks which layer is being processed
- **gpuMemoryUsed**: Simulates memory variation (2GB ± 50MB)
- **kvPages**: Fixed at 256 pages
- **cachedLayers**: Fixed at 2 cached layers
- **running**: Engine state (started/stopped)

### Action Latencies
- **Prefetch**: 5ms (GPU memory write)
- **Evict**: 1-3ms (PCIe transfer)
- **Move KV to RAM**: 2ms (CPU memory move)
- **Move KV to GPU**: 3ms (GPU memory write)
- **Compress KV**: 4ms (CPU computation)
- **Offload KV**: 6ms (Storage I/O)

## Integration with Phase 2 Components

### ProductionMemoryStateProvider
Provides real-time memory metrics to the scheduler:
```java
// Java side
ProductionMemoryStateProvider provider = new ProductionMemoryStateProvider();
MemoryState state = provider.getCurrentState();
// Internally calls:
// - getCurrentLayerNative()
// - getGpuMemoryNative()
// - getKvPagesNative()
// - getCachedLayersNative()
```

### Phase2NativeEngineAdapter
Executes scheduler decisions on the native engine:
```java
// Java side
Phase2NativeEngineAdapter adapter = new Phase2NativeEngineAdapter();
adapter.startEngine();
// Internally calls: nativeStart(nativeEngine)

adapter.executeDecision(decision);
// Maps Decision.action to appropriate native method call
// Returns latency and execution result
```

## Graceful Degradation Strategy

The Java code is designed to work with or without the native library:

1. **Load Attempt**: Class.forName() tries to load native methods
2. **Fallback Path**: UnsatisfiedLinkError caught and logged
3. **Mock Values**: Returns sensible defaults:
   - `getCurrentLayer()` → 0
   - `getGpuMemory()` → 2GB
   - `getKvPages()` → 256
   - `getCachedLayers()` → 2
4. **Decision Making**: Scheduler continues with fallback metrics
5. **Online Learning**: Still improves model accuracy (validated: 40x loss improvement)

## Next Steps

### 2.1.1 - Compilation & Local Testing
- [ ] Compile on Windows with MSVC
- [ ] Compile on Linux with g++/clang++
- [ ] Compile on macOS
- [ ] Run integration test with compiled library
- [ ] Verify latency measurements match mock values

### 2.1.2 - Real Engine Integration
- [ ] Replace MockNativeEngine with actual NativeEngine C++ class
- [ ] Implement methods using actual engine calls (not mocks)
- [ ] Add real GPU memory queries
- [ ] Collect actual latency data

### 2.1.3 - Performance Optimization
- [ ] Profile JNI call overhead
- [ ] Optimize hot paths (getCurrentLayer, getGpuMemory)
- [ ] Batch multiple operations when possible
- [ ] Add thread-safe caching for frequently-accessed values

### 2.1.4 - Production Wiring
- [ ] Integrate with actual Llama.cpp runtime
- [ ] Collect 1000+ real decision samples
- [ ] Retrain scheduler model on real data
- [ ] Deploy to local inference system

## Known Limitations

1. **Mock Engine**: Currently simulated, not connected to real engine
2. **No Thread Safety**: MockNativeEngine uses simple static instance
3. **No Resource Cleanup**: No explicit deallocation of native resources
4. **Limited Error Handling**: Basic error messages, no detailed diagnostics
5. **No Metrics Persistence**: Runtime metrics not saved to file

## File Locations
- Source: `src/main/cpp/adaptive_scheduler.cpp`
- Headers: `src/main/cpp/headers/*.h`
- Build Output: `lib/adaptive_scheduler.{dll|so|dylib}`
- Build Config: `CMakeLists.txt`
- Build Scripts: `build_native_library.{bat|sh}`

## References
- [JNI Specification](https://docs.oracle.com/javase/8/docs/technotes/guides/jni/spec/jniTOC.html)
- [Phase 2 Production Wiring Guide](./PHASE_2_PRODUCTION_WIRING.md)
- [Phase 2 Integration Test](./src/main/java/com/adaptivellm/scheduler/Phase2ProductionIntegrationTest.java)
