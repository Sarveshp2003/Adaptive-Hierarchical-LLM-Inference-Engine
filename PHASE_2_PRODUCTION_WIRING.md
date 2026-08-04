# Phase 2: Production Wiring Guide

**Status**: In Development  
**Last Updated**: 2024  
**Target Completion**: Production Deployment

## Overview

Phase 2 transitions the Adaptive Hierarchical LLM Inference Engine from simulation-based development to production-ready integration with the actual NativeEngine runtime. This guide outlines the implementation approach, JNI bindings, and deployment steps.

## Components

### 1. ProductionMemoryStateProvider
**File**: `ProductionMemoryStateProvider.java`  
**Purpose**: Fetches real runtime metrics instead of simulations  
**Replaces**: `RuntimeMemoryStateProvider.java` (simulation-based)

#### Real Metrics Collected
- **GPU Memory**: From NativeEngine via JNI
- **KV Cache Pages**: Actual count from runtime
- **Cached Layers**: Current layer prefetch state
- **Storage Latency**: SSD I/O performance measurement
- **JVM Heap**: Standard JVM memory management

#### Native Method Stubs (to be implemented via JNI)
```java
private native int getCurrentLayerNative();
private native long getGpuMemoryNative();
private native int getKvPagesNative();
private native int getCachedLayersNative();
```

### 2. Phase2NativeEngineAdapter
**File**: `Phase2NativeEngineAdapter.java`  
**Purpose**: JNI bridge for scheduler decisions to NativeEngine calls  
**Replaces**: `NativeEngineAdapter.java` (simulation-based)

#### Action-to-Native Mapping

| Scheduler Action | NativeEngine Method | Return Value |
|-----------------|-------------------|--------------|
| PREFETCH_LAYER  | requestLayer(layerId) | latency_ms |
| EVICT_LAYER     | evictLayer(layerId) | latency_ms |
| KEEP_LAYER      | keepLayer(layerId) | latency_ms |
| MOVE_KV_TO_RAM  | moveKvCache(pageId, RAM) | latency_ms |
| MOVE_KV_TO_GPU  | moveKvCache(pageId, GPU) | latency_ms |
| COMPRESS_KV     | compressKvCache(pageId) | latency_ms |
| OFFLOAD_KV      | offloadKvCache(pageId) | latency_ms |
| NO_ACTION       | (no-op) | 0 |

#### Native Method Stubs (to be implemented via JNI)
```java
private native void nativeStart(Object nativeEngine);
private native void nativeStop(Object nativeEngine);
private native long nativePrefetchLayer(Object nativeEngine, int layerId);
private native long nativeEvictLayer(Object nativeEngine, int layerId);
private native long nativeKeepLayer(Object nativeEngine, int layerId);
private native long nativeMoveKvToRam(Object nativeEngine, long kvPageId);
private native long nativeMoveKvToGpu(Object nativeEngine, long kvPageId);
private native long nativeCompressKv(Object nativeEngine, long kvPageId);
private native long nativeOffloadKv(Object nativeEngine, long kvPageId);
```

### 3. Phase2ProductionIntegrationTest
**File**: `Phase2ProductionIntegrationTest.java`  
**Purpose**: Validates production wiring components and error handling

#### Test Coverage
- ProductionMemoryStateProvider instantiation and metrics
- Phase2NativeEngineAdapter structure and state management
- End-to-end pipeline with real scheduler components
- Error handling for native library unavailability

## JNI Implementation Roadmap

### Step 1: C++ Native Methods (not included in this commit)
```cpp
// adaptive_scheduler.cpp
extern "C" {
    // Memory state
    JNIEXPORT jint JNICALL Java_com_adaptivellm_scheduler_ProductionMemoryStateProvider_getCurrentLayerNative
        (JNIEnv* env, jobject obj);
    
    JNIEXPORT jlong JNICALL Java_com_adaptivellm_scheduler_ProductionMemoryStateProvider_getGpuMemoryNative
        (JNIEnv* env, jobject obj);
    
    // Engine control
    JNIEXPORT void JNICALL Java_com_adaptivellm_scheduler_Phase2NativeEngineAdapter_nativeStart
        (JNIEnv* env, jobject obj, jobject nativeEngine);
    
    JNIEXPORT jlong JNICALL Java_com_adaptivellm_scheduler_Phase2NativeEngineAdapter_nativePrefetchLayer
        (JNIEnv* env, jobject obj, jobject nativeEngine, jint layerId);
    
    // ... other methods
}
```

### Step 2: CMake Configuration
```cmake
# CMakeLists.txt additions
find_package(JNI REQUIRED)
add_library(adaptive_scheduler SHARED adaptive_scheduler.cpp)
target_link_libraries(adaptive_scheduler ${JNI_LIBRARIES} native_engine)
```

### Step 3: Build and Link
```bash
# Build C++ with JNI
cmake -DCMAKE_BUILD_TYPE=Release ..
make

# Output: libadaptive_scheduler.so (Linux) or adaptive_scheduler.dll (Windows)
```

### Step 4: Java Library Loading
```bash
# Set library path during runtime
java -Djava.library.path=/path/to/native/libs MainClass
```

## Integration Steps

### Phase 2.1: Development Environment (Week 1)
- [ ] Implement C++ JNI stubs with mock returns
- [ ] Compile libadaptive_scheduler with mock implementations
- [ ] Run Phase2ProductionIntegrationTest
- [ ] Verify latency overhead < 1ms

### Phase 2.2: Staging Deployment (Week 2-3)
- [ ] Implement C++ native methods with real NativeEngine calls
- [ ] Deploy to staging cluster with actual Llama model
- [ ] Run canary test (10 concurrent requests)
- [ ] Collect 100+ real execution samples

### Phase 2.3: Data Collection (Week 3-4)
- [ ] Deploy to production subset (5% traffic)
- [ ] Collect 1000+ real execution samples
- [ ] Monitor decision latency, success rates, memory savings
- [ ] Identify hyperparameter tuning opportunities

### Phase 2.4: Optimization (Week 4-5)
- [ ] Analyze collected data for failure modes
- [ ] Retrain model with real data samples
- [ ] Tune learning rate, update frequency, action weights
- [ ] A/B test against rule-based baseline

### Phase 2.5: Full Rollout (Week 6)
- [ ] Deploy to 100% production traffic
- [ ] Monitor production metrics continuously
- [ ] Collect ongoing feedback samples for continuous learning
- [ ] Prepare Phase 3: Cloud-Native Orchestration

## Deployment Checklist

### Pre-Deployment
- [ ] C++ JNI methods implemented and tested
- [ ] libadaptive_scheduler compiled and signed
- [ ] Java classes compiled with Phase 2 adapters
- [ ] ProductionMemoryStateProvider tested with mock engine
- [ ] Phase2NativeEngineAdapter latency < 2ms
- [ ] Error handling for native library unavailability

### Deployment
- [ ] Deploy libadaptive_scheduler to all nodes
- [ ] Update java.library.path in production config
- [ ] Run Phase2ProductionIntegrationTest in staging
- [ ] Monitor startup logs for native library loading
- [ ] Verify first 100 decisions with real metrics

### Post-Deployment
- [ ] Monitor average decision latency (target: <10ms)
- [ ] Monitor native call success rates (target: >99%)
- [ ] Collect execution samples every 100 decisions
- [ ] Run batch retraining on real data weekly
- [ ] Compare against baseline in A/B testing

## Error Handling Strategy

### Native Library Unavailable
```
Fallback Behavior:
1. Try to load libadaptive_scheduler
2. If fails: Print warning, continue with null checks
3. When native method called: Return cached result or error
4. Fallback to RuntimeMemoryStateProvider (simulation)
```

### Native Call Failures
```
Graceful Degradation:
1. Native call fails -> return ExecutionResult.failed()
2. Scheduler uses failure signal as training example
3. Continue with next decision
4. Log error for monitoring
5. Eventually retry (exponential backoff)
```

## Monitoring & Observability

### Key Metrics
- **Decision Latency**: p50, p95, p99 (target: <10ms)
- **Native Call Success Rate**: % successful vs failed
- **Memory Savings**: Bytes freed per decision
- **GPU Utilization**: Before/after scheduler decisions
- **Model Loss**: Training loss on real samples

### Logging
```
[Phase2NativeEngineAdapter] PREFETCH_LAYER(layerId=15) latency=3ms
[Phase2NativeEngineAdapter] Native call failed: method not found
[ProductionMemoryStateProvider] GPU memory: 2.5GB / 8GB (31%)
```

### Dashboards (to be created)
- Real-time decision latency distribution
- GPU memory usage timeline
- KV cache size evolution
- Model loss convergence
- Native call success rates

## File Changes Summary

### New Files (Phase 2)
1. `ProductionMemoryStateProvider.java` (8.6 KB)
   - Real metrics from NativeEngine via JNI
   - Replaces simulation-based provider

2. `Phase2NativeEngineAdapter.java` (14.2 KB)
   - JNI bridge for scheduler decisions
   - Replaces simulation-based adapter

3. `Phase2ProductionIntegrationTest.java` (7.1 KB)
   - Validates production wiring components
   - Tests error handling and graceful fallback

### Existing Files (Unchanged for now)
- `NativeEngineAdapter.java` - Kept for reference and testing
- `RuntimeMemoryStateProvider.java` - Kept as fallback
- `SchedulerRuntimeController.java` - Works with both providers
- `AdaptiveScheduler.java` - Agnostic to provider implementation

## Next Steps (Phase 3)

After Phase 2 production deployment and data collection:

1. **Continuous Learning**: Automated retraining on new samples
2. **Distributed Orchestration**: Multi-node scheduler coordination
3. **Advanced Metrics**: Real-time memory pressure prediction
4. **Cloud Integration**: Kubernetes native scheduler

## References

- Java JNI Guide: https://docs.oracle.com/javase/10/docs/specs/jni/
- Phase 1 Summary: See README.md "Adaptive AI Scheduler" section
- Architecture: See README.md architecture diagram
- Performance Baseline: See progress.md "Performance Metrics" table

---

**Prepared by**: Adaptive LLM Team  
**Next Review**: After Phase 2.1 (Development Environment)  
**Escalation Contact**: Production Engineering Lead
