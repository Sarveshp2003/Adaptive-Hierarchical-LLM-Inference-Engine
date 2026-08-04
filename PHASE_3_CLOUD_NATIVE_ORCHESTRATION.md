# Phase 3: Cloud-Native Orchestration & Distributed Scheduling

**Status**: Planning  
**Target Start**: After Phase 2 production deployment  
**Objective**: Scale scheduler to multi-node clusters with Kubernetes orchestration

## Overview

Phase 3 extends the single-node AI scheduler to a distributed, cloud-native system with:
- **Multi-Node Coordination**: Scheduler decisions across cluster nodes
- **Kubernetes Integration**: Native scheduler plugin and CRD support
- **Federated Learning**: Aggregate model improvements across nodes
- **Dynamic Resource Allocation**: Auto-scaling based on inference load

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                 Kubernetes Master                       │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Adaptive Scheduler Controller (CRD)             │   │
│  │  - Monitor PodMemoryState resources              │   │
│  │  - Orchestrate decision propagation              │   │
│  │  - Aggregate metrics from worker nodes           │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
         │              │              │
         │              │              │
┌────────▼────┐  ┌──────▼──────┐  ┌───▼───────────┐
│  Worker 1   │  │  Worker 2   │  │  Worker N     │
│ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐   │
│ │ Phase2  │ │  │ │ Phase2  │ │  │ │ Phase2  │   │
│ │Adapter+ │ │  │ │Adapter+ │ │  │ │Adapter+ │   │
│ │Scheduler│ │  │ │Scheduler│ │  │ │Scheduler│   │
│ └─────────┘ │  │ └─────────┘ │  │ └─────────┘   │
│             │  │             │  │               │
└─────────────┘  └─────────────┘  └───────────────┘
```

## Key Components

### 1. Kubernetes Custom Resource Definition (CRD)
**Status**: To Be Implemented  
**Purpose**: Define PodMemoryState as native K8s resource

```yaml
apiVersion: adaptivellm.io/v1alpha1
kind: PodMemoryState
metadata:
  name: inference-pod-1
  namespace: llm-workloads
spec:
  podRef:
    name: llama-inference-1
    namespace: default
  modelLayers: 28
  gpuMemoryMb: 8192
  ramMemoryMb: 16384
status:
  currentLayer: 12
  gpuUtilization: 0.75
  ramUtilization: 0.80
  lastDecision: PREFETCH_LAYER
  lastDecisionTime: "2024-08-04T10:15:00Z"
```

### 2. Adaptive Scheduler Controller
**Status**: To Be Implemented  
**Purpose**: K8s operator managing distributed scheduling

#### Controller Responsibilities
- Watch PodMemoryState resources
- Collect memory metrics from worker nodes
- Run global optimization across cluster
- Apply decisions to individual pods
- Aggregate training samples for federated learning

#### Key Methods
```java
public class AdaptiveSchedulerController {
    public void watchPodMemoryStates();
    public void orchestrateDecisions(List<PodMemoryState> states);
    public void applyDecisionToPod(Pod pod, Decision decision);
    public void aggregateMetrics(List<MemoryMetrics> metrics);
    public void triggerFederatedRetraining();
}
```

### 3. Federated Learning Engine
**Status**: To Be Implemented  
**Purpose**: Train global model on aggregated cluster data

#### Training Strategy
```
┌─────────────────────┐
│ Each Worker Node    │
│ ┌───────────────┐   │
│ │ 100 samples   │   │
│ │ Local loss: X │   │
│ └───────────────┘   │
└──────────┬──────────┘
           │
           │ (send samples)
           ▼
┌─────────────────────┐
│ Master Node         │
│ ┌───────────────┐   │
│ │ 10K samples   │   │
│ │ Global model  │   │
│ │ Retrain (5s)  │   │
│ └───────────────┘   │
│                     │
│ ┌───────────────┐   │
│ │ New weights   │   │
│ │ Broadcast     │   │
│ └───────────────┘   │
└──────────┬──────────┘
           │
           │ (broadcast)
           ▼
┌─────────────────────┐
│ Each Worker Node    │
│ ┌───────────────┐   │
│ │ Update model  │   │
│ │ New loss: Y   │   │
│ └───────────────┘   │
└─────────────────────┘
```

### 4. Distributed Memory State Provider
**Status**: To Be Implemented  
**Purpose**: Fetch metrics from multiple nodes with gossip protocol

```java
public class DistributedMemoryStateProvider {
    // Gossip-based metric distribution
    public void publishMetrics(MemoryState state);
    
    // Aggregate view of cluster state
    public ClusterMemoryState getAggregateState();
    
    // Per-node state
    public Map<String, MemoryState> getNodeStates();
}
```

## Implementation Timeline

### Phase 3.1: CRD & Controller Scaffolding (Week 1)
- [ ] Design and implement PodMemoryState CRD
- [ ] Create AdaptiveSchedulerController base class
- [ ] Implement Kubernetes watch on PodMemoryState
- [ ] Set up test environment with minikube

### Phase 3.2: Single-Node Integration (Week 2)
- [ ] Integrate Phase 2 adapter with controller
- [ ] Test decision application via kubectl
- [ ] Implement Pod status update mechanism
- [ ] Validate end-to-end decision pipeline

### Phase 3.3: Multi-Node Coordination (Week 3)
- [ ] Implement gossip protocol for metric distribution
- [ ] Test metric aggregation across 5-node cluster
- [ ] Implement global decision orchestration
- [ ] Monitor cross-node latency

### Phase 3.4: Federated Learning (Week 4)
- [ ] Implement federated sample collection
- [ ] Create distributed retraining pipeline
- [ ] Broadcast model updates to all workers
- [ ] Validate loss convergence across cluster

### Phase 3.5: Auto-Scaling & Optimization (Week 5)
- [ ] Implement auto-scaling based on inference load
- [ ] Create performance dashboard
- [ ] Tune coordination overhead
- [ ] Production readiness review

## Deployment Architecture

### Kubernetes Manifests
```yaml
# 1. CRD Definition
---
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: podmemorystates.adaptivellm.io
spec:
  group: adaptivellm.io
  names:
    kind: PodMemoryState
    plural: podmemorystates

# 2. Controller Deployment
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: adaptive-scheduler-controller
  namespace: adaptivellm-system
spec:
  replicas: 1
  selector:
    matchLabels:
      app: adaptive-scheduler-controller
  template:
    metadata:
      labels:
        app: adaptive-scheduler-controller
    spec:
      serviceAccountName: adaptive-scheduler-controller
      containers:
      - name: controller
        image: gcr.io/adaptivellm/scheduler-controller:v1.0
        env:
        - name: WATCH_NAMESPACE
          value: ""
        - name: POD_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name

# 3. RBAC
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: adaptive-scheduler-controller
  namespace: adaptivellm-system

---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: adaptive-scheduler-controller
rules:
- apiGroups: ["adaptivellm.io"]
  resources: ["podmemorystates"]
  verbs: ["get", "list", "watch", "create", "update"]
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list", "watch"]
- apiGroups: [""]
  resources: ["nodes"]
  verbs: ["get", "list"]
```

## Scaling Strategy

### Cluster Size Benchmarks
| Nodes | Decisions/sec | Aggregation Latency | Training Samples/hour |
|-------|---------------|---------------------|------------------------|
| 1     | 100           | <1ms                | 3600                   |
| 5     | 400           | <10ms               | 18000                  |
| 10    | 800           | <20ms               | 36000                  |
| 50    | 4000          | <50ms               | 180000                 |

### Bottleneck Analysis
- **Decision Latency**: Network RTT across nodes (target: <50ms)
- **Model Synchronization**: Broadcast overhead (target: <5s per cycle)
- **Metric Aggregation**: Gossip protocol efficiency (target: 99% accuracy)

## Monitoring & Observability

### Key Metrics (Phase 3)
- **Cross-Node Latency**: Decision propagation time
- **Federated Learning Efficiency**: Model loss per training cycle
- **Cluster Utilization**: GPU/memory across all nodes
- **Controller Health**: CRD watch latency, sync failures

### Observability Stack
```yaml
# Prometheus ServiceMonitor
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: adaptive-scheduler
spec:
  selector:
    matchLabels:
      app: adaptive-scheduler-controller
  endpoints:
  - port: metrics
    interval: 30s

# Grafana Dashboard
- Title: "Adaptive Scheduler - Cluster Overview"
  Panels:
  - Decision latency p50/p95/p99
  - Model loss convergence
  - GPU utilization per node
  - Network throughput between nodes
```

## Production Readiness Checklist

### Code Quality
- [ ] Unit tests for all CRD operations
- [ ] Integration tests with etcd
- [ ] Load tests for 50+ node clusters
- [ ] Chaos engineering tests (node failures)

### Deployment
- [ ] Helm chart for easy installation
- [ ] Upgrade strategy (zero-downtime rolling)
- [ ] Backup/restore procedures for controller state
- [ ] Health check endpoints

### Operations
- [ ] Runbook for common issues
- [ ] Alert rules for critical failures
- [ ] Capacity planning guide
- [ ] Cost analysis for cloud deployment

## Post-Phase-3 Roadmap

### Phase 4: Advanced Features
- Predictive prefetching based on request patterns
- Multi-model scheduling (heterogeneous architectures)
- Cost optimization for cloud providers
- SLA-aware scheduling

### Phase 5: Research & Innovation
- Reinforcement learning for long-term optimization
- Generative model for architecture discovery
- Theoretical performance bounds
- Open-source contribution plan

## File Structure (Phase 3)

```
src/main/java/com/adaptivellm/
├── kubernetes/
│   ├── AdaptiveSchedulerController.java (new)
│   ├── PodMemoryStateWatch.java (new)
│   └── KubernetesIntegration.java (new)
├── distributed/
│   ├── DistributedMemoryStateProvider.java (new)
│   ├── GossipProtocol.java (new)
│   └── ClusterCoordinator.java (new)
├── learning/
│   ├── FederatedLearningEngine.java (new)
│   └── AggregatedTrainingData.java (new)
└── scheduler/
    └── (existing Phase 1-2 files)

k8s/
├── crd/
│   └── podmemorystate-crd.yaml (new)
├── controller/
│   ├── deployment.yaml (new)
│   ├── rbac.yaml (new)
│   └── service.yaml (new)
└── monitoring/
    ├── servicemonitor.yaml (new)
    └── grafana-dashboard.json (new)
```

## Success Criteria

- [ ] Scheduler running on 5-node Kubernetes cluster
- [ ] Cross-node decision latency < 50ms
- [ ] Federated model improves by 5% over single-node
- [ ] Zero performance degradation from coordination overhead
- [ ] All tests passing (unit, integration, load, chaos)
- [ ] Production deployment on cloud provider

---

**Phase 3 Owner**: Cloud Infrastructure Lead  
**Expected Completion**: 6 weeks  
**Milestone Reviews**: Weekly standups, phase gate at end of each week  
**Escalation**: Architecture review board
