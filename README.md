# ChaosAI — Resilience & Entropy Testing Hub

ChaosAI is an interactive experimentation sandbox and benchmarking platform engineered to evaluate autonomous AI agent architectures, multi-agent swarms, and RAG pipelines under chaotic and adversarial conditions.

## Features

- **Chaos Lab**:
  - Live interactive topology graph visualizing agent nodes, critics, and external tool bridges.
  - Real-time simulation of fault injection (latency spikes, schema degradation, semantic hallucination traps, and Byzantine rogue agents).
  - Live telemetry dashboard tracking Resilience Score, Hallucination Index, Token Burn, and Recovery Latency.
  - Streaming execution terminal showing agent reasoning cycles, failure detection, and self-healing.

- **Experiment Builder**:
  - Create and configure custom adversarial testing scenarios.
  - Target multiple architecture categories (Agentic Loop, Swarms, RAG, Tool Bridges, Alignment).
  - Configure failure vectors, entropy rates, and expected healing strategies (reflection, state rollbacks, quorum voting).

- **Test Vault & Benchmarks**:
  - Historical run logs persisted with full diagnostic metadata.
  - Inspector modal for reviewing telemetry breakdowns.
  - Benchmark export to CSV and JSON execution telemetry logs.

- **Fault Vector Taxonomy**:
  - Curated catalogue of AI failure vectors (FLT-01 through FLT-06) covering semantic noise, protocol rupture, memory truncation, and Byzantine partitions.

## Local Usage

Open `index.html` in any modern web browser or serve it via a static file server:

```bash
# Python 3
python3 -m http.server 8080

# Or npx serve
npx serve .
```
