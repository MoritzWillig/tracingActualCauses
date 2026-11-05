# Tracing Actual Causes with Counterfactual Witness Maps

Repository for the Paper **"Tracing Actual Causes with Counterfactual Witness Maps"** accepted at **NeurIPS 2026**.  
**Authors:** *Tim Woydt, Matej Zečević, Kristian Kersting, Moritz Willig*  
**Paper Link:** https://openreview.net/pdf?id=R0MXum0YPw

**Description:** TrAC is the first sound-and-complete procedure for enumerating *all* actual causes of a target event in a Markovian structural causal model (SCM), using the modified Halpern–Pearl (HP) definition.

**Abstract:** Halpernian actual causation, the formal study of which events in a specific situation caused a specific outcome in acyclic structural causal models, underpins legal responsibility, moral blame, and the assessment of causal harm. Despite the rich logical framework, no prior sound and complete enumeration procedure is known for finding *all* actual causes in a given setting, blocking downstream tasks such as responsibility attribution, blame assessment, and causal-harm grading that aggregate over the full cause set. We close this gap by introducing *witness targets*: sets of joint configurations containing every counterfactual witness. In combination with an *ancestor intervention grammar* we derive a finite directed acyclic *witness map*. We prove that for every actual cause there exists a contingency set such that the joint intervention set appears as a node of this map, and present an algorithm for *Tracing Actual Causes (TrAC)* that is sound and complete for enumerating all actual causes of any target event. We compare TrAC empirically with existing single-cause baselines on random Boolean structural causal models, demonstrating practical feasibility for graphs of up to 15 nodes.



This paper contains the implementation of the **TrAC algorithm** (Algorithm 1) from the paper and all additional experimental scripts to run the described experiments. Benchmark require the setup of external libraries (see 'Benchmarks' section below).




## Repository structure

```
trac/
  trac.py            # Core TrAC implementation (witness targets, witness map, Algorithm 1)
  baselines.py       # Timing wrappers for baseline solvers (actualcauses, HP2SAT)
  benchmark_trac.py  # Benchmark harness (parallel, per-cell CSV output)
  plot_results.py    # Result plotting

repos/               # External baseline repos (cloned separately, see below)
  ActualCausesIdentification/
  HP2SAT1.0/
  info.txt           # Clone instructions and required commits/patches

results/             # Pre-computed benchmark data and plots
  data/              # Per-cell CSVs (n<nodes>_d<density>.csv)
  benchmark_results.csv / *.png

run_benchmark.sh     # Convenience launcher for the benchmark
```

## Quick start

**Dependencies (TrAC only):**
```bash
pip install networkx
```

**Run the built-in examples** (Forest-fire, Billy-and-Suzy):
```bash
python trac/trac.py                  # all examples
python trac/trac.py --example bs     # Billy-and-Suzy only
python trac/trac.py --example ff-or  # Forest-fire (OR)
```

## Benchmarks

**Additional dependencies for baseline comparison:**
```bash
pip install actualcauses
# HP2SAT requires a compiled JAR – see repos/info.txt
```

**Set up baseline repos** (follow `repos/info.txt`):
```bash
cd repos
git clone https://github.com/SamuelReyd/ActualCausesIdentification
cd ActualCausesIdentification && git checkout e1f9b5f && git apply ../actualCauses.patch && cd ..
git clone -b hp-optimization-library https://github.com/amjadKhalifah/HP2SAT1.0
cd HP2SAT1.0 && git checkout be33600 && git apply ../HP2SAT.patch && cd ..
```

**Run benchmark:**
```bash
bash run_benchmark.sh [WORKERS [TIMEOUT [OUTPUT [MIN_NODES [MAX_NODES]]]]]
bash run_benchmark.sh --plot-only   # regenerate plots from existing data
bash run_benchmark.sh --update      # re-run and merge with existing results
```

Results are written to `results/` as per-cell CSVs and a summary PNG.

## License

This project is licensed under the [MIT License](LICENSE).

