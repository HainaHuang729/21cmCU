# 21cmFAST 4.3.2 GPU adapter

The v4.3.2 lightcone benchmark runs the upstream driver with CUDA perturbations and coeval brightness. Initial conditions, emissivity, radiation, spin temperature and ionization still use the upstream CPU routines. Redshift-space distortions use the upstream post-processing path. The v3 single-box launcher remains a separate entry point.

The adapter uses the v3 CUDA FFTW-plan compiler through the shared source in `lightcone/plan_cuda`. In the strict benchmark an unsupported FFTW plan raises an error. The v4 CIC kernel accumulates in double precision and checks the distance to both neighboring float32 rounding boundaries. For uncertain cells it revisits source particles in index order. The test exercises two CPU threads. OpenMP updates across slabs can interleave differently on other runs.

With 21cmFAST 4.3.2, CuPy and CUDA installed, put the upstream 4.3.2 checkout on `PYTHONPATH` so the benchmark can import its official integration test parameter sets. Run each mode in its own process:

```sh
python v4/lightcone_ps.py runs/v432-384 384 cpu point_config.json
python v4/lightcone_ps.py runs/v432-384 384 gpu point_config.json
python v4/lightcone_pair.py runs/v432-384 384 compare
python v4/compare_lightcone_ps.py runs/v432-384 runs/v432-384 runs/v432-384 runs/v432-384/acceptance.json
```

Use 192 in place of 384 and a separate output directory for the smaller case. The benchmark verifies the IC digest, redshift nodes, lightcone distances, all six fields and 32 brightness power-spectrum bins per node. The comparison uses `1e-8 + 1e-4 * abs(CPU)`. Timing in these scripts ends after the lightcone HDF5 is closed and synchronized. The PS estimator runs during evolution and its small JSON summary is saved afterwards.

See the [paired results](../reports/v432-ordered-brightness.json).
