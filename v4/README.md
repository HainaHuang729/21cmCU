# 21cmFAST 4.3.2 GPU adapter

The current Eulerian atomic-cooling lightcone passes the native 4.3.2 CPU comparison at both benchmark sizes. The table gives wall time through HDF5 synchronization on node a8 (RTX A6000), using archived CPU timings from the same cluster node in separate allocations. Each run matched its CPU reference's input parameters, initial-condition digest and comoving distances, and used two CPU threads and 43 redshift nodes. The six requested lightcone fields, line-of-sight velocity, optical depth, 903 global values, 1,376 32-bin power-spectrum samples, and 20,800 / 25,008 transverse power samples per run all passed `abs(GPU - CPU) <= 1e-8 + 1e-4 * abs(CPU)`. [Paired acceptance results](../reports/v432-current-large-lightcones.json) · [Resident optimization results](../reports/v432-resident-large-lightcones.json)

| Fine grid / evolved grid | Box length | CPU | Prior GPU | Current GPU | Speedup vs CPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| 192³ / 64³ | 200 Mpc | 65.78 s | 20.93 s | 15.58 s | 4.22× |
| 384³ / 96³ | 375 Mpc | 347.52 s | 45.14 s | 37.60 s | 9.24× |

The benchmark runs the upstream lightcone driver for Eulerian atomic-cooling sources. CUDA generates the IC from the seed, evaluates 1LPT and 2LPT, deposits particles, filters shells, builds conditional and unconditional source tables, solves X-ray optical depths and frequency integrals, updates thermal and ionization fields, and builds the MHR rate and derivative tables. CUDA also evaluates the initial RECFAST spline and collision-only spin temperature on the first snapshot. The rectilinear lightcone uses zero-upload GPU slice interpolation, optical-depth velocity correction, brightness evaluation directly on device fields, and the 32-bin brightness power spectrum. CPU single-step oracles for source tables and thermal evolution run in the explicit audit entry point.

This is a partial port. The host prepares the star-formation and escape-fraction scalings, the ionizing-background and radiation prefactors, radiation-shell redshifts, and the Astropy distance and Hubble functions. The GPU evaluates the scalar mass variances used by the source and ionization callers and the unconditional moments used to normalize the emissivity grids. Perturbation reuses the eight IC fields on the GPU and caches device perturbation arrays across nodes, while [device_registry.py](device_registry.py) retains intermediate fields on GPU memory (`3,509` hits / `3.90 GB` avoided at 192³ and `3,393` hits / `12.75 GB` avoided at 384³). Radiation skips redundant shell downloads (`3,262` and `3,258` avoided) and the Eulerian ionization loop keeps filtered fields resident across radii. [Constant-efficiency routing](../reports/v432-constant-lightcones.json) · [64³ Eulerian thermal node](../reports/v432-eulerian-resident-ionization.json)

All five source models—now including `CHMF-SAMPLER` (with 31-bit GSL `cmrg`/`mrg` normalization in [halo_catalog_random.h](halo_catalog_random.h))—support even cubic grids and `N_THREADS >= 1` across the upstream GSL PRNG types (`mt`, `gfsr`, `cmrg`, `mrg`, `taus2`). The 192³ and 384³ Eulerian timings use two threads, fixed seed 424242 and the atomic-cooling fixture, alongside an 8-thread 192³ `E-INTEGRAL` check on a8 (34.93 s 8T CPU vs 12.88 s GPU, 2.71× vs 8T EPYC CPU and 4.32× faster than the prior 55.69 s run), multithreaded 32³ and 96³ CHMF comparisons, and large-grid BBKS/Gaussian and molecular-cooling runs. [Multithread PRNG checks](../reports/v432-multithread-rng.json) · [CHMF multithread and hybrid speed](../reports/v432-chmf-multithread-and-speed.json) · [Gaussian matter window checks](../reports/v432-gaussian-matter-window.json) · [Large-grid new-physics checks](../reports/v432-large-new-physics.json)

The adapter uses the v3 CUDA FFTW-plan compiler through the shared source in `lightcone/plan_cuda`. In the strict benchmark an unsupported FFTW plan raises an error. At each redshift the host evaluates the three native perturbation growth coefficients once. The upstream velocity coefficient uses a finite difference at Δz = 10⁻¹⁰, which makes it sensitive to a one-ULP difference in the growth factor. The v4 CIC kernel accumulates in double precision and checks the distance to both neighboring float32 rounding boundaries. For uncertain cells a CUDA block searches for contributing particles in parallel and sums the accepted weights in native particle order. With four fine cells per coarse cell along each axis, the adapted v3 tiled scatter uses the 4.3.2 integer particle origins and nearest periodic displacement index. [CIC stages and forced replay comparisons](../reports/v432-cic-block-replay.json) · [Independent NumPy particle-order tests](../reports/v432-cic-replay-synthetic.json) The test exercises two CPU threads. OpenMP updates across slabs can interleave differently on other runs.

For both Eulerian models the GPU now normalizes emission while the mapped grids are still resident. It uses the v3 double-precision reduction and the native 4.3.2 local-density weighting. Standalone timings for mapping, normalization and returning the outputs were 3.66 → 1.53 ms at 64³ and 11.51 → 2.99 ms at 96³. These measurements used a tiled copy of a saved native thermal density field at fixed cell size. [Emissivity comparisons and stage timings](../reports/v432-eulerian-emissivity-means.json)

The mass-dependent `E-INTEGRAL` model also supports turning off both spin-temperature fluctuations and recombinations. A 32³ → 16³, 24 Mpc case with seed 812445 and log₁₀ escape fraction −0.7 passed all eight fields, 33 global values and 712 transverse power bins. Between redshifts 9.0 and 8.4 the count of fully ionized cells grew from 1,017 to 2,084. The native CPU took 4.37 s. The initial GPU run took 10.76 s in the same allocation. After skipping unused thermal setup, the final GPU replay on the same host took 9.89 s in a separate allocation. Each timer includes diagnostic cache writes. [Nonthermal mass-dependent lightcone](../reports/v432-e-integral-nonthermal-lightcone.json)

Nonthermal runs now omit the spin-temperature and radiation kernels and tables during backend setup. RECFAST interpolation and optionally the MHR recombination table remain available. With a CUDA context already created, backend entry on an A6000 took 93.64 → 21.89 s with empty compilation caches and 1.31 → 0.44 s with warm caches (median of four). Four single-node comparisons covered 98,304 field values. [Initialization timings and node results](../reports/v432-nonthermal-setup.json)

Both Eulerian source models support `RECOMB_MODEL="homogeneous"` with `CELL_RECOMB=True`, with spin fluctuations on or off. The native volume averages feed one scalar recombination history per epoch. The CUDA MHR kernel evaluates it at the stored redshift, and the barrier uses that history with the local density. A three-node thermal `E-INTEGRAL` run using explicit GSL-QAG passed 13 stored fields and 712 transverse power bins. The three cumulative histories matched the native float values exactly (0.0277841, 0.0788969 and 0.1396166 per mean baryon). [Single-node comparisons](../reports/v432-homogeneous-recombinations.json) · [Integrated lightcone](../reports/v432-eulerian-homogeneous-lightcone.json)

### Angular lightcones

Pass an upstream `AngularLightconer` to the same GPU lightcone driver. The native lightconer supplies comoving distances and shell selection. The adapter calls cosmotile once for the unit rays, then CUDA scales, rotates and translates them for each shell. At a rotation-sensitive nearest-neighbor boundary it uses the native coordinates for that shell. CUDA performs periodic spatial interpolation (orders 0, 1, 3 and 5), interpolation between coevals using the v3 weighted arithmetic, and projection of the three Cartesian velocities onto each sightline. It follows the native choice of sampling before or after interpolation in redshift, including the corner-wise sign rule for `z_reion`. Angular velocity corrections and RSD remapping require `KEEP_3D_VELOCITIES=True`. [Shell coordinate checks and timing](../reports/v432-angular-shell-geometry.json)

Angular and rectilinear finalization share the same device velocity data across the optical-depth correction and RSD fields. The driver evaluates the native Astropy Hubble factors once per finalization input, uploads the line-of-sight velocity once and retains the CUDA pixel displacement for the RSD fields. [Native comparisons and reuse counts](../reports/v432-lightcone-velocity-reuse.json)

A thermal constant-efficiency case at 32³ → 16³ tested 128 sky directions, 80 output slices, 14 fields, 63 global values and 800 angular power values with and without RSD. All passed the native tolerance. The unshifted brightness and the line-of-sight velocities also matched bit for bit. Each native coeval was computed once. Independent single-pair tests covered all four interpolation orders on rotated rays, reversed coeval ordering, periodic wrapping and empty shell selections. The checks use the native spline convention in cosmotile 0.2.5 (prefilter disabled) and 0.3.0 (periodic prefilter for orders 3 and 5). [Angular acceptance](../reports/v432-angular-lightcone.json)

### Molecular cooling and radiation feedback

The public driver accepts thermal `L-INTEGRAL` and `CHMF-SAMPLER` with `USE_MCGS=True`, photoheating feedback, metallicity, inhomogeneous recombinations and `V_CB_MODEL="AVG-DEBUG"`. This option uses the native fixed relative velocity. For both source models, `POWER_SPECTRUM="CLASS"` accepts `V_CB_MODEL="AVG-AUTO"` (cosmology-dependent mean) and `V_CB_MODEL="FLUCTS"` (a spatial relative-velocity field). The upstream CLASS solver supplies the transfer table. CUDA evaluates its interpolation, high-wavenumber continuation, mass variances, seeded initial conditions and perturbations. [CLASS initial fields and mass variances](../reports/v432-class-avg-auto-ic.json) Select a density grid with the same resolution as the feedback grids. The 4.3.2 deposition routine indexes the turnover mass at its input density-grid index.

The CUDA path constructs both populations’ conditional source tables and their global ionizing-photon histories, evaluates the two local turnover masses, looks up both galaxy populations and deposits them in source order. Ionization consumes the combined photon grid and the native high-redshift MCG floor. The six scalar photon-history integrals took 0.271 ms on the CPU and 0.541 ms on the GPU in a warm single-node measurement. [Ionization history](../reports/v432-ionization-global-nion.json) Pop III supplies its own stellar spectrum, star-formation history and Lyman–Werner feedback. The (z=11.5) replay uses the saved CPU density, source catalog and previous thermal and ionization fields. All five emissivity grids match bit for bit, including for 18,243 sampled halos. The separate SFRD and LW checks scale the tolerance to the native field amplitude. [Integrated source comparisons](../reports/v432-molecular-lightcones.json) · [Ionization barrier](../reports/v432-molecular-ionization.json) · [Sampled sources](../reports/v432-mcg-sampled-sources.json)

Both 24 Mpc tests use 32³ / 16³ grids, one thread, seed 129887 and nodes 12, 11.5 and 11. Each produces 97 lightcone slices, 15 fields, 63 global values and 776 transverse power values. The two populations' conditional photon, SFRD and metallicity-dependent X-ray tables use the native GL and adaptive high-density branches. Reuse the v3 frequency integrator with the 4.3.2 shell turnover history for the X-ray optical depth. [Conditional tables](../reports/v432-mcg-conditional-tables.json) · [Global photon history](../reports/v432-mcg-global-nion.json) · [Thermal feedback](../reports/v432-mcg-radiation-turnover.json) · [X-ray tables and stellar spectra](../reports/v432-molecular-radiation-setup.json)

For a matched CLASS rerun, use the saved native inputs (`p21.LightCone.from_file("cpu.h5").inputs`). Both saved lightcones contained the identical 231-point transfer table and mean relative speed of 27.03787802692852 km/s. A fresh 32³ / 16³ GPU lightcone took 136.53 s, including 124.39 s in a single host CLASS solve. The native CPU lightcone took 139.75 s. The verifier reuses the cosmology stored in the CPU HDF5 after validating the requested parameters. Replaying the five emissivity fields at z=11.5 took 1.36 s with zero CLASS calls and gave bitwise agreement. [CLASS lightcone and transfer hashes](../reports/v432-molecular-lightcones.json)

With spatial streaming enabled, all 4,096 velocities matched the CPU bit for bit and ranged from 0.7663 to 20.8592 km/s. The corresponding GPU lightcone reused the saved transfer tables, made zero CLASS calls and finished in 24.43 s. The CPU run included a 122.20 s CLASS solve in its 137.70 s total. The two SFRD fields, X-ray emissivity and LW intensity matched bit for bit over all 97 slices. [Relative-velocity initial fields](../reports/v432-class-flucts-ic.json) · [Spatial-streaming lightcone](../reports/v432-molecular-lightcones.json)

The sampled-halo molecular driver also accepts CLASS with spatial streaming. With the same native transfer tables loaded on both sides, the 32³ / 16³ CHMF lightcone took 30.74 s CPU and 18.87 s GPU. The three ordered catalogs matched in mass, coordinates and all three random fields (15,389 halos at z=12, 19,230 at 11.5 and 23,325 at 11). Replaying the saved z=11.5 inputs gave exact agreement for all 20,480 source values. The public driver passed all 15 fields, 63 global values and 776 transverse power values over 97 slices. The largest brightness difference was 7.63 × 10⁻⁶ mK. [Sampled-halo CLASS comparison and single-node replay](../reports/v432-molecular-lightcones.json)

The `CHMF-SAMPLER` CLASS `AVG-AUTO` comparison used the native mean of 27.037878 km/s and reused identical archived CLASS tables on both sides. The three ordered halo catalogs and the cached emissivity, thermal and ionization arrays matched bit for bit. The raw radiation arrays passed a per-value relative tolerance of 10⁻⁵. The 32³ / 16³ lightcone passed 15 fields, 63 global values and 776 power values, with 29.62 s CPU and 9.04 s GPU in one allocation. [Mean-streaming comparison](../reports/v432-chmf-class-avg-auto.json)

All five source models passed native CPU comparisons with atomic cooling, `POWER_SPECTRUM="CLASS"` and `V_CB_MODEL="NONE"`. The CLASS CUDA kernel applies the native streaming condition to density power, including normalization and mass variance. The 32³ / 16³ lightcones each passed 13 fields and 57 global values, with 4,008 transverse power values in total. The CHMF and DexM catalogs matched exactly at all six sampled nodes. Spectrum probes covered 237 wavenumbers per mode for `NONE`, `AVG-AUTO` and `FLUCTS`, with 600 mass-table values per mode matching the CPU bit for bit. [Density-only CLASS acceptance](../reports/v432-class-density-only.json)

The native `BBKS`, `EFSTATHIOU`, `PEEBLES` and `WHITE` transfer functions also use the CUDA initial-condition and mass-quadrature path. The adapter retains the native top-hat radii and σ₈ normalization. Across these four spectra, all 2,400 mass-table entries and 638,976 seeded initial-condition values matched the CPU bit for bit. Four 32³ / 16³ Eulerian thermal lightcones each passed 13 fields, 57 global values and 776 transverse power values. [Analytic spectra and lightcone acceptance](../reports/v432-analytic-spectra.json)

The Eulerian `E-INTEGRAL` molecular branch normalizes the two SFRD fields with the native density weights, computes current and previous photon counts at each radius and carries each population's history forward separately. Both filter radii used the previous-history correction at the two later redshifts. All 6,144 photon-history values matched the CPU bit for bit in isolated replays. Standalone emissivity checks cover luminosities with and without metallicity dependence. [Emissivity fields and tables](../reports/v432-eulerian-mcg-emissivity.json) · [Conditional photon history](../reports/v432-eulerian-mcg-conditional-history.json) · [Ionization and filter geometry](../reports/v432-eulerian-mcg-ionization-geometry.json)

Two public Eulerian molecular lightcones used 16³ / 8³ grids, 24 Mpc, seed 129887, one thread, photoheating, inhomogeneous recombinations and metallicity-independent luminosities. Each pair passed all 15 fields over 49 slices, 66 global values and 196 transverse brightness-power values. The timer covers the driver call on the same host, including coeval cache writes, and stops before the final lightcone save. Both spatial-streaming runs loaded the same native transfer tables and made zero CLASS calls. All 512 spatial relative velocities matched bit for bit, ranging from 1.5343 to 17.6129 km/s. [Matched Eulerian lightcones](../reports/v432-eulerian-mcg-lightcone.json)

| Transfer and relative velocity | 4.3.2 CPU | 4.3.2 GPU | Speedup | Largest brightness difference |
|---|---:|---:|---:|---:|
| EH, fixed mean (`AVG-DEBUG`) | 16.98 s | 5.07 s | 3.35× | 3.81 × 10⁻⁶ mK |
| CLASS, spatial field (`FLUCTS`) | 17.42 s | 6.00 s | 2.90× | 1.91 × 10⁻⁶ mK |

`LYA_MULTIPLE_SCATTERING=True` uses the shared CUDA FFT with the native 4.3.2 scattering window. The native shell scalars feed the double-precision CUDA recurrence. X rays and Lyman–Werner radiation use straight-line shell windows. In 30 window comparisons, all 265,920 Fourier values matched the native release build bit for bit. An Eulerian molecular lightcone passed the same 15 fields, 66 global values and 196 power values, with a maximum brightness difference of 1.91 × 10⁻⁶ mK. Turning on scattering changed the native brightness by up to 1.80 mK in that fixture. The driver took 18.65 s CPU and 20.93 s GPU, including the GPU's first use of the scattering kernel. [Scattering-window and public driver results](../reports/v432-lya-scattering.json)

A Lagrangian `L-INTEGRAL` replay at z=11.5 used the same saved emissivity history and previous ionization state for both backends. All 40,960 radiation and thermal values passed. The four output fields for spin temperature, kinetic temperature, electron fraction and LW intensity matched bit for bit. The straight-line case also passed its cached-node regression. [Lagrangian radiation replays](../reports/v432-lya-scattering.json)

The public molecular `L-INTEGRAL` lightcone with multiple scattering also passes the native 4.3.2 comparison. ACG and MCG emission use the same radii and scattering parameters at each shell, so the GPU retains one double-precision Fourier window and applies it to both source fields. The three-node 32³ → 16³ run reused 62 windows. For 30 pairs of 64³ Fourier inputs, filter time fell from 282 ms to 143 ms. [Lagrangian lightcone and window reuse](../reports/v432-lagrangian-lya-window-reuse.json)

Radiation-shell interpolation keeps the two bracketing emissivity snapshots on the device for reuse across shells, subject to the available memory budget. The adapter releases its references after each radiation call. The z=11 replay used two successive snapshot pairs across 34 shells. It reduced input uploads from 204 to 9, and all 417,792 interpolated values were bitwise equal to the native CPU calculations. The six radiation fields passed the same 10⁻⁵ per-value relative tolerance with normal capacity and with caching disabled. [Saved-node comparison and upload counters](../reports/v432-emissivity-history-reuse.json)

Shell filters also reuse the on-device interpolation result. The host copy serves the native array interface. For molecular sources with multiple scattering, each population can reuse one forward FFT across its Lyα and Lyman–Werner windows. The 34-shell Lagrangian replay avoided 170 filter input uploads and 66 forward FFTs. The 11-shell Eulerian straight-line replay avoided 33 filter input uploads. Both reproduced the prior GPU radiation fields bit for bit and passed the native 10⁻⁵ per-value relative tolerance. [Filter handoff and FFT checks](../reports/v432-emissivity-filter-handoff.json)

When the radiation workspace has capacity, the accumulator reads the filtered grids on CUDA and keeps the running radiation sums on the device through the final normalization. The adapter publishes the normalized fields and the last shell's scratch grids to the native arrays. Native debug logging keeps intermediate host arrays available. [Radiation accumulation and host transfer checks](../reports/v432-radiation-accumulation-residency.json)

The joined small-box acceptance exercised `L-INTEGRAL` and `E-INTEGRAL` with multiple scattering and `CHMF-SAMPLER` with CLASS spatial streaming. Each latest GPU run matched all 15 stored lightcone fields, all evaluated global quantities and all transverse brightness power bins against its saved native CPU run. Across the three cases, all 552,960 cached emissivity, raw radiation, thermal and ionization values passed a per-value relative tolerance of 10⁻⁵. The CLASS tables, relative-velocity grid and three ordered halo catalogs matched the native reference. [Joined lightcone and cached coeval results](../reports/v432-joined-small-lightcones.json)

### Photon conservation

`run_lightcone` and `generate_lightcone` accept a matching `photon_calibration` for Eulerian sources. `E-INTEGRAL` supports the redshift, escape-fraction amplitude and escape-fraction slope corrections. `CONST-ION-EFF` supports the redshift correction. The upstream host fit supplies the correction to the accelerated ionization calculation.

```python
from gpu_lightcone import run_lightcone
from photon_conservation import PhotonCalibration

# calibration_boxes contains native uncorrected IonizedBox snapshots,
# generated with photon_conservation.calibration_inputs(inputs).
calibration = PhotonCalibration.from_boxes(inputs, calibration_boxes)
cone = run_lightcone(
    inputs=inputs, lightconer=lightconer, photon_calibration=calibration
)
```

Each calibration sample pairs its actual redshift with the mean neutral fraction from that snapshot. The unmodified 4.3.2 Python calibration loop appends a final redshift without a corresponding measurement. These comparisons supplied paired arrays to both the native C calibration and the adapter. Eight thermal and nonthermal single-node cases compared the corrected fields and all fitted histories. A two-node public `E-INTEGRAL` redshift-corrected lightcone also passed. [Calibration, fit and corrected fields](../reports/v432-photon-conservation.json)

### Constant ionizing efficiency

`CONST-ION-EFF` uses the native conditional extended Press–Schechter collapse fraction. Its unconditional normalization integrates the requested Press–Schechter (PS) or Sheth–Tormen (ST) mass function over the native finite mass range. The ionizing yield uses `HII_EFF_FACTOR`. The thermal source uses the native redshift derivative of the collapse fraction with the X-ray threshold and the 4.3.2 time and density factors. The adapter shares the CUDA quadrature and ionization kernels with the other source paths. [Conditional emission](../reports/v432-constant-efficiency-conditional.json) · [Global integrals](../reports/v432-constant-global-integrals.json)

| 32³ → 16³, 24 Mpc, ST, seed 812445 | Stored fields | Global values | Transverse power bins | Largest brightness error |
| --- | ---: | ---: | ---: | ---: |
| Spin fluctuations off, recombinations off | 8 | 33 | 712 | 1.29 × 10⁻⁶ mK |
| Spin fluctuations on, inhomogeneous recombinations | 14 | 57 | 712 | 1.91 × 10⁻⁶ mK |

Both lightcones cover 89 slices from the same three nodes per configuration. All stored fields, per-node outputs (including the initial conditions), global values and power bins passed the per-value tolerance. Separate native/GPU PS node comparisons exercised the early neutral limit at redshift 45 and cell-based recombinations at redshifts 9 and 8.7. The PS checks cover individual nodes. [Lightcone and coeval comparisons](../reports/v432-constant-lightcones.json)

### Lagrangian source models

| Source model and settings | Native/GPU lightcone comparison |
| --- | --- |
| Default `CHMF-SAMPLER`, spin fluctuations off, recombination model `none` | 48³ → 16³, 42 Mpc, 23 slices. All six stored fields, 30 global values, 184 transverse power bins and the ordered masses, coordinates and galaxy random variables of 795,125 halo entries matched the reference criteria. [Acceptance and timing](../reports/v432-chmf-default-integrated.json) |
| `CHMF-SAMPLER`, `USE_TS_FLUCT=True`, `RECOMB_MODEL="inhomogeneous"` | 32³ → 16³, 24 Mpc, 30 slices from nodes 10 and 9.7. All 13 stored fields, 38 global values and 240 transverse power bins passed. Spin temperature and cumulative recombinations were bitwise equal. [Thermal lightcone](../reports/v432-chmf-thermal-integrated.json) |
| `DEXM-ESF` and `L-INTEGRAL`, spin fluctuations off, recombination model `none` | Each case used 32³ → 16³, 24 Mpc, 54 slices and passed all six fields, 20 global values and 432 transverse power bins. The DexM run emitted 175 catalog entries over two epochs. The Lagrangian integral model computes its source grid directly. [Both lightcones](../reports/v432-lagrangian-lightcones.json) |
| `DEXM-ESF` and `L-INTEGRAL`, spin fluctuations on with inhomogeneous recombinations | At 32³ → 16³, 24 Mpc and nodes 10 and 9.7, both models produced a nonzero 21 cm absorption signal. Each comparison passed 13 stored fields and 240 transverse power bins. Two independent seeds passed the two-thread Lagrangian integral test. [Thermal lightcones](../reports/v432-lagrangian-thermal-lightcones.json) |

The 42 Mpc default CHMF case took **24.66 s on the native CPU and 10.14 s on the GPU (2.43×)**. Both runs used node a9, one CPU thread, the same saved inputs and the same public-driver timer. The final GPU replay used warm compiler caches and reused the CPU time from an earlier allocation. Before the shared power-spectrum change, successive processes in a single allocation measured 24.66 s CPU and 9.64 s GPU. An older revision on node a8 measured 14.78 s CPU and 17.52 s GPU. [Timing and scope](../reports/v432-chmf-default-integrated.json) · [Earlier public runs](../reports/v432-chmf-lightcone.json)

The conditional inverse mass table accounted for 22.29 s in an earlier 26.20 s native a9 run. CUDA builds the conditional number and collapsed-fraction tables and identifies reachable rows. The sampler now extends a row only to the lowest probability bin requested so far, using native GSL Brent roots and the same descending brackets. Across four seeds it evaluated 125,011 of 274,078 entries in the selected rows (45.61%). Every evaluated double matched the full native table bit for bit. All 12 mass- and number-limited catalogs matched their respective ordered CPU digests. Eight catalog calls took 5.73 s versus 8.94 s with the earlier selected-row implementation in a single paired timing. A blind test of CUDA roots showed that a rounding difference could change the mass and daughter catalog. The production path uses native roots and executes the mass and merger decisions in native order. [Demand-driven table and catalog comparisons](../reports/v432-chmf-demand-inverse.json) · [CUDA root comparison](../reports/v432-chmf-brent-precision.json)

The CHMF sampler now refills its MT19937 and GFSR4 buffers from the same CUDA generators used by the initial conditions. Native GSL still determines how many words each Poisson, inverse-CDF, position or property draw consumes. Each catalog starts at the same native per-worker seed and can consume across arbitrary batch boundaries. The one- and two-thread number-limited cases and the one-thread CLASS molecular mass-limited case matched their archived native catalogs in all five ordered fields. [CUDA random-word routing and catalog digests](../reports/v432-chmf-cuda-random-words.json)

At the saved CHMF inverse-CDF divergence points, the shared CUDA QAG implementation reproduced all 161 GSL integrals bit for bit when given the exact native integrand values. It requested only recorded abscissae. The reordered CUDA formula, evaluated independently, reproduced 129 of those integrals exactly. At the 10,431 recorded integrand points, supplying the native derivative, Taylor factor, barrier and exponential made the remaining CUDA arithmetic exact. The next precision checks concern these elementary functions and the logarithm used in the Brent residual. [Inverse-CDF arithmetic replay](../reports/v432-chmf-inverse-cdf-arithmetic.json)

On the EPYC 9754 and glibc 2.39 reference host, a device-linked implementation matched 750,000 scalar results for `exp`, `pow(x,-0.81)` and `log`. FMA contraction is enabled in its elementary functions and disabled in the CUDA physics and solver. It independently reproduced all 161 QAG integrals and both 200×400 inverse tables bit for bit. The quantile loop multiplies by the reciprocal of the invariant mass, matching the native executable. [CUDA Brent and archived table comparisons](../reports/v432-chmf-cuda-brent-exact.json)

Set `P21_CUDA_GLIBC_SOURCE` to the glibc math source directory to select CUDA inverse-CDF roots in the public CHMF backend. It contains `math_config.h`, `e_exp.c`, `e_exp_data.c`, `e_log.c`, `e_log_data.c`, `e_pow.c` and `e_pow_log_data.c`. The module checks its elementary functions against the host at first use. CUDA evaluates the row normalization and roots. The native driver provides the condition grid and consumes the inverse table in its original sampling order. When the variable is unset, the driver uses demand-driven GSL roots. [Public CHMF comparisons and timing](../reports/v432-chmf-cuda-inverse-driver.json)

With CUDA roots enabled, the three CLASS molecular catalog calls took 2.16 s in total, compared with 3.56 s for the CUDA backend with GSL roots. The public 32³ / 16³ lightcone loaded the same CLASS tables as its archived native CPU reference and took 16.95 s GPU versus 30.74 s CPU. The catalogs matched bit for bit. All 15 fields, 63 global values and 776 transverse power values passed, with a maximum brightness difference of 7.63 × 10⁻⁶ mK. [Matched CLASS lightcone and catalog timing](../reports/v432-chmf-cuda-inverse-driver.json)

The CUDA mass-limited sampling routine reproduced the native ordered masses and exact random-word consumption for 2,046 progenitor conditions captured from the CLASS molecular run. This includes 277 zeroed masses from the correction. A separate native comparison covers 36 correction cases around the target threshold and exercises rejected uniform indices. [Mass sampling and correction replays](../reports/v432-chmf-mass-sample-replay.json)

With `P21_CUDA_GLIBC_SOURCE` configured, `P21_CUDA_PROGENITOR=1` enables the batched MASS-LIMITED progenitor path. Each window consumes resident random words through mass correction, accepted Gaussian pairs and position draws. The shared cursor advances only past complete conditions. A condition that straddles the available word block starts again in the native path from its original word. The one-thread CLASS and two-thread EH cases produced identical ordered catalogs. In one same-allocation CLASS comparison, the three catalog calls took 1.14 s with native sampling and 1.24 s with the batched CUDA path. Both runs used CUDA inverse roots. [Batched progenitor tests and timing](../reports/v432-chmf-batched-progenitors.json)

`P21_CUDA_GRID_SAMPLE=1` enables the initial grid window when CUDA inverse tables are configured. The window evaluates GSL's product method for Poisson means through 10 and then consumes the mass, position and property draws in order. `P21_CUDA_PROGENITOR=1` also enables this path for NUMBER-LIMITED progenitors. The original GSL gamma/binomial routine handles larger means when `P21_CUDA_GSL_SOURCE` is unset. The CLASS grid contains 1,802 means above 10, with a maximum of 116.42. [Production grid, Poisson oracle and large-mean traces](../reports/v432-chmf-grid-poisson.json)

Set `P21_CUDA_GSL_SOURCE` to the unpacked GSL 2.7 source tree to enable high-mean Poisson on CUDA. CUDA follows the native integer-gamma and BINV/BTPE decisions. Add glibc 2.39's `s_tan.c`, `utan.h`, `utan.tbl`, `dla.h` and `mydefs.h` from `sysdeps/ieee754/dbl-64` to the directory in `P21_CUDA_GLIBC_SOURCE` to evaluate the integer-gamma tangent on CUDA when GSL requests it. The adapter checks the tangent against the host's scalar libm when loading the module. With the original glibc file set, CUDA reads a tangent table evaluated using the scalar host function. Conditions that do not fit the current word window are replayed from their original cursor. [GSL replay and driver acceptance](../reports/v432-chmf-high-poisson.json) · [Device tangent and catalog checks](../reports/v432-chmf-device-tangent.json)

The CLASS small box used CUDA for 4,088 of 4,096 initial cells, including 1,797 high-mean conditions. Its ordered catalogs matched the CPU, and the full lightcone passed all 15 fields, 63 global quantities and 776 transverse power values. The matched driver calls took 30.74 s CPU and 18.27 s GPU, and used the same cached CLASS transfer tables. [Matched lightcone](../reports/v432-chmf-high-poisson.json)

Each worker keeps its condition block across sampling windows. Only complete conditions commit random words. With CUDA tangent enabled, the initial grid prepares each active block on CUDA using the resident density from inverse-table row selection and the resident number and collapsed-fraction table. Progenitor blocks reuse the resident parent masses, the same moment table and the retained σ(M) table. CUDA evaluates the native logarithm, interpolates σ(M) and computes the Sheth–Tormen collapse barrier at both redshifts. `MASS-LIMITED` can use device condition preparation without CUDA tangent. The two small boxes matched the saved ordered CPU catalogs and every audited condition. [CPU condition reuse](../reports/v432-chmf-condition-reuse.json) · [CUDA grid conditions](../reports/v432-chmf-grid-conditions.json) · [CUDA progenitors and integrated lightcone](../reports/v432-chmf-progenitor-conditions.json)

`BINARY-SPLIT` uses the same CUDA condition and random-word windows with the native Parkinson timestep and branch queue. It reads the 4.3.2 σ, slope and J tables and rounds active branch masses and barriers to float32 at the native update points. On the 32³ → 16³ test, CUDA committed 32,900 of 32,908 progenitor conditions. The ordered catalogs matched for `PARKINSON_y1` values 0, +0.2 and −0.2. The native Sheth–Tormen `PARTITION` case segfaulted in the progenitor call. Its rejection helper passes a null output to a function that writes the zeroth-order barrier. [Binary-split replay, catalog and lightcone checks](../reports/v432-chmf-binary-split.json)

`PARTITION` also uses CUDA condition and random-word windows with `P21_CUDA_PROGENITOR=1` and the existing glibc configuration. The sampler follows the GSL upper-tail branches and inverts the same σ table with the native linear interpolation. The adapter's CPU fallback and the independent CPU verifier supply storage for the discarded barrier in the affected Sheth–Tormen call. The verifier builds its CPU reference in a private shim and records the repair in its metadata. The installed 4.3.2 release stays unchanged. [Partition replay, catalog and lightcone checks](../reports/v432-chmf-partition.json)

The linked CHMF CUDA module is cached under CuPy's kernel-cache directory. Its key includes the two generated CUDA sources, their compile options, the GPU architecture and toolchain versions. The adapter runs the same host math comparisons when loading either a fresh or cached module. `CUPY_CACHE_IN_MEMORY=1` disables disk caching. [Cross-process setup measurements and lightcone check](../reports/v432-chmf-linked-cache.json)

CHMF sampling keeps one CUDA stream per native worker for the lifetime of the backend. A worker submits its condition preparation and sampling windows in order. Independent workers can overlap, and shared resident tables are ready before either stream reads them. [Paired stream timings and native catalog checks](../reports/v432-chmf-worker-streams.json)

Each native worker also reuses its device scratch arrays and pinned result buffers. A window reports the number of masses it committed with its random-word cursor. The adapter downloads only the committed payload and waits once for the payload copies. Empty windows return zero offsets without a payload transfer. [Window boundary checks and paired catalog timings](../reports/v432-chmf-window-buffers.json)

At a thermal CHMF node with 30 filter radii, the upstream setup also evaluated 31 mass-variance integrals whose values are unused with Lagrangian photons. Skipping those evaluations reduced median ionization setup time from 8.65 ms to 0.26 ms. All 24,576 values of the six ionization fields remained bitwise equal to the native node. [Source scaling and radius setup](../reports/v432-source-scaling-setup.json)

Lagrangian emissivity, radiation setup and ionization now share the CUDA cell mass variance through the native 4.3.2 scaling calculation. Both the molecular `L-INTEGRAL` EH case with Lyα multiple scattering and the `CHMF-SAMPLER` CLASS case passed all 17 grids against their saved 32³ → 16³ CPU nodes. The five emissivity and six ionization grids were bitwise equal in both cases. The six radiation grids passed a relative tolerance of 10⁻⁵ per value, including both raw Lyα fluxes with the previous ionized box supplied. [Source scaling and single-node results](../reports/v432-lagrangian-source-sigma.json)

Compiled-shim caching reduced CHMF backend entry from 4.67 s cold to 0.89 s warm in fresh processes. [Stage profile](../reports/v432-chmf-stage-profile.json) · [Mass and catalog checks](../reports/v432-halo-mass-function.json) · [Compiled-shim cache](../reports/v432-native-shim-cache.json)

CUDA filters and checks DexM candidates, generates the resolved catalog’s three Gaussian property streams, places sampled halos, maps halo and unresolved contributions to the emissivity grid and evolves the center-method ionization barrier. The thermal CHMF path reads the actual metallicity-dependent X-ray luminosity density in units of `10^38 erg s^-1 Mpc^-3`. The optical-depth calculation takes the homogeneous photon history from a CUDA integral over the native capped stellar and escape-fraction power laws. The two-node test exercised 28 radiation shells, the thermal step, the MHR recombination table and cumulative recombinations. Both drivers reuse the CUDA EH transfer and sigma mass-table kernels. All 1,800 mass-variance and derivative entries in three cosmologies reproduced the native float32 tables. Host work includes the mass-dependent scaling constants, the native CHMF GSL draws and selected roots, and the Astropy shell geometry. The default atomic-cooling metallicity-dependent X-ray table uses shared CUDA quadrature. [Shared power setup](../reports/v432-shared-power-setup.json) · [X-ray table checks](../reports/v432-halo-metallicity-xray.json). [DexM finder](../reports/v432-dexm-finder.json) · [DexM properties and 64-bit IC seeds](../reports/v432-dexm-property-rng.json) · [Optimized DexM exclusion replay](../reports/v432-dexm-optimized.json) · [Source advection](../reports/v432-chmf-source-advection.json) · [Radiation and heat](../reports/v432-chmf-radiation.json) · [Global photon history](../reports/v432-chmf-global-nion.json) · [Ionization and recombinations](../reports/v432-chmf-ionization.json)

The public CHMF driver accepts both mass-limited and number-limited progenitor sampling. Both use the native grid Poisson draws at the first node. With two threads and seed 424242, the same 32,908 initial halos gave 30,205 number-limited progenitors or 30,290 mass-limited progenitors. The number-limited catalogs matched the native CPU in ordered masses, coordinates and all three random properties for both one and two threads. [Sampling method checks](../reports/v432-chmf-number-limited.json)

Gauss–Legendre is the 4.3.2 default for the atomic-cooling mass-dependent models. Explicit `INTEGRATION_METHOD_ACGS="GSL-QAG"` now uses the shared adaptive CUDA quadrature for conditional photon and SFRD tables across the full density range. The comparison includes the high-density collapse limit and the log floor. A 32³ → 16³ thermal CHMF lightcone joined this quadrature, on-demand inverse tables and GPU property sampling. All 13 fields, 38 global values and 240 power bins passed. The 72,692 catalog entries matched in all five arrays. The native driver took 15.62 s and the accelerated driver took 7.28 s on a9 in separate allocations, with a single measurement per route. [Integrated small lightcone](../reports/v432-chmf-qag-integrated.json) · [Quadrature comparisons](../reports/v432-conditional-qag.json)

CHMF positions and three Gaussian property transforms, including the progenitor–descendant correlations, share one GPU batch. GSL consumes uniforms in the native per-thread draw order. A scalar libm evaluation resolves ambiguous final float rounding. Across 218,856 sampled halos, all five ordered arrays matched the native catalogs. [CHMF sampler comparisons](../reports/v432-chmf-property-rng.json)

### Device residency and zero-upload lightcones

[device_registry.py](device_registry.py) retains 3D field buffers on GPU memory under an automatic VRAM budget (`~12.66 GB` on an RTX A6000) and connects [perturb.py](perturb.py) caching, [evolution_backend.py](evolution_backend.py) radiation and ionization residency, [lightcone_device.py](lightcone_device.py), [angular_slices.py](angular_slices.py), and [brightness.py](brightness.py) so rectilinear and angular lightcones consume on-device grids without redundant host uploads. Across the 43-node Eulerian benchmarks on a8 (`192³ / 64³` and `384³ / 96³`), the registry recorded `3,509` hits (`3.90 GB`) and `3,393` hits (`12.75 GB`), while emissivity history skipped `3,262` (`3.42 GB`) and `3,258` (`11.53 GB`) shell downloads and kept the Eulerian ionization filter-radius loop resident (`682` and `620` calls). On the same host a8, `192³` BBKS + Gaussian window (`case2`) dropped from `49.93 s` (prior a9 warm GPU) to **`11.55 s` warm / `14.96 s` cold GPU** versus **`63.76 s` CPU (`5.52×`)** with zero failures; `192³` CLASS `FLUCTS` + MCGS + `LYA_MULTIPLE_SCATTERING` (`case1`, 10 nodes) ran in **`47.96 s` GPU** versus **`248.89 s` CPU** (`196.48 s` excl. CLASS -> **`4.10×` excl. CLASS / `5.19×` total**), passing all 10 coeval boxes, 220 globals, 11,520 transverse power bins, and 14/15 lightcone fields with the same `37` (`0.0025%`) near-zero `T_s ≈ T_CMB` voxels in `brightness_temp` (`max_abs = 3.05 × 10⁻⁵ mK`). [Resident optimization results](../reports/v432-resident-large-lightcones.json) · [Large-grid new-physics checks](../reports/v432-large-new-physics.json)

### Multithreaded CHMF-SAMPLER and hybrid vs prefix sampling

`CHMF-SAMPLER` now supports `N_THREADS >= 1` across all five upstream GSL PRNGs (`mt`, `gfsr`, `cmrg`, `mrg`, `taus2`). [halo_catalog_random.h](halo_catalog_random.h) adds exact 31-bit GSL normalization (`/ 2147483647.0`, `max = 2147483646`) for workers whose `worker % 5` selects `cmrg` or `mrg`; [halo_catalog.py](halo_catalog.py), [halo_grid.py](halo_grid.py), and [catalog_progenitor.py](catalog_progenitor.py) fix module stream pre-resolution, premature pre-subgrid registry binding, and `sigma` scoping when both `P21_CUDA_GLIBC_SOURCE` and `P21_CUDA_GSL_SOURCE` are set. Across `32³ / 16³` thermal lightcones (`verify_chmf_multithread.py --suite small`), all **8/8** cases (`1/2/4/8` threads × `hybrid` and `prefix`) passed with bitwise-identical ordered catalogs (`1.64–1.99 s` warm GPU, `2.73×–6.78×`). On `96³ / 32³` (`6.51M–6.68M` halos), all cases matched catalogs bit for bit and passed globals and 2,144 transverse power bins: `1T hybrid` passed cleanly in **`4.81 s`** (`4.94×` vs `23.78 s` CPU; **`15.4×` faster** than the prior `74.15 s` prefix run); `2T hybrid` (**`3.80 s`**, `3.78×` vs `14.37 s` CPU) and `8T hybrid` (**`5.46 s`**, `1.58×` vs `8.61 s` CPU) differed only by 1 and 2 ionization-boundary voxels from CPU OpenMP float32 atomic reduction order; `1T prefix` improved to `48.52 s` (`0.49×`), remaining bottlenecked by serial `(1,),(1,)` CUDA prefix sampling. [CHMF verification driver](verify_chmf_multithread.py) · [CHMF multithread and hybrid speed](../reports/v432-chmf-multithread-and-speed.json)

### Device LZF HDF5 I/O and concurrent batch runner

[lightcone_io.py](lightcone_io.py) adds direct chunked HDF5 writers for `none`, `parallel-lzf`, and `device-lzf` compression with aligned chunk layout and fsync synchronization. All three modes round-trip identically (`bitwise_equal: true` across all lightcone fields, globals, distances, and inputs): at `192³` (`650` slices), `none` writes `85.25 MB` in `0.382 s` while `device-lzf` writes **`63.34 MB` in `0.352 s`** (`1.35×`, `328` chunks, `0` skipped); at `384³` (`521` slices), `none` writes `153.70 MB` in `0.679 s` while `device-lzf` writes **`116.99 MB` in `0.685 s`** (`1.31×`, `1,056` chunks, `0` skipped). [run_batch.py](run_batch.py) provides multi-worker concurrent execution (`64³ verify` passed `2/2` in `25.40 s`). Single-A6000 throughput (`count=4`, `threads_per_worker=2`, `device-lzf`):

| Grid / Box | Concurrency | Wall time | Throughput | Peak VRAM | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| `192³ / 64³`, 200 Mpc | 1 | 61.50 s | 234.1 / h | 2,242 MiB | Serial baseline |
| `192³ / 64³`, 200 Mpc | 2 | 50.19 s | 286.9 / h | 4,480 MiB | Optimal (+22.5%) |
| `192³ / 64³`, 200 Mpc | 4 | 107.20 s | 134.3 / h | 3,081 MiB | Single-GPU scheduling contention |
| `384³ / 96³`, 375 Mpc | 1 | 146.96 s | 98.0 / h | 9,178 MiB | Serial baseline |
| `384³ / 96³`, 375 Mpc | 2 | 129.24 s | 111.4 / h | 18,354 MiB | 4 / 4 passed |
| `384³ / 96³`, 375 Mpc | 4 | 121.10 s | 118.9 / h | 35,885 MiB | Optimal (+21.3%) |

[LZF and batch report](../reports/v432-lzf-and-batch.json)

### Comparing recorded run times across versions

The four backends were also measured on one matched perturbation node. The 128³ → 32³ case used the same eight initial-condition arrays by SHA256, a 100 Mpc box, one thread, redshift 9.375, low-resolution 2LPT, and density and line-of-sight velocity outputs. The timer covered the public perturbation call, including synchronization. Initial-condition generation and compilation were excluded. Each entry is a median over 15 warm trials on a9.

| Matched single node | v3 CPU | v3 GPU | 4.3.2 CPU | 4.3.2 GPU |
| --- | ---: | ---: | ---: | ---: |
| 128³ / 32³, low-resolution 2LPT | 225.25 ms | 1.88 ms | 154.41 ms | 3.39 ms |

Each GPU reproduced its own native CPU’s 32,768 densities and 32,768 LOS velocities bit for bit. In the earlier 11.54 ms 4.3.2 GPU path, the search for particles contributing to two ambiguous cells took 8.34 ms. The final 3.39 ms path parallelizes that search, adds the selected terms in native order, and uses the upstream scalar growth rate. The two native CPU versions use different CIC origins and displacement grids. The newer code also applies displacement scalings at lookup, avoiding v3’s passes to scale and restore the six IC arrays. The contributions of those CPU changes to the overall 70.84 ms saving have not been measured separately. [Matched timings, profiler and precision checks](../reports/v3-v432-matched-perturbation.json)

The earlier end-to-end lightcone records are below. The physical parameters and timer boundaries differed between versions.

| Fine grid / evolved grid | v3 CPU | v3 GPU | 4.3.2 CPU | 4.3.2 GPU |
| --- | ---: | ---: | ---: | ---: |
| 192³ / 64³ | 114.87 s | 11.76 s | 65.78 s | 27.52 s |
| 384³ / 96³ | 340.60 s | 14.93 s | 347.52 s | 67.61 s |

Each CPU/GPU pair used its own version's benchmark. v3 used parameter point 80000, eight CPU threads, 92 nodes, a redshift range of 5–14, low-resolution perturbations and the line-of-sight velocity. The 4.3.2 Eulerian fixture used two threads, 43 nodes over a lightcone at 6–15, high-resolution perturbations and all three velocity components. The cosmology, seed, star-formation parameters and X-ray thresholds also differed. The v3 interval starts after IC preparation. The 4.3.2 interval includes generation and hashing of ICs plus per-node power spectra before HDF5 synchronization. The v3 small-box CPU timing record comes from job 490140 with seed 1850877075. The large-box v3 run uses seed 1850877074 and its GPU time is the median of three runs. These measurements support within-version speedups. A cross-version cost attribution requires a common source model, parameter mapping and timed interval. [v3 timing records](../reports/v3-single-box-acceptance.json) · [4.3.2 timing records](../reports/v432-gpu-lightcone-acceptance.json)

The recorded 4.3.2 CPU run is faster on the smaller grid and slightly slower on the larger grid. Within 4.3.2 alone, the 43 perturbation calls took 26.35 s CPU and 1.52 s GPU at 192³. At 384³ they took 227.50 s CPU and 15.07 s GPU, accounting for 212.43 s of the 279.90 s elapsed-time reduction. The other stages and output operations together took 120.02 s CPU and 52.55 s GPU on the larger grid. The saved inputs give the following workload differences.

| Input or measured output | v3 point 80000 | 4.3.2 Eulerian benchmark |
| --- | ---: | ---: |
| Redshift step factor, evolution nodes | 1.02, 92 | 1.04, 43 |
| Maximum ionization filter radius | 50 Mpc (effective default) | 15 Mpc |
| Heating shells | 40 | 40 |
| X-ray threshold | 1164.86 eV | 500 eV |
| log₁₀ stellar fraction at 10¹⁰ solar masses | −1.06263 | −1.3 |
| Star formation timescale in Hubble units | 0.27977 | 0.5 |
| EH spectrum σ₈ and nₛ | 0.815, 0.968 | 0.8102, 0.9665 |
| Stored slices, small / large grid | 761 / 609 | 650 / 521 |

The 4.3.2 driver builds emissivity fields at each redshift node and uses their history in radiation-shell interpolation. The v3 spin-temperature calculation builds its global Nion and SFRD tables and conditional SFRD tables by radius. Separating the timing contribution of that code change from the 49 fewer evolution nodes and the other input differences would require a matched full-lightcone v3 vs 4.3.2 benchmark, which was skipped by request; four-way matched timing currently rests on the single-node 2LPT benchmark above. For the v3 small pair, the native and accelerated HDF5 files from job 490140 have identical astrophysics, cosmology, flags, user parameters, point, seed, grids and box length.

### Porting and precision checks

See the [upstream integration guide](../docs/upstream-gpu-integration.md) for the v4 review references, build expectations and native regression cases.

1. Fix the input realization and save the native 4.3.2 parameters, source choices, redshift nodes, outputs and CPU coevals. Reuse the same cached initial, source, thermal and ionization fields for every single-step comparison that depends on them.
2. For each remaining calculation, match its inputs, units, grid indexing and update order against the v3 CUDA function and the 4.3.2 native caller. Adapt the caller when the existing kernel already performs the requested calculation. For example, lagrangian ionization copies the turnover means computed by the source stage. Only the Eulerian mass-dependent branch evaluates a separate ionization turnover grid. [Native branch replay](../reports/v432-mcg-radiation-turnover.json)
3. Compare each changed kernel on those cached inputs. The lightcone voxel criterion is `atol=1e-8, rtol=1e-4`. Include threshold crossings, low-amplitude source fields, nonzero molecular feedback, and exact catalog ordering where random draws affect later states. Read the production routing counters to verify the device call actually ran. [Ordered-deposition regression](../reports/v432-halo-grid-subnormals.json)
4. Run the joined GPU branch on a small box and compare every stored lightcone field, per-node global quantity and slice power against the saved CPU run. The `compare` mode reads only the output files. Reserve the large-box CPU/GPU comparison for the complete physics configuration.
5. Time a shared workload over the same interval when quoting speedups. Record host CLASS time, first compilation, device execution and transfers separately where measured.
6. Extend coverage to the remaining stochastic halo draws and validate Lagrangian scattering against a matching native reference. Each configuration needs its own native reference when its physical inputs change.

### Run a 4.3.2 lightcone

From the repository root, use Python 3.12 or later and a CUDA 12 environment. Install the native build prerequisites described by [21cmFAST](https://21cmfast.readthedocs.io/en/v4.3.2/installation.html), then install the adapter's dependencies and test tools.

```bash
python -m pip install -U pip
python -m pip install '.[cuda12]' '21cmfast==4.3.2' --group test
export PYTHONPATH="$PWD/v4${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest
```

The dependency package sets up the environment for the source checkout. CUDA uses native 21cmFAST 4.3.2. The pytest entry point includes CPU routing checks and GPU parity cases. The GPU cases skip when CuPy or a usable CUDA device is unavailable.

The `gpu_lightcone` module accepts native input and lightconer objects. The four run and generate entry points log a warning and call the native CPU driver when the adapter's preflight detects an unsupported configuration. Set `fallback_to_cpu=False` to require CUDA dispatch. Driver errors propagate, and an explicitly supplied GPU photon calibration requires GPU execution. GPU validation and batch workers use strict dispatch.

This example reuses the inputs, cosmology and comoving slice coordinates from a saved native CPU lightcone.

```python
import py21cmfast as p21
from py21cmfast.lightconers import RectilinearLightconer
from gpu_lightcone import run_lightcone

reference = p21.LightCone.from_file("cpu.h5")
inputs = reference.inputs
lightconer = RectilinearLightconer(
    lc_distances=reference.lightcone_distances,
    cosmo=inputs.cosmo_params.cosmo,
    quantities=("brightness_temp", "neutral_fraction", "density"),
)
cone = run_lightcone(
    inputs=inputs,
    lightconer=lightconer,
    diagnostics_dir="routing",
    regenerate=True,
    write=False,
)
cone.save("gpu-lightcone.h5")
```

You can also construct your own `InputParameters`, pass a previously generated `initial_conditions` object, use the upstream cache options or iterate `gpu_lightcone.generate_lightcone`. The supported modes use flat Lambda-CDM, even cubic grids, and `N_THREADS >= 1` across all five source models—including `CHMF-SAMPLER` (backed by CUDA ports of GSL `mt`, `gfsr`, `cmrg`, `mrg`, and `taus2`). All five source models accept linear, Zel’dovich and 2LPT perturbations. In 11 public first-node tests, 2,015,232 field values matched the native floats bit for bit. The perturbation code converts the box length to a C float before using it in double-precision displacements and mode spacing. [Perturbation algorithm checks](../reports/v432-perturb-algorithms.json) The supported mass function is Sheth–Tormen for all five source models. Constant efficiency also accepts Press–Schechter. The Lagrangian cases use metallicity-dependent sources with upper stellar turnover enabled for CHMF and DexM and disabled for `L-INTEGRAL`. Thermal Lagrangian runs use X-ray and Lyα heating. The original Lagrangian acceptance cases use straight-line Lyα propagation. Both thermal backends read RECFAST, stellar spectra, Lyα heating and secondary-electron tables from the configured 4.3.2 external table directory. All 18,332 secondary table values matched the native floats under both the default and modified inputs. [External thermal table comparisons](../reports/v432-external-thermal-tables.json) Extended large-box and multithreaded checks cover 8-thread `192³ / 64³` `E-INTEGRAL`, `192³ / 64³` BBKS + Gaussian `FILTER`, `192³ / 64³` molecular cooling + CLASS `FLUCTS` + `LYA_MULTIPLE_SCATTERING`, `32³ / 16³` 1/2/4/8-thread `CHMF-SAMPLER` (`hybrid` and `prefix`), and `96³ / 32³` thermal `CHMF-SAMPLER` (`1T` passing, `2T`/`8T` matching catalogs/globals/PS with 1–2 OpenMP float32 ionization-boundary voxels). For batch workflows, [lightcone_io.py](lightcone_io.py) writes aligned `device-lzf` / `parallel-lzf` HDF5 files and [run_batch.py](run_batch.py) schedules multi-worker concurrent GPU runs. [Multithread verification driver](verify_multithread_rng.py) · [CHMF multithread driver](verify_chmf_multithread.py) · [Large-grid new-physics driver](verify_large_new_physics.py) · [Multithread PRNG checks](../reports/v432-multithread-rng.json) · [CHMF multithread checks](../reports/v432-chmf-multithread-and-speed.json) · [Gaussian matter window checks](../reports/v432-gaussian-matter-window.json) · [Large-grid new-physics checks](../reports/v432-large-new-physics.json) · [LZF and batch checks](../reports/v432-lzf-and-batch.json) The public entry point validates the remaining option combinations. Run simultaneous simulations in separate processes or via `run_batch.py`. [Public option checks](gpu_lightcone.py)

### Reuse a native reference

For the default 48³ CHMF case, the first command creates the reference HDF5 and per-epoch catalog hashes. Subsequent changes to the GPU implementation need only the second command. It compares saved inputs, coordinates, all fields and global quantities, catalog hashes and transverse brightness spectra. Each distinct parameter set needs its own reference directory. Mode `compare` rereads the saved files and catalog digests without evolving either box.

```sh
python v4/verify_chmf_lightcone.py cpu runs/chmf
python v4/verify_chmf_lightcone.py gpu runs/chmf
python v4/verify_chmf_lightcone.py compare runs/chmf
```

For the two-node CHMF thermal case, use the same arguments for the initial CPU baseline and every GPU comparison.

```sh
run_thermal() {
    python v4/verify_chmf_lightcone.py "$1" runs/chmf-thermal --thermal \
        --dim 32 --hii-dim 16 --box-mpc 24 --threads 1 --seed 81033 \
        --nodes 10 9.7 --bounds 9.73 9.92
}
run_thermal cpu
run_thermal gpu
```

For a constant-efficiency comparison, the following commands use the same native reference through all GPU revisions. Add `--thermal` to each command and use its own directory for the case with spin fluctuations and inhomogeneous recombinations. Mode `compare` rereads the saved HDF5 outputs and per-node cache without running the simulation.

```sh
python v4/verify_constant_lightcone.py cpu runs/constant
python v4/verify_constant_lightcone.py gpu runs/constant
python v4/verify_constant_lightcone.py compare runs/constant
```

Native shims are reused from `$XDG_CACHE_HOME/21cmfast-gpu/native-shims` (default `~/.cache/21cmfast-gpu/native-shims`). Set `P21_GPU_SHIM_CACHE` to select another directory. Generated C, expanded headers, compiler options, toolchain and the upstream native library participate in the cache key. Each context loads a private copy for its callbacks. [Cache validation](../reports/v432-native-shim-cache.json)

### Compare against the native CPU

The benchmark additionally imports upstream integration-test parameters. Put the upstream 4.3.2 checkout on `PYTHONPATH` and run each mode in its own process.

```sh
python v4/lightcone_ps.py runs/v432-384 384 cpu point_config.json
python v4/lightcone_ps.py runs/v432-384 384 gpu point_config.json
python v4/lightcone_pair.py runs/v432-384 384 compare
python v4/compare_lightcone_ps.py runs/v432-384 runs/v432-384 runs/v432-384 runs/v432-384/acceptance.json
```

The CPU benchmark command reuses a complete native reference when its inputs, geometry, fields and power-spectrum bins match. Two saved 192³ native references passed this check in 0.124 s and 0.023 s after imports. The check inspected the HDF5 metadata, coordinates and the 32-bin spectra, with zero physics calls. Use `--rerun-cpu` for an intentional new CPU timing. [Reference reuse checks](../reports/v432-cpu-reference-reuse.json) Use 192 in place of 384 and a separate output directory for the smaller case. The benchmark verifies the IC digest, redshift nodes, lightcone distances, all six fields and 32 brightness power-spectrum bins per node. The comparison uses `1e-8 + 1e-4 * abs(CPU)`. Timing in these scripts ends after the lightcone HDF5 is closed and synchronized. The PS estimator runs during evolution and its small JSON summary is saved afterwards.

For a full trajectory audit, use `verify_evolution.py` with the same arguments as `lightcone_ps.py`. Run CPU first, then GPU in the same directory. The CPU run saves all allocated fields at four driver boundaries. The GPU run compares them and records an additional 12 thermal terms against the native single-step update. The adapter requires the 4.3.2 C sources and an unstripped Linux extension, a C compiler with OpenMP, and the GSL and FFTW headers. Set `CPPFLAGS` if headers are outside the compiler search path.

See the [GPU port measurements](../reports/v432-gpu-port.json), the [earlier hybrid evolution alignment results](../reports/v432-evolution-alignment.json) and the [earlier perturbation and brightness benchmark](../reports/v432-ordered-brightness.json).

### Single-node reuse checks

With the same upstream test checkout on `PYTHONPATH`, run
`python v4/verify_recomb_reuse.py recomb.json` for the inhomogeneous recombination update.
Run `python v4/verify_ion_reuse.py ion.json` for the center-method ionization
barrier and the final thermal update. The latter compiles a temporary copy of
the installed v4.3.2 C sources to capture the official inputs and outputs.
Set `CPPFLAGS` to the GSL and FFTW include paths if installed outside the
compiler search path.

The ionization comparison starts at each radius with the native incoming
neutral fraction and the normalized ACG and MCG photon counts. The thermal
comparison starts with the native electron and heating rates and spin coupling
coefficients. These checks run independently of the lightcone driver's backend
selection.

### v3 reuse in the Eulerian atomic-cooling driver

| 4.3.2 caller | Shared implementation and input contract |
| --- | --- |
| Stellar and escape-fraction scaling | In both versions the host prepares the scalar inputs to the CUDA mass integral. 4.3.2 `set_scaling_constants` and `evolve_scaling_constants_to_redshift` retain dimensionless efficiencies, halo-mass limits in solar masses and a star-formation timescale in seconds. For constant efficiency the SFR derivative uses `dt/dz`. The adapter evaluates mass variances on CUDA. [v3 host-to-kernel arguments](../lightcone/hmf_patch.py) · [4.3.2 source parameters](source_integrals.h) |
| Source and ionization mass variance | The v3 sigma hook integrates at a top-hat radius in comoving Mpc. 4.3.2 passes its native `MtoR` result to the existing GPU moment adapter. That adapter retains the 4.3.2 power normalization and window and uses the v3 adaptive quadrature. Mass cutoffs, stellar fractions and escape fractions enter from the native scaling setup. |
| Global source normalization | The v3 HMF kernel accepts growth, turnover mass and stellar/escape fractions prepared on the host. The 4.3.2 emissivity and ionization callers share the existing GPU global mass integral with the 4.3.2 logarithmic mass weights. For SFRD the escape exponent is zero and the prefactor includes the comoving baryon mass density divided by the star-formation timescale in seconds. |
| Ionization and gas temperature | The v3 assignment kernel takes collapse fraction, efficiency and a precomputed photoionization prefactor. The 4.3.2 Eulerian center method takes normalized photons per baryon, comoving radius in Mpc and its own `gamma_prefactor` to form an ionization rate in units of 10⁻¹² s⁻¹. The v3 host already folds radius and efficiency into the factor it supplies to the kernel. It calls the shared barrier and recombination code. The fully ionized temperature branch calls v3 `ion_temperature` directly with density contrast, neutral fraction, reionization redshift and temperatures in K. |
| Rectilinear lightcone | Both versions obtain comoving distances on the host. The v4 adapter uses pixel distances for weights, the 4.3.2 near-edge allowance of `1e-6`, an open far edge and the Astropy-weighted arithmetic type. It sends the selected LOS indices and weights to the same CUDA interpolation kernel as v3, including for LOS velocity. Requesting both rectangular LOS velocity and velocity_z gathers the field once per node pair. The two returned arrays are independently writable. [Regression comparison](../reports/v432-angular-lightcone.json) · [Shared interpolation](../lightcone/device_slices.py) · [4.3.2 slice selection](lightcone_device.py) |
| X-ray/heating shell geometry | Native 4.3.2 `RadiationFieldsSetup.setup_shells` sets shell radii in comoving Mpc and inverts Astropy distance using a 100-point interpolation grid. The v3 host advances each shell using the local distance derivative. Both GPU X-ray integrators consume prepared shell redshifts. The adapter uses the 4.3.2 values. [v3 X-ray call](../lightcone/xray_patch.py) · [4.3.2 radiation hooks](evolution_backend.py) |

Run `python v4/verify_scalar_slice_reuse.py single-step.json` for the same-input source, temperature and slice checks. For a joined lightcone check, place the existing matching `cpu.h5`, `cpu.json` and `cpu-ps.json` in the run directory, run GPU once and use the comparison commands above.

The [reuse acceptance report](../reports/v432-scalar-slice-reuse.json) records both sizes. All six fields reproduce the preceding GPU outputs value for value. Relative to native 4.3.2, the brightness field exceeds the voxel tolerance at 62 sites on 192 and 192 sites on 384. The other five fields and all 2,752 power-spectrum values pass. The final optical-depth velocity factors agree at every failing site when replayed from the saved rays.

The [subsequent 192³ run](../reports/v432-small192-thermal-fix.json) passed after two corrections to the X-ray heating history. The optical-depth cutoff needs the native float addition in the cosmological time derivative. The electron-fraction interpolation needs the inverse interval widths as rounded by the native OpenMP loop. The adapter prepares these 13 coefficients once per thread setting and uses the cached widths on the GPU. At 384³, two remaining brightness differences traced to the continuum Lyα flux. The native build rounds the reciprocal used to weight the partially illuminated outer shell to float32 before multiplication. Matching that operation gave the paired results above.

Given a native CPU lightcone and its saved evolution audit, the radiation table check replays one node using its stored electron fractions and star-formation data. It checks the interval widths, cell indices and interpolation weights for exact agreement. It compares the X-ray frequency tables and the stellar Lyα prefactors to the native result at relative tolerance `1e-12`.

```sh
python v4/verify_radiation_setup.py timing-192/cpu.h5 audit-192/reference-fields.h5 lookup-check
```
