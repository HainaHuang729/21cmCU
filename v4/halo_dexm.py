"""Native-order CUDA excursion-set halo search and fine-to-coarse overlap for 4.3.2."""

import ctypes as C
from pathlib import Path

import cupy as cp
import numpy as np

from evolution_backend import CB, view
from fft_plan_cuda import transform
from halo_ionization import halo_window


# The measured native loops round the density product and then use scalar
# single-precision division, also when resampling the occupancy grid.
normalize = cp.ElementwiseKernel(
    "float32 field, float32 growth, float32 volume",
    "float32 result",
    "result = __fdiv_rn(__fmul_rn(field, growth), volume);",
    "dexm_normalize",
)


def _alternative(source, begin, end, cuda, offset=0):
    start = source.index(begin, offset)
    stop = source.index(end, start)
    return (
        source[:start]
        + "if (gpu_dexm) {\n"
        + cuda
        + "\n} else {\n"
        + source[start:stop]
        + "\n}\n"
        + source[stop:]
    )


def finder_source(src):
    source = (src / "HaloCatalog.c").read_text()
    header = """
int (*dexm_dispatch)(int,int,double*,void**);
#define DEXM_CALL(stage,p,args) do { if (dexm_dispatch(stage,grid_dim,p,args)) Throw(ValueError); } while(0)
"""
    marker = "        // allocate array for the k-space box"
    source = source.replace(
        marker,
        """
        bool gpu_dexm = dexm_dispatch &&
            !matter_options_global->USE_FFTW_WISDOM && simulation_options_global->NON_CUBIC_FACTOR == 1. &&
            matter_options_global->HALO_FILTER >= FILTER_TOPHAT && matter_options_global->HALO_FILTER <= FILTER_GAUSSIAN;
"""
        + marker,
    )
    source = _alternative(
        source,
        "        density_field = (fftwf_complex",
        "        if (matter_options_global->DEXM_OPTIMIZE)",
        """
        density_field = NULL; density_field_saved = NULL; in_halo = NULL;
""",
    )
    source = source.replace(
        "forbidden = (char *)calloc(",
        "forbidden = gpu_dexm ? NULL : (char *)calloc(",
        1,
    )
    # Recover the exclusion radius once per distinct previous mass. Upstream
    # round-trips the catalogue's float mass through MtoR before adding overlap.
    marker = "int ComputeHaloCatalog("
    source = source.replace(
        marker,
        """
void dexm_rebuilt_radius(float mass, float trial, int *range, float *sq) {
    float prev = MtoR(mass);
    float r = prev + simulation_options_global->DEXM_R_OVERLAP * trial;
    float pixels = r / simulation_options_global->BOX_LEN * simulation_options_global->DIM;
    *range = ceil(pixels);
    *sq = pow(pixels, 2);
}
"""
        + marker,
        1,
    )
    source = _alternative(
        source,
        "#pragma omp parallel shared(boxes, density_field)",
        "        // ***************** END INITIALIZATION",
        """
        double p[]={simulation_options_global->N_THREADS};
        void *a[]={boxes->hires_density};
        DEXM_CALL(0,p,a);
""",
    )
    source = source.replace(
        "float *halo_catalog = calloc(",
        "float *halo_catalog = gpu_dexm ? NULL : calloc(",
    )
    source = _alternative(
        source,
        "            memcpy(density_field, density_field_saved,",
        "            total_halo_num += r_halo_num;",
        """
        float check_r = R * simulation_options_global->DEXM_R_OVERLAP;
        float paint_r = R / simulation_options_global->BOX_LEN * grid_dim;
        check_r = check_r / simulation_options_global->BOX_LEN * grid_dim;
        bool optimized = matter_options_global->DEXM_OPTIMIZE &&
                         M > simulation_options_global->DEXM_OPTIMIZE_MINMASS;
        float exclude_r = (1. + simulation_options_global->DEXM_R_OVERLAP) * R;
        exclude_r = exclude_r / simulation_options_global->BOX_LEN * grid_dim;
        double p[]={R, matter_options_global->HALO_FILTER, 2.*M_PI/simulation_options_global->BOX_LEN,
                    growth_factor, delta_crit, ceil(check_r), (float)pow(check_r,2),
                    ceil(paint_r), (float)pow(paint_r,2), M, optimized, ceil(exclude_r), (float)pow(exclude_r,2)};
        void *a[]={&r_halo_num};
        DEXM_CALL(1,p,a);
""",
    )
    source = _alternative(
        source,
        "        for (x = 0;",
        "        add_properties_cat(random_seed, redshift, halos_dexm);",
        """
        double p[]={cell_length,(double)total_halo_num};
        void *a[]={halos_dexm->halo_masses,halos_dexm->halo_coords};
        DEXM_CALL(2,p,a);
""",
        source.index("        size_huge count = 0;"),
    )
    source = source.replace(
        "        add_properties_cat(random_seed, redshift, halos_dexm);",
        """
        if (gpu_dexm && simulation_options_global->N_THREADS >= 1) {
            int workers = 1;
            if (halos_dexm->n_halos && simulation_options_global->N_THREADS > 1) {
                #pragma omp parallel
                {
                    #pragma omp single
                    workers = omp_get_num_threads();
                }
            }
            double p[] = {simulation_options_global->N_THREADS, workers};
            void *a[] = {&random_seed, &halos_dexm->n_halos, halos_dexm->star_rng,
                         halos_dexm->sfr_rng, halos_dexm->xray_rng};
            DEXM_CALL(4,p,a);
        } else {
            add_properties_cat(random_seed, redshift, halos_dexm);
        }
""",
        1,
    )
    begin = source.index("            // First we construct a grid")
    end = source.index("            stochastic_halofield(", begin)
    marker = "float *halo_overlap_box = calloc(HII_TOT_NUM_PIXELS, sizeof(float));"
    native = source[begin:end].replace(marker, "")
    source = (
        source[:begin]
        + marker
        + """
            if (gpu_dexm) {
                double p[]={simulation_options_global->HII_DIM,
                            (float)(physconst.l_factor*simulation_options_global->BOX_LEN/(simulation_options_global->HII_DIM+0.)),
                            2.*M_PI/simulation_options_global->BOX_LEN};
                void *a[]={halo_overlap_box};
                DEXM_CALL(3,p,a);
            } else {
"""
        + native
        + "\n}\n"
        + source[end:]
    )
    return header + source


class HaloFinder:
    def __init__(self, report, library):
        self.report = report
        module = cp.RawModule(
            code=Path(__file__).with_suffix(".cu").read_text(),
            options=("--fmad=false",),
        )
        self._ordered = module.get_function("ordered_dexm")
        self._optimized = module.get_function("optimized_dexm")
        self._rebuild = module.get_function("rebuild_forbidden")
        self._prior_radius = library.dexm_rebuilt_radius
        self._prior_radius.argtypes = [
            C.c_float,
            C.c_float,
            C.POINTER(C.c_int),
            C.POINTER(C.c_float),
        ]
        self._prior_radius.restype = None
        self.clear()

    def clear(self):
        self.spectrum = self.occupied = self.masses = None
        self.previous_groups = []

    def rebuild_forbidden(self, side, trial):
        mask = cp.zeros((side**3 + 31) // 32, dtype="uint32")
        for mass, centres in self.previous_groups:
            width, squared = C.c_int(), C.c_float()
            self._prior_radius(mass, trial, C.byref(width), C.byref(squared))
            self._rebuild(
                (centres.size,),
                (256,),
                (
                    centres,
                    mask,
                    np.int32(side),
                    np.int32(width.value),
                    np.float32(squared.value),
                ),
            )
            self.record("dexm_rebuilt_centres", centres.size)
        return mask

    def record(self, name, value=1):
        self.report[name] = self.report.get(name, 0) + int(value)

    def select(self, side, p):
        spectrum = self.spectrum.copy()
        self.apply_window(spectrum, side, p[2], p[0], int(p[1]))
        filtered = transform(spectrum, side, True, self.threads)
        normalized = normalize(filtered, np.float32(p[3]), np.float32(side**3))
        candidates = cp.flatnonzero(normalized > np.float32(p[4]))
        count = cp.zeros(1, dtype="uint64")
        optimized = bool(p[10])
        branch = "dexm_optimized" if optimized else "dexm_ordinary"
        self.record(branch + "_radii")
        self.record(branch + "_candidates", candidates.size)
        if not optimized:
            # The native radii decrease monotonically. Only larger, optimized
            # radii use the previous groups; the ordinary search uses occupancy.
            self.previous_groups.clear()
        if candidates.size:
            common = (
                candidates,
                np.int64(candidates.size),
                self.occupied,
                self.masses,
                count,
                np.int32(side),
            )
            paint = (np.int32(p[7]), np.float32(p[8]), np.float32(p[9]))
            if optimized:
                mask = self.rebuild_forbidden(side, p[0])
                new_centres = cp.empty(candidates.size, "int64")
                self._optimized(
                    (1,),
                    (256,),
                    (
                        *common,
                        *paint,
                        np.int32(p[11]),
                        np.float32(p[12]),
                        mask,
                        new_centres,
                    ),
                )
            else:
                self._ordered(
                    (1,), (256,), (*common, np.int32(p[5]), np.float32(p[6]), *paint)
                )
        self.record("dexm_radius_searches")
        self.record("dexm_candidates", candidates.size)
        accepted = int(count.item())
        if optimized and accepted:
            self.previous_groups.append((p[9], new_centres[:accepted].copy()))
        self.record(branch + "_accepted", accepted)
        self.record("dexm_accepted", accepted)
        return accepted

    @staticmethod
    def apply_window(spectrum, side, dk, radius, kind):
        halo_window()(
            ((spectrum.size + 255) // 256,),
            (256,),
            (
                spectrum,
                np.int32(side),
                np.float64(dk),
                np.float32(radius),
                np.float32(0),
                np.float64(0),
                np.int32(kind),
            ),
        )

    def dispatch(self, stage, side, p, a):
        shape = (side,) * 3
        if stage == 0:
            self.clear()
            self.threads = int(p[0])
            density = cp.asarray(view(a[0], side**3, "f").reshape(shape))
            self.spectrum = cp.ascontiguousarray(
                transform(density, side, False, self.threads)
            )
            self.occupied = cp.zeros(shape, dtype="B")
            self.masses = cp.zeros(shape, dtype="f")
            self.record("dexm_density_forward_fft")
        elif stage == 1:
            view(a[0], 1, "uint64")[0] = self.select(side, p)
        elif stage == 2:
            indices = cp.flatnonzero(self.masses)
            self.has_halos = bool(indices.size)
            if indices.size != int(p[1]):
                raise ValueError("DexM catalogue count disagrees with occupied centres")
            if indices.size:
                masses = self.masses.ravel()[indices]
                masses.get(out=view(a[0], indices.size, "f"))
                coords = cp.stack(cp.unravel_index(indices, shape), axis=1)
                coords = cp.ascontiguousarray((coords.astype("d") * p[0]).astype("f"))
                coords.get(out=view(a[1], indices.size * 3, "f").reshape(-1, 3))
            self.spectrum = self.masses = None
            self.previous_groups.clear()
        elif stage == 3:
            coarse = int(p[0])
            if not self.has_halos:
                view(a[0], coarse**3, "f").fill(0.0)
                self.clear()
                self.record("dexm_empty_overlap_grids")
                return
            spectrum = cp.ascontiguousarray(
                transform(self.occupied.astype("f"), side, False, self.threads)
            )
            if coarse != side:
                self.apply_window(spectrum, side, p[2], p[1], 0)
            field = transform(spectrum, side, True, self.threads)
            ratio = np.float32(side) / np.float32(coarse)
            idx = ((cp.arange(coarse, dtype="f") * ratio).astype("d") + 0.5).astype("i")
            field = normalize(
                field[cp.ix_(idx, idx, idx)], np.float32(1.0), np.float32(side**3)
            )
            cp.ascontiguousarray(cp.clip(field, 0.0, 1.0)).get(
                out=view(a[0], coarse**3, "f").reshape((coarse,) * 3)
            )
            self.clear()
            self.record("dexm_overlap_grids")
        elif stage == 4:
            from halo_rng import assign_properties

            count = int(view(a[1], 1, "uint64")[0])
            if count:
                seed = int(view(a[0], 1, "uint64")[0])
                targets = [view(a[i], count, "f") for i in (2, 3, 4)]
                assign_properties(seed, int(p[0]), int(p[1]), targets)
                self.record("dexm_property_rng_halos", count)
            else:
                self.record("dexm_empty_property_catalogs")
        else:
            raise ValueError(f"Unknown DexM stage {stage}")


def install_finder(library, report, errors):
    finder = HaloFinder(report, library)

    @CB
    def callback(stage, side, scalars, fields):
        try:
            finder.dispatch(stage, side, scalars, fields)
            return 0
        except BaseException as error:
            errors.append(error)
            return 1

    C.c_void_p.in_dll(library, "dexm_dispatch").value = C.cast(
        callback, C.c_void_p
    ).value
    return finder, callback
