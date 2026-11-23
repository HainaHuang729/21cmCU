"""Build a private library with a hook at the normal perturb_field entry point."""
from pathlib import Path
import platform
import shutil
import sys

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from cffi import FFI
from env_detect import detect_native_deps

root, work = (Path(arg).resolve() for arg in sys.argv[1:])
package = work / "package"
py21 = package / "py21cmfast"
shutil.copytree(root / "resident_filter_20261118/fix/package/py21cmfast", py21)
for extension in py21.glob("c_21cmfast*.so"):
    extension.unlink()
src = py21 / "src"
output_source=py21/"outputs.py"
outputs=output_source.read_text()
before='        # Expose all the fields of the structs to the surface of the Coeval object'
assert outputs.count(before)==1
outputs=outputs.replace(before,
    '        import os\n        lazy_initial_fields = os.environ.get("PS_LAZY_COEVAL_IC") == "1"\n'+before,1)
after='            if box is None:\n                continue\n            for field in box._get_box_structures():'
assert outputs.count(after)==1
outputs=outputs.replace(after,
    '            if box is None or (lazy_initial_fields and box is initial_conditions):\n                continue\n            for field in box._get_box_structures():',1)
before='    @classmethod\n    def get_fields(cls, spin_temp: bool = True)'
assert outputs.count(before)==1
outputs=outputs.replace(before,"""    def __getattr__(self, name):
        initial = self.__dict__.get("init_struct")
        if initial is not None and name in initial.__dict__.get("_array_state", {}):
            return getattr(initial, name)
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

"""+before,1)
compile(outputs,str(output_source),"exec")
output_source.write_text(outputs)
wrapper_path = py21 / "wrapper.py"
wrapper_text = wrapper_path.read_text()
assert wrapper_text.count("import os\n") == 1
wrapper_text = wrapper_text.replace("import os\n", "import os\nimport sys\n", 1)
wrap_old = "            if flag_options.USE_MINI_HALOS:\n                prev_perturb = pf2"
wrap_new = (
    '            if flag_options.USE_MINI_HALOS or (os.environ.get("PS_REUSE_PREV_PERTURB") == "1" and not user_params.MINIMIZE_MEMORY):\n'
    "                prev_perturb = pf2"
)
assert wrapper_text.count(wrap_old) == 1
wrapper_text = wrapper_text.replace(wrap_old, wrap_new, 1)
lc_loop_old = """            if prev_coeval is not None:
                for quantity, idx, this_lc in lightconer.make_lightcone_slices(
                    coeval, prev_coeval
                ):
                    if this_lc is not None:
                        lightcone.lightcones[quantity][..., idx] = this_lc
                        lc_index = idx"""
lc_loop_new = """            if prev_coeval is not None:
                if (
                    os.environ.get("PS_FAST_LIGHTCONE", "1") == "1"
                    and hasattr(lightconer, "fill_lightcone_slices")
                    and not getattr(lightconer, "get_los_velocity", False)
                ):
                    _ps_last_idx = lightconer.fill_lightcone_slices(
                        coeval, prev_coeval, lightcone.lightcones
                    )
                    if _ps_last_idx is not None:
                        lc_index = _ps_last_idx
                else:
                    for quantity, idx, this_lc in lightconer.make_lightcone_slices(
                        coeval, prev_coeval
                    ):
                        if this_lc is not None:
                            lightcone.lightcones[quantity][..., idx] = this_lc
                            lc_index = idx"""
assert wrapper_text.count(lc_loop_old) == 1
wrapper_text = wrapper_text.replace(lc_loop_old, lc_loop_new, 1)
loop_start = "        for iz, z in enumerate(scrollz):\n"
loop_end = "        if flag_options.PHOTON_CONS:\n"
assert wrapper_text.count(loop_start) == 1
ls_idx = wrapper_text.index(loop_start)
assert wrapper_text[ls_idx:].count(loop_end) == 1
le_idx = wrapper_text.index(loop_end, ls_idx)
loop_block = wrapper_text[ls_idx:le_idx]
indented_block = "".join(("    " + ln if ln.strip() else ln) for ln in loop_block.splitlines(True))
wrapped_loop = (
    '        _ps_py21 = sys.modules.get("py21cmfast")\n'
    '        _ps_sched = getattr(_ps_py21, "_ps_set_lightcone_schedule", None)\n'
    "        if _ps_sched is not None:\n"
    "            _ps_sched(scrollz, perturb, user_params)\n"
    "        try:\n"
    + indented_block
    + '        finally:\n'
    + '            _ps_clear = getattr(_ps_py21, "_ps_clear_lightcone_schedule", None)\n'
    + "            if _ps_clear is not None:\n"
    + "                _ps_clear()\n"
)
wrapper_text = wrapper_text[:ls_idx] + wrapped_loop + wrapper_text[le_idx:]
compile(wrapper_text, str(wrapper_path), "exec")
wrapper_path.write_text(wrapper_text)

lc_path = py21 / "lightcones.py"
lc_text = lc_path.read_text()
lc_imp_old = "import attr\n"
lc_imp_new = "import attr\nimport os\n"
assert lc_text.count(lc_imp_old) == 1
lc_text = lc_text.replace(lc_imp_old, lc_imp_new, 1)
lc_sub_old = '''    def coeval_subselect(
        self, lcd: Quantity[pixel], coeval: np.ndarray, coeval_res: Quantity[_LENGTH]
    ):
        """Sub-select the coeval slice corresponding to this coeval distance."""
        # This makes the back of the lightcone exactly line up with the back of the
        # coeval box at that redshift, modulo the index_offset.
        lcpix = self.get_lc_distances_in_pixels(coeval_res)
        lcidx = int((lcpix.max() - lcd + 1 * pixel).to_value(pixel))
        return coeval.take(-lcidx + self.index_offset, axis=2, mode="wrap")'''
lc_sub_new = '''    def coeval_subselect(
        self, lcd: Quantity[pixel], coeval: np.ndarray, coeval_res: Quantity[_LENGTH]
    ):
        """Sub-select the coeval slice corresponding to this coeval distance."""
        # This makes the back of the lightcone exactly line up with the back of the
        # coeval box at that redshift, modulo the index_offset.
        if os.environ.get("PS_FAST_LIGHTCONE", "1") == "1":
            cache = self.__dict__.get("_lcpix_max_cache")
            if cache is None:
                cache = {}
                self._lcpix_max_cache = cache
            res_key = float(coeval_res.to_value(Mpc))
            lcpix_max_val = cache.get(res_key)
            if lcpix_max_val is None:
                lcpix = self.get_lc_distances_in_pixels(coeval_res)
                lcpix_max_val = float(lcpix.max().to_value(pixel))
                cache[res_key] = lcpix_max_val
            lc_val = float(lcd.to_value(pixel)) if hasattr(lcd, "to_value") else float(lcd)
            lcidx = int(lcpix_max_val - lc_val + 1.0)
            return coeval[:, :, (-lcidx + self.index_offset) % coeval.shape[2]]
        lcpix = self.get_lc_distances_in_pixels(coeval_res)
        lcidx = int((lcpix.max() - lcd + 1 * pixel).to_value(pixel))
        return coeval.take(-lcidx + self.index_offset, axis=2, mode="wrap")'''
assert lc_text.count(lc_sub_old) == 1
lc_text = lc_text.replace(lc_sub_old, lc_sub_new, 1)
lc_make_old = '''        lc_distances = pixlcdist[lcidx]

        for idx, lcd in zip(lcidx, lc_distances):'''
lc_make_new = '''        lc_distances = pixlcdist[lcidx]

        if (
            os.environ.get("PS_FAST_LIGHTCONE", "1") == "1"
            and isinstance(self, RectilinearLightconer)
            and not self.get_los_velocity
        ):
            dc1_v = float(dc1.to_value(pixel))
            dc2_v = float(dc2.to_value(pixel))
            denom = abs(dc1_v - dc2_v)
            lcpix_max_v = float(pixlcdist.max().to_value(pixel))
            lc_vals = np.asarray(lc_distances.to_value(pixel), dtype=np.float64)
            q_info = [
                (q, getattr(c1, q), getattr(c2, q), self.interp_kinds.get(q, "mean"))
                for q in self.quantities
            ]
            for idx, lc_val in zip(lcidx, lc_vals):
                lc_f = float(lc_val)
                lcidx_v = int(lcpix_max_v - lc_f + 1.0)
                w1 = abs(dc2_v - lc_f)
                w2 = abs(dc1_v - lc_f)
                for q, c1_arr, c2_arr, kind in q_info:
                    k1 = (-lcidx_v + self.index_offset) % c1_arr.shape[2]
                    k2 = (-lcidx_v + self.index_offset) % c2_arr.shape[2]
                    box1 = c1_arr[:, :, k1]
                    box2 = c2_arr[:, :, k2]
                    box = (w1 * box1 + w2 * box2) / denom
                    if kind == "mean_max":
                        flag = box1 * box2 < 0
                        box[flag] = np.maximum(box1, box2)[flag]
                    yield q, idx, box
            return

        for idx, lcd in zip(lcidx, lc_distances):'''
assert lc_text.count(lc_make_old) == 1
lc_text = lc_text.replace(lc_make_old, lc_make_new, 1)

lc_ri_old = '''    def redshift_interpolation(
        self,
        dc: float,
        coeval_a: np.ndarray,
        coeval_b: np.ndarray,
        dc_a: float,
        dc_b: float,
        kind: str = "mean",
    ) -> np.ndarray:
        """Perform redshift interpolation to a new box given two bracketing coevals."""
        if coeval_a.shape != coeval_b.shape:
            raise ValueError("coeval_a and coeval_b must have the same shape")

        out = (np.abs(dc_b - dc) * coeval_a + np.abs(dc_a - dc) * coeval_b) / np.abs(
            dc_a - dc_b
        )'''
lc_ri_new = '''    def redshift_interpolation(
        self,
        dc: float,
        coeval_a: np.ndarray,
        coeval_b: np.ndarray,
        dc_a: float,
        dc_b: float,
        kind: str = "mean",
    ) -> np.ndarray:
        """Perform redshift interpolation to a new box given two bracketing coevals."""
        if coeval_a.shape != coeval_b.shape:
            raise ValueError("coeval_a and coeval_b must have the same shape")

        if os.environ.get("PS_FAST_LIGHTCONE", "1") == "1":
            dc_v = float(dc.to_value(pixel)) if hasattr(dc, "to_value") else float(dc)
            dca_v = float(dc_a.to_value(pixel)) if hasattr(dc_a, "to_value") else float(dc_a)
            dcb_v = float(dc_b.to_value(pixel)) if hasattr(dc_b, "to_value") else float(dc_b)
            out = (abs(dcb_v - dc_v) * coeval_a + abs(dca_v - dc_v) * coeval_b) / abs(
                dca_v - dcb_v
            )
        else:
            out = (np.abs(dc_b - dc) * coeval_a + np.abs(dc_a - dc) * coeval_b) / np.abs(
                dc_a - dc_b
            )'''
assert lc_text.count(lc_ri_old) == 1
lc_text = lc_text.replace(lc_ri_old, lc_ri_new, 1)
lc_rect_old = """    def construct_lightcone(
        self,
        lcd: np.ndarray,
        box: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:"""

lc_rect_new = """    def fill_lightcone_slices(self, c1: Coeval, c2: Coeval, lc_dict: dict):
        if c1.user_params != c2.user_params:
            raise ValueError("c1 and c2 must have the same user parameters")
        if c1.cosmo_params != c2.cosmo_params:
            raise ValueError("c1 and c2 must have the same cosmological parameters")
        if self.get_los_velocity or self.line_of_sight_axis not in (-1, 2):
            last_idx = None
            for quantity, idx, this_lc in self.make_lightcone_slices(c1, c2):
                if this_lc is not None:
                    lc_dict[quantity][..., idx] = this_lc
                    last_idx = idx
            return last_idx

        cosmo = c1.cosmo_params.cosmo
        pixeleq = pixel_scale(c1.user_params.cell_size / pixel)
        dc1 = cosmo.comoving_distance(c1.redshift).to(pixel, equivalencies=pixeleq)
        dc2 = cosmo.comoving_distance(c2.redshift).to(pixel, equivalencies=pixeleq)
        dcmin = min(dc1, dc2)
        dcmax = max(dc1, dc2)
        pixlcdist = self.get_lc_distances_in_pixels(c1.user_params.cell_size)
        lcidx = np.nonzero((pixlcdist >= dcmin * 0.9999) & (pixlcdist < dcmax))[0]
        if len(lcidx) == 0:
            return None

        dc1_v = float(dc1.to_value(pixel))
        dc2_v = float(dc2.to_value(pixel))
        denom = abs(dc1_v - dc2_v)
        cache = self.__dict__.get("_lcpix_max_cache")
        if cache is None:
            cache = {}
            self._lcpix_max_cache = cache
        res_key = float(c1.user_params.cell_size.to_value(Mpc))
        lcpix_max_v = cache.get(res_key)
        if lcpix_max_v is None:
            lcpix_max_v = float(pixlcdist.max().to_value(pixel))
            cache[res_key] = lcpix_max_v

        lc_vals = np.asarray(pixlcdist[lcidx].to_value(pixel), dtype=np.float64)
        first_c1 = getattr(c1, self.quantities[0])
        side = first_c1.shape[2]
        n_slices = int(len(lcidx))
        lc_indices = np.ascontiguousarray(lcidx, dtype=np.int32)
        k_indices = np.empty(n_slices, dtype=np.int32)
        w1_arr = np.empty(n_slices, dtype=np.float64)
        w2_arr = np.empty(n_slices, dtype=np.float64)
        for i, lc_f in enumerate(lc_vals):
            lcidx_v = int(lcpix_max_v - float(lc_f) + 1.0)
            k_indices[i] = (-lcidx_v + self.index_offset) % side
            w1_arr[i] = abs(dc2_v - float(lc_f))
            w2_arr[i] = abs(dc1_v - float(lc_f))

        c_arrays = []
        for q in self.quantities:
            a1 = getattr(c1, q)
            a2 = getattr(c2, q)
            lco = lc_dict[q]
            if not (
                isinstance(a1, np.ndarray) and isinstance(a2, np.ndarray) and isinstance(lco, np.ndarray)
                and a1.dtype == np.float32 and a2.dtype == np.float32 and lco.dtype == np.float32
                and a1.flags.c_contiguous and a2.flags.c_contiguous and lco.flags.c_contiguous
                and a1.shape == (side, side, side) and a2.shape == (side, side, side)
                and lco.ndim == 3 and lco.shape[0] == side and lco.shape[1] == side
            ):
                last_idx = None
                for quantity, idx, this_lc in self.make_lightcone_slices(c1, c2):
                    if this_lc is not None:
                        lc_dict[quantity][..., idx] = this_lc
                        last_idx = idx
                return last_idx
            c_arrays.append((a1, a2, lco, 1 if self.interp_kinds.get(q, "mean") == "mean_max" else 0))

        import sys
        device_fill = getattr(sys.modules["py21cmfast"], "_ps_device_fill", None)
        if device_fill is not None:
            device_fill(self.quantities, c_arrays, lc_indices, k_indices, w1_arr, w2_arr, denom)
            return int(lcidx[-1])

        from .c_21cmfast import ffi, lib
        nt = max(1, min(8, int(c1.user_params.N_THREADS)))
        lc_ptr = ffi.cast("const int *", lc_indices.ctypes.data)
        k_ptr = ffi.cast("const int *", k_indices.ctypes.data)
        w1_ptr = ffi.cast("const double *", w1_arr.ctypes.data)
        w2_ptr = ffi.cast("const double *", w2_arr.ctypes.data)
        for a1, a2, lco, is_mean_max in c_arrays:
            lib.PSFillLightconeSlices(
                ffi.cast("const float *", a1.ctypes.data),
                ffi.cast("const float *", a2.ctypes.data),
                ffi.cast("float *", lco.ctypes.data),
                side,
                int(lco.shape[2]),
                n_slices,
                lc_ptr,
                k_ptr,
                w1_ptr,
                w2_ptr,
                denom,
                is_mean_max,
                nt,
            )
        return int(lcidx[-1])

""" + lc_rect_old
assert lc_text.count(lc_rect_old) == 1
lc_text = lc_text.replace(lc_rect_old, lc_rect_new, 1)

compile(lc_text, str(lc_path), "exec")
lc_path.write_text(lc_text)

path = src / "PerturbField.c"
source = path.read_text()

def splice(before, after):
    global source
    assert source.count(before) == 1, before
    source = source.replace(before, after, 1)

splice("int ComputePerturbField(", """#include <stdint.h>
static int ps_clamp_threads(int threads) {
    return threads < 1 ? 1 : (threads > 8 ? 8 : threads);
}

void PSParallelMemcpy(void *dst, const void *src, unsigned long long nbytes, int threads) {
    if (nbytes == 0 || dst == src) return;
    if (nbytes < (1ULL << 20) || threads <= 1) {
        memcpy(dst, src, (size_t)nbytes);
        return;
    }
    const int nt = ps_clamp_threads(threads);
    char *d = (char *)dst;
    const char *s = (const char *)src;
    #pragma omp parallel num_threads(nt)
    {
        int tid = omp_get_thread_num();
        int nthr = omp_get_num_threads();
        unsigned long long start = (nbytes * (unsigned long long)tid) / (unsigned long long)nthr;
        unsigned long long end = (nbytes * (unsigned long long)(tid + 1)) / (unsigned long long)nthr;
        if (end > start) memcpy(d + start, s + start, (size_t)(end - start));
    }
}

int PSArrayAllFinite(const float *arr, unsigned long long n, int threads) {
    const int nt = ps_clamp_threads(threads);
    int ok = 1;
    #pragma omp parallel for reduction(&:ok) num_threads(nt)
    for (unsigned long long i = 0; i < n; ++i) {
        if (!isfinite(arr[i])) ok = 0;
    }
    return ok;
}

void PSFillLightconeSlices(
    const float *c1, const float *c2, float *lc_out,
    int side, int depth, int n_slices,
    const int *lc_indices, const int *k_indices,
    const double *w1, const double *w2, double denom,
    int is_mean_max, int threads
) {
    if (n_slices <= 0) return;
    const int nt = ps_clamp_threads(threads);
    const float denom_f = (float)denom;
    #pragma omp parallel for collapse(2) num_threads(nt)
    for (int ix = 0; ix < side; ++ix) {
        for (int iy = 0; iy < side; ++iy) {
            const float *p1 = c1 + ((size_t)ix * (size_t)side + (size_t)iy) * (size_t)side;
            const float *p2 = c2 + ((size_t)ix * (size_t)side + (size_t)iy) * (size_t)side;
            float *po = lc_out + ((size_t)ix * (size_t)side + (size_t)iy) * (size_t)depth;
            for (int s = 0; s < n_slices; ++s) {
                const int k = k_indices[s];
                const float a_f = p1[k];
                const float b_f = p2[k];
                const float w1_f = (float)w1[s];
                const float w2_f = (float)w2[s];
                const float t1 = w1_f * a_f;
                const float t2 = w2_f * b_f;
                float out_f = (t1 + t2) / denom_f;
                if (is_mean_max && (a_f * b_f) < 0.0f) {
                    out_f = (a_f > b_f) ? a_f : b_f;
                }
                po[lc_indices[s]] = out_f;
            }
        }
    }
}

/* Shared FFTW plan lifecycle across C phases when SetPSKeepFFTWPlans(1) is active. */
static int ps_keep_fftw_plans = 0;
static void (*ps_destroy_ion_fftw_plans_cb)(void) = NULL;
static void (*ps_destroy_heat_fftw_plans_cb)(void) = NULL;
static fftwf_complex *ps_perturb_fftw_box = NULL;
static fftwf_complex *ps_perturb_fftw_saved = NULL;
static fftwf_plan ps_perturb_plan_r2c = NULL;
static fftwf_plan ps_perturb_plan_c2r = NULL;
static int ps_perturb_plan_side = 0;
static int ps_perturb_plan_para = 0;
static int ps_perturb_plan_threads = 0;

static unsigned ps_fftw_plan_flags(void) {
    const char *env = getenv("PS_FFTW_MEASURE");
    return (env && env[0] == '1' && env[1] == '\0') ? FFTW_MEASURE : FFTW_ESTIMATE;
}

static unsigned ps_fftw_heat_plan_flags(void) {
    const char *env = getenv("PS_FFTW_MEASURE_HEAT");
    if (env && env[0] == '1' && env[1] == '\0') return FFTW_MEASURE;
    return ps_fftw_plan_flags();
}

static void ps_destroy_cached_fftw_plans(void) {
    if (ps_perturb_plan_r2c) { fftwf_destroy_plan(ps_perturb_plan_r2c); ps_perturb_plan_r2c = NULL; }
    if (ps_perturb_plan_c2r) { fftwf_destroy_plan(ps_perturb_plan_c2r); ps_perturb_plan_c2r = NULL; }
    if (ps_perturb_fftw_box) { fftwf_free(ps_perturb_fftw_box); ps_perturb_fftw_box = NULL; }
    if (ps_perturb_fftw_saved) { fftwf_free(ps_perturb_fftw_saved); ps_perturb_fftw_saved = NULL; }
    ps_perturb_plan_side = 0;
    ps_perturb_plan_para = 0;
    ps_perturb_plan_threads = 0;
    if (ps_destroy_ion_fftw_plans_cb) ps_destroy_ion_fftw_plans_cb();
    if (ps_destroy_heat_fftw_plans_cb) ps_destroy_heat_fftw_plans_cb();
}

void SetPSKeepFFTWPlans(int keep) {
    if (!keep && ps_keep_fftw_plans) ps_destroy_cached_fftw_plans();
    ps_keep_fftw_plans = keep;
}

static int EnsurePerturbFFTWPlans(struct UserParams *user_params, int need_saved) {
    const int side = user_params->HII_DIM;
    const int para = (int)HII_D_PARA;
    const int nt = user_params->N_THREADS < 1 ? 1 : (user_params->N_THREADS > 8 ? 8 : user_params->N_THREADS);
    if (!ps_perturb_fftw_box || ps_perturb_plan_side != side || ps_perturb_plan_para != para || ps_perturb_plan_threads != nt) {
        if (ps_perturb_plan_r2c) { fftwf_destroy_plan(ps_perturb_plan_r2c); ps_perturb_plan_r2c = NULL; }
        if (ps_perturb_plan_c2r) { fftwf_destroy_plan(ps_perturb_plan_c2r); ps_perturb_plan_c2r = NULL; }
        if (ps_perturb_fftw_box) { fftwf_free(ps_perturb_fftw_box); ps_perturb_fftw_box = NULL; }
        if (ps_perturb_fftw_saved) { fftwf_free(ps_perturb_fftw_saved); ps_perturb_fftw_saved = NULL; }
        ps_perturb_fftw_box = fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
        if (!ps_perturb_fftw_box) return 1;
        fftwf_init_threads();
        fftwf_plan_with_nthreads(nt);
        const unsigned pflags = ps_fftw_plan_flags();
        ps_perturb_plan_r2c = fftwf_plan_dft_r2c_3d(side, side, para, (float *)ps_perturb_fftw_box, ps_perturb_fftw_box, pflags);
        ps_perturb_plan_c2r = fftwf_plan_dft_c2r_3d(side, side, para, ps_perturb_fftw_box, (float *)ps_perturb_fftw_box, pflags);
        fftwf_plan_with_nthreads(1);
        if (!ps_perturb_plan_r2c || !ps_perturb_plan_c2r) {
            ps_destroy_cached_fftw_plans();
            return 1;
        }
        ps_perturb_plan_side = side;
        ps_perturb_plan_para = para;
        ps_perturb_plan_threads = nt;
    }
    if (need_saved && !ps_perturb_fftw_saved) {
        ps_perturb_fftw_saved = fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
        if (!ps_perturb_fftw_saved) return 1;
    }
    return 0;
}

static int UncachedMaterializePerturbDensityAndVel(
    struct UserParams *user_params, float *density, float *velocity_z, float dDdt_over_D, int compute_vel
) {
    const int side = user_params->HII_DIM;
    fftwf_complex *box = fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
    fftwf_complex *saved = compute_vel ? fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS) : NULL;
    if (!box || (compute_vel && !saved)) {
        if (box) fftwf_free(box);
        if (saved) fftwf_free(saved);
        return 1;
    }
    #pragma omp parallel for num_threads(user_params->N_THREADS)
    for (int i=0; i<side; ++i)
        for (int j=0; j<side; ++j)
            for (int k=0; k<HII_D_PARA; ++k)
                ((float *)box)[HII_R_FFT_INDEX(i,j,k)] = density[HII_R_INDEX(i,j,k)];
    int status = dft_r2c_cube(user_params->USE_FFTW_WISDOM,side,HII_D_PARA,user_params->N_THREADS,box);
    if (!status && compute_vel) {
        memcpy(saved, box, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
    }
    if (!status) status = dft_c2r_cube(user_params->USE_FFTW_WISDOM,side,HII_D_PARA,user_params->N_THREADS,box);
    if (!status) {
        #pragma omp parallel for num_threads(user_params->N_THREADS)
        for (int i=0; i<side; ++i)
            for (int j=0; j<side; ++j)
                for (int k=0; k<HII_D_PARA; ++k) {
                    float value = ((float *)box)[HII_R_FFT_INDEX(i,j,k)] / (float)HII_TOT_NUM_PIXELS;
                    density[HII_R_INDEX(i,j,k)] = value < -1.0 ? -1+FRACT_FLOAT_ERR : value;
                }
    }
    if (!status && compute_vel) {
        float k_x, k_y, k_z, k_sq;
        int n_x, n_y, n_z;
        #pragma omp parallel shared(saved,dDdt_over_D,side) private(n_x,n_y,n_z,k_x,k_y,k_z,k_sq) num_threads(user_params->N_THREADS)
        {
            #pragma omp for
            for (n_x=0; n_x<side; n_x++) {
                if (n_x>HII_MIDDLE) k_x = (n_x-side) * DELTA_K;
                else k_x = n_x * DELTA_K;
                for (n_y=0; n_y<side; n_y++) {
                    if (n_y>HII_MIDDLE) k_y = (n_y-side) * DELTA_K;
                    else k_y = n_y * DELTA_K;
                    for (n_z=0; n_z<=(unsigned long long)(user_params->NON_CUBIC_FACTOR*HII_MIDDLE); n_z++) {
                        k_z = n_z * DELTA_K_PARA;
                        k_sq = k_x*k_x + k_y*k_y + k_z*k_z;
                        if ((n_x==0) && (n_y==0) && (n_z==0)) {
                            saved[0] = 0;
                        } else {
                            saved[HII_C_INDEX(n_x,n_y,n_z)] *= dDdt_over_D*k_z*I/k_sq/(HII_TOT_NUM_PIXELS+0.0);
                        }
                    }
                }
            }
        }
        status = dft_c2r_cube(user_params->USE_FFTW_WISDOM,side,HII_D_PARA,user_params->N_THREADS,saved);
        if (!status) {
            #pragma omp parallel for num_threads(user_params->N_THREADS)
            for (int i=0; i<side; ++i)
                for (int j=0; j<side; ++j)
                    for (int k=0; k<HII_D_PARA; ++k)
                        velocity_z[HII_R_INDEX(i,j,k)] = ((float *)saved)[HII_R_FFT_INDEX(i,j,k)];
        }
    }
    fftwf_free(box);
    if (saved) fftwf_free(saved);
    if (!ps_keep_fftw_plans) {
        ps_destroy_cached_fftw_plans();
        fftwf_cleanup_threads();
        fftwf_cleanup();
        fftwf_forget_wisdom();
    }
    return status;
}

/* Keep the CPU consumers on the original FFTW rounding path using cached plans. */
static int MaterializePerturbDensityAndVel(
    struct UserParams *user_params, float *density, float *velocity_z, float dDdt_over_D, int compute_vel
) {
    if (user_params->USE_FFTW_WISDOM) {
        return UncachedMaterializePerturbDensityAndVel(user_params, density, velocity_z, dDdt_over_D, compute_vel);
    }
    if (EnsurePerturbFFTWPlans(user_params, compute_vel)) return 1;
    const int side = user_params->HII_DIM;
    const int para = (int)HII_D_PARA;
    float *fbox = (float *)ps_perturb_fftw_box;
    const size_t row_bytes = (size_t)para * sizeof(float);
    #pragma omp parallel for collapse(2) num_threads(user_params->N_THREADS)
    for (int i=0; i<side; ++i)
        for (int j=0; j<side; ++j)
            memcpy(&fbox[HII_R_FFT_INDEX(i,j,0)], &density[HII_R_INDEX(i,j,0)], row_bytes);
    fftwf_execute(ps_perturb_plan_r2c);
    if (compute_vel) {
        float k_x, k_y, k_z, k_sq;
        int n_x, n_y, n_z;
        #pragma omp parallel shared(ps_perturb_fftw_saved,ps_perturb_fftw_box,dDdt_over_D,side) private(n_x,n_y,n_z,k_x,k_y,k_z,k_sq) num_threads(user_params->N_THREADS)
        {
            #pragma omp for
            for (n_x=0; n_x<side; n_x++) {
                if (n_x>HII_MIDDLE) k_x = (n_x-side) * DELTA_K;
                else k_x = n_x * DELTA_K;
                for (n_y=0; n_y<side; n_y++) {
                    if (n_y>HII_MIDDLE) k_y = (n_y-side) * DELTA_K;
                    else k_y = n_y * DELTA_K;
                    for (n_z=0; n_z<=(unsigned long long)(user_params->NON_CUBIC_FACTOR*HII_MIDDLE); n_z++) {
                        k_z = n_z * DELTA_K_PARA;
                        k_sq = k_x*k_x + k_y*k_y + k_z*k_z;
                        if ((n_x==0) && (n_y==0) && (n_z==0)) {
                            ps_perturb_fftw_saved[0] = 0;
                        } else {
                            ps_perturb_fftw_saved[HII_C_INDEX(n_x,n_y,n_z)] =
                                ps_perturb_fftw_box[HII_C_INDEX(n_x,n_y,n_z)] * (dDdt_over_D*k_z*I/k_sq/(HII_TOT_NUM_PIXELS+0.0));
                        }
                    }
                }
            }
        }
        fftwf_execute_dft_c2r(ps_perturb_plan_c2r, ps_perturb_fftw_saved, (float *)ps_perturb_fftw_saved);
        float *fsaved = (float *)ps_perturb_fftw_saved;
        #pragma omp parallel for collapse(2) num_threads(user_params->N_THREADS)
        for (int i=0; i<side; ++i)
            for (int j=0; j<side; ++j)
                memcpy(&velocity_z[HII_R_INDEX(i,j,0)], &fsaved[HII_R_FFT_INDEX(i,j,0)], row_bytes);
    }
    fftwf_execute(ps_perturb_plan_c2r);
    const float norm = (float)HII_TOT_NUM_PIXELS;
    #pragma omp parallel for collapse(2) num_threads(user_params->N_THREADS)
    for (int i=0; i<side; ++i)
        for (int j=0; j<side; ++j) {
            const float *row_in = &fbox[HII_R_FFT_INDEX(i,j,0)];
            float *row_out = &density[HII_R_INDEX(i,j,0)];
            for (int k=0; k<para; ++k) {
                float value = row_in[k] / norm;
                row_out[k] = value < -1.0 ? -1+FRACT_FLOAT_ERR : value;
            }
        }
    if (!ps_keep_fftw_plans) {
        ps_destroy_cached_fftw_plans();
        fftwf_cleanup_threads();
        fftwf_cleanup();
        fftwf_forget_wisdom();
    }
    return 0;
}

static int MaterializePerturbDensity(struct UserParams *user_params, float *density) {
    return MaterializePerturbDensityAndVel(user_params, density, NULL, 0.0f, 0);
}

static int UncachedMaterializePerturbDensity(struct UserParams *user_params, float *density) {
    return UncachedMaterializePerturbDensityAndVel(user_params, density, NULL, 0.0f, 0);
}

char *PSPerturbPlanText(int inverse) {
    if (!user_params_ufunc || user_params_ufunc->USE_FFTW_WISDOM ||
        EnsurePerturbFFTWPlans(user_params_ufunc, 0)) return NULL;
    return fftwf_sprint_plan(inverse ? ps_perturb_plan_c2r : ps_perturb_plan_r2c);
}

int TestMaterializePerturbDensity(int cached, int threads, float *density) {
    struct UserParams tmp = *user_params_ufunc;
    tmp.N_THREADS = threads;
    return cached ? MaterializePerturbDensity(&tmp, density) : UncachedMaterializePerturbDensity(&tmp, density);
}
typedef int (*PerturbGPUHook)(float, int, int, double,
    float, float, float, float, const float **, float *, float *);
static PerturbGPUHook perturb_gpu_hook = NULL;
typedef int (*PerturbTraceObserver)(int,const float *,const float *);
static PerturbTraceObserver perturb_trace_observer;
void SetPerturbTraceObserver(void *p) { perturb_trace_observer = (PerturbTraceObserver)p; }

void SetPerturbGPUHook(void *hook) { perturb_gpu_hook = (PerturbGPUHook)hook; }

int ComputePerturbField(""")
anchor = "    mass_factor = pow(f_pixel_factor, 3);"
splice(anchor, anchor + """
    if (perturb_gpu_hook && !global_params.EVOLVE_DENSITY_LINEARLY &&
        !global_params.SMOOTH_EVOLVED_DENSITY_FIELD &&
        !user_params->PERTURB_ON_HIGH_RES && user_params->USE_2LPT &&
        user_params->NON_CUBIC_FACTOR == 1.0 && !user_params->KEEP_3D_VELOCITIES &&
        user_params->DIM % user_params->HII_DIM == 0) {
        const float *fields[] = {boxes->hires_density, boxes->lowres_vx, boxes->lowres_vy,
            boxes->lowres_vz, boxes->lowres_vx_2LPT, boxes->lowres_vy_2LPT, boxes->lowres_vz_2LPT};
        const float first = (growth_factor - init_growth_factor) / user_params->BOX_LEN;
        const float second = (displacement_factor_2LPT - init_displacement_factor_2LPT) / user_params->BOX_LEN;
        const int result = perturb_gpu_hook(redshift, user_params->DIM, user_params->HII_DIM,
            DELTA_K, first, second, init_growth_factor, dDdt/growth_factor,
            fields, perturbed_field->density, perturbed_field->velocity_z);
        if (result == 4) {
            if (MaterializePerturbDensityAndVel(user_params, perturbed_field->density,
                                                perturbed_field->velocity_z, dDdt/growth_factor, 1)) Throw(ValueError);
            goto perturb_success;
        }
        if (result == 3) {
            if (MaterializePerturbDensity(user_params, perturbed_field->density)) Throw(ValueError);
            goto perturb_success;
        }
        if (result == 0) goto perturb_success;
        if (result != 2) Throw(ValueError);
    }
""")
splice("    fftwf_cleanup_threads();\n    fftwf_cleanup();\n    fftwf_forget_wisdom();", """    if (!ps_keep_fftw_plans) {
        ps_destroy_cached_fftw_plans();
        fftwf_cleanup_threads();
        fftwf_cleanup();
        fftwf_forget_wisdom();
    }""")
splice("        fftwf_free(HIRES_density_perturb_saved);\n    }\n    fftwf_cleanup();", """        fftwf_free(HIRES_density_perturb_saved);
    }
    if (!ps_keep_fftw_plans) {
        ps_destroy_cached_fftw_plans();
        fftwf_cleanup();
    }""")
splice("    } // End of Try{}", "    perturb_success: ;\n    if (perturb_trace_observer && perturb_trace_observer(user_params->HII_DIM, perturbed_field->density, perturbed_field->velocity_z)) Throw(ValueError);\n    } // End of Try{}")
path.write_text(source)

ip = src / "IonisationBox.c"
ion = ip.read_text()
ion = ion.replace("int ComputeIonizedBox(", """typedef int (*IonObserver)(float, float, unsigned long long, const float *, const float *);
static IonObserver ion_observer = NULL;
void SetIonObserver(void *hook) { ion_observer = (IonObserver)hook; }

int ComputeIonizedBox(""", 1)
anchor = '    free(overdense_int_boundexceeded_threaded);'
assert ion.count(anchor) == 1
ion = ion.replace(anchor, """    if (ion_observer && ion_observer(redshift, prev_redshift, HII_TOT_NUM_PIXELS,
        box->xH_box, box->dNrec_box)) Throw(ValueError);
""" + anchor)
from ion_patch import patch_ion
ion = patch_ion(ion)
from trajectory_patch import patch_trajectory
ion = patch_trajectory(ion)
ion_clean = """        fftwf_cleanup_threads();
        fftwf_cleanup();
        fftwf_forget_wisdom();"""
assert ion.count(ion_clean) == 1
ion = ion.replace(ion_clean, """        if (!ps_keep_fftw_plans) {
            ps_destroy_cached_fftw_plans();
            fftwf_cleanup_threads();
            fftwf_cleanup();
            fftwf_forget_wisdom();
        }""", 1)
ip.write_text(ion)
from heat_seed_patch import patch_heat
sp=src / "SpinTemperatureBox.c"
from xray_patch import patch_xray
spin=patch_xray(patch_heat(sp.read_text()))
anchor="/* The GPU filter cache belongs"
assert spin.count(anchor)==1
spin=spin.replace(anchor,
    "typedef int (*GPUPreviousHeatMeans)(float,const float *,double *,double *);\n"
    "static GPUPreviousHeatMeans previous_heat_means;\n"
    "void SetGPUPreviousHeatMeans(void *ptr) {previous_heat_means=(GPUPreviousHeatMeans)ptr;}\n"
    +anchor,1)
anchor="        #pragma omp parallel shared(previous_spin_temp) private(ct)"
assert spin.count(anchor)==1
spin=spin.replace(anchor,
    "        if (!previous_heat_means || previous_heat_means(prev_redshift,previous_spin_temp->x_e_box,&x_e_ave,&Tk_ave)) {\n"
    +anchor,1)
anchor="        x_e_ave /= (float)HII_TOT_NUM_PIXELS;"
assert spin.count(anchor)==1
spin=spin.replace(anchor,"        }\n"+anchor,1)
xe_declaration = """
static long xe_trace_id = -1;
static float xe_trace_z = -1;
void SetXeTraceCell(long i, float z) {xe_trace_id=i; xe_trace_z=z;}
float CompareXeStep(float rho,float px,float pt,double raw,double growth,double dt,double dz,
                    double pref,double clump,double fH,double p1) {
    double x=px;
    double sink=alpha_A(pt)*clump*x*x*fH*p1*(1.+rho*growth);
    x += dt*(raw*pref-sink)*dz;
    if(x>1.) x=1.-FRACT_FLOAT_ERR;
    if(x<0.) x=0.;
    return (float)x;
}
"""
assert spin.count("} GPUHeatArgs;")==1
spin=spin.replace("} GPUHeatArgs;","} GPUHeatArgs;"+xe_declaration+"""
static double *heat_win_cache = NULL;
static float *heat_win_radii = NULL;
static int heat_win_side = 0;
static int heat_win_count = 0;
static double heat_win_dk = 0.0;
static fftwf_plan heat_plan_r2c = NULL;
static int heat_plan_side = 0;
static int heat_plan_threads = 0;
static const void *heat_plan_buf = NULL;
static fftwf_plan heat_plan_c2r = NULL;
static int heat_plan_c2r_side = 0;
static fftwf_plan heat_plan_c2r_mt = NULL;
static int heat_plan_c2r_mt_side = 0;
static int heat_plan_c2r_mt_threads = 0;
static fftwf_complex *heat_ref_work = NULL;
static fftwf_complex *heat_thread_buf[16] = {0};
static int heat_thread_slots = 0;
static int heat_buf_side = 0;

static void HeatDestroyWindowCache(void) {
    if (heat_win_cache) { free(heat_win_cache); heat_win_cache = NULL; }
    if (heat_win_radii) { free(heat_win_radii); heat_win_radii = NULL; }
    heat_win_side = 0;
    heat_win_count = 0;
    heat_win_dk = 0.0;
}

static void HeatDestroyFFTWPlans(void) {
    if (heat_plan_r2c) { fftwf_destroy_plan(heat_plan_r2c); heat_plan_r2c = NULL; }
    if (heat_plan_c2r) { fftwf_destroy_plan(heat_plan_c2r); heat_plan_c2r = NULL; }
    if (heat_plan_c2r_mt) { fftwf_destroy_plan(heat_plan_c2r_mt); heat_plan_c2r_mt = NULL; }
    heat_plan_side = 0;
    heat_plan_threads = 0;
    heat_plan_buf = NULL;
    heat_plan_c2r_side = 0;
    heat_plan_c2r_mt_side = 0;
    heat_plan_c2r_mt_threads = 0;
    if (heat_ref_work) { fftwf_free(heat_ref_work); heat_ref_work = NULL; }
    for (int t = 0; t < 16; ++t) {
        if (heat_thread_buf[t]) { fftwf_free(heat_thread_buf[t]); heat_thread_buf[t] = NULL; }
    }
    heat_thread_slots = 0;
    heat_buf_side = 0;
    HeatDestroyWindowCache();
}

static int HeatEnsureBuffers(int side, int nthreads) {
    ps_destroy_heat_fftw_plans_cb = HeatDestroyFFTWPlans;
    size_t kz_len = (size_t)(side / 2 + 1);
    size_t bytes = sizeof(fftwf_complex) * (size_t)side * (size_t)side * kz_len;
    if (heat_buf_side != side) {
        if (heat_ref_work) { fftwf_free(heat_ref_work); heat_ref_work = NULL; }
        for (int t = 0; t < 16; ++t) {
            if (heat_thread_buf[t]) { fftwf_free(heat_thread_buf[t]); heat_thread_buf[t] = NULL; }
        }
        heat_thread_slots = 0;
        heat_ref_work = (fftwf_complex *)fftwf_malloc(bytes);
        if (!heat_ref_work) return 1;
        heat_buf_side = side;
    }
    if (!heat_ref_work) return 1;
    int need_slots = nthreads < 2 ? 2 : (nthreads > 16 ? 16 : nthreads);
    while (heat_thread_slots < need_slots) {
        heat_thread_buf[heat_thread_slots] = (fftwf_complex *)fftwf_malloc(bytes);
        if (!heat_thread_buf[heat_thread_slots]) return 1;
        ++heat_thread_slots;
    }
    return 0;
}

static int HeatEnsureWindowCache(int side, int count, double dk, const float *radii, int nthreads) {
    ps_destroy_heat_fftw_plans_cb = HeatDestroyFFTWPlans;
    if (heat_win_cache && heat_win_side == side && heat_win_count == count && heat_win_dk == dk &&
        heat_win_radii && memcmp(heat_win_radii, radii, (size_t)count * sizeof(float)) == 0) {
        return 0;
    }
    HeatDestroyWindowCache();
    heat_win_radii = (float *)malloc((size_t)count * sizeof(float));
    if (!heat_win_radii) return 1;
    memcpy(heat_win_radii, radii, (size_t)count * sizeof(float));
    int mid = side / 2;
    int kz_len = mid + 1;
    if (count > 1) {
        size_t oct_plane = (size_t)kz_len * kz_len;
        size_t oct_vol = (size_t)kz_len * oct_plane;
        heat_win_cache = (double *)malloc((size_t)(count - 1) * oct_vol * sizeof(double));
        if (!heat_win_cache) {
            HeatDestroyWindowCache();
            return 1;
        }
        int nt = nthreads < 1 ? 1 : (nthreads > 16 ? 16 : nthreads);
        #pragma omp parallel for schedule(static) num_threads(nt)
        for (int s = 1; s < count; ++s) {
            double *wscale = heat_win_cache + (size_t)(s - 1) * oct_vol;
            const float R = radii[s];
            for (int ax = 0; ax < kz_len; ++ax) {
                double *wx = wscale + (size_t)ax * oct_plane;
                for (int ay = 0; ay < kz_len; ++ay) {
                    const float *km_row = ion_kmag_table + ((size_t)ax * side + ay) * kz_len;
                    double *wrow = wx + (size_t)ay * kz_len;
                    for (int iz = 0; iz < kz_len; ++iz) {
                        const float kR = km_row[iz] * R;
                        if (kR > 1e-4) {
                            wrow[iz] = 3.0 * pow(kR, -3) * (sin(kR) - cos(kR) * kR);
                        } else {
                            wrow[iz] = 1.0;
                        }
                    }
                }
            }
        }
    }
    heat_win_side = side;
    heat_win_count = count;
    heat_win_dk = dk;
    ps_destroy_heat_fftw_plans_cb = HeatDestroyFFTWPlans;
    return 0;
}

static int heat_batch_exec(
    int side, int count, float inverse_growth,
    const float *density, float *out_batch, float *out_minima, float *out_maxima, int nthreads
) {
    const int mid = side / 2;
    const int kz_len = mid + 1;
    const int padded = 2 * kz_len;
    const size_t n = (size_t)side * side * side;
    const size_t kn = (size_t)side * side * kz_len;
    float *fsrc = (float *)heat_ref_work;
    const size_t row_bytes = (size_t)side * sizeof(float);
    const int copy_nt = nthreads < 1 ? 1 : (nthreads > 16 ? 16 : nthreads);
    #pragma omp parallel for collapse(2) num_threads(copy_nt)
    for (int i = 0; i < side; ++i)
        for (int j = 0; j < side; ++j)
            memcpy(&fsrc[((size_t)i * side + j) * padded], &density[((size_t)i * side + j) * side], row_bytes);

    fftwf_execute_dft_r2c(heat_plan_r2c, fsrc, heat_ref_work);
    const float norm = (float)n;
    float *norm_ptr = (float *)heat_ref_work;
    const size_t fkn = 2 * kn;
    #pragma omp parallel for num_threads(copy_nt)
    for (size_t ct = 0; ct < fkn; ++ct) {
        norm_ptr[ct] /= norm;
    }

    const int nt = copy_nt < heat_thread_slots ? copy_nt : heat_thread_slots;
    if (nt < 1) return 6;
    const size_t oct_plane = (size_t)kz_len * kz_len;
    const size_t oct_vol = (size_t)kz_len * oct_plane;
    #pragma omp parallel num_threads(nt)
    {
        int tid = omp_get_thread_num();
        fftwf_complex *work = heat_thread_buf[tid];
        #pragma omp for schedule(dynamic, 1)
        for (int s = 0; s < count; ++s) {
            if (s == 0) {
                memcpy(work, heat_ref_work, sizeof(fftwf_complex) * kn);
            } else {
                const double *wscale = heat_win_cache + (size_t)(s - 1) * oct_vol;
                for (int nx = 0; nx < side; ++nx) {
                    const int ax = (nx > mid) ? (side - nx) : nx;
                    const double *wx = wscale + (size_t)ax * oct_plane;
                    for (int ny = 0; ny < side; ++ny) {
                        const int ay = (ny > mid) ? (side - ny) : ny;
                        const double *wrow = wx + (size_t)ay * kz_len;
                        const size_t base = ((size_t)nx * side + ny) * kz_len;
                        const float *srow = (const float *)(heat_ref_work + base);
                        float *drow = (float *)(work + base);
                        for (int nz = 0; nz < kz_len; ++nz) {
                            const double w = wrow[nz];
                            drow[2 * nz] = (float)((double)srow[2 * nz] * w);
                            drow[2 * nz + 1] = (float)((double)srow[2 * nz + 1] * w);
                        }
                    }
                }
            }
            fftwf_execute_dft_c2r(heat_plan_c2r, work, (float *)work);
            const float *fwork = (const float *)work;
            float *dst = out_batch + (size_t)s * n;
            float min_d = 0.0f, max_d = 0.0f;
            for (int i = 0; i < side; ++i) {
                for (int j = 0; j < side; ++j) {
                    const float *rin = fwork + ((size_t)i * side + j) * padded;
                    float *rout = dst + ((size_t)i * side + j) * side;
                    for (int k = 0; k < side; ++k) {
                        double curr = (double)rin[k];
                        if (curr <= -1.0) curr = -1.0 + FRACT_FLOAT_ERR;
                        curr *= inverse_growth;
                        rout[k] = (float)curr;
                        if (curr < min_d) min_d = (float)curr;
                        if (curr > max_d) max_d = (float)curr;
                    }
                }
            }
            out_minima[s] = min_d;
            out_maxima[s] = max_d;
        }
    }
    return 0;
}

int HeatReferenceBatchDensity(
    int side, int count, double dk, float inverse_growth,
    const float *density, const float *radii,
    float *out_batch, float *out_minima, float *out_maxima
) {
    struct UserParams *user_params = user_params_ufunc;
    if (!user_params || count < 1 || !density || !radii || !out_batch || !out_minima || !out_maxima) return 1;
    int nthreads = omp_get_max_threads();
    if (nthreads < 1) nthreads = 1;
    if (nthreads > 16) nthreads = 16;
    if (HeatEnsureBuffers(side, nthreads)) return 2;
    if (IonEnsureKmag(side, dk, nthreads)) return 3;
    if (HeatEnsureWindowCache(side, count, dk, radii, nthreads)) return 4;
    const int r2c_nt = nthreads < 1 ? 1 : (nthreads > 8 ? 8 : nthreads);
    if (!heat_plan_r2c || heat_plan_side != side || heat_plan_threads != r2c_nt || heat_plan_buf != heat_ref_work) {
        if (heat_plan_r2c) { fftwf_destroy_plan(heat_plan_r2c); heat_plan_r2c = NULL; }
        fftwf_init_threads();
        fftwf_plan_with_nthreads(r2c_nt);
        heat_plan_r2c = fftwf_plan_dft_r2c_3d(side, side, side, (float *)heat_ref_work, heat_ref_work, ps_fftw_heat_plan_flags());
        fftwf_plan_with_nthreads(1);
        if (!heat_plan_r2c) return 5;
        heat_plan_side = side;
        heat_plan_threads = r2c_nt;
        heat_plan_buf = heat_ref_work;
    }
    if (!heat_plan_c2r || heat_plan_c2r_side != side) {
        if (heat_plan_c2r) { fftwf_destroy_plan(heat_plan_c2r); heat_plan_c2r = NULL; }
        fftwf_plan_with_nthreads(1);
        heat_plan_c2r = fftwf_plan_dft_c2r_3d(side, side, side, heat_ref_work, (float *)heat_ref_work, ps_fftw_heat_plan_flags());
        if (!heat_plan_c2r) return 5;
        heat_plan_c2r_side = side;
    }
    ps_destroy_heat_fftw_plans_cb = HeatDestroyFFTWPlans;

    int rc = heat_batch_exec(side, count, inverse_growth, density, out_batch, out_minima, out_maxima, nthreads);
    if (rc != 0) return rc;

    if (!ps_keep_fftw_plans) {
        ps_destroy_cached_fftw_plans();
        fftwf_cleanup_threads();
        fftwf_cleanup();
        fftwf_forget_wisdom();
    }
    return 0;
}

int HeatReferenceBatchDensityPreplanned(
    int side, int count, double dk, float inverse_growth,
    const float *density, const float *radii,
    float *out_batch, float *out_minima, float *out_maxima, int nthreads
) {
    if (!ps_keep_fftw_plans || count < 1 || !density || !radii || !out_batch || !out_minima || !out_maxima) return 9;
    if (!heat_plan_r2c || !heat_plan_c2r || heat_plan_side != side || heat_plan_c2r_side != side ||
        heat_buf_side != side || !heat_ref_work || heat_thread_slots < 1) return 9;
    if (!ion_kmag_table || ion_kmag_side != side || ion_kmag_dk != dk) return 9;
    if (!heat_win_cache || heat_win_side != side || heat_win_count != count || heat_win_dk != dk ||
        !heat_win_radii || memcmp(heat_win_radii, radii, (size_t)count * sizeof(float)) != 0) return 9;
    int nt = nthreads < 1 ? 1 : (nthreads > heat_thread_slots ? heat_thread_slots : nthreads);
    return heat_batch_exec(side, count, inverse_growth, density, out_batch, out_minima, out_maxima, nt);
}

static void heat_apply_window_mt(
    const fftwf_complex *src, fftwf_complex *dst,
    const double *wscale, int side, int nt
) {
    const int mid = side / 2;
    const int kz_len = mid + 1;
    const size_t oct_plane = (size_t)kz_len * kz_len;
    #pragma omp parallel for collapse(2) num_threads(nt)
    for (int nx = 0; nx < side; ++nx) {
        for (int ny = 0; ny < side; ++ny) {
            const int ax = (nx > mid) ? (side - nx) : nx;
            const int ay = (ny > mid) ? (side - ny) : ny;
            const double *wrow = wscale + (size_t)ax * oct_plane + (size_t)ay * kz_len;
            const size_t base = ((size_t)nx * side + ny) * kz_len;
            const float *srow = (const float *)(src + base);
            float *drow = (float *)(dst + base);
            int nz = 0;
#ifdef __AVX2__
            for (; nz + 2 <= kz_len; nz += 2) {
                __m128 vf = _mm_loadu_ps(srow + 2 * nz);
                __m256d vd = _mm256_cvtps_pd(vf);
                __m128d vw = _mm_loadu_pd(wrow + nz);
                __m256d vw_rep = _mm256_permute4x64_pd(_mm256_castpd128_pd256(vw), 0x50);
                __m256d vres = _mm256_mul_pd(vd, vw_rep);
                _mm_storeu_ps(drow + 2 * nz, _mm256_cvtpd_ps(vres));
            }
#endif
            for (; nz < kz_len; ++nz) {
                const double w = wrow[nz];
                drow[2 * nz] = (float)((double)srow[2 * nz] * w);
                drow[2 * nz + 1] = (float)((double)srow[2 * nz + 1] * w);
            }
        }
    }
}

static int heat_batch_exec_padded(
    int side, int count, const float *density, float *out_padded, int nthreads
) {
    if (((uintptr_t)out_padded & 15) != 0) return 10;
    const int mid = side / 2;
    const int kz_len = mid + 1;
    const int padded = 2 * kz_len;
    const size_t n = (size_t)side * side * side;
    const size_t kn = (size_t)side * side * kz_len;
    const size_t padded_vol = (size_t)side * side * padded;
    if ((padded_vol * sizeof(float)) % 16 != 0) return 10;
    const int nt = ps_clamp_threads(nthreads);

    float *fsrc = (float *)heat_ref_work;
    const size_t row_bytes = (size_t)side * sizeof(float);
    #pragma omp parallel for collapse(2) num_threads(nt)
    for (int i = 0; i < side; ++i)
        for (int j = 0; j < side; ++j)
            memcpy(&fsrc[((size_t)i * side + j) * padded], &density[((size_t)i * side + j) * side], row_bytes);

    fftwf_execute_dft_r2c(heat_plan_r2c, fsrc, heat_ref_work);
    const float norm = (float)n;
    float *norm_ptr = (float *)heat_ref_work;
    const size_t fkn = 2 * kn;
    #pragma omp parallel for num_threads(nt)
    for (size_t ct = 0; ct < fkn; ++ct) {
        norm_ptr[ct] /= norm;
    }

    const size_t oct_vol = (size_t)kz_len * kz_len * kz_len;
    for (int s = 0; s < count; ++s) {
        fftwf_complex *dst_s = (fftwf_complex *)(out_padded + (size_t)s * padded_vol);
        if (s == 0) {
            PSParallelMemcpy(dst_s, heat_ref_work, sizeof(fftwf_complex) * kn, nt);
        } else {
            heat_apply_window_mt(heat_ref_work, dst_s, heat_win_cache + (size_t)(s - 1) * oct_vol, side, nt);
        }
        fftwf_execute_dft_c2r(heat_plan_c2r_mt, dst_s, (float *)dst_s);
    }
    return 0;
}

int HeatReferenceBatchDensityPadded(
    int side, int count, double dk, const float *density, const float *radii,
    float *out_padded, int nthreads
) {
    struct UserParams *user_params = user_params_ufunc;
    if (!user_params || count < 1 || !density || !radii || !out_padded) return 1;
    int nt = ps_clamp_threads(nthreads > 0 ? nthreads : omp_get_max_threads());
    if (HeatEnsureBuffers(side, nt)) return 2;
    if (IonEnsureKmag(side, dk, nt)) return 3;
    if (HeatEnsureWindowCache(side, count, dk, radii, nt)) return 4;
    if (!heat_plan_r2c || heat_plan_side != side || heat_plan_threads != nt || heat_plan_buf != heat_ref_work) {
        if (heat_plan_r2c) { fftwf_destroy_plan(heat_plan_r2c); heat_plan_r2c = NULL; }
        fftwf_init_threads();
        fftwf_plan_with_nthreads(nt);
        heat_plan_r2c = fftwf_plan_dft_r2c_3d(side, side, side, (float *)heat_ref_work, heat_ref_work, ps_fftw_heat_plan_flags());
        fftwf_plan_with_nthreads(1);
        if (!heat_plan_r2c) return 5;
        heat_plan_side = side;
        heat_plan_threads = nt;
        heat_plan_buf = heat_ref_work;
    }
    if (!heat_plan_c2r_mt || heat_plan_c2r_mt_side != side || heat_plan_c2r_mt_threads != nt) {
        if (heat_plan_c2r_mt) { fftwf_destroy_plan(heat_plan_c2r_mt); heat_plan_c2r_mt = NULL; }
        fftwf_init_threads();
        fftwf_plan_with_nthreads(nt);
        heat_plan_c2r_mt = fftwf_plan_dft_c2r_3d(side, side, side, heat_ref_work, (float *)heat_ref_work, ps_fftw_heat_plan_flags());
        fftwf_plan_with_nthreads(1);
        if (!heat_plan_c2r_mt) return 5;
        heat_plan_c2r_mt_side = side;
        heat_plan_c2r_mt_threads = nt;
    }
    ps_destroy_heat_fftw_plans_cb = HeatDestroyFFTWPlans;
    int rc = heat_batch_exec_padded(side, count, density, out_padded, nt);
    if (rc != 0) return rc;
    if (!ps_keep_fftw_plans) {
        ps_destroy_cached_fftw_plans();
        fftwf_cleanup_threads();
        fftwf_cleanup();
        fftwf_forget_wisdom();
    }
    return 0;
}

int HeatReferenceBatchDensityPreplannedPadded(
    int side, int count, double dk, const float *density, const float *radii,
    float *out_padded, int nthreads
) {
    int nt = ps_clamp_threads(nthreads);
    if (!ps_keep_fftw_plans || count < 1 || !density || !radii || !out_padded) return 9;
    if (!heat_plan_r2c || !heat_plan_c2r_mt || heat_plan_side != side || heat_plan_threads != nt ||
        heat_plan_c2r_mt_side != side || heat_plan_c2r_mt_threads != nt ||
        heat_buf_side != side || !heat_ref_work) return 9;
    if (!ion_kmag_table || ion_kmag_side != side || ion_kmag_dk != dk) return 9;
    if (!heat_win_cache || heat_win_side != side || heat_win_count != count || heat_win_dk != dk ||
        !heat_win_radii || memcmp(heat_win_radii, radii, (size_t)count * sizeof(float)) != 0) return 9;
    return heat_batch_exec_padded(side, count, density, out_padded, nt);
}
int HeatReferenceUncachedFilter(
    int side, int count, float inverse_growth,
    const float *density, const float *radii,
    float *out_batch, float *out_minima, float *out_maxima
) {
    struct UserParams *user_params = user_params_ufunc;
    if (!user_params || count < 1) return 1;
    const int nthreads = user_params->N_THREADS;
    const size_t n = (size_t)side * side * side;
    fftwf_complex *unfiltered_box = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * HII_KSPACE_NUM_PIXELS);
    fftwf_complex *box = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * HII_KSPACE_NUM_PIXELS);
    if (!unfiltered_box || !box) {
        if (unfiltered_box) fftwf_free(unfiltered_box);
        if (box) fftwf_free(box);
        return 1;
    }
    #pragma omp parallel for num_threads(nthreads)
    for (int i = 0; i < side; ++i)
        for (int j = 0; j < side; ++j)
            for (int k = 0; k < side; ++k)
                *((float *)unfiltered_box + HII_R_FFT_INDEX(i, j, k)) = density[HII_R_INDEX(i, j, k)];
    if (dft_r2c_cube(user_params->USE_FFTW_WISDOM, side, side, nthreads, unfiltered_box)) {
        fftwf_free(unfiltered_box); fftwf_free(box);
        return 2;
    }
    #pragma omp parallel for num_threads(nthreads)
    for (unsigned long long ct = 0; ct < HII_KSPACE_NUM_PIXELS; ++ct)
        unfiltered_box[ct] /= (float)n;

    for (int s = 0; s < count; ++s) {
        memcpy(box, unfiltered_box, sizeof(fftwf_complex) * HII_KSPACE_NUM_PIXELS);
        if (s > 0) filter_box(box, 1, 0, radii[s]);
        if (dft_c2r_cube(user_params->USE_FFTW_WISDOM, side, side, nthreads, box)) {
            fftwf_free(unfiltered_box); fftwf_free(box);
            return 3;
        }
        float min_d = 0.0f, max_d = 0.0f;
        float *dst = out_batch + (size_t)s * n;
        #pragma omp parallel for reduction(min:min_d) reduction(max:max_d) num_threads(nthreads)
        for (int i = 0; i < side; ++i) {
            for (int j = 0; j < side; ++j) {
                for (int k = 0; k < side; ++k) {
                    double curr = *((float *)box + HII_R_FFT_INDEX(i, j, k));
                    if (curr <= -1.0) curr = -1.0 + FRACT_FLOAT_ERR;
                    curr *= inverse_growth;
                    dst[HII_R_INDEX(i, j, k)] = (float)curr;
                    if (curr < min_d) min_d = (float)curr;
                    if (curr > max_d) max_d = (float)curr;
                }
            }
        }
        out_minima[s] = min_d;
        out_maxima[s] = max_d;
    }
    fftwf_free(unfiltered_box);
    fftwf_free(box);
    return 0;
}
""",1)
anchor="                dfcoll_dz_val = (ave_fcoll_inv/pow(10.,10.))*ST_over_PS[R_ct]*SFR_timescale_factor[R_ct]/astro_params->t_STAR;"
assert spin.count(anchor)==1
spin=spin.replace(anchor,anchor+"""
                if (xe_trace_id>=0 && fabs(zp-xe_trace_z)<1.e-6)
                    printf("CPU_XE_SCALE %.9g %d %.17g %.17g %.17g %.17g %.17g %.17g\\n",
                      zp,R_ct,(double)delNL0[R_ct][xe_trace_id],(double)zpp_growth[R_ct],
                      (double)del_fcoll_Rct[xe_trace_id],ave_fcoll,dfcoll_dz_val,
                      freq_int_ion_tbl_diff[m_xHII_low_box[xe_trace_id]][R_ct]*inverse_val_box[xe_trace_id]+
                      freq_int_ion_tbl[m_xHII_low_box[xe_trace_id]][R_ct]);
""",1)
anchor="                            this_spin_temp->x_e_box[box_ct] = x_e;"
assert spin.count(anchor)==1
spin=spin.replace(anchor,"""
                            if (box_ct==xe_trace_id && fabs(zp-xe_trace_z)<1.e-6)
                                printf("CPU_XE_STEP %.9g %.17g %.17g %.17g %.17g %.17g\\n",zp,
                                    (double)previous_spin_temp->x_e_box[box_ct],dxion_source_dt_box[box_ct],
                                    dxion_sink_dt,dxe_dzp,x_e);
"""+anchor,1)
spin_clean = """    fftwf_forget_wisdom();
    fftwf_cleanup_threads();
    fftwf_cleanup();"""
assert spin.count(spin_clean) == 1
spin = spin.replace(spin_clean, """    if (!ps_keep_fftw_plans) {
        ps_destroy_cached_fftw_plans();
        fftwf_forget_wisdom();
        fftwf_cleanup_threads();
        fftwf_cleanup();
    }""", 1)
sp.write_text(spin)
from brightness_patch import patch_brightness
bp=src / "BrightnessTemperatureBox.c"
bp_text = patch_brightness(bp.read_text())
bp_clean = """    fftwf_cleanup_threads();
    fftwf_cleanup();"""
assert bp_text.count(bp_clean) == 1
bp_text = bp_text.replace(bp_clean, """    if (!ps_keep_fftw_plans) {
        ps_destroy_cached_fftw_plans();
        fftwf_cleanup_threads();
        fftwf_cleanup();
    }""", 1)
bp.write_text(bp_text)

from ic_seed_patch import patch_seed
ic_path=src / "GenerateICs.c"
ic_src = patch_seed(ic_path.read_text())
assert ic_src.count("#include <fftw3.h>") == 1
ic_src = ic_src.replace("#include <fftw3.h>", "#include <fftw3.h>\n#ifdef __AVX2__\n#include <immintrin.h>\n#endif", 1)
ic_path.write_text(ic_src)
from conditional_patch import patch_condition
ps=src/"ps.c"
from sigma_patch import patch_sigma
from hmf_patch import patch_hmf
ps.write_text(patch_hmf(patch_sigma(patch_condition(ps.read_text()))))
from mhr_patch import patch_mhr
rec=src/"recombinations.c"
rec.write_text(patch_mhr(rec.read_text()))
from heat_table_cache_patch import patch_heat_table_cache
heat_helpers=src/"heating_helper_progs.c"
heat_helpers.write_text(patch_heat_table_cache(heat_helpers.read_text()))
from native_profile import patch_profile
patch_profile(src)
deps = detect_native_deps(root=root)
missing = []
if not deps["gsl_found"]:
    missing.append("GSL (gsl/gsl_integration.h)")
if not deps["fftw_found"]:
    missing.append("FFTW3f (fftw3.h)")
if missing:
    raise RuntimeError(
        "Missing native build dependencies: "
        + ", ".join(missing)
        + ". Install them via `conda env create -f environment.yml` or pass "
        + "PS_GSL_PREFIX / PS_FFTW_PREFIX."
    )
extra_compile_args = [
    "-fopenmp",
    "-O3",
    *(["-mavx2"] if platform.machine() in ("x86_64", "AMD64") else []),
    "-ffp-contract=off",
    "-fno-fast-math",
    "-fno-finite-math-only",
]
ffi = FFI()
ffi.set_source(
    "py21cmfast.c_21cmfast",
    '#define LOG_LEVEL 1\n#include "GenerateICs.c"',
    include_dirs=[str(src), *deps["include_dirs"]],
    library_dirs=deps["library_dirs"],
    libraries=["m","gsl","gslcblas","fftw3f_omp","fftw3f"],
    extra_compile_args=extra_compile_args,
    extra_link_args=["-fopenmp", *[f"-Wl,-rpath,{d}" for d in deps["library_dirs"]]],
)
for name in ("21cmFAST.h","Globals.h"):
    ffi.cdef((src/name).read_text())
ffi.cdef("void SetPerturbGPUHook(void *);")
ffi.cdef("""
void PSParallelMemcpy(void *dst, const void *src, unsigned long long nbytes, int threads);
int PSArrayAllFinite(const float *arr, unsigned long long n, int threads);
int TestIonPrunedC2RCompare(int side, double dk, float radius, int nt, const void *src, float *out_pruned, float *out_full);
void PSFillLightconeSlices(
    const float *c1, const float *c2, float *lc_out,
    int side, int depth, int n_slices,
    const int *lc_indices, const int *k_indices,
    const double *w1, const double *w2, double denom,
    int is_mean_max, int threads
);
""")
ffi.cdef("void SetGPUFilterHook(void *); void SetGPUHeatHook(void *); void SetHeatObserver(void *); int ResolveCollapseDensity(const float *, float, const long *, long, float *);")
ffi.compile(tmpdir=str(package),verbose=True)
