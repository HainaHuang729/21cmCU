/*
 * Copyright (c) 2003, 2007-14 Matteo Frigo
 * Copyright (c) 2003, 2007-14 Massachusetts Institute of Technology
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 */

/* This file was automatically generated --- DO NOT EDIT */
/* Generated on Tue Sep 14 10:45:00 EDT 2021 */

// dft/simd/common/n1fv_15.c (non-FMA codelet, CUDA adaptation)
__device__ void n1fv_15(const R *ri, const R *ii, R *ro, R *io, stride is, stride os, INT v,
                        INT ivs, INT ovs) {
    DVK(KP216506350, +0.216506350946109661690930792688234045867850657);
    DVK(KP509036960, +0.509036960455127183450980863393907648510733164);
    DVK(KP823639103, +0.823639103546331925877420039278190003029660514);
    DVK(KP587785252, +0.587785252292473129168705954639072768597652438);
    DVK(KP951056516, +0.951056516295153572116439333379382143405698634);
    DVK(KP250000000, +0.250000000000000000000000000000000000000000000);
    DVK(KP559016994, +0.559016994374947424102293417182819058860154590);
    DVK(KP866025403, +0.866025403784438646763723170752936183471402627);
    DVK(KP484122918, +0.484122918275927110647408174972799951354115213);
    DVK(KP500000000, +0.500000000000000000000000000000000000000000000);
    {
        INT i;
        const R *xi;
        R *xo;
        xi = ri;
        xo = ro;
        for (i = v; i > 0; i = i - VL, xi = xi + (VL * ivs), xo = xo + (VL * ovs),
            MAKE_VOLATILE_STRIDE(30, is), MAKE_VOLATILE_STRIDE(30, os)) {
            V T5, T10, TB, TO, TU, TV, TR, Ta, Tf, Tg, Tl, Tq, Tr, TE, TH;
            V TI, TZ, T11, T1f, T1g;
            {
                V T1, T2, T3, T4;
                T1 = LD(&(xi[0]), ivs, &(xi[0]));
                T2 = LD(&(xi[WS(is, 5)]), ivs, &(xi[WS(is, 1)]));
                T3 = LD(&(xi[WS(is, 10)]), ivs, &(xi[0]));
                T4 = VADD(T2, T3);
                T5 = VADD(T1, T4);
                T10 = VSUB(T3, T2);
                TB = VFNMS(LDK(KP500000000), T4, T1);
            }
            {
                V T6, T9, TC, TP, Tm, Tp, TG, TN, Tb, Te, TD, TQ, Th, Tk, TF;
                V TM, TX, TY;
                {
                    V T7, T8, Tn, To;
                    T6 = LD(&(xi[WS(is, 3)]), ivs, &(xi[WS(is, 1)]));
                    T7 = LD(&(xi[WS(is, 8)]), ivs, &(xi[0]));
                    T8 = LD(&(xi[WS(is, 13)]), ivs, &(xi[WS(is, 1)]));
                    T9 = VADD(T7, T8);
                    TC = VFNMS(LDK(KP500000000), T9, T6);
                    TP = VSUB(T8, T7);
                    Tm = LD(&(xi[WS(is, 9)]), ivs, &(xi[WS(is, 1)]));
                    Tn = LD(&(xi[WS(is, 14)]), ivs, &(xi[0]));
                    To = LD(&(xi[WS(is, 4)]), ivs, &(xi[0]));
                    Tp = VADD(Tn, To);
                    TG = VFNMS(LDK(KP500000000), Tp, Tm);
                    TN = VSUB(To, Tn);
                }
                {
                    V Tc, Td, Ti, Tj;
                    Tb = LD(&(xi[WS(is, 12)]), ivs, &(xi[0]));
                    Tc = LD(&(xi[WS(is, 2)]), ivs, &(xi[0]));
                    Td = LD(&(xi[WS(is, 7)]), ivs, &(xi[WS(is, 1)]));
                    Te = VADD(Tc, Td);
                    TD = VFNMS(LDK(KP500000000), Te, Tb);
                    TQ = VSUB(Td, Tc);
                    Th = LD(&(xi[WS(is, 6)]), ivs, &(xi[0]));
                    Ti = LD(&(xi[WS(is, 11)]), ivs, &(xi[WS(is, 1)]));
                    Tj = LD(&(xi[WS(is, 1)]), ivs, &(xi[WS(is, 1)]));
                    Tk = VADD(Ti, Tj);
                    TF = VFNMS(LDK(KP500000000), Tk, Th);
                    TM = VSUB(Tj, Ti);
                }
                TO = VSUB(TM, TN);
                TU = VSUB(TF, TG);
                TV = VSUB(TC, TD);
                TR = VSUB(TP, TQ);
                Ta = VADD(T6, T9);
                Tf = VADD(Tb, Te);
                Tg = VADD(Ta, Tf);
                Tl = VADD(Th, Tk);
                Tq = VADD(Tm, Tp);
                Tr = VADD(Tl, Tq);
                TE = VADD(TC, TD);
                TH = VADD(TF, TG);
                TI = VADD(TE, TH);
                TX = VADD(TP, TQ);
                TY = VADD(TM, TN);
                TZ = VMUL(LDK(KP484122918), VSUB(TX, TY));
                T11 = VADD(TX, TY);
            }
            T1f = VADD(TB, TI);
            T1g = VBYI(VMUL(LDK(KP866025403), VADD(T10, T11)));
            ST(&(xo[WS(os, 5)]), VSUB(T1f, T1g), ovs, &(xo[WS(os, 1)]));
            ST(&(xo[WS(os, 10)]), VADD(T1f, T1g), ovs, &(xo[0]));
            {
                V Tu, Ts, Tt, Ty, TA, Tw, Tx, Tz, Tv;
                Tu = VMUL(LDK(KP559016994), VSUB(Tg, Tr));
                Ts = VADD(Tg, Tr);
                Tt = VFNMS(LDK(KP250000000), Ts, T5);
                Tw = VSUB(Tl, Tq);
                Tx = VSUB(Ta, Tf);
                Ty = VBYI(VFNMS(LDK(KP587785252), Tx, VMUL(LDK(KP951056516), Tw)));
                TA = VBYI(VFMA(LDK(KP951056516), Tx, VMUL(LDK(KP587785252), Tw)));
                ST(&(xo[0]), VADD(T5, Ts), ovs, &(xo[0]));
                Tz = VADD(Tu, Tt);
                ST(&(xo[WS(os, 6)]), VSUB(Tz, TA), ovs, &(xo[0]));
                ST(&(xo[WS(os, 9)]), VADD(TA, Tz), ovs, &(xo[WS(os, 1)]));
                Tv = VSUB(Tt, Tu);
                ST(&(xo[WS(os, 3)]), VSUB(Tv, Ty), ovs, &(xo[WS(os, 1)]));
                ST(&(xo[WS(os, 12)]), VADD(Ty, Tv), ovs, &(xo[0]));
            }
            {
                V TS, TW, T1b, T18, T13, T1a, TL, T17, T12, TJ, TK;
                TS = VFNMS(LDK(KP509036960), TR, VMUL(LDK(KP823639103), TO));
                TW = VFNMS(LDK(KP587785252), TV, VMUL(LDK(KP951056516), TU));
                T1b = VFMA(LDK(KP951056516), TV, VMUL(LDK(KP587785252), TU));
                T18 = VFMA(LDK(KP823639103), TR, VMUL(LDK(KP509036960), TO));
                T12 = VFNMS(LDK(KP216506350), T11, VMUL(LDK(KP866025403), T10));
                T13 = VSUB(TZ, T12);
                T1a = VADD(TZ, T12);
                TJ = VFNMS(LDK(KP250000000), TI, TB);
                TK = VMUL(LDK(KP559016994), VSUB(TE, TH));
                TL = VSUB(TJ, TK);
                T17 = VADD(TK, TJ);
                {
                    V TT, T14, T1d, T1e;
                    TT = VSUB(TL, TS);
                    T14 = VBYI(VSUB(TW, T13));
                    ST(&(xo[WS(os, 8)]), VSUB(TT, T14), ovs, &(xo[0]));
                    ST(&(xo[WS(os, 7)]), VADD(TT, T14), ovs, &(xo[WS(os, 1)]));
                    T1d = VSUB(T17, T18);
                    T1e = VBYI(VADD(T1b, T1a));
                    ST(&(xo[WS(os, 11)]), VSUB(T1d, T1e), ovs, &(xo[WS(os, 1)]));
                    ST(&(xo[WS(os, 4)]), VADD(T1d, T1e), ovs, &(xo[0]));
                }
                {
                    V T15, T16, T19, T1c;
                    T15 = VADD(TL, TS);
                    T16 = VBYI(VADD(TW, T13));
                    ST(&(xo[WS(os, 13)]), VSUB(T15, T16), ovs, &(xo[WS(os, 1)]));
                    ST(&(xo[WS(os, 2)]), VADD(T15, T16), ovs, &(xo[0]));
                    T19 = VADD(T17, T18);
                    T1c = VBYI(VSUB(T1a, T1b));
                    ST(&(xo[WS(os, 14)]), VSUB(T19, T1c), ovs, &(xo[0]));
                    ST(&(xo[WS(os, 1)]), VADD(T19, T1c), ovs, &(xo[WS(os, 1)]));
                }
            }
        }
    }
    VLEAVE();
}

// dft/simd/common/n1fv_5.c (non-FMA codelet, CUDA adaptation)
__device__ void n1fv_5(const R *ri, const R *ii, R *ro, R *io, stride is, stride os, INT v, INT ivs,
                       INT ovs) {
    DVK(KP250000000, +0.250000000000000000000000000000000000000000000);
    DVK(KP587785252, +0.587785252292473129168705954639072768597652438);
    DVK(KP951056516, +0.951056516295153572116439333379382143405698634);
    DVK(KP559016994, +0.559016994374947424102293417182819058860154590);
    {
        INT i;
        const R *xi;
        R *xo;
        xi = ri;
        xo = ro;
        for (i = v; i > 0; i = i - VL, xi = xi + (VL * ivs), xo = xo + (VL * ovs),
            MAKE_VOLATILE_STRIDE(10, is), MAKE_VOLATILE_STRIDE(10, os)) {
            V T8, T7, Td, T9, Tc;
            T8 = LD(&(xi[0]), ivs, &(xi[0]));
            {
                V T1, T2, T3, T4, T5, T6;
                T1 = LD(&(xi[WS(is, 1)]), ivs, &(xi[WS(is, 1)]));
                T2 = LD(&(xi[WS(is, 4)]), ivs, &(xi[0]));
                T3 = VADD(T1, T2);
                T4 = LD(&(xi[WS(is, 2)]), ivs, &(xi[0]));
                T5 = LD(&(xi[WS(is, 3)]), ivs, &(xi[WS(is, 1)]));
                T6 = VADD(T4, T5);
                T7 = VMUL(LDK(KP559016994), VSUB(T3, T6));
                Td = VSUB(T4, T5);
                T9 = VADD(T3, T6);
                Tc = VSUB(T1, T2);
            }
            ST(&(xo[0]), VADD(T8, T9), ovs, &(xo[0]));
            {
                V Te, Tf, Tb, Tg, Ta;
                Te = VBYI(VFMA(LDK(KP951056516), Tc, VMUL(LDK(KP587785252), Td)));
                Tf = VBYI(VFNMS(LDK(KP587785252), Tc, VMUL(LDK(KP951056516), Td)));
                Ta = VFNMS(LDK(KP250000000), T9, T8);
                Tb = VADD(T7, Ta);
                Tg = VSUB(Ta, T7);
                ST(&(xo[WS(os, 1)]), VSUB(Tb, Te), ovs, &(xo[WS(os, 1)]));
                ST(&(xo[WS(os, 3)]), VSUB(Tg, Tf), ovs, &(xo[WS(os, 1)]));
                ST(&(xo[WS(os, 4)]), VADD(Te, Tb), ovs, &(xo[0]));
                ST(&(xo[WS(os, 2)]), VADD(Tf, Tg), ovs, &(xo[0]));
            }
        }
    }
    VLEAVE();
}

// dft/simd/common/n1fv_2.c (non-FMA codelet, CUDA adaptation)
__device__ void n1fv_2(const R *ri, const R *ii, R *ro, R *io, stride is, stride os, INT v, INT ivs,
                       INT ovs) {
    {
        INT i;
        const R *xi;
        R *xo;
        xi = ri;
        xo = ro;
        for (i = v; i > 0; i = i - VL, xi = xi + (VL * ivs), xo = xo + (VL * ovs),
            MAKE_VOLATILE_STRIDE(4, is), MAKE_VOLATILE_STRIDE(4, os)) {
            V T1, T2;
            T1 = LD(&(xi[0]), ivs, &(xi[0]));
            T2 = LD(&(xi[WS(is, 1)]), ivs, &(xi[WS(is, 1)]));
            ST(&(xo[WS(os, 1)]), VSUB(T1, T2), ovs, &(xo[WS(os, 1)]));
            ST(&(xo[0]), VADD(T1, T2), ovs, &(xo[0]));
        }
    }
    VLEAVE();
}

// dft/simd/common/t1fuv_10.c (non-FMA codelet, CUDA adaptation)
__device__ void t1fuv_10(R *ri, R *ii, const R *W, stride rs, INT mb, INT me, INT ms) {
    DVK(KP587785252, +0.587785252292473129168705954639072768597652438);
    DVK(KP951056516, +0.951056516295153572116439333379382143405698634);
    DVK(KP250000000, +0.250000000000000000000000000000000000000000000);
    DVK(KP559016994, +0.559016994374947424102293417182819058860154590);
    {
        INT m;
        R *x;
        x = ri;
        for (m = mb, W = W + (mb * ((TWVL / VL) * 18)); m < me;
             m = m + VL, x = x + (VL * ms), W = W + (TWVL * 18), MAKE_VOLATILE_STRIDE(10, rs)) {
            V Tr, TH, Tg, Tl, Tm, TA, TB, TJ, T5, Ta, Tb, TD, TE, TI, To;
            V Tq, Tp;
            To = LD(&(x[0]), ms, &(x[0]));
            Tp = LD(&(x[WS(rs, 5)]), ms, &(x[WS(rs, 1)]));
            Tq = BYTWJ(&(W[TWVL * 8]), Tp);
            Tr = VSUB(To, Tq);
            TH = VADD(To, Tq);
            {
                V Td, Tk, Tf, Ti;
                {
                    V Tc, Tj, Te, Th;
                    Tc = LD(&(x[WS(rs, 4)]), ms, &(x[0]));
                    Td = BYTWJ(&(W[TWVL * 6]), Tc);
                    Tj = LD(&(x[WS(rs, 1)]), ms, &(x[WS(rs, 1)]));
                    Tk = BYTWJ(&(W[0]), Tj);
                    Te = LD(&(x[WS(rs, 9)]), ms, &(x[WS(rs, 1)]));
                    Tf = BYTWJ(&(W[TWVL * 16]), Te);
                    Th = LD(&(x[WS(rs, 6)]), ms, &(x[0]));
                    Ti = BYTWJ(&(W[TWVL * 10]), Th);
                }
                Tg = VSUB(Td, Tf);
                Tl = VSUB(Ti, Tk);
                Tm = VADD(Tg, Tl);
                TA = VADD(Td, Tf);
                TB = VADD(Ti, Tk);
                TJ = VADD(TA, TB);
            }
            {
                V T2, T9, T4, T7;
                {
                    V T1, T8, T3, T6;
                    T1 = LD(&(x[WS(rs, 2)]), ms, &(x[0]));
                    T2 = BYTWJ(&(W[TWVL * 2]), T1);
                    T8 = LD(&(x[WS(rs, 3)]), ms, &(x[WS(rs, 1)]));
                    T9 = BYTWJ(&(W[TWVL * 4]), T8);
                    T3 = LD(&(x[WS(rs, 7)]), ms, &(x[WS(rs, 1)]));
                    T4 = BYTWJ(&(W[TWVL * 12]), T3);
                    T6 = LD(&(x[WS(rs, 8)]), ms, &(x[0]));
                    T7 = BYTWJ(&(W[TWVL * 14]), T6);
                }
                T5 = VSUB(T2, T4);
                Ta = VSUB(T7, T9);
                Tb = VADD(T5, Ta);
                TD = VADD(T2, T4);
                TE = VADD(T7, T9);
                TI = VADD(TD, TE);
            }
            {
                V Tn, Ts, Tt, Tx, Tz, Tv, Tw, Ty, Tu;
                Tn = VMUL(LDK(KP559016994), VSUB(Tb, Tm));
                Ts = VADD(Tb, Tm);
                Tt = VFNMS(LDK(KP250000000), Ts, Tr);
                Tv = VSUB(T5, Ta);
                Tw = VSUB(Tg, Tl);
                Tx = VBYI(VFMA(LDK(KP951056516), Tv, VMUL(LDK(KP587785252), Tw)));
                Tz = VBYI(VFNMS(LDK(KP587785252), Tv, VMUL(LDK(KP951056516), Tw)));
                ST(&(x[WS(rs, 5)]), VADD(Tr, Ts), ms, &(x[WS(rs, 1)]));
                Ty = VSUB(Tt, Tn);
                ST(&(x[WS(rs, 3)]), VSUB(Ty, Tz), ms, &(x[WS(rs, 1)]));
                ST(&(x[WS(rs, 7)]), VADD(Tz, Ty), ms, &(x[WS(rs, 1)]));
                Tu = VADD(Tn, Tt);
                ST(&(x[WS(rs, 1)]), VSUB(Tu, Tx), ms, &(x[WS(rs, 1)]));
                ST(&(x[WS(rs, 9)]), VADD(Tx, Tu), ms, &(x[WS(rs, 1)]));
            }
            {
                V TM, TK, TL, TG, TO, TC, TF, TP, TN;
                TM = VMUL(LDK(KP559016994), VSUB(TI, TJ));
                TK = VADD(TI, TJ);
                TL = VFNMS(LDK(KP250000000), TK, TH);
                TC = VSUB(TA, TB);
                TF = VSUB(TD, TE);
                TG = VBYI(VFNMS(LDK(KP587785252), TF, VMUL(LDK(KP951056516), TC)));
                TO = VBYI(VFMA(LDK(KP951056516), TF, VMUL(LDK(KP587785252), TC)));
                ST(&(x[0]), VADD(TH, TK), ms, &(x[0]));
                TP = VADD(TM, TL);
                ST(&(x[WS(rs, 4)]), VADD(TO, TP), ms, &(x[0]));
                ST(&(x[WS(rs, 6)]), VSUB(TP, TO), ms, &(x[0]));
                TN = VSUB(TL, TM);
                ST(&(x[WS(rs, 2)]), VADD(TG, TN), ms, &(x[0]));
                ST(&(x[WS(rs, 8)]), VSUB(TN, TG), ms, &(x[0]));
            }
        }
    }
    VLEAVE();
}

// dft/simd/common/t1fuv_5.c (non-FMA codelet, CUDA adaptation)
__device__ void t1fuv_5(R *ri, R *ii, const R *W, stride rs, INT mb, INT me, INT ms) {
    DVK(KP250000000, +0.250000000000000000000000000000000000000000000);
    DVK(KP559016994, +0.559016994374947424102293417182819058860154590);
    DVK(KP587785252, +0.587785252292473129168705954639072768597652438);
    DVK(KP951056516, +0.951056516295153572116439333379382143405698634);
    {
        INT m;
        R *x;
        x = ri;
        for (m = mb, W = W + (mb * ((TWVL / VL) * 8)); m < me;
             m = m + VL, x = x + (VL * ms), W = W + (TWVL * 8), MAKE_VOLATILE_STRIDE(5, rs)) {
            V Tc, Tg, Th, T5, Ta, Td;
            Tc = LD(&(x[0]), ms, &(x[0]));
            {
                V T2, T9, T4, T7;
                {
                    V T1, T8, T3, T6;
                    T1 = LD(&(x[WS(rs, 1)]), ms, &(x[WS(rs, 1)]));
                    T2 = BYTWJ(&(W[0]), T1);
                    T8 = LD(&(x[WS(rs, 3)]), ms, &(x[WS(rs, 1)]));
                    T9 = BYTWJ(&(W[TWVL * 4]), T8);
                    T3 = LD(&(x[WS(rs, 4)]), ms, &(x[0]));
                    T4 = BYTWJ(&(W[TWVL * 6]), T3);
                    T6 = LD(&(x[WS(rs, 2)]), ms, &(x[0]));
                    T7 = BYTWJ(&(W[TWVL * 2]), T6);
                }
                Tg = VSUB(T2, T4);
                Th = VSUB(T7, T9);
                T5 = VADD(T2, T4);
                Ta = VADD(T7, T9);
                Td = VADD(T5, Ta);
            }
            ST(&(x[0]), VADD(Tc, Td), ms, &(x[0]));
            {
                V Ti, Tj, Tf, Tk, Tb, Te;
                Ti = VBYI(VFMA(LDK(KP951056516), Tg, VMUL(LDK(KP587785252), Th)));
                Tj = VBYI(VFNMS(LDK(KP587785252), Tg, VMUL(LDK(KP951056516), Th)));
                Tb = VMUL(LDK(KP559016994), VSUB(T5, Ta));
                Te = VFNMS(LDK(KP250000000), Td, Tc);
                Tf = VADD(Tb, Te);
                Tk = VSUB(Te, Tb);
                ST(&(x[WS(rs, 1)]), VSUB(Tf, Ti), ms, &(x[WS(rs, 1)]));
                ST(&(x[WS(rs, 3)]), VSUB(Tk, Tj), ms, &(x[WS(rs, 1)]));
                ST(&(x[WS(rs, 4)]), VADD(Ti, Tf), ms, &(x[0]));
                ST(&(x[WS(rs, 2)]), VADD(Tj, Tk), ms, &(x[0]));
            }
        }
    }
    VLEAVE();
}

// dft/simd/common/q1fv_5.c (non-FMA codelet, CUDA adaptation)
__device__ void q1fv_5(R *ri, R *ii, const R *W, stride rs, stride vs, INT mb, INT me, INT ms) {
    DVK(KP250000000, +0.250000000000000000000000000000000000000000000);
    DVK(KP587785252, +0.587785252292473129168705954639072768597652438);
    DVK(KP951056516, +0.951056516295153572116439333379382143405698634);
    DVK(KP559016994, +0.559016994374947424102293417182819058860154590);
    {
        INT m;
        R *x;
        x = ri;
        for (m = mb, W = W + (mb * ((TWVL / VL) * 8)); m < me; m = m + VL, x = x + (VL * ms),
            W = W + (TWVL * 8), MAKE_VOLATILE_STRIDE(10, rs), MAKE_VOLATILE_STRIDE(10, vs)) {
            V T8, T7, Th, Te, T9, Ta, T1q, T1p, T1z, T1w, T1r, T1s, Ts, Tr, TB;
            V Ty, Tt, Tu, TM, TL, TV, TS, TN, TO, T16, T15, T1f, T1c, T17, T18;
            {
                V T6, Td, T3, Tc;
                T8 = LD(&(x[0]), ms, &(x[0]));
                {
                    V T4, T5, T1, T2;
                    T4 = LD(&(x[WS(rs, 2)]), ms, &(x[0]));
                    T5 = LD(&(x[WS(rs, 3)]), ms, &(x[WS(rs, 1)]));
                    T6 = VADD(T4, T5);
                    Td = VSUB(T4, T5);
                    T1 = LD(&(x[WS(rs, 1)]), ms, &(x[WS(rs, 1)]));
                    T2 = LD(&(x[WS(rs, 4)]), ms, &(x[0]));
                    T3 = VADD(T1, T2);
                    Tc = VSUB(T1, T2);
                }
                T7 = VMUL(LDK(KP559016994), VSUB(T3, T6));
                Th = VBYI(VFNMS(LDK(KP587785252), Tc, VMUL(LDK(KP951056516), Td)));
                Te = VBYI(VFMA(LDK(KP951056516), Tc, VMUL(LDK(KP587785252), Td)));
                T9 = VADD(T3, T6);
                Ta = VFNMS(LDK(KP250000000), T9, T8);
            }
            {
                V T1o, T1v, T1l, T1u;
                T1q = LD(&(x[WS(vs, 4)]), ms, &(x[WS(vs, 4)]));
                {
                    V T1m, T1n, T1j, T1k;
                    T1m = LD(&(x[WS(vs, 4) + WS(rs, 2)]), ms, &(x[WS(vs, 4)]));
                    T1n = LD(&(x[WS(vs, 4) + WS(rs, 3)]), ms, &(x[WS(vs, 4) + WS(rs, 1)]));
                    T1o = VADD(T1m, T1n);
                    T1v = VSUB(T1m, T1n);
                    T1j = LD(&(x[WS(vs, 4) + WS(rs, 1)]), ms, &(x[WS(vs, 4) + WS(rs, 1)]));
                    T1k = LD(&(x[WS(vs, 4) + WS(rs, 4)]), ms, &(x[WS(vs, 4)]));
                    T1l = VADD(T1j, T1k);
                    T1u = VSUB(T1j, T1k);
                }
                T1p = VMUL(LDK(KP559016994), VSUB(T1l, T1o));
                T1z = VBYI(VFNMS(LDK(KP587785252), T1u, VMUL(LDK(KP951056516), T1v)));
                T1w = VBYI(VFMA(LDK(KP951056516), T1u, VMUL(LDK(KP587785252), T1v)));
                T1r = VADD(T1l, T1o);
                T1s = VFNMS(LDK(KP250000000), T1r, T1q);
            }
            {
                V Tq, Tx, Tn, Tw;
                Ts = LD(&(x[WS(vs, 1)]), ms, &(x[WS(vs, 1)]));
                {
                    V To, Tp, Tl, Tm;
                    To = LD(&(x[WS(vs, 1) + WS(rs, 2)]), ms, &(x[WS(vs, 1)]));
                    Tp = LD(&(x[WS(vs, 1) + WS(rs, 3)]), ms, &(x[WS(vs, 1) + WS(rs, 1)]));
                    Tq = VADD(To, Tp);
                    Tx = VSUB(To, Tp);
                    Tl = LD(&(x[WS(vs, 1) + WS(rs, 1)]), ms, &(x[WS(vs, 1) + WS(rs, 1)]));
                    Tm = LD(&(x[WS(vs, 1) + WS(rs, 4)]), ms, &(x[WS(vs, 1)]));
                    Tn = VADD(Tl, Tm);
                    Tw = VSUB(Tl, Tm);
                }
                Tr = VMUL(LDK(KP559016994), VSUB(Tn, Tq));
                TB = VBYI(VFNMS(LDK(KP587785252), Tw, VMUL(LDK(KP951056516), Tx)));
                Ty = VBYI(VFMA(LDK(KP951056516), Tw, VMUL(LDK(KP587785252), Tx)));
                Tt = VADD(Tn, Tq);
                Tu = VFNMS(LDK(KP250000000), Tt, Ts);
            }
            {
                V TK, TR, TH, TQ;
                TM = LD(&(x[WS(vs, 2)]), ms, &(x[WS(vs, 2)]));
                {
                    V TI, TJ, TF, TG;
                    TI = LD(&(x[WS(vs, 2) + WS(rs, 2)]), ms, &(x[WS(vs, 2)]));
                    TJ = LD(&(x[WS(vs, 2) + WS(rs, 3)]), ms, &(x[WS(vs, 2) + WS(rs, 1)]));
                    TK = VADD(TI, TJ);
                    TR = VSUB(TI, TJ);
                    TF = LD(&(x[WS(vs, 2) + WS(rs, 1)]), ms, &(x[WS(vs, 2) + WS(rs, 1)]));
                    TG = LD(&(x[WS(vs, 2) + WS(rs, 4)]), ms, &(x[WS(vs, 2)]));
                    TH = VADD(TF, TG);
                    TQ = VSUB(TF, TG);
                }
                TL = VMUL(LDK(KP559016994), VSUB(TH, TK));
                TV = VBYI(VFNMS(LDK(KP587785252), TQ, VMUL(LDK(KP951056516), TR)));
                TS = VBYI(VFMA(LDK(KP951056516), TQ, VMUL(LDK(KP587785252), TR)));
                TN = VADD(TH, TK);
                TO = VFNMS(LDK(KP250000000), TN, TM);
            }
            {
                V T14, T1b, T11, T1a;
                T16 = LD(&(x[WS(vs, 3)]), ms, &(x[WS(vs, 3)]));
                {
                    V T12, T13, TZ, T10;
                    T12 = LD(&(x[WS(vs, 3) + WS(rs, 2)]), ms, &(x[WS(vs, 3)]));
                    T13 = LD(&(x[WS(vs, 3) + WS(rs, 3)]), ms, &(x[WS(vs, 3) + WS(rs, 1)]));
                    T14 = VADD(T12, T13);
                    T1b = VSUB(T12, T13);
                    TZ = LD(&(x[WS(vs, 3) + WS(rs, 1)]), ms, &(x[WS(vs, 3) + WS(rs, 1)]));
                    T10 = LD(&(x[WS(vs, 3) + WS(rs, 4)]), ms, &(x[WS(vs, 3)]));
                    T11 = VADD(TZ, T10);
                    T1a = VSUB(TZ, T10);
                }
                T15 = VMUL(LDK(KP559016994), VSUB(T11, T14));
                T1f = VBYI(VFNMS(LDK(KP587785252), T1a, VMUL(LDK(KP951056516), T1b)));
                T1c = VBYI(VFMA(LDK(KP951056516), T1a, VMUL(LDK(KP587785252), T1b)));
                T17 = VADD(T11, T14);
                T18 = VFNMS(LDK(KP250000000), T17, T16);
            }
            ST(&(x[0]), VADD(T8, T9), ms, &(x[0]));
            ST(&(x[WS(rs, 4)]), VADD(T1q, T1r), ms, &(x[0]));
            ST(&(x[WS(rs, 2)]), VADD(TM, TN), ms, &(x[0]));
            ST(&(x[WS(rs, 3)]), VADD(T16, T17), ms, &(x[WS(rs, 1)]));
            ST(&(x[WS(rs, 1)]), VADD(Ts, Tt), ms, &(x[WS(rs, 1)]));
            {
                V Tj, Tk, Ti, T1B, T1C, T1A;
                Ti = VSUB(Ta, T7);
                Tj = BYTWJ(&(W[TWVL * 2]), VADD(Th, Ti));
                Tk = BYTWJ(&(W[TWVL * 4]), VSUB(Ti, Th));
                ST(&(x[WS(vs, 2)]), Tj, ms, &(x[WS(vs, 2)]));
                ST(&(x[WS(vs, 3)]), Tk, ms, &(x[WS(vs, 3)]));
                T1A = VSUB(T1s, T1p);
                T1B = BYTWJ(&(W[TWVL * 2]), VADD(T1z, T1A));
                T1C = BYTWJ(&(W[TWVL * 4]), VSUB(T1A, T1z));
                ST(&(x[WS(vs, 2) + WS(rs, 4)]), T1B, ms, &(x[WS(vs, 2)]));
                ST(&(x[WS(vs, 3) + WS(rs, 4)]), T1C, ms, &(x[WS(vs, 3)]));
            }
            {
                V T1h, T1i, T1g, TD, TE, TC;
                T1g = VSUB(T18, T15);
                T1h = BYTWJ(&(W[TWVL * 2]), VADD(T1f, T1g));
                T1i = BYTWJ(&(W[TWVL * 4]), VSUB(T1g, T1f));
                ST(&(x[WS(vs, 2) + WS(rs, 3)]), T1h, ms, &(x[WS(vs, 2) + WS(rs, 1)]));
                ST(&(x[WS(vs, 3) + WS(rs, 3)]), T1i, ms, &(x[WS(vs, 3) + WS(rs, 1)]));
                TC = VSUB(Tu, Tr);
                TD = BYTWJ(&(W[TWVL * 2]), VADD(TB, TC));
                TE = BYTWJ(&(W[TWVL * 4]), VSUB(TC, TB));
                ST(&(x[WS(vs, 2) + WS(rs, 1)]), TD, ms, &(x[WS(vs, 2) + WS(rs, 1)]));
                ST(&(x[WS(vs, 3) + WS(rs, 1)]), TE, ms, &(x[WS(vs, 3) + WS(rs, 1)]));
            }
            {
                V TX, TY, TW, TT, TU, TP;
                TW = VSUB(TO, TL);
                TX = BYTWJ(&(W[TWVL * 2]), VADD(TV, TW));
                TY = BYTWJ(&(W[TWVL * 4]), VSUB(TW, TV));
                ST(&(x[WS(vs, 2) + WS(rs, 2)]), TX, ms, &(x[WS(vs, 2)]));
                ST(&(x[WS(vs, 3) + WS(rs, 2)]), TY, ms, &(x[WS(vs, 3)]));
                TP = VADD(TL, TO);
                TT = BYTWJ(&(W[0]), VSUB(TP, TS));
                TU = BYTWJ(&(W[TWVL * 6]), VADD(TS, TP));
                ST(&(x[WS(vs, 1) + WS(rs, 2)]), TT, ms, &(x[WS(vs, 1)]));
                ST(&(x[WS(vs, 4) + WS(rs, 2)]), TU, ms, &(x[WS(vs, 4)]));
            }
            {
                V Tf, Tg, Tb, Tz, TA, Tv;
                Tb = VADD(T7, Ta);
                Tf = BYTWJ(&(W[0]), VSUB(Tb, Te));
                Tg = BYTWJ(&(W[TWVL * 6]), VADD(Te, Tb));
                ST(&(x[WS(vs, 1)]), Tf, ms, &(x[WS(vs, 1)]));
                ST(&(x[WS(vs, 4)]), Tg, ms, &(x[WS(vs, 4)]));
                Tv = VADD(Tr, Tu);
                Tz = BYTWJ(&(W[0]), VSUB(Tv, Ty));
                TA = BYTWJ(&(W[TWVL * 6]), VADD(Ty, Tv));
                ST(&(x[WS(vs, 1) + WS(rs, 1)]), Tz, ms, &(x[WS(vs, 1) + WS(rs, 1)]));
                ST(&(x[WS(vs, 4) + WS(rs, 1)]), TA, ms, &(x[WS(vs, 4) + WS(rs, 1)]));
            }
            {
                V T1d, T1e, T19, T1x, T1y, T1t;
                T19 = VADD(T15, T18);
                T1d = BYTWJ(&(W[0]), VSUB(T19, T1c));
                T1e = BYTWJ(&(W[TWVL * 6]), VADD(T1c, T19));
                ST(&(x[WS(vs, 1) + WS(rs, 3)]), T1d, ms, &(x[WS(vs, 1) + WS(rs, 1)]));
                ST(&(x[WS(vs, 4) + WS(rs, 3)]), T1e, ms, &(x[WS(vs, 4) + WS(rs, 1)]));
                T1t = VADD(T1p, T1s);
                T1x = BYTWJ(&(W[0]), VSUB(T1t, T1w));
                T1y = BYTWJ(&(W[TWVL * 6]), VADD(T1w, T1t));
                ST(&(x[WS(vs, 1) + WS(rs, 4)]), T1x, ms, &(x[WS(vs, 1)]));
                ST(&(x[WS(vs, 4) + WS(rs, 4)]), T1y, ms, &(x[WS(vs, 4)]));
            }
        }
    }
    VLEAVE();
}

// rdft/simd/common/hc2cfdftv_2.c (non-FMA codelet, CUDA adaptation)
__device__ void hc2cfdftv_2(R *Rp, R *Ip, R *Rm, R *Im, const R *W, stride rs, INT mb, INT me,
                            INT ms) {
    DVK(KP500000000, +0.500000000000000000000000000000000000000000000);
    {
        INT m;
        for (m = mb, W = W + ((mb - 1) * ((TWVL / VL) * 2)); m < me; m = m + VL,
            Rp = Rp + (VL * ms), Ip = Ip + (VL * ms), Rm = Rm - (VL * ms), Im = Im - (VL * ms),
            W = W + (TWVL * 2), MAKE_VOLATILE_STRIDE(8, rs)) {
            V T4, T6, T1, T3, T2, T5, T7, T8;
            T1 = LD(&(Rp[0]), ms, &(Rp[0]));
            T2 = LD(&(Rm[0]), -ms, &(Rm[0]));
            T3 = VCONJ(T2);
            T4 = VADD(T1, T3);
            T5 = LDW(&(W[0]));
            T6 = VZMULIJ(T5, VSUB(T3, T1));
            T7 = VCONJ(VMUL(LDK(KP500000000), VSUB(T4, T6)));
            ST(&(Rm[0]), T7, -ms, &(Rm[0]));
            T8 = VMUL(LDK(KP500000000), VADD(T4, T6));
            ST(&(Rp[0]), T8, ms, &(Rp[0]));
        }
    }
    VLEAVE();
}

// rdft/scalar/r2cf/r2cf_2.c (non-FMA codelet, CUDA adaptation)
__device__ void r2cf_2(R *R0, R *R1, R *Cr, R *Ci, stride rs, stride csr, stride csi, INT v,
                       INT ivs, INT ovs) {
    {
        INT i;
        for (i = v; i > 0; i = i - 1, R0 = R0 + ivs, R1 = R1 + ivs, Cr = Cr + ovs, Ci = Ci + ovs,
            MAKE_VOLATILE_STRIDE(8, rs), MAKE_VOLATILE_STRIDE(8, csr),
            MAKE_VOLATILE_STRIDE(8, csi)) {
            E T1, T2;
            T1 = R0[0];
            T2 = R1[0];
            Cr[WS(csr, 1)] = T1 - T2;
            Cr[0] = T1 + T2;
        }
    }
}

// rdft/simd/common/hc2cfdftv_10.c (non-FMA codelet, CUDA adaptation)
__device__ void hc2cfdftv_10(R *Rp, R *Ip, R *Rm, R *Im, const R *W, stride rs, INT mb, INT me,
                             INT ms) {
    DVK(KP125000000, +0.125000000000000000000000000000000000000000000);
    DVK(KP279508497, +0.279508497187473712051146708591409529430077295);
    DVK(KP587785252, +0.587785252292473129168705954639072768597652438);
    DVK(KP951056516, +0.951056516295153572116439333379382143405698634);
    DVK(KP500000000, +0.500000000000000000000000000000000000000000000);
    {
        INT m;
        for (m = mb, W = W + ((mb - 1) * ((TWVL / VL) * 18)); m < me; m = m + VL,
            Rp = Rp + (VL * ms), Ip = Ip + (VL * ms), Rm = Rm - (VL * ms), Im = Im - (VL * ms),
            W = W + (TWVL * 18), MAKE_VOLATILE_STRIDE(40, rs)) {
            V Tl, Tt, Tu, TY, TZ, T10, Tz, TE, TF, TV, TW, TX, Ta, TU, TN;
            V TR, TH, TQ, TK, TL, TM, TI, TG, TJ, TT, TO, TP, TS, T18, T1c;
            V T12, T1b, T15, T16, T17, T14, T11, T13, T1e, T19, T1a, T1d;
            {
                V T1, T3, Ty, T8, T7, TB, Tf, Ts, Tk, Tw, Tq, TD, T2, Tx, T6;
                V TA, Tc, Te, Td, Tb, Tr, Tj, Ti, Th, Tg, Tv, Tn, Tp, To, Tm;
                V TC, T4, T9, T5;
                T1 = LD(&(Rp[0]), ms, &(Rp[0]));
                T2 = LD(&(Rm[0]), -ms, &(Rm[0]));
                T3 = VCONJ(T2);
                Tx = LDW(&(W[0]));
                Ty = VZMULIJ(Tx, VSUB(T3, T1));
                T8 = LD(&(Rp[WS(rs, 2)]), ms, &(Rp[0]));
                T6 = LD(&(Rm[WS(rs, 2)]), -ms, &(Rm[0]));
                T7 = VCONJ(T6);
                TA = LDW(&(W[TWVL * 6]));
                TB = VZMULJ(TA, VADD(T7, T8));
                Tc = LD(&(Rp[WS(rs, 1)]), ms, &(Rp[WS(rs, 1)]));
                Td = LD(&(Rm[WS(rs, 1)]), -ms, &(Rm[WS(rs, 1)]));
                Te = VCONJ(Td);
                Tb = LDW(&(W[TWVL * 2]));
                Tf = VZMULJ(Tb, VADD(Tc, Te));
                Tr = LDW(&(W[TWVL * 4]));
                Ts = VZMULIJ(Tr, VSUB(Te, Tc));
                Tj = LD(&(Rp[WS(rs, 3)]), ms, &(Rp[WS(rs, 1)]));
                Th = LD(&(Rm[WS(rs, 3)]), -ms, &(Rm[WS(rs, 1)]));
                Ti = VCONJ(Th);
                Tg = LDW(&(W[TWVL * 12]));
                Tk = VZMULIJ(Tg, VSUB(Ti, Tj));
                Tv = LDW(&(W[TWVL * 10]));
                Tw = VZMULJ(Tv, VADD(Ti, Tj));
                Tn = LD(&(Rp[WS(rs, 4)]), ms, &(Rp[0]));
                To = LD(&(Rm[WS(rs, 4)]), -ms, &(Rm[0]));
                Tp = VCONJ(To);
                Tm = LDW(&(W[TWVL * 14]));
                Tq = VZMULJ(Tm, VADD(Tn, Tp));
                TC = LDW(&(W[TWVL * 16]));
                TD = VZMULIJ(TC, VSUB(Tp, Tn));
                Tl = VSUB(Tf, Tk);
                Tt = VSUB(Tq, Ts);
                Tu = VADD(Tl, Tt);
                TY = VADD(Ty, Tw);
                TZ = VADD(TB, TD);
                T10 = VADD(TY, TZ);
                Tz = VSUB(Tw, Ty);
                TE = VSUB(TB, TD);
                TF = VADD(Tz, TE);
                TV = VADD(Tf, Tk);
                TW = VADD(Ts, Tq);
                TX = VADD(TV, TW);
                T4 = VADD(T1, T3);
                T5 = LDW(&(W[TWVL * 8]));
                T9 = VZMULIJ(T5, VSUB(T7, T8));
                Ta = VSUB(T4, T9);
                TU = VADD(T4, T9);
            }
            TL = VSUB(Tl, Tt);
            TM = VSUB(TE, Tz);
            TN = VMUL(LDK(KP500000000),
                      VBYI(VFMA(LDK(KP951056516), TL, VMUL(LDK(KP587785252), TM))));
            TR = VMUL(LDK(KP500000000),
                      VBYI(VFNMS(LDK(KP587785252), TL, VMUL(LDK(KP951056516), TM))));
            TI = VMUL(LDK(KP279508497), VSUB(Tu, TF));
            TG = VADD(Tu, TF);
            TJ = VFNMS(LDK(KP125000000), TG, VMUL(LDK(KP500000000), Ta));
            TH = VCONJ(VMUL(LDK(KP500000000), VADD(Ta, TG)));
            TQ = VSUB(TJ, TI);
            TK = VADD(TI, TJ);
            ST(&(Rm[WS(rs, 4)]), TH, -ms, &(Rm[0]));
            TT = VCONJ(VADD(TQ, TR));
            ST(&(Rm[WS(rs, 2)]), TT, -ms, &(Rm[0]));
            TO = VSUB(TK, TN);
            ST(&(Rp[WS(rs, 1)]), TO, ms, &(Rp[WS(rs, 1)]));
            TP = VCONJ(VADD(TK, TN));
            ST(&(Rm[0]), TP, -ms, &(Rm[0]));
            TS = VSUB(TQ, TR);
            ST(&(Rp[WS(rs, 3)]), TS, ms, &(Rp[WS(rs, 1)]));
            T16 = VSUB(TZ, TY);
            T17 = VSUB(TV, TW);
            T18 = VMUL(LDK(KP500000000),
                       VBYI(VFNMS(LDK(KP587785252), T17, VMUL(LDK(KP951056516), T16))));
            T1c = VMUL(LDK(KP500000000),
                       VBYI(VFMA(LDK(KP951056516), T17, VMUL(LDK(KP587785252), T16))));
            T14 = VMUL(LDK(KP279508497), VSUB(TX, T10));
            T11 = VADD(TX, T10);
            T13 = VFNMS(LDK(KP125000000), T11, VMUL(LDK(KP500000000), TU));
            T12 = VMUL(LDK(KP500000000), VADD(TU, T11));
            T1b = VADD(T14, T13);
            T15 = VSUB(T13, T14);
            ST(&(Rp[0]), T12, ms, &(Rp[0]));
            T1e = VADD(T1b, T1c);
            ST(&(Rp[WS(rs, 4)]), T1e, ms, &(Rp[0]));
            T19 = VCONJ(VSUB(T15, T18));
            ST(&(Rm[WS(rs, 1)]), T19, -ms, &(Rm[WS(rs, 1)]));
            T1a = VADD(T15, T18);
            ST(&(Rp[WS(rs, 2)]), T1a, ms, &(Rp[0]));
            T1d = VCONJ(VSUB(T1b, T1c));
            ST(&(Rm[WS(rs, 3)]), T1d, -ms, &(Rm[WS(rs, 1)]));
        }
    }
    VLEAVE();
}

// rdft/scalar/r2cf/r2cf_10.c (non-FMA codelet, CUDA adaptation)
__device__ void r2cf_10(R *R0, R *R1, R *Cr, R *Ci, stride rs, stride csr, stride csi, INT v,
                        INT ivs, INT ovs) {
    DK(KP250000000, +0.250000000000000000000000000000000000000000000);
    DK(KP559016994, +0.559016994374947424102293417182819058860154590);
    DK(KP951056516, +0.951056516295153572116439333379382143405698634);
    DK(KP587785252, +0.587785252292473129168705954639072768597652438);
    {
        INT i;
        for (i = v; i > 0; i = i - 1, R0 = R0 + ivs, R1 = R1 + ivs, Cr = Cr + ovs, Ci = Ci + ovs,
            MAKE_VOLATILE_STRIDE(40, rs), MAKE_VOLATILE_STRIDE(40, csr),
            MAKE_VOLATILE_STRIDE(40, csi)) {
            E Ti, Tt, Ta, Tn, Td, To, Te, Tv, T3, Tq, T6, Tr, T7, Tu, Tg;
            E Th;
            Tg = R0[0];
            Th = R1[WS(rs, 2)];
            Ti = Tg - Th;
            Tt = Tg + Th;
            {
                E T8, T9, Tb, Tc;
                T8 = R0[WS(rs, 2)];
                T9 = R1[WS(rs, 4)];
                Ta = T8 - T9;
                Tn = T8 + T9;
                Tb = R0[WS(rs, 3)];
                Tc = R1[0];
                Td = Tb - Tc;
                To = Tb + Tc;
            }
            Te = Ta + Td;
            Tv = Tn + To;
            {
                E T1, T2, T4, T5;
                T1 = R0[WS(rs, 1)];
                T2 = R1[WS(rs, 3)];
                T3 = T1 - T2;
                Tq = T1 + T2;
                T4 = R0[WS(rs, 4)];
                T5 = R1[WS(rs, 1)];
                T6 = T4 - T5;
                Tr = T4 + T5;
            }
            T7 = T3 + T6;
            Tu = Tq + Tr;
            {
                E Tl, Tm, Tf, Tj, Tk;
                Tl = Td - Ta;
                Tm = T3 - T6;
                Ci[WS(csi, 1)] = FNMS(KP951056516, Tm, KP587785252 * Tl);
                Ci[WS(csi, 3)] = FMA(KP587785252, Tm, KP951056516 * Tl);
                Tf = KP559016994 * (T7 - Te);
                Tj = T7 + Te;
                Tk = FNMS(KP250000000, Tj, Ti);
                Cr[WS(csr, 1)] = Tf + Tk;
                Cr[WS(csr, 5)] = Ti + Tj;
                Cr[WS(csr, 3)] = Tk - Tf;
            }
            {
                E Tp, Ts, Ty, Tw, Tx;
                Tp = Tn - To;
                Ts = Tq - Tr;
                Ci[WS(csi, 2)] = FNMS(KP587785252, Ts, KP951056516 * Tp);
                Ci[WS(csi, 4)] = FMA(KP951056516, Ts, KP587785252 * Tp);
                Ty = KP559016994 * (Tu - Tv);
                Tw = Tu + Tv;
                Tx = FNMS(KP250000000, Tw, Tt);
                Cr[WS(csr, 2)] = Tx - Ty;
                Cr[0] = Tt + Tw;
                Cr[WS(csr, 4)] = Ty + Tx;
            }
        }
    }
}

// dft/simd/common/n1bv_15.c (non-FMA codelet, CUDA adaptation)
__device__ void n1bv_15(const R *ri, const R *ii, R *ro, R *io, stride is, stride os, INT v,
                        INT ivs, INT ovs) {
    DVK(KP216506350, +0.216506350946109661690930792688234045867850657);
    DVK(KP509036960, +0.509036960455127183450980863393907648510733164);
    DVK(KP823639103, +0.823639103546331925877420039278190003029660514);
    DVK(KP951056516, +0.951056516295153572116439333379382143405698634);
    DVK(KP587785252, +0.587785252292473129168705954639072768597652438);
    DVK(KP250000000, +0.250000000000000000000000000000000000000000000);
    DVK(KP559016994, +0.559016994374947424102293417182819058860154590);
    DVK(KP866025403, +0.866025403784438646763723170752936183471402627);
    DVK(KP484122918, +0.484122918275927110647408174972799951354115213);
    DVK(KP500000000, +0.500000000000000000000000000000000000000000000);
    {
        INT i;
        const R *xi;
        R *xo;
        xi = ii;
        xo = io;
        for (i = v; i > 0; i = i - VL, xi = xi + (VL * ivs), xo = xo + (VL * ovs),
            MAKE_VOLATILE_STRIDE(30, is), MAKE_VOLATILE_STRIDE(30, os)) {
            V Ti, T11, TH, Ts, TL, TM, Tz, TC, TD, TI, T12, T13, T14, T15, T16;
            V T17, Tf, Tj, TZ, T10;
            {
                V TF, Tg, Th, TG;
                TF = LD(&(xi[0]), ivs, &(xi[0]));
                Tg = LD(&(xi[WS(is, 5)]), ivs, &(xi[WS(is, 1)]));
                Th = LD(&(xi[WS(is, 10)]), ivs, &(xi[0]));
                TG = VADD(Tg, Th);
                Ti = VSUB(Tg, Th);
                T11 = VADD(TF, TG);
                TH = VFNMS(LDK(KP500000000), TG, TF);
            }
            {
                V Tm, Tn, T3, To, Tw, Tx, Td, Ty, Tp, Tq, T6, Tr, Tt, Tu, Ta;
                V Tv, T7, Te;
                {
                    V T1, T2, Tb, Tc;
                    Tm = LD(&(xi[WS(is, 3)]), ivs, &(xi[WS(is, 1)]));
                    T1 = LD(&(xi[WS(is, 8)]), ivs, &(xi[0]));
                    T2 = LD(&(xi[WS(is, 13)]), ivs, &(xi[WS(is, 1)]));
                    Tn = VADD(T1, T2);
                    T3 = VSUB(T1, T2);
                    To = VFNMS(LDK(KP500000000), Tn, Tm);
                    Tw = LD(&(xi[WS(is, 9)]), ivs, &(xi[WS(is, 1)]));
                    Tb = LD(&(xi[WS(is, 14)]), ivs, &(xi[0]));
                    Tc = LD(&(xi[WS(is, 4)]), ivs, &(xi[0]));
                    Tx = VADD(Tb, Tc);
                    Td = VSUB(Tb, Tc);
                    Ty = VFNMS(LDK(KP500000000), Tx, Tw);
                }
                {
                    V T4, T5, T8, T9;
                    Tp = LD(&(xi[WS(is, 12)]), ivs, &(xi[0]));
                    T4 = LD(&(xi[WS(is, 2)]), ivs, &(xi[0]));
                    T5 = LD(&(xi[WS(is, 7)]), ivs, &(xi[WS(is, 1)]));
                    Tq = VADD(T4, T5);
                    T6 = VSUB(T4, T5);
                    Tr = VFNMS(LDK(KP500000000), Tq, Tp);
                    Tt = LD(&(xi[WS(is, 6)]), ivs, &(xi[0]));
                    T8 = LD(&(xi[WS(is, 11)]), ivs, &(xi[WS(is, 1)]));
                    T9 = LD(&(xi[WS(is, 1)]), ivs, &(xi[WS(is, 1)]));
                    Tu = VADD(T8, T9);
                    Ta = VSUB(T8, T9);
                    Tv = VFNMS(LDK(KP500000000), Tu, Tt);
                }
                Ts = VSUB(To, Tr);
                TL = VSUB(T3, T6);
                TM = VSUB(Ta, Td);
                Tz = VSUB(Tv, Ty);
                TC = VADD(To, Tr);
                TD = VADD(Tv, Ty);
                TI = VADD(TC, TD);
                T12 = VADD(Tm, Tn);
                T13 = VADD(Tp, Tq);
                T14 = VADD(T12, T13);
                T15 = VADD(Tt, Tu);
                T16 = VADD(Tw, Tx);
                T17 = VADD(T15, T16);
                T7 = VADD(T3, T6);
                Te = VADD(Ta, Td);
                Tf = VMUL(LDK(KP484122918), VSUB(T7, Te));
                Tj = VADD(T7, Te);
            }
            TZ = VADD(TH, TI);
            T10 = VBYI(VMUL(LDK(KP866025403), VADD(Ti, Tj)));
            ST(&(xo[WS(os, 5)]), VSUB(TZ, T10), ovs, &(xo[WS(os, 1)]));
            ST(&(xo[WS(os, 10)]), VADD(T10, TZ), ovs, &(xo[0]));
            {
                V T1a, T18, T19, T1e, T1f, T1c, T1d, T1g, T1b;
                T1a = VMUL(LDK(KP559016994), VSUB(T14, T17));
                T18 = VADD(T14, T17);
                T19 = VFNMS(LDK(KP250000000), T18, T11);
                T1c = VSUB(T12, T13);
                T1d = VSUB(T15, T16);
                T1e = VBYI(VFNMS(LDK(KP951056516), T1d, VMUL(LDK(KP587785252), T1c)));
                T1f = VBYI(VFMA(LDK(KP951056516), T1c, VMUL(LDK(KP587785252), T1d)));
                ST(&(xo[0]), VADD(T11, T18), ovs, &(xo[0]));
                T1g = VADD(T1a, T19);
                ST(&(xo[WS(os, 6)]), VADD(T1f, T1g), ovs, &(xo[0]));
                ST(&(xo[WS(os, 9)]), VSUB(T1g, T1f), ovs, &(xo[WS(os, 1)]));
                T1b = VSUB(T19, T1a);
                ST(&(xo[WS(os, 3)]), VSUB(T1b, T1e), ovs, &(xo[WS(os, 1)]));
                ST(&(xo[WS(os, 12)]), VADD(T1e, T1b), ovs, &(xo[0]));
            }
            {
                V TA, TN, TU, TS, Tl, TR, TK, TV, Tk, TE, TJ;
                TA = VFMA(LDK(KP951056516), Ts, VMUL(LDK(KP587785252), Tz));
                TN = VFMA(LDK(KP823639103), TL, VMUL(LDK(KP509036960), TM));
                TU = VFNMS(LDK(KP823639103), TM, VMUL(LDK(KP509036960), TL));
                TS = VFNMS(LDK(KP951056516), Tz, VMUL(LDK(KP587785252), Ts));
                Tk = VFNMS(LDK(KP216506350), Tj, VMUL(LDK(KP866025403), Ti));
                Tl = VADD(Tf, Tk);
                TR = VSUB(Tf, Tk);
                TE = VMUL(LDK(KP559016994), VSUB(TC, TD));
                TJ = VFNMS(LDK(KP250000000), TI, TH);
                TK = VADD(TE, TJ);
                TV = VSUB(TJ, TE);
                {
                    V TB, TO, TX, TY;
                    TB = VBYI(VADD(Tl, TA));
                    TO = VSUB(TK, TN);
                    ST(&(xo[WS(os, 1)]), VADD(TB, TO), ovs, &(xo[WS(os, 1)]));
                    ST(&(xo[WS(os, 14)]), VSUB(TO, TB), ovs, &(xo[0]));
                    TX = VBYI(VSUB(TS, TR));
                    TY = VSUB(TV, TU);
                    ST(&(xo[WS(os, 7)]), VADD(TX, TY), ovs, &(xo[WS(os, 1)]));
                    ST(&(xo[WS(os, 8)]), VSUB(TY, TX), ovs, &(xo[0]));
                }
                {
                    V TP, TQ, TT, TW;
                    TP = VBYI(VSUB(Tl, TA));
                    TQ = VADD(TN, TK);
                    ST(&(xo[WS(os, 4)]), VADD(TP, TQ), ovs, &(xo[0]));
                    ST(&(xo[WS(os, 11)]), VSUB(TQ, TP), ovs, &(xo[WS(os, 1)]));
                    TT = VBYI(VADD(TR, TS));
                    TW = VADD(TU, TV);
                    ST(&(xo[WS(os, 2)]), VADD(TT, TW), ovs, &(xo[0]));
                    ST(&(xo[WS(os, 13)]), VSUB(TW, TT), ovs, &(xo[WS(os, 1)]));
                }
            }
        }
    }
    VLEAVE();
}

// dft/simd/common/n1bv_5.c (non-FMA codelet, CUDA adaptation)
__device__ void n1bv_5(const R *ri, const R *ii, R *ro, R *io, stride is, stride os, INT v, INT ivs,
                       INT ovs) {
    DVK(KP250000000, +0.250000000000000000000000000000000000000000000);
    DVK(KP587785252, +0.587785252292473129168705954639072768597652438);
    DVK(KP951056516, +0.951056516295153572116439333379382143405698634);
    DVK(KP559016994, +0.559016994374947424102293417182819058860154590);
    {
        INT i;
        const R *xi;
        R *xo;
        xi = ii;
        xo = io;
        for (i = v; i > 0; i = i - VL, xi = xi + (VL * ivs), xo = xo + (VL * ovs),
            MAKE_VOLATILE_STRIDE(10, is), MAKE_VOLATILE_STRIDE(10, os)) {
            V Tb, T3, Tc, T6, Ta;
            Tb = LD(&(xi[0]), ivs, &(xi[0]));
            {
                V T1, T2, T8, T4, T5, T9;
                T1 = LD(&(xi[WS(is, 1)]), ivs, &(xi[WS(is, 1)]));
                T2 = LD(&(xi[WS(is, 4)]), ivs, &(xi[0]));
                T8 = VADD(T1, T2);
                T4 = LD(&(xi[WS(is, 2)]), ivs, &(xi[0]));
                T5 = LD(&(xi[WS(is, 3)]), ivs, &(xi[WS(is, 1)]));
                T9 = VADD(T4, T5);
                T3 = VSUB(T1, T2);
                Tc = VADD(T8, T9);
                T6 = VSUB(T4, T5);
                Ta = VMUL(LDK(KP559016994), VSUB(T8, T9));
            }
            ST(&(xo[0]), VADD(Tb, Tc), ovs, &(xo[0]));
            {
                V T7, Tf, Te, Tg, Td;
                T7 = VBYI(VFMA(LDK(KP951056516), T3, VMUL(LDK(KP587785252), T6)));
                Tf = VBYI(VFNMS(LDK(KP951056516), T6, VMUL(LDK(KP587785252), T3)));
                Td = VFNMS(LDK(KP250000000), Tc, Tb);
                Te = VADD(Ta, Td);
                Tg = VSUB(Td, Ta);
                ST(&(xo[WS(os, 1)]), VADD(T7, Te), ovs, &(xo[WS(os, 1)]));
                ST(&(xo[WS(os, 3)]), VSUB(Tg, Tf), ovs, &(xo[WS(os, 1)]));
                ST(&(xo[WS(os, 4)]), VSUB(Te, T7), ovs, &(xo[0]));
                ST(&(xo[WS(os, 2)]), VADD(Tf, Tg), ovs, &(xo[0]));
            }
        }
    }
    VLEAVE();
}

// dft/simd/common/n1bv_2.c (non-FMA codelet, CUDA adaptation)
__device__ void n1bv_2(const R *ri, const R *ii, R *ro, R *io, stride is, stride os, INT v, INT ivs,
                       INT ovs) {
    {
        INT i;
        const R *xi;
        R *xo;
        xi = ii;
        xo = io;
        for (i = v; i > 0; i = i - VL, xi = xi + (VL * ivs), xo = xo + (VL * ovs),
            MAKE_VOLATILE_STRIDE(4, is), MAKE_VOLATILE_STRIDE(4, os)) {
            V T1, T2;
            T1 = LD(&(xi[0]), ivs, &(xi[0]));
            T2 = LD(&(xi[WS(is, 1)]), ivs, &(xi[WS(is, 1)]));
            ST(&(xo[WS(os, 1)]), VSUB(T1, T2), ovs, &(xo[WS(os, 1)]));
            ST(&(xo[0]), VADD(T1, T2), ovs, &(xo[0]));
        }
    }
    VLEAVE();
}

// dft/simd/common/t1buv_10.c (non-FMA codelet, CUDA adaptation)
__device__ void t1buv_10(R *ri, R *ii, const R *W, stride rs, INT mb, INT me, INT ms) {
    DVK(KP587785252, +0.587785252292473129168705954639072768597652438);
    DVK(KP951056516, +0.951056516295153572116439333379382143405698634);
    DVK(KP250000000, +0.250000000000000000000000000000000000000000000);
    DVK(KP559016994, +0.559016994374947424102293417182819058860154590);
    {
        INT m;
        R *x;
        x = ii;
        for (m = mb, W = W + (mb * ((TWVL / VL) * 18)); m < me;
             m = m + VL, x = x + (VL * ms), W = W + (TWVL * 18), MAKE_VOLATILE_STRIDE(10, rs)) {
            V Tu, TH, Tg, Tl, Tp, TD, TE, TJ, T5, Ta, To, TA, TB, TI, Tr;
            V Tt, Ts;
            Tr = LD(&(x[0]), ms, &(x[0]));
            Ts = LD(&(x[WS(rs, 5)]), ms, &(x[WS(rs, 1)]));
            Tt = BYTW(&(W[TWVL * 8]), Ts);
            Tu = VSUB(Tr, Tt);
            TH = VADD(Tr, Tt);
            {
                V Td, Tk, Tf, Ti;
                {
                    V Tc, Tj, Te, Th;
                    Tc = LD(&(x[WS(rs, 4)]), ms, &(x[0]));
                    Td = BYTW(&(W[TWVL * 6]), Tc);
                    Tj = LD(&(x[WS(rs, 1)]), ms, &(x[WS(rs, 1)]));
                    Tk = BYTW(&(W[0]), Tj);
                    Te = LD(&(x[WS(rs, 9)]), ms, &(x[WS(rs, 1)]));
                    Tf = BYTW(&(W[TWVL * 16]), Te);
                    Th = LD(&(x[WS(rs, 6)]), ms, &(x[0]));
                    Ti = BYTW(&(W[TWVL * 10]), Th);
                }
                Tg = VSUB(Td, Tf);
                Tl = VSUB(Ti, Tk);
                Tp = VADD(Tg, Tl);
                TD = VADD(Td, Tf);
                TE = VADD(Ti, Tk);
                TJ = VADD(TD, TE);
            }
            {
                V T2, T9, T4, T7;
                {
                    V T1, T8, T3, T6;
                    T1 = LD(&(x[WS(rs, 2)]), ms, &(x[0]));
                    T2 = BYTW(&(W[TWVL * 2]), T1);
                    T8 = LD(&(x[WS(rs, 3)]), ms, &(x[WS(rs, 1)]));
                    T9 = BYTW(&(W[TWVL * 4]), T8);
                    T3 = LD(&(x[WS(rs, 7)]), ms, &(x[WS(rs, 1)]));
                    T4 = BYTW(&(W[TWVL * 12]), T3);
                    T6 = LD(&(x[WS(rs, 8)]), ms, &(x[0]));
                    T7 = BYTW(&(W[TWVL * 14]), T6);
                }
                T5 = VSUB(T2, T4);
                Ta = VSUB(T7, T9);
                To = VADD(T5, Ta);
                TA = VADD(T2, T4);
                TB = VADD(T7, T9);
                TI = VADD(TA, TB);
            }
            {
                V Tq, Tv, Tw, Tn, Tz, Tb, Tm, Ty, Tx;
                Tq = VMUL(LDK(KP559016994), VSUB(To, Tp));
                Tv = VADD(To, Tp);
                Tw = VFNMS(LDK(KP250000000), Tv, Tu);
                Tb = VSUB(T5, Ta);
                Tm = VSUB(Tg, Tl);
                Tn = VBYI(VFMA(LDK(KP951056516), Tb, VMUL(LDK(KP587785252), Tm)));
                Tz = VBYI(VFNMS(LDK(KP951056516), Tm, VMUL(LDK(KP587785252), Tb)));
                ST(&(x[WS(rs, 5)]), VADD(Tu, Tv), ms, &(x[WS(rs, 1)]));
                Ty = VSUB(Tw, Tq);
                ST(&(x[WS(rs, 3)]), VSUB(Ty, Tz), ms, &(x[WS(rs, 1)]));
                ST(&(x[WS(rs, 7)]), VADD(Tz, Ty), ms, &(x[WS(rs, 1)]));
                Tx = VADD(Tq, Tw);
                ST(&(x[WS(rs, 1)]), VADD(Tn, Tx), ms, &(x[WS(rs, 1)]));
                ST(&(x[WS(rs, 9)]), VSUB(Tx, Tn), ms, &(x[WS(rs, 1)]));
            }
            {
                V TM, TK, TL, TG, TP, TC, TF, TO, TN;
                TM = VMUL(LDK(KP559016994), VSUB(TI, TJ));
                TK = VADD(TI, TJ);
                TL = VFNMS(LDK(KP250000000), TK, TH);
                TC = VSUB(TA, TB);
                TF = VSUB(TD, TE);
                TG = VBYI(VFNMS(LDK(KP951056516), TF, VMUL(LDK(KP587785252), TC)));
                TP = VBYI(VFMA(LDK(KP951056516), TC, VMUL(LDK(KP587785252), TF)));
                ST(&(x[0]), VADD(TH, TK), ms, &(x[0]));
                TO = VADD(TM, TL);
                ST(&(x[WS(rs, 4)]), VSUB(TO, TP), ms, &(x[0]));
                ST(&(x[WS(rs, 6)]), VADD(TP, TO), ms, &(x[0]));
                TN = VSUB(TL, TM);
                ST(&(x[WS(rs, 2)]), VADD(TG, TN), ms, &(x[0]));
                ST(&(x[WS(rs, 8)]), VSUB(TN, TG), ms, &(x[0]));
            }
        }
    }
    VLEAVE();
}

// dft/simd/common/t1buv_5.c (non-FMA codelet, CUDA adaptation)
__device__ void t1buv_5(R *ri, R *ii, const R *W, stride rs, INT mb, INT me, INT ms) {
    DVK(KP250000000, +0.250000000000000000000000000000000000000000000);
    DVK(KP559016994, +0.559016994374947424102293417182819058860154590);
    DVK(KP587785252, +0.587785252292473129168705954639072768597652438);
    DVK(KP951056516, +0.951056516295153572116439333379382143405698634);
    {
        INT m;
        R *x;
        x = ii;
        for (m = mb, W = W + (mb * ((TWVL / VL) * 8)); m < me;
             m = m + VL, x = x + (VL * ms), W = W + (TWVL * 8), MAKE_VOLATILE_STRIDE(5, rs)) {
            V Tf, T5, Ta, Tc, Td, Tg;
            Tf = LD(&(x[0]), ms, &(x[0]));
            {
                V T2, T9, T4, T7;
                {
                    V T1, T8, T3, T6;
                    T1 = LD(&(x[WS(rs, 1)]), ms, &(x[WS(rs, 1)]));
                    T2 = BYTW(&(W[0]), T1);
                    T8 = LD(&(x[WS(rs, 3)]), ms, &(x[WS(rs, 1)]));
                    T9 = BYTW(&(W[TWVL * 4]), T8);
                    T3 = LD(&(x[WS(rs, 4)]), ms, &(x[0]));
                    T4 = BYTW(&(W[TWVL * 6]), T3);
                    T6 = LD(&(x[WS(rs, 2)]), ms, &(x[0]));
                    T7 = BYTW(&(W[TWVL * 2]), T6);
                }
                T5 = VSUB(T2, T4);
                Ta = VSUB(T7, T9);
                Tc = VADD(T2, T4);
                Td = VADD(T7, T9);
                Tg = VADD(Tc, Td);
            }
            ST(&(x[0]), VADD(Tf, Tg), ms, &(x[0]));
            {
                V Tb, Tj, Ti, Tk, Te, Th;
                Tb = VBYI(VFMA(LDK(KP951056516), T5, VMUL(LDK(KP587785252), Ta)));
                Tj = VBYI(VFNMS(LDK(KP951056516), Ta, VMUL(LDK(KP587785252), T5)));
                Te = VMUL(LDK(KP559016994), VSUB(Tc, Td));
                Th = VFNMS(LDK(KP250000000), Tg, Tf);
                Ti = VADD(Te, Th);
                Tk = VSUB(Th, Te);
                ST(&(x[WS(rs, 1)]), VADD(Tb, Ti), ms, &(x[WS(rs, 1)]));
                ST(&(x[WS(rs, 3)]), VSUB(Tk, Tj), ms, &(x[WS(rs, 1)]));
                ST(&(x[WS(rs, 4)]), VSUB(Ti, Tb), ms, &(x[0]));
                ST(&(x[WS(rs, 2)]), VADD(Tj, Tk), ms, &(x[0]));
            }
        }
    }
    VLEAVE();
}

// dft/simd/common/q1bv_5.c (non-FMA codelet, CUDA adaptation)
__device__ void q1bv_5(R *ri, R *ii, const R *W, stride rs, stride vs, INT mb, INT me, INT ms) {
    DVK(KP250000000, +0.250000000000000000000000000000000000000000000);
    DVK(KP559016994, +0.559016994374947424102293417182819058860154590);
    DVK(KP587785252, +0.587785252292473129168705954639072768597652438);
    DVK(KP951056516, +0.951056516295153572116439333379382143405698634);
    {
        INT m;
        R *x;
        x = ii;
        for (m = mb, W = W + (mb * ((TWVL / VL) * 8)); m < me; m = m + VL, x = x + (VL * ms),
            W = W + (TWVL * 8), MAKE_VOLATILE_STRIDE(10, rs), MAKE_VOLATILE_STRIDE(10, vs)) {
            V Tb, T7, Th, Ta, Tc, Td, T1t, T1p, T1z, T1s, T1u, T1v, Tv, Tr, TB;
            V Tu, Tw, Tx, TP, TL, TV, TO, TQ, TR, T19, T15, T1f, T18, T1a, T1b;
            {
                V T6, T9, T3, T8;
                Tb = LD(&(x[0]), ms, &(x[0]));
                {
                    V T4, T5, T1, T2;
                    T4 = LD(&(x[WS(rs, 2)]), ms, &(x[0]));
                    T5 = LD(&(x[WS(rs, 3)]), ms, &(x[WS(rs, 1)]));
                    T6 = VSUB(T4, T5);
                    T9 = VADD(T4, T5);
                    T1 = LD(&(x[WS(rs, 1)]), ms, &(x[WS(rs, 1)]));
                    T2 = LD(&(x[WS(rs, 4)]), ms, &(x[0]));
                    T3 = VSUB(T1, T2);
                    T8 = VADD(T1, T2);
                }
                T7 = VBYI(VFMA(LDK(KP951056516), T3, VMUL(LDK(KP587785252), T6)));
                Th = VBYI(VFNMS(LDK(KP951056516), T6, VMUL(LDK(KP587785252), T3)));
                Ta = VMUL(LDK(KP559016994), VSUB(T8, T9));
                Tc = VADD(T8, T9);
                Td = VFNMS(LDK(KP250000000), Tc, Tb);
            }
            {
                V T1o, T1r, T1l, T1q;
                T1t = LD(&(x[WS(vs, 4)]), ms, &(x[WS(vs, 4)]));
                {
                    V T1m, T1n, T1j, T1k;
                    T1m = LD(&(x[WS(vs, 4) + WS(rs, 2)]), ms, &(x[WS(vs, 4)]));
                    T1n = LD(&(x[WS(vs, 4) + WS(rs, 3)]), ms, &(x[WS(vs, 4) + WS(rs, 1)]));
                    T1o = VSUB(T1m, T1n);
                    T1r = VADD(T1m, T1n);
                    T1j = LD(&(x[WS(vs, 4) + WS(rs, 1)]), ms, &(x[WS(vs, 4) + WS(rs, 1)]));
                    T1k = LD(&(x[WS(vs, 4) + WS(rs, 4)]), ms, &(x[WS(vs, 4)]));
                    T1l = VSUB(T1j, T1k);
                    T1q = VADD(T1j, T1k);
                }
                T1p = VBYI(VFMA(LDK(KP951056516), T1l, VMUL(LDK(KP587785252), T1o)));
                T1z = VBYI(VFNMS(LDK(KP951056516), T1o, VMUL(LDK(KP587785252), T1l)));
                T1s = VMUL(LDK(KP559016994), VSUB(T1q, T1r));
                T1u = VADD(T1q, T1r);
                T1v = VFNMS(LDK(KP250000000), T1u, T1t);
            }
            {
                V Tq, Tt, Tn, Ts;
                Tv = LD(&(x[WS(vs, 1)]), ms, &(x[WS(vs, 1)]));
                {
                    V To, Tp, Tl, Tm;
                    To = LD(&(x[WS(vs, 1) + WS(rs, 2)]), ms, &(x[WS(vs, 1)]));
                    Tp = LD(&(x[WS(vs, 1) + WS(rs, 3)]), ms, &(x[WS(vs, 1) + WS(rs, 1)]));
                    Tq = VSUB(To, Tp);
                    Tt = VADD(To, Tp);
                    Tl = LD(&(x[WS(vs, 1) + WS(rs, 1)]), ms, &(x[WS(vs, 1) + WS(rs, 1)]));
                    Tm = LD(&(x[WS(vs, 1) + WS(rs, 4)]), ms, &(x[WS(vs, 1)]));
                    Tn = VSUB(Tl, Tm);
                    Ts = VADD(Tl, Tm);
                }
                Tr = VBYI(VFMA(LDK(KP951056516), Tn, VMUL(LDK(KP587785252), Tq)));
                TB = VBYI(VFNMS(LDK(KP951056516), Tq, VMUL(LDK(KP587785252), Tn)));
                Tu = VMUL(LDK(KP559016994), VSUB(Ts, Tt));
                Tw = VADD(Ts, Tt);
                Tx = VFNMS(LDK(KP250000000), Tw, Tv);
            }
            {
                V TK, TN, TH, TM;
                TP = LD(&(x[WS(vs, 2)]), ms, &(x[WS(vs, 2)]));
                {
                    V TI, TJ, TF, TG;
                    TI = LD(&(x[WS(vs, 2) + WS(rs, 2)]), ms, &(x[WS(vs, 2)]));
                    TJ = LD(&(x[WS(vs, 2) + WS(rs, 3)]), ms, &(x[WS(vs, 2) + WS(rs, 1)]));
                    TK = VSUB(TI, TJ);
                    TN = VADD(TI, TJ);
                    TF = LD(&(x[WS(vs, 2) + WS(rs, 1)]), ms, &(x[WS(vs, 2) + WS(rs, 1)]));
                    TG = LD(&(x[WS(vs, 2) + WS(rs, 4)]), ms, &(x[WS(vs, 2)]));
                    TH = VSUB(TF, TG);
                    TM = VADD(TF, TG);
                }
                TL = VBYI(VFMA(LDK(KP951056516), TH, VMUL(LDK(KP587785252), TK)));
                TV = VBYI(VFNMS(LDK(KP951056516), TK, VMUL(LDK(KP587785252), TH)));
                TO = VMUL(LDK(KP559016994), VSUB(TM, TN));
                TQ = VADD(TM, TN);
                TR = VFNMS(LDK(KP250000000), TQ, TP);
            }
            {
                V T14, T17, T11, T16;
                T19 = LD(&(x[WS(vs, 3)]), ms, &(x[WS(vs, 3)]));
                {
                    V T12, T13, TZ, T10;
                    T12 = LD(&(x[WS(vs, 3) + WS(rs, 2)]), ms, &(x[WS(vs, 3)]));
                    T13 = LD(&(x[WS(vs, 3) + WS(rs, 3)]), ms, &(x[WS(vs, 3) + WS(rs, 1)]));
                    T14 = VSUB(T12, T13);
                    T17 = VADD(T12, T13);
                    TZ = LD(&(x[WS(vs, 3) + WS(rs, 1)]), ms, &(x[WS(vs, 3) + WS(rs, 1)]));
                    T10 = LD(&(x[WS(vs, 3) + WS(rs, 4)]), ms, &(x[WS(vs, 3)]));
                    T11 = VSUB(TZ, T10);
                    T16 = VADD(TZ, T10);
                }
                T15 = VBYI(VFMA(LDK(KP951056516), T11, VMUL(LDK(KP587785252), T14)));
                T1f = VBYI(VFNMS(LDK(KP951056516), T14, VMUL(LDK(KP587785252), T11)));
                T18 = VMUL(LDK(KP559016994), VSUB(T16, T17));
                T1a = VADD(T16, T17);
                T1b = VFNMS(LDK(KP250000000), T1a, T19);
            }
            ST(&(x[0]), VADD(Tb, Tc), ms, &(x[0]));
            ST(&(x[WS(rs, 4)]), VADD(T1t, T1u), ms, &(x[0]));
            ST(&(x[WS(rs, 2)]), VADD(TP, TQ), ms, &(x[0]));
            ST(&(x[WS(rs, 3)]), VADD(T19, T1a), ms, &(x[WS(rs, 1)]));
            ST(&(x[WS(rs, 1)]), VADD(Tv, Tw), ms, &(x[WS(rs, 1)]));
            {
                V Tj, Tk, Ti, T1B, T1C, T1A;
                Ti = VSUB(Td, Ta);
                Tj = BYTW(&(W[TWVL * 2]), VADD(Th, Ti));
                Tk = BYTW(&(W[TWVL * 4]), VSUB(Ti, Th));
                ST(&(x[WS(vs, 2)]), Tj, ms, &(x[WS(vs, 2)]));
                ST(&(x[WS(vs, 3)]), Tk, ms, &(x[WS(vs, 3)]));
                T1A = VSUB(T1v, T1s);
                T1B = BYTW(&(W[TWVL * 2]), VADD(T1z, T1A));
                T1C = BYTW(&(W[TWVL * 4]), VSUB(T1A, T1z));
                ST(&(x[WS(vs, 2) + WS(rs, 4)]), T1B, ms, &(x[WS(vs, 2)]));
                ST(&(x[WS(vs, 3) + WS(rs, 4)]), T1C, ms, &(x[WS(vs, 3)]));
            }
            {
                V T1h, T1i, T1g, TD, TE, TC;
                T1g = VSUB(T1b, T18);
                T1h = BYTW(&(W[TWVL * 2]), VADD(T1f, T1g));
                T1i = BYTW(&(W[TWVL * 4]), VSUB(T1g, T1f));
                ST(&(x[WS(vs, 2) + WS(rs, 3)]), T1h, ms, &(x[WS(vs, 2) + WS(rs, 1)]));
                ST(&(x[WS(vs, 3) + WS(rs, 3)]), T1i, ms, &(x[WS(vs, 3) + WS(rs, 1)]));
                TC = VSUB(Tx, Tu);
                TD = BYTW(&(W[TWVL * 2]), VADD(TB, TC));
                TE = BYTW(&(W[TWVL * 4]), VSUB(TC, TB));
                ST(&(x[WS(vs, 2) + WS(rs, 1)]), TD, ms, &(x[WS(vs, 2) + WS(rs, 1)]));
                ST(&(x[WS(vs, 3) + WS(rs, 1)]), TE, ms, &(x[WS(vs, 3) + WS(rs, 1)]));
            }
            {
                V TX, TY, TW, TT, TU, TS;
                TW = VSUB(TR, TO);
                TX = BYTW(&(W[TWVL * 2]), VADD(TV, TW));
                TY = BYTW(&(W[TWVL * 4]), VSUB(TW, TV));
                ST(&(x[WS(vs, 2) + WS(rs, 2)]), TX, ms, &(x[WS(vs, 2)]));
                ST(&(x[WS(vs, 3) + WS(rs, 2)]), TY, ms, &(x[WS(vs, 3)]));
                TS = VADD(TO, TR);
                TT = BYTW(&(W[0]), VADD(TL, TS));
                TU = BYTW(&(W[TWVL * 6]), VSUB(TS, TL));
                ST(&(x[WS(vs, 1) + WS(rs, 2)]), TT, ms, &(x[WS(vs, 1)]));
                ST(&(x[WS(vs, 4) + WS(rs, 2)]), TU, ms, &(x[WS(vs, 4)]));
            }
            {
                V Tf, Tg, Te, Tz, TA, Ty;
                Te = VADD(Ta, Td);
                Tf = BYTW(&(W[0]), VADD(T7, Te));
                Tg = BYTW(&(W[TWVL * 6]), VSUB(Te, T7));
                ST(&(x[WS(vs, 1)]), Tf, ms, &(x[WS(vs, 1)]));
                ST(&(x[WS(vs, 4)]), Tg, ms, &(x[WS(vs, 4)]));
                Ty = VADD(Tu, Tx);
                Tz = BYTW(&(W[0]), VADD(Tr, Ty));
                TA = BYTW(&(W[TWVL * 6]), VSUB(Ty, Tr));
                ST(&(x[WS(vs, 1) + WS(rs, 1)]), Tz, ms, &(x[WS(vs, 1) + WS(rs, 1)]));
                ST(&(x[WS(vs, 4) + WS(rs, 1)]), TA, ms, &(x[WS(vs, 4) + WS(rs, 1)]));
            }
            {
                V T1d, T1e, T1c, T1x, T1y, T1w;
                T1c = VADD(T18, T1b);
                T1d = BYTW(&(W[0]), VADD(T15, T1c));
                T1e = BYTW(&(W[TWVL * 6]), VSUB(T1c, T15));
                ST(&(x[WS(vs, 1) + WS(rs, 3)]), T1d, ms, &(x[WS(vs, 1) + WS(rs, 1)]));
                ST(&(x[WS(vs, 4) + WS(rs, 3)]), T1e, ms, &(x[WS(vs, 4) + WS(rs, 1)]));
                T1w = VADD(T1s, T1v);
                T1x = BYTW(&(W[0]), VADD(T1p, T1w));
                T1y = BYTW(&(W[TWVL * 6]), VSUB(T1w, T1p));
                ST(&(x[WS(vs, 1) + WS(rs, 4)]), T1x, ms, &(x[WS(vs, 1)]));
                ST(&(x[WS(vs, 4) + WS(rs, 4)]), T1y, ms, &(x[WS(vs, 4)]));
            }
        }
    }
    VLEAVE();
}

// rdft/simd/common/hc2cbdftv_2.c (non-FMA codelet, CUDA adaptation)
__device__ void hc2cbdftv_2(R *Rp, R *Ip, R *Rm, R *Im, const R *W, stride rs, INT mb, INT me,
                            INT ms) {
    {
        INT m;
        for (m = mb, W = W + ((mb - 1) * ((TWVL / VL) * 2)); m < me; m = m + VL,
            Rp = Rp + (VL * ms), Ip = Ip + (VL * ms), Rm = Rm - (VL * ms), Im = Im - (VL * ms),
            W = W + (TWVL * 2), MAKE_VOLATILE_STRIDE(8, rs)) {
            V T6, T5, T2, T4, T3, T1, T7, T8;
            T2 = LD(&(Rp[0]), ms, &(Rp[0]));
            T3 = LD(&(Rm[0]), -ms, &(Rm[0]));
            T4 = VCONJ(T3);
            T6 = VADD(T2, T4);
            T1 = LDW(&(W[0]));
            T5 = VZMULI(T1, VSUB(T2, T4));
            T7 = VADD(T5, T6);
            ST(&(Rp[0]), T7, ms, &(Rp[0]));
            T8 = VCONJ(VSUB(T6, T5));
            ST(&(Rm[0]), T8, -ms, &(Rm[0]));
        }
    }
    VLEAVE();
}

// rdft/scalar/r2cb/r2cb_2.c (non-FMA codelet, CUDA adaptation)
__device__ void r2cb_2(R *R0, R *R1, R *Cr, R *Ci, stride rs, stride csr, stride csi, INT v,
                       INT ivs, INT ovs) {
    {
        INT i;
        for (i = v; i > 0; i = i - 1, R0 = R0 + ovs, R1 = R1 + ovs, Cr = Cr + ivs, Ci = Ci + ivs,
            MAKE_VOLATILE_STRIDE(8, rs), MAKE_VOLATILE_STRIDE(8, csr),
            MAKE_VOLATILE_STRIDE(8, csi)) {
            E T1, T2;
            T1 = Cr[0];
            T2 = Cr[WS(csr, 1)];
            R1[0] = T1 - T2;
            R0[0] = T1 + T2;
        }
    }
}

// rdft/simd/common/hc2cbdftv_10.c (non-FMA codelet, CUDA adaptation)
__device__ void hc2cbdftv_10(R *Rp, R *Ip, R *Rm, R *Im, const R *W, stride rs, INT mb, INT me,
                             INT ms) {
    DVK(KP250000000, +0.250000000000000000000000000000000000000000000);
    DVK(KP951056516, +0.951056516295153572116439333379382143405698634);
    DVK(KP587785252, +0.587785252292473129168705954639072768597652438);
    DVK(KP559016994, +0.559016994374947424102293417182819058860154590);
    {
        INT m;
        for (m = mb, W = W + ((mb - 1) * ((TWVL / VL) * 18)); m < me; m = m + VL,
            Rp = Rp + (VL * ms), Ip = Ip + (VL * ms), Rm = Rm - (VL * ms), Im = Im - (VL * ms),
            W = W + (TWVL * 18), MAKE_VOLATILE_STRIDE(40, rs)) {
            V T5, TE, Ts, Tt, TC, Tz, TH, TJ, To, Tq, T2, T4, T3, T9, Tx;
            V Tm, TB, Td, Ty, Ti, TA, T6, T8, T7, Tl, Tk, Tj, Tc, Tb, Ta;
            V Tf, Th, Tg, TF, TG, Te, Tn;
            T2 = LD(&(Rp[0]), ms, &(Rp[0]));
            T3 = LD(&(Rm[WS(rs, 4)]), -ms, &(Rm[0]));
            T4 = VCONJ(T3);
            T5 = VSUB(T2, T4);
            TE = VADD(T2, T4);
            T6 = LD(&(Rp[WS(rs, 2)]), ms, &(Rp[0]));
            T7 = LD(&(Rm[WS(rs, 2)]), -ms, &(Rm[0]));
            T8 = VCONJ(T7);
            T9 = VSUB(T6, T8);
            Tx = VADD(T6, T8);
            Tl = LD(&(Rp[WS(rs, 1)]), ms, &(Rp[WS(rs, 1)]));
            Tj = LD(&(Rm[WS(rs, 3)]), -ms, &(Rm[WS(rs, 1)]));
            Tk = VCONJ(Tj);
            Tm = VSUB(Tk, Tl);
            TB = VADD(Tk, Tl);
            Tc = LD(&(Rp[WS(rs, 3)]), ms, &(Rp[WS(rs, 1)]));
            Ta = LD(&(Rm[WS(rs, 1)]), -ms, &(Rm[WS(rs, 1)]));
            Tb = VCONJ(Ta);
            Td = VSUB(Tb, Tc);
            Ty = VADD(Tb, Tc);
            Tf = LD(&(Rp[WS(rs, 4)]), ms, &(Rp[0]));
            Tg = LD(&(Rm[0]), -ms, &(Rm[0]));
            Th = VCONJ(Tg);
            Ti = VSUB(Tf, Th);
            TA = VADD(Tf, Th);
            Ts = VSUB(T9, Td);
            Tt = VSUB(Ti, Tm);
            TC = VSUB(TA, TB);
            Tz = VSUB(Tx, Ty);
            TF = VADD(Tx, Ty);
            TG = VADD(TA, TB);
            TH = VADD(TF, TG);
            TJ = VMUL(LDK(KP559016994), VSUB(TF, TG));
            Te = VADD(T9, Td);
            Tn = VADD(Ti, Tm);
            To = VADD(Te, Tn);
            Tq = VMUL(LDK(KP559016994), VSUB(Te, Tn));
            {
                V T1c, TX, Tv, T1b, TR, T15, TL, T17, TT, T11, TW, Tu, TQ, Tr, TP;
                V Tp, T1, T1a, TO, T14, TD, T10, TK, TZ, TI, Tw, T16, TS, TY, TM;
                V TU, T1e, TN, T1d, T19, T13, TV, T18, T12;
                T1c = VADD(TE, TH);
                TW = LDW(&(W[TWVL * 8]));
                TX = VZMULI(TW, VADD(T5, To));
                Tu = VBYI(VFNMS(LDK(KP951056516), Tt, VMUL(LDK(KP587785252), Ts)));
                TQ = VBYI(VFMA(LDK(KP951056516), Ts, VMUL(LDK(KP587785252), Tt)));
                Tp = VFNMS(LDK(KP250000000), To, T5);
                Tr = VSUB(Tp, Tq);
                TP = VADD(Tq, Tp);
                T1 = LDW(&(W[TWVL * 4]));
                Tv = VZMULI(T1, VSUB(Tr, Tu));
                T1a = LDW(&(W[0]));
                T1b = VZMULI(T1a, VADD(TQ, TP));
                TO = LDW(&(W[TWVL * 16]));
                TR = VZMULI(TO, VSUB(TP, TQ));
                T14 = LDW(&(W[TWVL * 12]));
                T15 = VZMULI(T14, VADD(Tu, Tr));
                TD = VBYI(VFNMS(LDK(KP951056516), TC, VMUL(LDK(KP587785252), Tz)));
                T10 = VBYI(VFMA(LDK(KP951056516), Tz, VMUL(LDK(KP587785252), TC)));
                TI = VFNMS(LDK(KP250000000), TH, TE);
                TK = VSUB(TI, TJ);
                TZ = VADD(TJ, TI);
                Tw = LDW(&(W[TWVL * 2]));
                TL = VZMUL(Tw, VADD(TD, TK));
                T16 = LDW(&(W[TWVL * 10]));
                T17 = VZMUL(T16, VADD(T10, TZ));
                TS = LDW(&(W[TWVL * 14]));
                TT = VZMUL(TS, VSUB(TK, TD));
                TY = LDW(&(W[TWVL * 6]));
                T11 = VZMUL(TY, VSUB(TZ, T10));
                TM = VADD(Tv, TL);
                ST(&(Rp[WS(rs, 1)]), TM, ms, &(Rp[WS(rs, 1)]));
                TU = VADD(TR, TT);
                ST(&(Rp[WS(rs, 4)]), TU, ms, &(Rp[0]));
                T1e = VCONJ(VSUB(T1c, T1b));
                ST(&(Rm[0]), T1e, -ms, &(Rm[0]));
                TN = VCONJ(VSUB(TL, Tv));
                ST(&(Rm[WS(rs, 1)]), TN, -ms, &(Rm[WS(rs, 1)]));
                T1d = VADD(T1b, T1c);
                ST(&(Rp[0]), T1d, ms, &(Rp[0]));
                T19 = VCONJ(VSUB(T17, T15));
                ST(&(Rm[WS(rs, 3)]), T19, -ms, &(Rm[WS(rs, 1)]));
                T13 = VCONJ(VSUB(T11, TX));
                ST(&(Rm[WS(rs, 2)]), T13, -ms, &(Rm[0]));
                TV = VCONJ(VSUB(TT, TR));
                ST(&(Rm[WS(rs, 4)]), TV, -ms, &(Rm[0]));
                T18 = VADD(T15, T17);
                ST(&(Rp[WS(rs, 3)]), T18, ms, &(Rp[WS(rs, 1)]));
                T12 = VADD(TX, T11);
                ST(&(Rp[WS(rs, 2)]), T12, ms, &(Rp[0]));
            }
        }
    }
    VLEAVE();
}

// rdft/scalar/r2cb/r2cb_10.c (non-FMA codelet, CUDA adaptation)
__device__ void r2cb_10(R *R0, R *R1, R *Cr, R *Ci, stride rs, stride csr, stride csi, INT v,
                        INT ivs, INT ovs) {
    DK(KP500000000, +0.500000000000000000000000000000000000000000000);
    DK(KP1_902113032, +1.902113032590307144232878666758764286811397268);
    DK(KP1_175570504, +1.175570504584946258337411909278145537195304875);
    DK(KP2_000000000, +2.000000000000000000000000000000000000000000000);
    DK(KP1_118033988, +1.118033988749894848204586834365638117720309180);
    {
        INT i;
        for (i = v; i > 0; i = i - 1, R0 = R0 + ovs, R1 = R1 + ovs, Cr = Cr + ivs, Ci = Ci + ivs,
            MAKE_VOLATILE_STRIDE(40, rs), MAKE_VOLATILE_STRIDE(40, csr),
            MAKE_VOLATILE_STRIDE(40, csi)) {
            E T3, Tb, Tn, Tv, Tk, Tu, Ta, Ts, Te, Tg, Ti, Tj;
            {
                E T1, T2, Tl, Tm;
                T1 = Cr[0];
                T2 = Cr[WS(csr, 5)];
                T3 = T1 - T2;
                Tb = T1 + T2;
                Tl = Ci[WS(csi, 4)];
                Tm = Ci[WS(csi, 1)];
                Tn = Tl - Tm;
                Tv = Tl + Tm;
            }
            Ti = Ci[WS(csi, 2)];
            Tj = Ci[WS(csi, 3)];
            Tk = Ti - Tj;
            Tu = Ti + Tj;
            {
                E T6, Tc, T9, Td;
                {
                    E T4, T5, T7, T8;
                    T4 = Cr[WS(csr, 2)];
                    T5 = Cr[WS(csr, 3)];
                    T6 = T4 - T5;
                    Tc = T4 + T5;
                    T7 = Cr[WS(csr, 4)];
                    T8 = Cr[WS(csr, 1)];
                    T9 = T7 - T8;
                    Td = T7 + T8;
                }
                Ta = T6 + T9;
                Ts = KP1_118033988 * (T6 - T9);
                Te = Tc + Td;
                Tg = KP1_118033988 * (Tc - Td);
            }
            R1[WS(rs, 2)] = FMA(KP2_000000000, Ta, T3);
            R0[0] = FMA(KP2_000000000, Te, Tb);
            {
                E To, Tq, Th, Tp, Tf;
                To = FNMS(KP1_902113032, Tn, KP1_175570504 * Tk);
                Tq = FMA(KP1_902113032, Tk, KP1_175570504 * Tn);
                Tf = FNMS(KP500000000, Te, Tb);
                Th = Tf - Tg;
                Tp = Tg + Tf;
                R0[WS(rs, 1)] = Th - To;
                R0[WS(rs, 2)] = Tp + Tq;
                R0[WS(rs, 4)] = Th + To;
                R0[WS(rs, 3)] = Tp - Tq;
            }
            {
                E Tw, Ty, Tt, Tx, Tr;
                Tw = FNMS(KP1_902113032, Tv, KP1_175570504 * Tu);
                Ty = FMA(KP1_902113032, Tu, KP1_175570504 * Tv);
                Tr = FNMS(KP500000000, Ta, T3);
                Tt = Tr - Ts;
                Tx = Ts + Tr;
                R1[WS(rs, 3)] = Tt - Tw;
                R1[WS(rs, 4)] = Tx + Ty;
                R1[WS(rs, 1)] = Tt + Tw;
                R1[0] = Tx - Ty;
            }
        }
    }
}
