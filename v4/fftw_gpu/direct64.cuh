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

// dft/simd/common/n1fv_64.c
__device__ void n1fv_64(const R *ri, const R *ii, R *ro, R *io, stride is, stride os, INT v,
                        INT ivs, INT ovs) {
    DVK(KP471396736, +0.471396736825997648556387625905254377657460319);
    DVK(KP881921264, +0.881921264348355029712756863660388349508442621);
    DVK(KP098017140, +0.098017140329560601994195563888641845861136673);
    DVK(KP995184726, +0.995184726672196886244836953109479921575474869);
    DVK(KP290284677, +0.290284677254462367636192375817395274691476278);
    DVK(KP956940335, +0.956940335732208864935797886980269969482849206);
    DVK(KP634393284, +0.634393284163645498215171613225493370675687095);
    DVK(KP773010453, +0.773010453362736960810906609758469800971041293);
    DVK(KP555570233, +0.555570233019602224742830813948532874374937191);
    DVK(KP831469612, +0.831469612302545237078788377617905756738560812);
    DVK(KP980785280, +0.980785280403230449126182236134239036973933731);
    DVK(KP195090322, +0.195090322016128267848284868477022240927691618);
    DVK(KP382683432, +0.382683432365089771728459984030398866761344562);
    DVK(KP923879532, +0.923879532511286756128183189396788286822416626);
    DVK(KP707106781, +0.707106781186547524400844362104849039284835938);
    {
        INT i;
        const R *xi;
        R *xo;
        xi = ri;
        xo = ro;
        for (i = v; i > 0; i = i - VL, xi = xi + (VL * ivs), xo = xo + (VL * ovs),
            MAKE_VOLATILE_STRIDE(128, is), MAKE_VOLATILE_STRIDE(128, os)) {
            V T4p, T5q, Tb, T39, T2n, T3A, T6f, T6T, Tq, T3B, T6i, T76, T2i, T3a, T4w;
            V T5r, TI, T2p, T6C, T6V, T3h, T3E, T4L, T5u, TZ, T2q, T6F, T6U, T3e, T3D;
            V T4E, T5t, T23, T2N, T6t, T71, T6w, T72, T2c, T2O, T3t, T41, T5f, T5R, T5k;
            V T5S, T3w, T42, T1s, T2K, T6m, T6Y, T6p, T6Z, T1B, T2L, T3m, T3Y, T4Y, T5O;
            V T53, T5P, T3p, T3Z;
            {
                V T3, T4n, T2m, T4o, T6, T5p, T9, T5o;
                {
                    V T1, T2, T2k, T2l;
                    T1 = LD(&(xi[0]), ivs, &(xi[0]));
                    T2 = LD(&(xi[WS(is, 32)]), ivs, &(xi[0]));
                    T3 = VSUB(T1, T2);
                    T4n = VADD(T1, T2);
                    T2k = LD(&(xi[WS(is, 16)]), ivs, &(xi[0]));
                    T2l = LD(&(xi[WS(is, 48)]), ivs, &(xi[0]));
                    T2m = VSUB(T2k, T2l);
                    T4o = VADD(T2k, T2l);
                }
                {
                    V T4, T5, T7, T8;
                    T4 = LD(&(xi[WS(is, 8)]), ivs, &(xi[0]));
                    T5 = LD(&(xi[WS(is, 40)]), ivs, &(xi[0]));
                    T6 = VSUB(T4, T5);
                    T5p = VADD(T4, T5);
                    T7 = LD(&(xi[WS(is, 56)]), ivs, &(xi[0]));
                    T8 = LD(&(xi[WS(is, 24)]), ivs, &(xi[0]));
                    T9 = VSUB(T7, T8);
                    T5o = VADD(T7, T8);
                }
                T4p = VSUB(T4n, T4o);
                T5q = VSUB(T5o, T5p);
                {
                    V Ta, T2j, T6d, T6e;
                    Ta = VMUL(LDK(KP707106781), VADD(T6, T9));
                    Tb = VADD(T3, Ta);
                    T39 = VSUB(T3, Ta);
                    T2j = VMUL(LDK(KP707106781), VSUB(T9, T6));
                    T2n = VSUB(T2j, T2m);
                    T3A = VADD(T2m, T2j);
                    T6d = VADD(T4n, T4o);
                    T6e = VADD(T5p, T5o);
                    T6f = VADD(T6d, T6e);
                    T6T = VSUB(T6d, T6e);
                }
            }
            {
                V Te, T4q, To, T4u, Th, T4r, Tl, T4t;
                {
                    V Tc, Td, Tm, Tn;
                    Tc = LD(&(xi[WS(is, 4)]), ivs, &(xi[0]));
                    Td = LD(&(xi[WS(is, 36)]), ivs, &(xi[0]));
                    Te = VSUB(Tc, Td);
                    T4q = VADD(Tc, Td);
                    Tm = LD(&(xi[WS(is, 12)]), ivs, &(xi[0]));
                    Tn = LD(&(xi[WS(is, 44)]), ivs, &(xi[0]));
                    To = VSUB(Tm, Tn);
                    T4u = VADD(Tm, Tn);
                }
                {
                    V Tf, Tg, Tj, Tk;
                    Tf = LD(&(xi[WS(is, 20)]), ivs, &(xi[0]));
                    Tg = LD(&(xi[WS(is, 52)]), ivs, &(xi[0]));
                    Th = VSUB(Tf, Tg);
                    T4r = VADD(Tf, Tg);
                    Tj = LD(&(xi[WS(is, 60)]), ivs, &(xi[0]));
                    Tk = LD(&(xi[WS(is, 28)]), ivs, &(xi[0]));
                    Tl = VSUB(Tj, Tk);
                    T4t = VADD(Tj, Tk);
                }
                {
                    V Ti, Tp, T6g, T6h;
                    Ti = VFNMS(LDK(KP382683432), Th, VMUL(LDK(KP923879532), Te));
                    Tp = VFMA(LDK(KP923879532), Tl, VMUL(LDK(KP382683432), To));
                    Tq = VADD(Ti, Tp);
                    T3B = VSUB(Tp, Ti);
                    T6g = VADD(T4q, T4r);
                    T6h = VADD(T4t, T4u);
                    T6i = VADD(T6g, T6h);
                    T76 = VSUB(T6h, T6g);
                }
                {
                    V T2g, T2h, T4s, T4v;
                    T2g = VFNMS(LDK(KP923879532), To, VMUL(LDK(KP382683432), Tl));
                    T2h = VFMA(LDK(KP382683432), Te, VMUL(LDK(KP923879532), Th));
                    T2i = VSUB(T2g, T2h);
                    T3a = VADD(T2h, T2g);
                    T4s = VSUB(T4q, T4r);
                    T4v = VSUB(T4t, T4u);
                    T4w = VMUL(LDK(KP707106781), VADD(T4s, T4v));
                    T5r = VMUL(LDK(KP707106781), VSUB(T4v, T4s));
                }
            }
            {
                V Tu, T4F, TG, T4G, TB, T4J, TD, T4I;
                {
                    V Ts, Tt, TE, TF;
                    Ts = LD(&(xi[WS(is, 62)]), ivs, &(xi[0]));
                    Tt = LD(&(xi[WS(is, 30)]), ivs, &(xi[0]));
                    Tu = VSUB(Ts, Tt);
                    T4F = VADD(Ts, Tt);
                    TE = LD(&(xi[WS(is, 14)]), ivs, &(xi[0]));
                    TF = LD(&(xi[WS(is, 46)]), ivs, &(xi[0]));
                    TG = VSUB(TE, TF);
                    T4G = VADD(TE, TF);
                    {
                        V Tv, Tw, Tx, Ty, Tz, TA;
                        Tv = LD(&(xi[WS(is, 6)]), ivs, &(xi[0]));
                        Tw = LD(&(xi[WS(is, 38)]), ivs, &(xi[0]));
                        Tx = VSUB(Tv, Tw);
                        Ty = LD(&(xi[WS(is, 54)]), ivs, &(xi[0]));
                        Tz = LD(&(xi[WS(is, 22)]), ivs, &(xi[0]));
                        TA = VSUB(Ty, Tz);
                        TB = VMUL(LDK(KP707106781), VADD(Tx, TA));
                        T4J = VADD(Tv, Tw);
                        TD = VMUL(LDK(KP707106781), VSUB(TA, Tx));
                        T4I = VADD(Ty, Tz);
                    }
                }
                {
                    V TC, TH, T6A, T6B;
                    TC = VADD(Tu, TB);
                    TH = VSUB(TD, TG);
                    TI = VFMA(LDK(KP195090322), TC, VMUL(LDK(KP980785280), TH));
                    T2p = VFNMS(LDK(KP195090322), TH, VMUL(LDK(KP980785280), TC));
                    T6A = VADD(T4F, T4G);
                    T6B = VADD(T4J, T4I);
                    T6C = VADD(T6A, T6B);
                    T6V = VSUB(T6A, T6B);
                }
                {
                    V T3f, T3g, T4H, T4K;
                    T3f = VSUB(Tu, TB);
                    T3g = VADD(TG, TD);
                    T3h = VFNMS(LDK(KP555570233), T3g, VMUL(LDK(KP831469612), T3f));
                    T3E = VFMA(LDK(KP555570233), T3f, VMUL(LDK(KP831469612), T3g));
                    T4H = VSUB(T4F, T4G);
                    T4K = VSUB(T4I, T4J);
                    T4L = VFNMS(LDK(KP382683432), T4K, VMUL(LDK(KP923879532), T4H));
                    T5u = VFMA(LDK(KP382683432), T4H, VMUL(LDK(KP923879532), T4K));
                }
            }
            {
                V TS, T4z, TW, T4y, TP, T4C, TX, T4B;
                {
                    V TQ, TR, TU, TV;
                    TQ = LD(&(xi[WS(is, 18)]), ivs, &(xi[0]));
                    TR = LD(&(xi[WS(is, 50)]), ivs, &(xi[0]));
                    TS = VSUB(TQ, TR);
                    T4z = VADD(TQ, TR);
                    TU = LD(&(xi[WS(is, 2)]), ivs, &(xi[0]));
                    TV = LD(&(xi[WS(is, 34)]), ivs, &(xi[0]));
                    TW = VSUB(TU, TV);
                    T4y = VADD(TU, TV);
                    {
                        V TJ, TK, TL, TM, TN, TO;
                        TJ = LD(&(xi[WS(is, 58)]), ivs, &(xi[0]));
                        TK = LD(&(xi[WS(is, 26)]), ivs, &(xi[0]));
                        TL = VSUB(TJ, TK);
                        TM = LD(&(xi[WS(is, 10)]), ivs, &(xi[0]));
                        TN = LD(&(xi[WS(is, 42)]), ivs, &(xi[0]));
                        TO = VSUB(TM, TN);
                        TP = VMUL(LDK(KP707106781), VSUB(TL, TO));
                        T4C = VADD(TM, TN);
                        TX = VMUL(LDK(KP707106781), VADD(TO, TL));
                        T4B = VADD(TJ, TK);
                    }
                }
                {
                    V TT, TY, T6D, T6E;
                    TT = VSUB(TP, TS);
                    TY = VADD(TW, TX);
                    TZ = VFNMS(LDK(KP195090322), TY, VMUL(LDK(KP980785280), TT));
                    T2q = VFMA(LDK(KP980785280), TY, VMUL(LDK(KP195090322), TT));
                    T6D = VADD(T4y, T4z);
                    T6E = VADD(T4C, T4B);
                    T6F = VADD(T6D, T6E);
                    T6U = VSUB(T6D, T6E);
                }
                {
                    V T3c, T3d, T4A, T4D;
                    T3c = VSUB(TW, TX);
                    T3d = VADD(TS, TP);
                    T3e = VFMA(LDK(KP831469612), T3c, VMUL(LDK(KP555570233), T3d));
                    T3D = VFNMS(LDK(KP555570233), T3c, VMUL(LDK(KP831469612), T3d));
                    T4A = VSUB(T4y, T4z);
                    T4D = VSUB(T4B, T4C);
                    T4E = VFMA(LDK(KP923879532), T4A, VMUL(LDK(KP382683432), T4D));
                    T5t = VFNMS(LDK(KP382683432), T4A, VMUL(LDK(KP923879532), T4D));
                }
            }
            {
                V T1F, T55, T2a, T56, T1M, T5h, T27, T5g, T58, T59, T1U, T5a, T25, T5b, T5c;
                V T21, T5d, T24;
                {
                    V T1D, T1E, T28, T29;
                    T1D = LD(&(xi[WS(is, 63)]), ivs, &(xi[WS(is, 1)]));
                    T1E = LD(&(xi[WS(is, 31)]), ivs, &(xi[WS(is, 1)]));
                    T1F = VSUB(T1D, T1E);
                    T55 = VADD(T1D, T1E);
                    T28 = LD(&(xi[WS(is, 15)]), ivs, &(xi[WS(is, 1)]));
                    T29 = LD(&(xi[WS(is, 47)]), ivs, &(xi[WS(is, 1)]));
                    T2a = VSUB(T28, T29);
                    T56 = VADD(T28, T29);
                }
                {
                    V T1G, T1H, T1I, T1J, T1K, T1L;
                    T1G = LD(&(xi[WS(is, 7)]), ivs, &(xi[WS(is, 1)]));
                    T1H = LD(&(xi[WS(is, 39)]), ivs, &(xi[WS(is, 1)]));
                    T1I = VSUB(T1G, T1H);
                    T1J = LD(&(xi[WS(is, 55)]), ivs, &(xi[WS(is, 1)]));
                    T1K = LD(&(xi[WS(is, 23)]), ivs, &(xi[WS(is, 1)]));
                    T1L = VSUB(T1J, T1K);
                    T1M = VMUL(LDK(KP707106781), VADD(T1I, T1L));
                    T5h = VADD(T1G, T1H);
                    T27 = VMUL(LDK(KP707106781), VSUB(T1L, T1I));
                    T5g = VADD(T1J, T1K);
                }
                {
                    V T1Q, T1T, T1X, T20;
                    {
                        V T1O, T1P, T1R, T1S;
                        T1O = LD(&(xi[WS(is, 3)]), ivs, &(xi[WS(is, 1)]));
                        T1P = LD(&(xi[WS(is, 35)]), ivs, &(xi[WS(is, 1)]));
                        T1Q = VSUB(T1O, T1P);
                        T58 = VADD(T1O, T1P);
                        T1R = LD(&(xi[WS(is, 19)]), ivs, &(xi[WS(is, 1)]));
                        T1S = LD(&(xi[WS(is, 51)]), ivs, &(xi[WS(is, 1)]));
                        T1T = VSUB(T1R, T1S);
                        T59 = VADD(T1R, T1S);
                    }
                    T1U = VFNMS(LDK(KP382683432), T1T, VMUL(LDK(KP923879532), T1Q));
                    T5a = VSUB(T58, T59);
                    T25 = VFMA(LDK(KP382683432), T1Q, VMUL(LDK(KP923879532), T1T));
                    {
                        V T1V, T1W, T1Y, T1Z;
                        T1V = LD(&(xi[WS(is, 59)]), ivs, &(xi[WS(is, 1)]));
                        T1W = LD(&(xi[WS(is, 27)]), ivs, &(xi[WS(is, 1)]));
                        T1X = VSUB(T1V, T1W);
                        T5b = VADD(T1V, T1W);
                        T1Y = LD(&(xi[WS(is, 11)]), ivs, &(xi[WS(is, 1)]));
                        T1Z = LD(&(xi[WS(is, 43)]), ivs, &(xi[WS(is, 1)]));
                        T20 = VSUB(T1Y, T1Z);
                        T5c = VADD(T1Y, T1Z);
                    }
                    T21 = VFMA(LDK(KP923879532), T1X, VMUL(LDK(KP382683432), T20));
                    T5d = VSUB(T5b, T5c);
                    T24 = VFNMS(LDK(KP923879532), T20, VMUL(LDK(KP382683432), T1X));
                }
                {
                    V T1N, T22, T6r, T6s;
                    T1N = VADD(T1F, T1M);
                    T22 = VADD(T1U, T21);
                    T23 = VSUB(T1N, T22);
                    T2N = VADD(T1N, T22);
                    T6r = VADD(T55, T56);
                    T6s = VADD(T5h, T5g);
                    T6t = VADD(T6r, T6s);
                    T71 = VSUB(T6r, T6s);
                }
                {
                    V T6u, T6v, T26, T2b;
                    T6u = VADD(T58, T59);
                    T6v = VADD(T5b, T5c);
                    T6w = VADD(T6u, T6v);
                    T72 = VSUB(T6v, T6u);
                    T26 = VSUB(T24, T25);
                    T2b = VSUB(T27, T2a);
                    T2c = VSUB(T26, T2b);
                    T2O = VADD(T2b, T26);
                }
                {
                    V T3r, T3s, T57, T5e;
                    T3r = VSUB(T1F, T1M);
                    T3s = VADD(T25, T24);
                    T3t = VADD(T3r, T3s);
                    T41 = VSUB(T3r, T3s);
                    T57 = VSUB(T55, T56);
                    T5e = VMUL(LDK(KP707106781), VADD(T5a, T5d));
                    T5f = VADD(T57, T5e);
                    T5R = VSUB(T57, T5e);
                }
                {
                    V T5i, T5j, T3u, T3v;
                    T5i = VSUB(T5g, T5h);
                    T5j = VMUL(LDK(KP707106781), VSUB(T5d, T5a));
                    T5k = VADD(T5i, T5j);
                    T5S = VSUB(T5j, T5i);
                    T3u = VADD(T2a, T27);
                    T3v = VSUB(T21, T1U);
                    T3w = VADD(T3u, T3v);
                    T42 = VSUB(T3v, T3u);
                }
            }
            {
                V T1q, T4P, T1v, T4O, T1n, T50, T1w, T4Z, T4U, T4V, T18, T4W, T1z, T4R, T4S;
                V T1f, T4T, T1y;
                {
                    V T1o, T1p, T1t, T1u;
                    T1o = LD(&(xi[WS(is, 17)]), ivs, &(xi[WS(is, 1)]));
                    T1p = LD(&(xi[WS(is, 49)]), ivs, &(xi[WS(is, 1)]));
                    T1q = VSUB(T1o, T1p);
                    T4P = VADD(T1o, T1p);
                    T1t = LD(&(xi[WS(is, 1)]), ivs, &(xi[WS(is, 1)]));
                    T1u = LD(&(xi[WS(is, 33)]), ivs, &(xi[WS(is, 1)]));
                    T1v = VSUB(T1t, T1u);
                    T4O = VADD(T1t, T1u);
                }
                {
                    V T1h, T1i, T1j, T1k, T1l, T1m;
                    T1h = LD(&(xi[WS(is, 57)]), ivs, &(xi[WS(is, 1)]));
                    T1i = LD(&(xi[WS(is, 25)]), ivs, &(xi[WS(is, 1)]));
                    T1j = VSUB(T1h, T1i);
                    T1k = LD(&(xi[WS(is, 9)]), ivs, &(xi[WS(is, 1)]));
                    T1l = LD(&(xi[WS(is, 41)]), ivs, &(xi[WS(is, 1)]));
                    T1m = VSUB(T1k, T1l);
                    T1n = VMUL(LDK(KP707106781), VSUB(T1j, T1m));
                    T50 = VADD(T1k, T1l);
                    T1w = VMUL(LDK(KP707106781), VADD(T1m, T1j));
                    T4Z = VADD(T1h, T1i);
                }
                {
                    V T14, T17, T1b, T1e;
                    {
                        V T12, T13, T15, T16;
                        T12 = LD(&(xi[WS(is, 61)]), ivs, &(xi[WS(is, 1)]));
                        T13 = LD(&(xi[WS(is, 29)]), ivs, &(xi[WS(is, 1)]));
                        T14 = VSUB(T12, T13);
                        T4U = VADD(T12, T13);
                        T15 = LD(&(xi[WS(is, 13)]), ivs, &(xi[WS(is, 1)]));
                        T16 = LD(&(xi[WS(is, 45)]), ivs, &(xi[WS(is, 1)]));
                        T17 = VSUB(T15, T16);
                        T4V = VADD(T15, T16);
                    }
                    T18 = VFNMS(LDK(KP923879532), T17, VMUL(LDK(KP382683432), T14));
                    T4W = VSUB(T4U, T4V);
                    T1z = VFMA(LDK(KP923879532), T14, VMUL(LDK(KP382683432), T17));
                    {
                        V T19, T1a, T1c, T1d;
                        T19 = LD(&(xi[WS(is, 5)]), ivs, &(xi[WS(is, 1)]));
                        T1a = LD(&(xi[WS(is, 37)]), ivs, &(xi[WS(is, 1)]));
                        T1b = VSUB(T19, T1a);
                        T4R = VADD(T19, T1a);
                        T1c = LD(&(xi[WS(is, 21)]), ivs, &(xi[WS(is, 1)]));
                        T1d = LD(&(xi[WS(is, 53)]), ivs, &(xi[WS(is, 1)]));
                        T1e = VSUB(T1c, T1d);
                        T4S = VADD(T1c, T1d);
                    }
                    T1f = VFMA(LDK(KP382683432), T1b, VMUL(LDK(KP923879532), T1e));
                    T4T = VSUB(T4R, T4S);
                    T1y = VFNMS(LDK(KP382683432), T1e, VMUL(LDK(KP923879532), T1b));
                }
                {
                    V T1g, T1r, T6k, T6l;
                    T1g = VSUB(T18, T1f);
                    T1r = VSUB(T1n, T1q);
                    T1s = VSUB(T1g, T1r);
                    T2K = VADD(T1r, T1g);
                    T6k = VADD(T4O, T4P);
                    T6l = VADD(T50, T4Z);
                    T6m = VADD(T6k, T6l);
                    T6Y = VSUB(T6k, T6l);
                }
                {
                    V T6n, T6o, T1x, T1A;
                    T6n = VADD(T4R, T4S);
                    T6o = VADD(T4U, T4V);
                    T6p = VADD(T6n, T6o);
                    T6Z = VSUB(T6o, T6n);
                    T1x = VADD(T1v, T1w);
                    T1A = VADD(T1y, T1z);
                    T1B = VSUB(T1x, T1A);
                    T2L = VADD(T1x, T1A);
                }
                {
                    V T3k, T3l, T4Q, T4X;
                    T3k = VSUB(T1v, T1w);
                    T3l = VADD(T1f, T18);
                    T3m = VADD(T3k, T3l);
                    T3Y = VSUB(T3k, T3l);
                    T4Q = VSUB(T4O, T4P);
                    T4X = VMUL(LDK(KP707106781), VADD(T4T, T4W));
                    T4Y = VADD(T4Q, T4X);
                    T5O = VSUB(T4Q, T4X);
                }
                {
                    V T51, T52, T3n, T3o;
                    T51 = VSUB(T4Z, T50);
                    T52 = VMUL(LDK(KP707106781), VSUB(T4W, T4T));
                    T53 = VADD(T51, T52);
                    T5P = VSUB(T52, T51);
                    T3n = VADD(T1q, T1n);
                    T3o = VSUB(T1z, T1y);
                    T3p = VADD(T3n, T3o);
                    T3Z = VSUB(T3o, T3n);
                }
            }
            {
                V T6N, T6R, T6Q, T6S;
                {
                    V T6L, T6M, T6O, T6P;
                    T6L = VADD(T6f, T6i);
                    T6M = VADD(T6F, T6C);
                    T6N = VADD(T6L, T6M);
                    T6R = VSUB(T6L, T6M);
                    T6O = VADD(T6m, T6p);
                    T6P = VADD(T6t, T6w);
                    T6Q = VADD(T6O, T6P);
                    T6S = VBYI(VSUB(T6P, T6O));
                }
                ST(&(xo[WS(os, 32)]), VSUB(T6N, T6Q), ovs, &(xo[0]));
                ST(&(xo[WS(os, 16)]), VADD(T6R, T6S), ovs, &(xo[0]));
                ST(&(xo[0]), VADD(T6N, T6Q), ovs, &(xo[0]));
                ST(&(xo[WS(os, 48)]), VSUB(T6R, T6S), ovs, &(xo[0]));
            }
            {
                V T6j, T6G, T6y, T6H, T6q, T6x;
                T6j = VSUB(T6f, T6i);
                T6G = VSUB(T6C, T6F);
                T6q = VSUB(T6m, T6p);
                T6x = VSUB(T6t, T6w);
                T6y = VMUL(LDK(KP707106781), VADD(T6q, T6x));
                T6H = VMUL(LDK(KP707106781), VSUB(T6x, T6q));
                {
                    V T6z, T6I, T6J, T6K;
                    T6z = VADD(T6j, T6y);
                    T6I = VBYI(VADD(T6G, T6H));
                    ST(&(xo[WS(os, 56)]), VSUB(T6z, T6I), ovs, &(xo[0]));
                    ST(&(xo[WS(os, 8)]), VADD(T6z, T6I), ovs, &(xo[0]));
                    T6J = VSUB(T6j, T6y);
                    T6K = VBYI(VSUB(T6H, T6G));
                    ST(&(xo[WS(os, 40)]), VSUB(T6J, T6K), ovs, &(xo[0]));
                    ST(&(xo[WS(os, 24)]), VADD(T6J, T6K), ovs, &(xo[0]));
                }
            }
            {
                V T6X, T7i, T78, T7g, T74, T7f, T7b, T7j, T6W, T77;
                T6W = VMUL(LDK(KP707106781), VADD(T6U, T6V));
                T6X = VADD(T6T, T6W);
                T7i = VSUB(T6T, T6W);
                T77 = VMUL(LDK(KP707106781), VSUB(T6V, T6U));
                T78 = VADD(T76, T77);
                T7g = VSUB(T77, T76);
                {
                    V T70, T73, T79, T7a;
                    T70 = VFMA(LDK(KP923879532), T6Y, VMUL(LDK(KP382683432), T6Z));
                    T73 = VFNMS(LDK(KP382683432), T72, VMUL(LDK(KP923879532), T71));
                    T74 = VADD(T70, T73);
                    T7f = VSUB(T73, T70);
                    T79 = VFNMS(LDK(KP382683432), T6Y, VMUL(LDK(KP923879532), T6Z));
                    T7a = VFMA(LDK(KP382683432), T71, VMUL(LDK(KP923879532), T72));
                    T7b = VADD(T79, T7a);
                    T7j = VSUB(T7a, T79);
                }
                {
                    V T75, T7c, T7l, T7m;
                    T75 = VADD(T6X, T74);
                    T7c = VBYI(VADD(T78, T7b));
                    ST(&(xo[WS(os, 60)]), VSUB(T75, T7c), ovs, &(xo[0]));
                    ST(&(xo[WS(os, 4)]), VADD(T75, T7c), ovs, &(xo[0]));
                    T7l = VBYI(VADD(T7g, T7f));
                    T7m = VADD(T7i, T7j);
                    ST(&(xo[WS(os, 12)]), VADD(T7l, T7m), ovs, &(xo[0]));
                    ST(&(xo[WS(os, 52)]), VSUB(T7m, T7l), ovs, &(xo[0]));
                }
                {
                    V T7d, T7e, T7h, T7k;
                    T7d = VSUB(T6X, T74);
                    T7e = VBYI(VSUB(T7b, T78));
                    ST(&(xo[WS(os, 36)]), VSUB(T7d, T7e), ovs, &(xo[0]));
                    ST(&(xo[WS(os, 28)]), VADD(T7d, T7e), ovs, &(xo[0]));
                    T7h = VBYI(VSUB(T7f, T7g));
                    T7k = VSUB(T7i, T7j);
                    ST(&(xo[WS(os, 20)]), VADD(T7h, T7k), ovs, &(xo[0]));
                    ST(&(xo[WS(os, 44)]), VSUB(T7k, T7h), ovs, &(xo[0]));
                }
            }
            {
                V T5N, T68, T61, T69, T5U, T65, T5Y, T66;
                {
                    V T5L, T5M, T5Z, T60;
                    T5L = VSUB(T4p, T4w);
                    T5M = VSUB(T5u, T5t);
                    T5N = VADD(T5L, T5M);
                    T68 = VSUB(T5L, T5M);
                    T5Z = VFNMS(LDK(KP555570233), T5O, VMUL(LDK(KP831469612), T5P));
                    T60 = VFMA(LDK(KP555570233), T5R, VMUL(LDK(KP831469612), T5S));
                    T61 = VADD(T5Z, T60);
                    T69 = VSUB(T60, T5Z);
                }
                {
                    V T5Q, T5T, T5W, T5X;
                    T5Q = VFMA(LDK(KP831469612), T5O, VMUL(LDK(KP555570233), T5P));
                    T5T = VFNMS(LDK(KP555570233), T5S, VMUL(LDK(KP831469612), T5R));
                    T5U = VADD(T5Q, T5T);
                    T65 = VSUB(T5T, T5Q);
                    T5W = VSUB(T5r, T5q);
                    T5X = VSUB(T4L, T4E);
                    T5Y = VADD(T5W, T5X);
                    T66 = VSUB(T5X, T5W);
                }
                {
                    V T5V, T62, T6b, T6c;
                    T5V = VADD(T5N, T5U);
                    T62 = VBYI(VADD(T5Y, T61));
                    ST(&(xo[WS(os, 58)]), VSUB(T5V, T62), ovs, &(xo[0]));
                    ST(&(xo[WS(os, 6)]), VADD(T5V, T62), ovs, &(xo[0]));
                    T6b = VBYI(VADD(T66, T65));
                    T6c = VADD(T68, T69);
                    ST(&(xo[WS(os, 10)]), VADD(T6b, T6c), ovs, &(xo[0]));
                    ST(&(xo[WS(os, 54)]), VSUB(T6c, T6b), ovs, &(xo[0]));
                }
                {
                    V T63, T64, T67, T6a;
                    T63 = VSUB(T5N, T5U);
                    T64 = VBYI(VSUB(T61, T5Y));
                    ST(&(xo[WS(os, 38)]), VSUB(T63, T64), ovs, &(xo[0]));
                    ST(&(xo[WS(os, 26)]), VADD(T63, T64), ovs, &(xo[0]));
                    T67 = VBYI(VSUB(T65, T66));
                    T6a = VSUB(T68, T69);
                    ST(&(xo[WS(os, 22)]), VADD(T67, T6a), ovs, &(xo[0]));
                    ST(&(xo[WS(os, 42)]), VSUB(T6a, T67), ovs, &(xo[0]));
                }
            }
            {
                V T11, T2C, T2v, T2D, T2e, T2z, T2s, T2A;
                {
                    V Tr, T10, T2t, T2u;
                    Tr = VSUB(Tb, Tq);
                    T10 = VSUB(TI, TZ);
                    T11 = VADD(Tr, T10);
                    T2C = VSUB(Tr, T10);
                    T2t = VFNMS(LDK(KP634393284), T1B, VMUL(LDK(KP773010453), T1s));
                    T2u = VFMA(LDK(KP773010453), T2c, VMUL(LDK(KP634393284), T23));
                    T2v = VADD(T2t, T2u);
                    T2D = VSUB(T2u, T2t);
                }
                {
                    V T1C, T2d, T2o, T2r;
                    T1C = VFMA(LDK(KP634393284), T1s, VMUL(LDK(KP773010453), T1B));
                    T2d = VFNMS(LDK(KP634393284), T2c, VMUL(LDK(KP773010453), T23));
                    T2e = VADD(T1C, T2d);
                    T2z = VSUB(T2d, T1C);
                    T2o = VSUB(T2i, T2n);
                    T2r = VSUB(T2p, T2q);
                    T2s = VADD(T2o, T2r);
                    T2A = VSUB(T2r, T2o);
                }
                {
                    V T2f, T2w, T2F, T2G;
                    T2f = VADD(T11, T2e);
                    T2w = VBYI(VADD(T2s, T2v));
                    ST(&(xo[WS(os, 57)]), VSUB(T2f, T2w), ovs, &(xo[WS(os, 1)]));
                    ST(&(xo[WS(os, 7)]), VADD(T2f, T2w), ovs, &(xo[WS(os, 1)]));
                    T2F = VBYI(VADD(T2A, T2z));
                    T2G = VADD(T2C, T2D);
                    ST(&(xo[WS(os, 9)]), VADD(T2F, T2G), ovs, &(xo[WS(os, 1)]));
                    ST(&(xo[WS(os, 55)]), VSUB(T2G, T2F), ovs, &(xo[WS(os, 1)]));
                }
                {
                    V T2x, T2y, T2B, T2E;
                    T2x = VSUB(T11, T2e);
                    T2y = VBYI(VSUB(T2v, T2s));
                    ST(&(xo[WS(os, 39)]), VSUB(T2x, T2y), ovs, &(xo[WS(os, 1)]));
                    ST(&(xo[WS(os, 25)]), VADD(T2x, T2y), ovs, &(xo[WS(os, 1)]));
                    T2B = VBYI(VSUB(T2z, T2A));
                    T2E = VSUB(T2C, T2D);
                    ST(&(xo[WS(os, 23)]), VADD(T2B, T2E), ovs, &(xo[WS(os, 1)]));
                    ST(&(xo[WS(os, 41)]), VSUB(T2E, T2B), ovs, &(xo[WS(os, 1)]));
                }
            }
            {
                V T3j, T3Q, T3J, T3R, T3y, T3N, T3G, T3O;
                {
                    V T3b, T3i, T3H, T3I;
                    T3b = VADD(T39, T3a);
                    T3i = VADD(T3e, T3h);
                    T3j = VADD(T3b, T3i);
                    T3Q = VSUB(T3b, T3i);
                    T3H = VFNMS(LDK(KP290284677), T3m, VMUL(LDK(KP956940335), T3p));
                    T3I = VFMA(LDK(KP290284677), T3t, VMUL(LDK(KP956940335), T3w));
                    T3J = VADD(T3H, T3I);
                    T3R = VSUB(T3I, T3H);
                }
                {
                    V T3q, T3x, T3C, T3F;
                    T3q = VFMA(LDK(KP956940335), T3m, VMUL(LDK(KP290284677), T3p));
                    T3x = VFNMS(LDK(KP290284677), T3w, VMUL(LDK(KP956940335), T3t));
                    T3y = VADD(T3q, T3x);
                    T3N = VSUB(T3x, T3q);
                    T3C = VADD(T3A, T3B);
                    T3F = VADD(T3D, T3E);
                    T3G = VADD(T3C, T3F);
                    T3O = VSUB(T3F, T3C);
                }
                {
                    V T3z, T3K, T3T, T3U;
                    T3z = VADD(T3j, T3y);
                    T3K = VBYI(VADD(T3G, T3J));
                    ST(&(xo[WS(os, 61)]), VSUB(T3z, T3K), ovs, &(xo[WS(os, 1)]));
                    ST(&(xo[WS(os, 3)]), VADD(T3z, T3K), ovs, &(xo[WS(os, 1)]));
                    T3T = VBYI(VADD(T3O, T3N));
                    T3U = VADD(T3Q, T3R);
                    ST(&(xo[WS(os, 13)]), VADD(T3T, T3U), ovs, &(xo[WS(os, 1)]));
                    ST(&(xo[WS(os, 51)]), VSUB(T3U, T3T), ovs, &(xo[WS(os, 1)]));
                }
                {
                    V T3L, T3M, T3P, T3S;
                    T3L = VSUB(T3j, T3y);
                    T3M = VBYI(VSUB(T3J, T3G));
                    ST(&(xo[WS(os, 35)]), VSUB(T3L, T3M), ovs, &(xo[WS(os, 1)]));
                    ST(&(xo[WS(os, 29)]), VADD(T3L, T3M), ovs, &(xo[WS(os, 1)]));
                    T3P = VBYI(VSUB(T3N, T3O));
                    T3S = VSUB(T3Q, T3R);
                    ST(&(xo[WS(os, 19)]), VADD(T3P, T3S), ovs, &(xo[WS(os, 1)]));
                    ST(&(xo[WS(os, 45)]), VSUB(T3S, T3P), ovs, &(xo[WS(os, 1)]));
                }
            }
            {
                V T4N, T5G, T5z, T5H, T5m, T5D, T5w, T5E;
                {
                    V T4x, T4M, T5x, T5y;
                    T4x = VADD(T4p, T4w);
                    T4M = VADD(T4E, T4L);
                    T4N = VADD(T4x, T4M);
                    T5G = VSUB(T4x, T4M);
                    T5x = VFNMS(LDK(KP195090322), T4Y, VMUL(LDK(KP980785280), T53));
                    T5y = VFMA(LDK(KP195090322), T5f, VMUL(LDK(KP980785280), T5k));
                    T5z = VADD(T5x, T5y);
                    T5H = VSUB(T5y, T5x);
                }
                {
                    V T54, T5l, T5s, T5v;
                    T54 = VFMA(LDK(KP980785280), T4Y, VMUL(LDK(KP195090322), T53));
                    T5l = VFNMS(LDK(KP195090322), T5k, VMUL(LDK(KP980785280), T5f));
                    T5m = VADD(T54, T5l);
                    T5D = VSUB(T5l, T54);
                    T5s = VADD(T5q, T5r);
                    T5v = VADD(T5t, T5u);
                    T5w = VADD(T5s, T5v);
                    T5E = VSUB(T5v, T5s);
                }
                {
                    V T5n, T5A, T5J, T5K;
                    T5n = VADD(T4N, T5m);
                    T5A = VBYI(VADD(T5w, T5z));
                    ST(&(xo[WS(os, 62)]), VSUB(T5n, T5A), ovs, &(xo[0]));
                    ST(&(xo[WS(os, 2)]), VADD(T5n, T5A), ovs, &(xo[0]));
                    T5J = VBYI(VADD(T5E, T5D));
                    T5K = VADD(T5G, T5H);
                    ST(&(xo[WS(os, 14)]), VADD(T5J, T5K), ovs, &(xo[0]));
                    ST(&(xo[WS(os, 50)]), VSUB(T5K, T5J), ovs, &(xo[0]));
                }
                {
                    V T5B, T5C, T5F, T5I;
                    T5B = VSUB(T4N, T5m);
                    T5C = VBYI(VSUB(T5z, T5w));
                    ST(&(xo[WS(os, 34)]), VSUB(T5B, T5C), ovs, &(xo[0]));
                    ST(&(xo[WS(os, 30)]), VADD(T5B, T5C), ovs, &(xo[0]));
                    T5F = VBYI(VSUB(T5D, T5E));
                    T5I = VSUB(T5G, T5H);
                    ST(&(xo[WS(os, 18)]), VADD(T5F, T5I), ovs, &(xo[0]));
                    ST(&(xo[WS(os, 46)]), VSUB(T5I, T5F), ovs, &(xo[0]));
                }
            }
            {
                V T2J, T34, T2X, T35, T2Q, T31, T2U, T32;
                {
                    V T2H, T2I, T2V, T2W;
                    T2H = VADD(Tb, Tq);
                    T2I = VADD(T2q, T2p);
                    T2J = VADD(T2H, T2I);
                    T34 = VSUB(T2H, T2I);
                    T2V = VFNMS(LDK(KP098017140), T2L, VMUL(LDK(KP995184726), T2K));
                    T2W = VFMA(LDK(KP995184726), T2O, VMUL(LDK(KP098017140), T2N));
                    T2X = VADD(T2V, T2W);
                    T35 = VSUB(T2W, T2V);
                }
                {
                    V T2M, T2P, T2S, T2T;
                    T2M = VFMA(LDK(KP098017140), T2K, VMUL(LDK(KP995184726), T2L));
                    T2P = VFNMS(LDK(KP098017140), T2O, VMUL(LDK(KP995184726), T2N));
                    T2Q = VADD(T2M, T2P);
                    T31 = VSUB(T2P, T2M);
                    T2S = VADD(T2n, T2i);
                    T2T = VADD(TZ, TI);
                    T2U = VADD(T2S, T2T);
                    T32 = VSUB(T2T, T2S);
                }
                {
                    V T2R, T2Y, T37, T38;
                    T2R = VADD(T2J, T2Q);
                    T2Y = VBYI(VADD(T2U, T2X));
                    ST(&(xo[WS(os, 63)]), VSUB(T2R, T2Y), ovs, &(xo[WS(os, 1)]));
                    ST(&(xo[WS(os, 1)]), VADD(T2R, T2Y), ovs, &(xo[WS(os, 1)]));
                    T37 = VBYI(VADD(T32, T31));
                    T38 = VADD(T34, T35);
                    ST(&(xo[WS(os, 15)]), VADD(T37, T38), ovs, &(xo[WS(os, 1)]));
                    ST(&(xo[WS(os, 49)]), VSUB(T38, T37), ovs, &(xo[WS(os, 1)]));
                }
                {
                    V T2Z, T30, T33, T36;
                    T2Z = VSUB(T2J, T2Q);
                    T30 = VBYI(VSUB(T2X, T2U));
                    ST(&(xo[WS(os, 33)]), VSUB(T2Z, T30), ovs, &(xo[WS(os, 1)]));
                    ST(&(xo[WS(os, 31)]), VADD(T2Z, T30), ovs, &(xo[WS(os, 1)]));
                    T33 = VBYI(VSUB(T31, T32));
                    T36 = VSUB(T34, T35);
                    ST(&(xo[WS(os, 17)]), VADD(T33, T36), ovs, &(xo[WS(os, 1)]));
                    ST(&(xo[WS(os, 47)]), VSUB(T36, T33), ovs, &(xo[WS(os, 1)]));
                }
            }
            {
                V T3X, T4i, T4b, T4j, T44, T4f, T48, T4g;
                {
                    V T3V, T3W, T49, T4a;
                    T3V = VSUB(T39, T3a);
                    T3W = VSUB(T3E, T3D);
                    T3X = VADD(T3V, T3W);
                    T4i = VSUB(T3V, T3W);
                    T49 = VFNMS(LDK(KP471396736), T3Y, VMUL(LDK(KP881921264), T3Z));
                    T4a = VFMA(LDK(KP471396736), T41, VMUL(LDK(KP881921264), T42));
                    T4b = VADD(T49, T4a);
                    T4j = VSUB(T4a, T49);
                }
                {
                    V T40, T43, T46, T47;
                    T40 = VFMA(LDK(KP881921264), T3Y, VMUL(LDK(KP471396736), T3Z));
                    T43 = VFNMS(LDK(KP471396736), T42, VMUL(LDK(KP881921264), T41));
                    T44 = VADD(T40, T43);
                    T4f = VSUB(T43, T40);
                    T46 = VSUB(T3B, T3A);
                    T47 = VSUB(T3h, T3e);
                    T48 = VADD(T46, T47);
                    T4g = VSUB(T47, T46);
                }
                {
                    V T45, T4c, T4l, T4m;
                    T45 = VADD(T3X, T44);
                    T4c = VBYI(VADD(T48, T4b));
                    ST(&(xo[WS(os, 59)]), VSUB(T45, T4c), ovs, &(xo[WS(os, 1)]));
                    ST(&(xo[WS(os, 5)]), VADD(T45, T4c), ovs, &(xo[WS(os, 1)]));
                    T4l = VBYI(VADD(T4g, T4f));
                    T4m = VADD(T4i, T4j);
                    ST(&(xo[WS(os, 11)]), VADD(T4l, T4m), ovs, &(xo[WS(os, 1)]));
                    ST(&(xo[WS(os, 53)]), VSUB(T4m, T4l), ovs, &(xo[WS(os, 1)]));
                }
                {
                    V T4d, T4e, T4h, T4k;
                    T4d = VSUB(T3X, T44);
                    T4e = VBYI(VSUB(T4b, T48));
                    ST(&(xo[WS(os, 37)]), VSUB(T4d, T4e), ovs, &(xo[WS(os, 1)]));
                    ST(&(xo[WS(os, 27)]), VADD(T4d, T4e), ovs, &(xo[WS(os, 1)]));
                    T4h = VBYI(VSUB(T4f, T4g));
                    T4k = VSUB(T4i, T4j);
                    ST(&(xo[WS(os, 21)]), VADD(T4h, T4k), ovs, &(xo[WS(os, 1)]));
                    ST(&(xo[WS(os, 43)]), VSUB(T4k, T4h), ovs, &(xo[WS(os, 1)]));
                }
            }
        }
    }
    VLEAVE();
}
// rdft/scalar/r2cf/r2cf_64.c
__device__ void r2cf_64(R *R0, R *R1, R *Cr, R *Ci, stride rs, stride csr, stride csi, INT v,
                        INT ivs, INT ovs) {
    DK(KP773010453, +0.773010453362736960810906609758469800971041293);
    DK(KP634393284, +0.634393284163645498215171613225493370675687095);
    DK(KP098017140, +0.098017140329560601994195563888641845861136673);
    DK(KP995184726, +0.995184726672196886244836953109479921575474869);
    DK(KP290284677, +0.290284677254462367636192375817395274691476278);
    DK(KP956940335, +0.956940335732208864935797886980269969482849206);
    DK(KP471396736, +0.471396736825997648556387625905254377657460319);
    DK(KP881921264, +0.881921264348355029712756863660388349508442621);
    DK(KP195090322, +0.195090322016128267848284868477022240927691618);
    DK(KP980785280, +0.980785280403230449126182236134239036973933731);
    DK(KP555570233, +0.555570233019602224742830813948532874374937191);
    DK(KP831469612, +0.831469612302545237078788377617905756738560812);
    DK(KP382683432, +0.382683432365089771728459984030398866761344562);
    DK(KP923879532, +0.923879532511286756128183189396788286822416626);
    DK(KP707106781, +0.707106781186547524400844362104849039284835938);
    {
        INT i;
        for (i = v; i > 0; i = i - 1, R0 = R0 + ivs, R1 = R1 + ivs, Cr = Cr + ovs, Ci = Ci + ovs,
            MAKE_VOLATILE_STRIDE(256, rs), MAKE_VOLATILE_STRIDE(256, csr),
            MAKE_VOLATILE_STRIDE(256, csi)) {
            E T4l, T5a, T15, T3n, T2T, T3Q, T7, Te, Tf, T4A, T4L, T1X, T3B, T23, T3y;
            E T5I, T66, T4R, T52, T2j, T3F, T2H, T3I, T5P, T69, T1i, T3t, T1l, T3u, TZ;
            E T63, T4v, T58, T1r, T3r, T1u, T3q, TK, T62, T4s, T57, Tm, Tt, Tu, T4o;
            E T5b, T1c, T3R, T2Q, T3o, T1M, T3z, T5L, T67, T26, T3C, T4H, T4M, T2y, T3J;
            E T5S, T6a, T2C, T3G, T4Y, T53;
            {
                E T3, T11, Td, T13, T6, T2S, Ta, T12, T14, T2R;
                {
                    E T1, T2, Tb, Tc;
                    T1 = R0[0];
                    T2 = R0[WS(rs, 16)];
                    T3 = T1 + T2;
                    T11 = T1 - T2;
                    Tb = R0[WS(rs, 28)];
                    Tc = R0[WS(rs, 12)];
                    Td = Tb + Tc;
                    T13 = Tb - Tc;
                }
                {
                    E T4, T5, T8, T9;
                    T4 = R0[WS(rs, 8)];
                    T5 = R0[WS(rs, 24)];
                    T6 = T4 + T5;
                    T2S = T4 - T5;
                    T8 = R0[WS(rs, 4)];
                    T9 = R0[WS(rs, 20)];
                    Ta = T8 + T9;
                    T12 = T8 - T9;
                }
                T4l = T3 - T6;
                T5a = Td - Ta;
                T14 = KP707106781 * (T12 + T13);
                T15 = T11 + T14;
                T3n = T11 - T14;
                T2R = KP707106781 * (T13 - T12);
                T2T = T2R - T2S;
                T3Q = T2S + T2R;
                T7 = T3 + T6;
                Te = Ta + Td;
                Tf = T7 + Te;
            }
            {
                E T1P, T4J, T21, T4y, T1S, T4K, T1W, T4z;
                {
                    E T1N, T1O, T1Z, T20;
                    T1N = R1[WS(rs, 28)];
                    T1O = R1[WS(rs, 12)];
                    T1P = T1N - T1O;
                    T4J = T1N + T1O;
                    T1Z = R1[0];
                    T20 = R1[WS(rs, 16)];
                    T21 = T1Z - T20;
                    T4y = T1Z + T20;
                }
                {
                    E T1Q, T1R, T1U, T1V;
                    T1Q = R1[WS(rs, 4)];
                    T1R = R1[WS(rs, 20)];
                    T1S = T1Q - T1R;
                    T4K = T1Q + T1R;
                    T1U = R1[WS(rs, 8)];
                    T1V = R1[WS(rs, 24)];
                    T1W = T1U - T1V;
                    T4z = T1U + T1V;
                }
                T4A = T4y - T4z;
                T4L = T4J - T4K;
                {
                    E T1T, T22, T5G, T5H;
                    T1T = KP707106781 * (T1P - T1S);
                    T1X = T1T - T1W;
                    T3B = T1W + T1T;
                    T22 = KP707106781 * (T1S + T1P);
                    T23 = T21 + T22;
                    T3y = T21 - T22;
                    T5G = T4y + T4z;
                    T5H = T4K + T4J;
                    T5I = T5G + T5H;
                    T66 = T5G - T5H;
                }
            }
            {
                E T2b, T4P, T2G, T4Q, T2e, T51, T2h, T50;
                {
                    E T29, T2a, T2E, T2F;
                    T29 = R1[WS(rs, 31)];
                    T2a = R1[WS(rs, 15)];
                    T2b = T29 - T2a;
                    T4P = T29 + T2a;
                    T2E = R1[WS(rs, 7)];
                    T2F = R1[WS(rs, 23)];
                    T2G = T2E - T2F;
                    T4Q = T2E + T2F;
                }
                {
                    E T2c, T2d, T2f, T2g;
                    T2c = R1[WS(rs, 3)];
                    T2d = R1[WS(rs, 19)];
                    T2e = T2c - T2d;
                    T51 = T2c + T2d;
                    T2f = R1[WS(rs, 27)];
                    T2g = R1[WS(rs, 11)];
                    T2h = T2f - T2g;
                    T50 = T2f + T2g;
                }
                T4R = T4P - T4Q;
                T52 = T50 - T51;
                {
                    E T2i, T2D, T5N, T5O;
                    T2i = KP707106781 * (T2e + T2h);
                    T2j = T2b + T2i;
                    T3F = T2b - T2i;
                    T2D = KP707106781 * (T2h - T2e);
                    T2H = T2D - T2G;
                    T3I = T2G + T2D;
                    T5N = T4P + T4Q;
                    T5O = T51 + T50;
                    T5P = T5N + T5O;
                    T69 = T5N - T5O;
                }
            }
            {
                E TN, T1e, TX, T1g, TQ, T1k, TU, T1f, T1h, T1j;
                {
                    E TL, TM, TV, TW;
                    TL = R0[WS(rs, 31)];
                    TM = R0[WS(rs, 15)];
                    TN = TL + TM;
                    T1e = TL - TM;
                    TV = R0[WS(rs, 27)];
                    TW = R0[WS(rs, 11)];
                    TX = TV + TW;
                    T1g = TV - TW;
                }
                {
                    E TO, TP, TS, TT;
                    TO = R0[WS(rs, 7)];
                    TP = R0[WS(rs, 23)];
                    TQ = TO + TP;
                    T1k = TO - TP;
                    TS = R0[WS(rs, 3)];
                    TT = R0[WS(rs, 19)];
                    TU = TS + TT;
                    T1f = TS - TT;
                }
                T1h = KP707106781 * (T1f + T1g);
                T1i = T1e + T1h;
                T3t = T1e - T1h;
                T1j = KP707106781 * (T1g - T1f);
                T1l = T1j - T1k;
                T3u = T1k + T1j;
                {
                    E TR, TY, T4t, T4u;
                    TR = TN + TQ;
                    TY = TU + TX;
                    TZ = TR + TY;
                    T63 = TR - TY;
                    T4t = TN - TQ;
                    T4u = TX - TU;
                    T4v = FNMS(KP382683432, T4u, KP923879532 * T4t);
                    T58 = FMA(KP382683432, T4t, KP923879532 * T4u);
                }
            }
            {
                E Ty, T1s, TI, T1n, TB, T1q, TF, T1o, T1p, T1t;
                {
                    E Tw, Tx, TG, TH;
                    Tw = R0[WS(rs, 1)];
                    Tx = R0[WS(rs, 17)];
                    Ty = Tw + Tx;
                    T1s = Tw - Tx;
                    TG = R0[WS(rs, 29)];
                    TH = R0[WS(rs, 13)];
                    TI = TG + TH;
                    T1n = TG - TH;
                }
                {
                    E Tz, TA, TD, TE;
                    Tz = R0[WS(rs, 9)];
                    TA = R0[WS(rs, 25)];
                    TB = Tz + TA;
                    T1q = Tz - TA;
                    TD = R0[WS(rs, 5)];
                    TE = R0[WS(rs, 21)];
                    TF = TD + TE;
                    T1o = TD - TE;
                }
                T1p = KP707106781 * (T1n - T1o);
                T1r = T1p - T1q;
                T3r = T1q + T1p;
                T1t = KP707106781 * (T1o + T1n);
                T1u = T1s + T1t;
                T3q = T1s - T1t;
                {
                    E TC, TJ, T4q, T4r;
                    TC = Ty + TB;
                    TJ = TF + TI;
                    TK = TC + TJ;
                    T62 = TC - TJ;
                    T4q = Ty - TB;
                    T4r = TI - TF;
                    T4s = FMA(KP923879532, T4q, KP382683432 * T4r);
                    T57 = FNMS(KP382683432, T4q, KP923879532 * T4r);
                }
            }
            {
                E Ti, T16, Ts, T1a, Tl, T17, Tp, T19, T4m, T4n;
                {
                    E Tg, Th, Tq, Tr;
                    Tg = R0[WS(rs, 2)];
                    Th = R0[WS(rs, 18)];
                    Ti = Tg + Th;
                    T16 = Tg - Th;
                    Tq = R0[WS(rs, 6)];
                    Tr = R0[WS(rs, 22)];
                    Ts = Tq + Tr;
                    T1a = Tq - Tr;
                }
                {
                    E Tj, Tk, Tn, To;
                    Tj = R0[WS(rs, 10)];
                    Tk = R0[WS(rs, 26)];
                    Tl = Tj + Tk;
                    T17 = Tj - Tk;
                    Tn = R0[WS(rs, 30)];
                    To = R0[WS(rs, 14)];
                    Tp = Tn + To;
                    T19 = Tn - To;
                }
                Tm = Ti + Tl;
                Tt = Tp + Ts;
                Tu = Tm + Tt;
                T4m = Ti - Tl;
                T4n = Tp - Ts;
                T4o = KP707106781 * (T4m + T4n);
                T5b = KP707106781 * (T4n - T4m);
                {
                    E T18, T1b, T2O, T2P;
                    T18 = FNMS(KP382683432, T17, KP923879532 * T16);
                    T1b = FMA(KP923879532, T19, KP382683432 * T1a);
                    T1c = T18 + T1b;
                    T3R = T1b - T18;
                    T2O = FNMS(KP923879532, T1a, KP382683432 * T19);
                    T2P = FMA(KP382683432, T16, KP923879532 * T17);
                    T2Q = T2O - T2P;
                    T3o = T2P + T2O;
                }
            }
            {
                E T1A, T4E, T1K, T4C, T1D, T4F, T1H, T4B;
                {
                    E T1y, T1z, T1I, T1J;
                    T1y = R1[WS(rs, 30)];
                    T1z = R1[WS(rs, 14)];
                    T1A = T1y - T1z;
                    T4E = T1y + T1z;
                    T1I = R1[WS(rs, 10)];
                    T1J = R1[WS(rs, 26)];
                    T1K = T1I - T1J;
                    T4C = T1I + T1J;
                }
                {
                    E T1B, T1C, T1F, T1G;
                    T1B = R1[WS(rs, 6)];
                    T1C = R1[WS(rs, 22)];
                    T1D = T1B - T1C;
                    T4F = T1B + T1C;
                    T1F = R1[WS(rs, 2)];
                    T1G = R1[WS(rs, 18)];
                    T1H = T1F - T1G;
                    T4B = T1F + T1G;
                }
                {
                    E T1E, T1L, T5J, T5K;
                    T1E = FNMS(KP923879532, T1D, KP382683432 * T1A);
                    T1L = FMA(KP382683432, T1H, KP923879532 * T1K);
                    T1M = T1E - T1L;
                    T3z = T1L + T1E;
                    T5J = T4B + T4C;
                    T5K = T4E + T4F;
                    T5L = T5J + T5K;
                    T67 = T5K - T5J;
                }
                {
                    E T24, T25, T4D, T4G;
                    T24 = FNMS(KP382683432, T1K, KP923879532 * T1H);
                    T25 = FMA(KP923879532, T1A, KP382683432 * T1D);
                    T26 = T24 + T25;
                    T3C = T25 - T24;
                    T4D = T4B - T4C;
                    T4G = T4E - T4F;
                    T4H = KP707106781 * (T4D + T4G);
                    T4M = KP707106781 * (T4G - T4D);
                }
            }
            {
                E T2m, T4S, T2w, T4W, T2p, T4T, T2t, T4V;
                {
                    E T2k, T2l, T2u, T2v;
                    T2k = R1[WS(rs, 1)];
                    T2l = R1[WS(rs, 17)];
                    T2m = T2k - T2l;
                    T4S = T2k + T2l;
                    T2u = R1[WS(rs, 5)];
                    T2v = R1[WS(rs, 21)];
                    T2w = T2u - T2v;
                    T4W = T2u + T2v;
                }
                {
                    E T2n, T2o, T2r, T2s;
                    T2n = R1[WS(rs, 9)];
                    T2o = R1[WS(rs, 25)];
                    T2p = T2n - T2o;
                    T4T = T2n + T2o;
                    T2r = R1[WS(rs, 29)];
                    T2s = R1[WS(rs, 13)];
                    T2t = T2r - T2s;
                    T4V = T2r + T2s;
                }
                {
                    E T2q, T2x, T5Q, T5R;
                    T2q = FNMS(KP382683432, T2p, KP923879532 * T2m);
                    T2x = FMA(KP923879532, T2t, KP382683432 * T2w);
                    T2y = T2q + T2x;
                    T3J = T2x - T2q;
                    T5Q = T4S + T4T;
                    T5R = T4V + T4W;
                    T5S = T5Q + T5R;
                    T6a = T5R - T5Q;
                }
                {
                    E T2A, T2B, T4U, T4X;
                    T2A = FNMS(KP923879532, T2w, KP382683432 * T2t);
                    T2B = FMA(KP382683432, T2m, KP923879532 * T2p);
                    T2C = T2A - T2B;
                    T3G = T2B + T2A;
                    T4U = T4S - T4T;
                    T4X = T4V - T4W;
                    T4Y = KP707106781 * (T4U + T4X);
                    T53 = KP707106781 * (T4X - T4U);
                }
            }
            {
                E Tv, T10, T5X, T5Y, T5Z, T60;
                Tv = Tf + Tu;
                T10 = TK + TZ;
                T5X = Tv + T10;
                T5Y = T5I + T5L;
                T5Z = T5P + T5S;
                T60 = T5Y + T5Z;
                Cr[WS(csr, 16)] = Tv - T10;
                Ci[WS(csi, 16)] = T5Z - T5Y;
                Cr[WS(csr, 32)] = T5X - T60;
                Cr[0] = T5X + T60;
            }
            {
                E T5F, T5V, T5U, T5W, T5M, T5T;
                T5F = Tf - Tu;
                T5V = TZ - TK;
                T5M = T5I - T5L;
                T5T = T5P - T5S;
                T5U = KP707106781 * (T5M + T5T);
                T5W = KP707106781 * (T5T - T5M);
                Cr[WS(csr, 24)] = T5F - T5U;
                Ci[WS(csi, 24)] = T5W - T5V;
                Cr[WS(csr, 8)] = T5F + T5U;
                Ci[WS(csi, 8)] = T5V + T5W;
            }
            {
                E T65, T6l, T6k, T6m, T6c, T6g, T6f, T6h;
                {
                    E T61, T64, T6i, T6j;
                    T61 = T7 - Te;
                    T64 = KP707106781 * (T62 + T63);
                    T65 = T61 + T64;
                    T6l = T61 - T64;
                    T6i = FNMS(KP382683432, T66, KP923879532 * T67);
                    T6j = FMA(KP382683432, T69, KP923879532 * T6a);
                    T6k = T6i + T6j;
                    T6m = T6j - T6i;
                }
                {
                    E T68, T6b, T6d, T6e;
                    T68 = FMA(KP923879532, T66, KP382683432 * T67);
                    T6b = FNMS(KP382683432, T6a, KP923879532 * T69);
                    T6c = T68 + T6b;
                    T6g = T6b - T68;
                    T6d = KP707106781 * (T63 - T62);
                    T6e = Tt - Tm;
                    T6f = T6d - T6e;
                    T6h = T6e + T6d;
                }
                Cr[WS(csr, 28)] = T65 - T6c;
                Ci[WS(csi, 28)] = T6k - T6h;
                Cr[WS(csr, 4)] = T65 + T6c;
                Ci[WS(csi, 4)] = T6h + T6k;
                Ci[WS(csi, 12)] = T6f + T6g;
                Cr[WS(csr, 12)] = T6l + T6m;
                Ci[WS(csi, 20)] = T6g - T6f;
                Cr[WS(csr, 20)] = T6l - T6m;
            }
            {
                E T5n, T5D, T5x, T5z, T5q, T5A, T5t, T5B;
                {
                    E T5l, T5m, T5v, T5w;
                    T5l = T4l - T4o;
                    T5m = T58 - T57;
                    T5n = T5l + T5m;
                    T5D = T5l - T5m;
                    T5v = T4v - T4s;
                    T5w = T5b - T5a;
                    T5x = T5v - T5w;
                    T5z = T5w + T5v;
                }
                {
                    E T5o, T5p, T5r, T5s;
                    T5o = T4A - T4H;
                    T5p = T4M - T4L;
                    T5q = FMA(KP831469612, T5o, KP555570233 * T5p);
                    T5A = FNMS(KP555570233, T5o, KP831469612 * T5p);
                    T5r = T4R - T4Y;
                    T5s = T53 - T52;
                    T5t = FNMS(KP555570233, T5s, KP831469612 * T5r);
                    T5B = FMA(KP555570233, T5r, KP831469612 * T5s);
                }
                {
                    E T5u, T5C, T5y, T5E;
                    T5u = T5q + T5t;
                    Cr[WS(csr, 26)] = T5n - T5u;
                    Cr[WS(csr, 6)] = T5n + T5u;
                    T5C = T5A + T5B;
                    Ci[WS(csi, 6)] = T5z + T5C;
                    Ci[WS(csi, 26)] = T5C - T5z;
                    T5y = T5t - T5q;
                    Ci[WS(csi, 10)] = T5x + T5y;
                    Ci[WS(csi, 22)] = T5y - T5x;
                    T5E = T5B - T5A;
                    Cr[WS(csr, 22)] = T5D - T5E;
                    Cr[WS(csr, 10)] = T5D + T5E;
                }
            }
            {
                E T4x, T5j, T5d, T5f, T4O, T5g, T55, T5h;
                {
                    E T4p, T4w, T59, T5c;
                    T4p = T4l + T4o;
                    T4w = T4s + T4v;
                    T4x = T4p + T4w;
                    T5j = T4p - T4w;
                    T59 = T57 + T58;
                    T5c = T5a + T5b;
                    T5d = T59 - T5c;
                    T5f = T5c + T59;
                }
                {
                    E T4I, T4N, T4Z, T54;
                    T4I = T4A + T4H;
                    T4N = T4L + T4M;
                    T4O = FMA(KP980785280, T4I, KP195090322 * T4N);
                    T5g = FNMS(KP195090322, T4I, KP980785280 * T4N);
                    T4Z = T4R + T4Y;
                    T54 = T52 + T53;
                    T55 = FNMS(KP195090322, T54, KP980785280 * T4Z);
                    T5h = FMA(KP195090322, T4Z, KP980785280 * T54);
                }
                {
                    E T56, T5i, T5e, T5k;
                    T56 = T4O + T55;
                    Cr[WS(csr, 30)] = T4x - T56;
                    Cr[WS(csr, 2)] = T4x + T56;
                    T5i = T5g + T5h;
                    Ci[WS(csi, 2)] = T5f + T5i;
                    Ci[WS(csi, 30)] = T5i - T5f;
                    T5e = T55 - T4O;
                    Ci[WS(csi, 14)] = T5d + T5e;
                    Ci[WS(csi, 18)] = T5e - T5d;
                    T5k = T5h - T5g;
                    Cr[WS(csr, 18)] = T5j - T5k;
                    Cr[WS(csr, 14)] = T5j + T5k;
                }
            }
            {
                E T3p, T41, T4c, T3S, T3w, T4b, T49, T4h, T3P, T42, T3E, T3W, T46, T4g, T3L;
                E T3X;
                {
                    E T3s, T3v, T3A, T3D;
                    T3p = T3n + T3o;
                    T41 = T3n - T3o;
                    T4c = T3R - T3Q;
                    T3S = T3Q + T3R;
                    T3s = FMA(KP831469612, T3q, KP555570233 * T3r);
                    T3v = FNMS(KP555570233, T3u, KP831469612 * T3t);
                    T3w = T3s + T3v;
                    T4b = T3v - T3s;
                    {
                        E T47, T48, T3N, T3O;
                        T47 = T3F - T3G;
                        T48 = T3J - T3I;
                        T49 = FNMS(KP471396736, T48, KP881921264 * T47);
                        T4h = FMA(KP471396736, T47, KP881921264 * T48);
                        T3N = FNMS(KP555570233, T3q, KP831469612 * T3r);
                        T3O = FMA(KP555570233, T3t, KP831469612 * T3u);
                        T3P = T3N + T3O;
                        T42 = T3O - T3N;
                    }
                    T3A = T3y + T3z;
                    T3D = T3B + T3C;
                    T3E = FMA(KP956940335, T3A, KP290284677 * T3D);
                    T3W = FNMS(KP290284677, T3A, KP956940335 * T3D);
                    {
                        E T44, T45, T3H, T3K;
                        T44 = T3y - T3z;
                        T45 = T3C - T3B;
                        T46 = FMA(KP881921264, T44, KP471396736 * T45);
                        T4g = FNMS(KP471396736, T44, KP881921264 * T45);
                        T3H = T3F + T3G;
                        T3K = T3I + T3J;
                        T3L = FNMS(KP290284677, T3K, KP956940335 * T3H);
                        T3X = FMA(KP290284677, T3H, KP956940335 * T3K);
                    }
                }
                {
                    E T3x, T3M, T3V, T3Y;
                    T3x = T3p + T3w;
                    T3M = T3E + T3L;
                    Cr[WS(csr, 29)] = T3x - T3M;
                    Cr[WS(csr, 3)] = T3x + T3M;
                    T3V = T3S + T3P;
                    T3Y = T3W + T3X;
                    Ci[WS(csi, 3)] = T3V + T3Y;
                    Ci[WS(csi, 29)] = T3Y - T3V;
                }
                {
                    E T3T, T3U, T3Z, T40;
                    T3T = T3P - T3S;
                    T3U = T3L - T3E;
                    Ci[WS(csi, 13)] = T3T + T3U;
                    Ci[WS(csi, 19)] = T3U - T3T;
                    T3Z = T3p - T3w;
                    T40 = T3X - T3W;
                    Cr[WS(csr, 19)] = T3Z - T40;
                    Cr[WS(csr, 13)] = T3Z + T40;
                }
                {
                    E T43, T4a, T4f, T4i;
                    T43 = T41 + T42;
                    T4a = T46 + T49;
                    Cr[WS(csr, 27)] = T43 - T4a;
                    Cr[WS(csr, 5)] = T43 + T4a;
                    T4f = T4c + T4b;
                    T4i = T4g + T4h;
                    Ci[WS(csi, 5)] = T4f + T4i;
                    Ci[WS(csi, 27)] = T4i - T4f;
                }
                {
                    E T4d, T4e, T4j, T4k;
                    T4d = T4b - T4c;
                    T4e = T49 - T46;
                    Ci[WS(csi, 11)] = T4d + T4e;
                    Ci[WS(csi, 21)] = T4e - T4d;
                    T4j = T41 - T42;
                    T4k = T4h - T4g;
                    Cr[WS(csr, 21)] = T4j - T4k;
                    Cr[WS(csr, 11)] = T4j + T4k;
                }
            }
            {
                E T1d, T33, T3e, T2U, T1w, T3d, T3b, T3j, T2N, T34, T28, T2Y, T38, T3i, T2J;
                E T2Z;
                {
                    E T1m, T1v, T1Y, T27;
                    T1d = T15 - T1c;
                    T33 = T15 + T1c;
                    T3e = T2T + T2Q;
                    T2U = T2Q - T2T;
                    T1m = FMA(KP195090322, T1i, KP980785280 * T1l);
                    T1v = FNMS(KP195090322, T1u, KP980785280 * T1r);
                    T1w = T1m - T1v;
                    T3d = T1v + T1m;
                    {
                        E T39, T3a, T2L, T2M;
                        T39 = T2j + T2y;
                        T3a = T2H + T2C;
                        T3b = FNMS(KP098017140, T3a, KP995184726 * T39);
                        T3j = FMA(KP995184726, T3a, KP098017140 * T39);
                        T2L = FNMS(KP195090322, T1l, KP980785280 * T1i);
                        T2M = FMA(KP980785280, T1u, KP195090322 * T1r);
                        T2N = T2L - T2M;
                        T34 = T2M + T2L;
                    }
                    T1Y = T1M - T1X;
                    T27 = T23 - T26;
                    T28 = FMA(KP634393284, T1Y, KP773010453 * T27);
                    T2Y = FNMS(KP634393284, T27, KP773010453 * T1Y);
                    {
                        E T36, T37, T2z, T2I;
                        T36 = T1X + T1M;
                        T37 = T23 + T26;
                        T38 = FMA(KP098017140, T36, KP995184726 * T37);
                        T3i = FNMS(KP098017140, T37, KP995184726 * T36);
                        T2z = T2j - T2y;
                        T2I = T2C - T2H;
                        T2J = FNMS(KP634393284, T2I, KP773010453 * T2z);
                        T2Z = FMA(KP773010453, T2I, KP634393284 * T2z);
                    }
                }
                {
                    E T1x, T2K, T2X, T30;
                    T1x = T1d + T1w;
                    T2K = T28 + T2J;
                    Cr[WS(csr, 25)] = T1x - T2K;
                    Cr[WS(csr, 7)] = T1x + T2K;
                    T2X = T2U + T2N;
                    T30 = T2Y + T2Z;
                    Ci[WS(csi, 7)] = T2X + T30;
                    Ci[WS(csi, 25)] = T30 - T2X;
                }
                {
                    E T2V, T2W, T31, T32;
                    T2V = T2N - T2U;
                    T2W = T2J - T28;
                    Ci[WS(csi, 9)] = T2V + T2W;
                    Ci[WS(csi, 23)] = T2W - T2V;
                    T31 = T1d - T1w;
                    T32 = T2Z - T2Y;
                    Cr[WS(csr, 23)] = T31 - T32;
                    Cr[WS(csr, 9)] = T31 + T32;
                }
                {
                    E T35, T3c, T3h, T3k;
                    T35 = T33 + T34;
                    T3c = T38 + T3b;
                    Cr[WS(csr, 31)] = T35 - T3c;
                    Cr[WS(csr, 1)] = T35 + T3c;
                    T3h = T3e + T3d;
                    T3k = T3i + T3j;
                    Ci[WS(csi, 1)] = T3h + T3k;
                    Ci[WS(csi, 31)] = T3k - T3h;
                }
                {
                    E T3f, T3g, T3l, T3m;
                    T3f = T3d - T3e;
                    T3g = T3b - T38;
                    Ci[WS(csi, 15)] = T3f + T3g;
                    Ci[WS(csi, 17)] = T3g - T3f;
                    T3l = T33 - T34;
                    T3m = T3j - T3i;
                    Cr[WS(csr, 17)] = T3l - T3m;
                    Cr[WS(csr, 15)] = T3l + T3m;
                }
            }
        }
    }
}
// dft/simd/common/n1bv_64.c
__device__ void n1bv_64(const R *ri, const R *ii, R *ro, R *io, stride is, stride os, INT v,
                        INT ivs, INT ovs) {
    DVK(KP634393284, +0.634393284163645498215171613225493370675687095);
    DVK(KP773010453, +0.773010453362736960810906609758469800971041293);
    DVK(KP290284677, +0.290284677254462367636192375817395274691476278);
    DVK(KP956940335, +0.956940335732208864935797886980269969482849206);
    DVK(KP098017140, +0.098017140329560601994195563888641845861136673);
    DVK(KP995184726, +0.995184726672196886244836953109479921575474869);
    DVK(KP471396736, +0.471396736825997648556387625905254377657460319);
    DVK(KP881921264, +0.881921264348355029712756863660388349508442621);
    DVK(KP195090322, +0.195090322016128267848284868477022240927691618);
    DVK(KP980785280, +0.980785280403230449126182236134239036973933731);
    DVK(KP555570233, +0.555570233019602224742830813948532874374937191);
    DVK(KP831469612, +0.831469612302545237078788377617905756738560812);
    DVK(KP923879532, +0.923879532511286756128183189396788286822416626);
    DVK(KP382683432, +0.382683432365089771728459984030398866761344562);
    DVK(KP707106781, +0.707106781186547524400844362104849039284835938);
    {
        INT i;
        const R *xi;
        R *xo;
        xi = ii;
        xo = io;
        for (i = v; i > 0; i = i - VL, xi = xi + (VL * ivs), xo = xo + (VL * ovs),
            MAKE_VOLATILE_STRIDE(128, is), MAKE_VOLATILE_STRIDE(128, os)) {
            V T4p, T5u, Tb, T3A, T2q, T3v, T6G, T78, Tq, T3w, T6B, T79, T2l, T3B, T4w;
            V T5r, TI, T2g, T6u, T74, T3q, T3D, T4E, T5o, TZ, T2h, T6x, T75, T3t, T3E;
            V T4L, T5p, T23, T2N, T6m, T70, T6p, T71, T2c, T2O, T3i, T3Y, T5f, T5R, T5k;
            V T5S, T3l, T3Z, T1s, T2K, T6f, T6X, T6i, T6Y, T1B, T2L, T3b, T3V, T4Y, T5O;
            V T53, T5P, T3e, T3W;
            {
                V T3, T4n, T2p, T4o, T6, T5s, T9, T5t;
                {
                    V T1, T2, T2n, T2o;
                    T1 = LD(&(xi[0]), ivs, &(xi[0]));
                    T2 = LD(&(xi[WS(is, 32)]), ivs, &(xi[0]));
                    T3 = VSUB(T1, T2);
                    T4n = VADD(T1, T2);
                    T2n = LD(&(xi[WS(is, 16)]), ivs, &(xi[0]));
                    T2o = LD(&(xi[WS(is, 48)]), ivs, &(xi[0]));
                    T2p = VSUB(T2n, T2o);
                    T4o = VADD(T2n, T2o);
                }
                {
                    V T4, T5, T7, T8;
                    T4 = LD(&(xi[WS(is, 8)]), ivs, &(xi[0]));
                    T5 = LD(&(xi[WS(is, 40)]), ivs, &(xi[0]));
                    T6 = VSUB(T4, T5);
                    T5s = VADD(T4, T5);
                    T7 = LD(&(xi[WS(is, 56)]), ivs, &(xi[0]));
                    T8 = LD(&(xi[WS(is, 24)]), ivs, &(xi[0]));
                    T9 = VSUB(T7, T8);
                    T5t = VADD(T7, T8);
                }
                T4p = VSUB(T4n, T4o);
                T5u = VSUB(T5s, T5t);
                {
                    V Ta, T2m, T6E, T6F;
                    Ta = VMUL(LDK(KP707106781), VADD(T6, T9));
                    Tb = VSUB(T3, Ta);
                    T3A = VADD(T3, Ta);
                    T2m = VMUL(LDK(KP707106781), VSUB(T6, T9));
                    T2q = VSUB(T2m, T2p);
                    T3v = VADD(T2p, T2m);
                    T6E = VADD(T4n, T4o);
                    T6F = VADD(T5s, T5t);
                    T6G = VSUB(T6E, T6F);
                    T78 = VADD(T6E, T6F);
                }
            }
            {
                V Te, T4q, To, T4t, Th, T4r, Tl, T4u;
                {
                    V Tc, Td, Tm, Tn;
                    Tc = LD(&(xi[WS(is, 4)]), ivs, &(xi[0]));
                    Td = LD(&(xi[WS(is, 36)]), ivs, &(xi[0]));
                    Te = VSUB(Tc, Td);
                    T4q = VADD(Tc, Td);
                    Tm = LD(&(xi[WS(is, 60)]), ivs, &(xi[0]));
                    Tn = LD(&(xi[WS(is, 28)]), ivs, &(xi[0]));
                    To = VSUB(Tm, Tn);
                    T4t = VADD(Tm, Tn);
                }
                {
                    V Tf, Tg, Tj, Tk;
                    Tf = LD(&(xi[WS(is, 20)]), ivs, &(xi[0]));
                    Tg = LD(&(xi[WS(is, 52)]), ivs, &(xi[0]));
                    Th = VSUB(Tf, Tg);
                    T4r = VADD(Tf, Tg);
                    Tj = LD(&(xi[WS(is, 12)]), ivs, &(xi[0]));
                    Tk = LD(&(xi[WS(is, 44)]), ivs, &(xi[0]));
                    Tl = VSUB(Tj, Tk);
                    T4u = VADD(Tj, Tk);
                }
                {
                    V Ti, Tp, T6z, T6A;
                    Ti = VFMA(LDK(KP382683432), Te, VMUL(LDK(KP923879532), Th));
                    Tp = VFNMS(LDK(KP382683432), To, VMUL(LDK(KP923879532), Tl));
                    Tq = VSUB(Ti, Tp);
                    T3w = VADD(Ti, Tp);
                    T6z = VADD(T4q, T4r);
                    T6A = VADD(T4t, T4u);
                    T6B = VSUB(T6z, T6A);
                    T79 = VADD(T6z, T6A);
                }
                {
                    V T2j, T2k, T4s, T4v;
                    T2j = VFNMS(LDK(KP382683432), Th, VMUL(LDK(KP923879532), Te));
                    T2k = VFMA(LDK(KP923879532), To, VMUL(LDK(KP382683432), Tl));
                    T2l = VSUB(T2j, T2k);
                    T3B = VADD(T2j, T2k);
                    T4s = VSUB(T4q, T4r);
                    T4v = VSUB(T4t, T4u);
                    T4w = VMUL(LDK(KP707106781), VADD(T4s, T4v));
                    T5r = VMUL(LDK(KP707106781), VSUB(T4s, T4v));
                }
            }
            {
                V TB, T4z, TF, T4y, Ty, T4C, TG, T4B;
                {
                    V Tz, TA, TD, TE;
                    Tz = LD(&(xi[WS(is, 18)]), ivs, &(xi[0]));
                    TA = LD(&(xi[WS(is, 50)]), ivs, &(xi[0]));
                    TB = VSUB(Tz, TA);
                    T4z = VADD(Tz, TA);
                    TD = LD(&(xi[WS(is, 2)]), ivs, &(xi[0]));
                    TE = LD(&(xi[WS(is, 34)]), ivs, &(xi[0]));
                    TF = VSUB(TD, TE);
                    T4y = VADD(TD, TE);
                    {
                        V Ts, Tt, Tu, Tv, Tw, Tx;
                        Ts = LD(&(xi[WS(is, 10)]), ivs, &(xi[0]));
                        Tt = LD(&(xi[WS(is, 42)]), ivs, &(xi[0]));
                        Tu = VSUB(Ts, Tt);
                        Tv = LD(&(xi[WS(is, 58)]), ivs, &(xi[0]));
                        Tw = LD(&(xi[WS(is, 26)]), ivs, &(xi[0]));
                        Tx = VSUB(Tv, Tw);
                        Ty = VMUL(LDK(KP707106781), VSUB(Tu, Tx));
                        T4C = VADD(Tv, Tw);
                        TG = VMUL(LDK(KP707106781), VADD(Tu, Tx));
                        T4B = VADD(Ts, Tt);
                    }
                }
                {
                    V TC, TH, T6s, T6t;
                    TC = VSUB(Ty, TB);
                    TH = VSUB(TF, TG);
                    TI = VFMA(LDK(KP831469612), TC, VMUL(LDK(KP555570233), TH));
                    T2g = VFNMS(LDK(KP555570233), TC, VMUL(LDK(KP831469612), TH));
                    T6s = VADD(T4y, T4z);
                    T6t = VADD(T4B, T4C);
                    T6u = VSUB(T6s, T6t);
                    T74 = VADD(T6s, T6t);
                }
                {
                    V T3o, T3p, T4A, T4D;
                    T3o = VADD(TB, Ty);
                    T3p = VADD(TF, TG);
                    T3q = VFMA(LDK(KP980785280), T3o, VMUL(LDK(KP195090322), T3p));
                    T3D = VFNMS(LDK(KP195090322), T3o, VMUL(LDK(KP980785280), T3p));
                    T4A = VSUB(T4y, T4z);
                    T4D = VSUB(T4B, T4C);
                    T4E = VFMA(LDK(KP382683432), T4A, VMUL(LDK(KP923879532), T4D));
                    T5o = VFNMS(LDK(KP382683432), T4D, VMUL(LDK(KP923879532), T4A));
                }
            }
            {
                V TS, T4J, TW, T4I, TP, T4G, TX, T4F;
                {
                    V TQ, TR, TU, TV;
                    TQ = LD(&(xi[WS(is, 14)]), ivs, &(xi[0]));
                    TR = LD(&(xi[WS(is, 46)]), ivs, &(xi[0]));
                    TS = VSUB(TQ, TR);
                    T4J = VADD(TQ, TR);
                    TU = LD(&(xi[WS(is, 62)]), ivs, &(xi[0]));
                    TV = LD(&(xi[WS(is, 30)]), ivs, &(xi[0]));
                    TW = VSUB(TU, TV);
                    T4I = VADD(TU, TV);
                    {
                        V TJ, TK, TL, TM, TN, TO;
                        TJ = LD(&(xi[WS(is, 6)]), ivs, &(xi[0]));
                        TK = LD(&(xi[WS(is, 38)]), ivs, &(xi[0]));
                        TL = VSUB(TJ, TK);
                        TM = LD(&(xi[WS(is, 54)]), ivs, &(xi[0]));
                        TN = LD(&(xi[WS(is, 22)]), ivs, &(xi[0]));
                        TO = VSUB(TM, TN);
                        TP = VMUL(LDK(KP707106781), VSUB(TL, TO));
                        T4G = VADD(TM, TN);
                        TX = VMUL(LDK(KP707106781), VADD(TL, TO));
                        T4F = VADD(TJ, TK);
                    }
                }
                {
                    V TT, TY, T6v, T6w;
                    TT = VSUB(TP, TS);
                    TY = VSUB(TW, TX);
                    TZ = VFNMS(LDK(KP555570233), TY, VMUL(LDK(KP831469612), TT));
                    T2h = VFMA(LDK(KP555570233), TT, VMUL(LDK(KP831469612), TY));
                    T6v = VADD(T4I, T4J);
                    T6w = VADD(T4F, T4G);
                    T6x = VSUB(T6v, T6w);
                    T75 = VADD(T6v, T6w);
                }
                {
                    V T3r, T3s, T4H, T4K;
                    T3r = VADD(TS, TP);
                    T3s = VADD(TW, TX);
                    T3t = VFNMS(LDK(KP195090322), T3s, VMUL(LDK(KP980785280), T3r));
                    T3E = VFMA(LDK(KP195090322), T3r, VMUL(LDK(KP980785280), T3s));
                    T4H = VSUB(T4F, T4G);
                    T4K = VSUB(T4I, T4J);
                    T4L = VFNMS(LDK(KP382683432), T4K, VMUL(LDK(KP923879532), T4H));
                    T5p = VFMA(LDK(KP923879532), T4K, VMUL(LDK(KP382683432), T4H));
                }
            }
            {
                V T21, T5h, T26, T5g, T1Y, T5d, T27, T5c, T55, T56, T1J, T57, T29, T58, T59;
                V T1Q, T5a, T2a;
                {
                    V T1Z, T20, T24, T25;
                    T1Z = LD(&(xi[WS(is, 15)]), ivs, &(xi[WS(is, 1)]));
                    T20 = LD(&(xi[WS(is, 47)]), ivs, &(xi[WS(is, 1)]));
                    T21 = VSUB(T1Z, T20);
                    T5h = VADD(T1Z, T20);
                    T24 = LD(&(xi[WS(is, 63)]), ivs, &(xi[WS(is, 1)]));
                    T25 = LD(&(xi[WS(is, 31)]), ivs, &(xi[WS(is, 1)]));
                    T26 = VSUB(T24, T25);
                    T5g = VADD(T24, T25);
                }
                {
                    V T1S, T1T, T1U, T1V, T1W, T1X;
                    T1S = LD(&(xi[WS(is, 7)]), ivs, &(xi[WS(is, 1)]));
                    T1T = LD(&(xi[WS(is, 39)]), ivs, &(xi[WS(is, 1)]));
                    T1U = VSUB(T1S, T1T);
                    T1V = LD(&(xi[WS(is, 55)]), ivs, &(xi[WS(is, 1)]));
                    T1W = LD(&(xi[WS(is, 23)]), ivs, &(xi[WS(is, 1)]));
                    T1X = VSUB(T1V, T1W);
                    T1Y = VMUL(LDK(KP707106781), VSUB(T1U, T1X));
                    T5d = VADD(T1V, T1W);
                    T27 = VMUL(LDK(KP707106781), VADD(T1U, T1X));
                    T5c = VADD(T1S, T1T);
                }
                {
                    V T1F, T1I, T1M, T1P;
                    {
                        V T1D, T1E, T1G, T1H;
                        T1D = LD(&(xi[WS(is, 3)]), ivs, &(xi[WS(is, 1)]));
                        T1E = LD(&(xi[WS(is, 35)]), ivs, &(xi[WS(is, 1)]));
                        T1F = VSUB(T1D, T1E);
                        T55 = VADD(T1D, T1E);
                        T1G = LD(&(xi[WS(is, 19)]), ivs, &(xi[WS(is, 1)]));
                        T1H = LD(&(xi[WS(is, 51)]), ivs, &(xi[WS(is, 1)]));
                        T1I = VSUB(T1G, T1H);
                        T56 = VADD(T1G, T1H);
                    }
                    T1J = VFNMS(LDK(KP382683432), T1I, VMUL(LDK(KP923879532), T1F));
                    T57 = VSUB(T55, T56);
                    T29 = VFMA(LDK(KP382683432), T1F, VMUL(LDK(KP923879532), T1I));
                    {
                        V T1K, T1L, T1N, T1O;
                        T1K = LD(&(xi[WS(is, 59)]), ivs, &(xi[WS(is, 1)]));
                        T1L = LD(&(xi[WS(is, 27)]), ivs, &(xi[WS(is, 1)]));
                        T1M = VSUB(T1K, T1L);
                        T58 = VADD(T1K, T1L);
                        T1N = LD(&(xi[WS(is, 11)]), ivs, &(xi[WS(is, 1)]));
                        T1O = LD(&(xi[WS(is, 43)]), ivs, &(xi[WS(is, 1)]));
                        T1P = VSUB(T1N, T1O);
                        T59 = VADD(T1N, T1O);
                    }
                    T1Q = VFMA(LDK(KP923879532), T1M, VMUL(LDK(KP382683432), T1P));
                    T5a = VSUB(T58, T59);
                    T2a = VFNMS(LDK(KP382683432), T1M, VMUL(LDK(KP923879532), T1P));
                }
                {
                    V T1R, T22, T6k, T6l;
                    T1R = VSUB(T1J, T1Q);
                    T22 = VSUB(T1Y, T21);
                    T23 = VSUB(T1R, T22);
                    T2N = VADD(T22, T1R);
                    T6k = VADD(T5g, T5h);
                    T6l = VADD(T5c, T5d);
                    T6m = VSUB(T6k, T6l);
                    T70 = VADD(T6k, T6l);
                }
                {
                    V T6n, T6o, T28, T2b;
                    T6n = VADD(T55, T56);
                    T6o = VADD(T58, T59);
                    T6p = VSUB(T6n, T6o);
                    T71 = VADD(T6n, T6o);
                    T28 = VSUB(T26, T27);
                    T2b = VSUB(T29, T2a);
                    T2c = VSUB(T28, T2b);
                    T2O = VADD(T28, T2b);
                }
                {
                    V T3g, T3h, T5b, T5e;
                    T3g = VADD(T26, T27);
                    T3h = VADD(T1J, T1Q);
                    T3i = VADD(T3g, T3h);
                    T3Y = VSUB(T3g, T3h);
                    T5b = VMUL(LDK(KP707106781), VSUB(T57, T5a));
                    T5e = VSUB(T5c, T5d);
                    T5f = VSUB(T5b, T5e);
                    T5R = VADD(T5e, T5b);
                }
                {
                    V T5i, T5j, T3j, T3k;
                    T5i = VSUB(T5g, T5h);
                    T5j = VMUL(LDK(KP707106781), VADD(T57, T5a));
                    T5k = VSUB(T5i, T5j);
                    T5S = VADD(T5i, T5j);
                    T3j = VADD(T21, T1Y);
                    T3k = VADD(T29, T2a);
                    T3l = VADD(T3j, T3k);
                    T3Z = VSUB(T3k, T3j);
                }
            }
            {
                V T1q, T50, T1v, T4Z, T1n, T4W, T1w, T4V, T4O, T4P, T18, T4Q, T1y, T4R, T4S;
                V T1f, T4T, T1z;
                {
                    V T1o, T1p, T1t, T1u;
                    T1o = LD(&(xi[WS(is, 17)]), ivs, &(xi[WS(is, 1)]));
                    T1p = LD(&(xi[WS(is, 49)]), ivs, &(xi[WS(is, 1)]));
                    T1q = VSUB(T1o, T1p);
                    T50 = VADD(T1o, T1p);
                    T1t = LD(&(xi[WS(is, 1)]), ivs, &(xi[WS(is, 1)]));
                    T1u = LD(&(xi[WS(is, 33)]), ivs, &(xi[WS(is, 1)]));
                    T1v = VSUB(T1t, T1u);
                    T4Z = VADD(T1t, T1u);
                }
                {
                    V T1h, T1i, T1j, T1k, T1l, T1m;
                    T1h = LD(&(xi[WS(is, 9)]), ivs, &(xi[WS(is, 1)]));
                    T1i = LD(&(xi[WS(is, 41)]), ivs, &(xi[WS(is, 1)]));
                    T1j = VSUB(T1h, T1i);
                    T1k = LD(&(xi[WS(is, 57)]), ivs, &(xi[WS(is, 1)]));
                    T1l = LD(&(xi[WS(is, 25)]), ivs, &(xi[WS(is, 1)]));
                    T1m = VSUB(T1k, T1l);
                    T1n = VMUL(LDK(KP707106781), VSUB(T1j, T1m));
                    T4W = VADD(T1k, T1l);
                    T1w = VMUL(LDK(KP707106781), VADD(T1j, T1m));
                    T4V = VADD(T1h, T1i);
                }
                {
                    V T14, T17, T1b, T1e;
                    {
                        V T12, T13, T15, T16;
                        T12 = LD(&(xi[WS(is, 5)]), ivs, &(xi[WS(is, 1)]));
                        T13 = LD(&(xi[WS(is, 37)]), ivs, &(xi[WS(is, 1)]));
                        T14 = VSUB(T12, T13);
                        T4O = VADD(T12, T13);
                        T15 = LD(&(xi[WS(is, 21)]), ivs, &(xi[WS(is, 1)]));
                        T16 = LD(&(xi[WS(is, 53)]), ivs, &(xi[WS(is, 1)]));
                        T17 = VSUB(T15, T16);
                        T4P = VADD(T15, T16);
                    }
                    T18 = VFNMS(LDK(KP382683432), T17, VMUL(LDK(KP923879532), T14));
                    T4Q = VSUB(T4O, T4P);
                    T1y = VFMA(LDK(KP382683432), T14, VMUL(LDK(KP923879532), T17));
                    {
                        V T19, T1a, T1c, T1d;
                        T19 = LD(&(xi[WS(is, 61)]), ivs, &(xi[WS(is, 1)]));
                        T1a = LD(&(xi[WS(is, 29)]), ivs, &(xi[WS(is, 1)]));
                        T1b = VSUB(T19, T1a);
                        T4R = VADD(T19, T1a);
                        T1c = LD(&(xi[WS(is, 13)]), ivs, &(xi[WS(is, 1)]));
                        T1d = LD(&(xi[WS(is, 45)]), ivs, &(xi[WS(is, 1)]));
                        T1e = VSUB(T1c, T1d);
                        T4S = VADD(T1c, T1d);
                    }
                    T1f = VFMA(LDK(KP923879532), T1b, VMUL(LDK(KP382683432), T1e));
                    T4T = VSUB(T4R, T4S);
                    T1z = VFNMS(LDK(KP382683432), T1b, VMUL(LDK(KP923879532), T1e));
                }
                {
                    V T1g, T1r, T6d, T6e;
                    T1g = VSUB(T18, T1f);
                    T1r = VSUB(T1n, T1q);
                    T1s = VSUB(T1g, T1r);
                    T2K = VADD(T1r, T1g);
                    T6d = VADD(T4Z, T50);
                    T6e = VADD(T4V, T4W);
                    T6f = VSUB(T6d, T6e);
                    T6X = VADD(T6d, T6e);
                }
                {
                    V T6g, T6h, T1x, T1A;
                    T6g = VADD(T4O, T4P);
                    T6h = VADD(T4R, T4S);
                    T6i = VSUB(T6g, T6h);
                    T6Y = VADD(T6g, T6h);
                    T1x = VSUB(T1v, T1w);
                    T1A = VSUB(T1y, T1z);
                    T1B = VSUB(T1x, T1A);
                    T2L = VADD(T1x, T1A);
                }
                {
                    V T39, T3a, T4U, T4X;
                    T39 = VADD(T1v, T1w);
                    T3a = VADD(T18, T1f);
                    T3b = VADD(T39, T3a);
                    T3V = VSUB(T39, T3a);
                    T4U = VMUL(LDK(KP707106781), VSUB(T4Q, T4T));
                    T4X = VSUB(T4V, T4W);
                    T4Y = VSUB(T4U, T4X);
                    T5O = VADD(T4X, T4U);
                }
                {
                    V T51, T52, T3c, T3d;
                    T51 = VSUB(T4Z, T50);
                    T52 = VMUL(LDK(KP707106781), VADD(T4Q, T4T));
                    T53 = VSUB(T51, T52);
                    T5P = VADD(T51, T52);
                    T3c = VADD(T1q, T1n);
                    T3d = VADD(T1y, T1z);
                    T3e = VADD(T3c, T3d);
                    T3W = VSUB(T3d, T3c);
                }
            }
            {
                V T7h, T7l, T7k, T7m;
                {
                    V T7f, T7g, T7i, T7j;
                    T7f = VADD(T78, T79);
                    T7g = VADD(T74, T75);
                    T7h = VSUB(T7f, T7g);
                    T7l = VADD(T7f, T7g);
                    T7i = VADD(T6X, T6Y);
                    T7j = VADD(T70, T71);
                    T7k = VBYI(VSUB(T7i, T7j));
                    T7m = VADD(T7i, T7j);
                }
                ST(&(xo[WS(os, 48)]), VSUB(T7h, T7k), ovs, &(xo[0]));
                ST(&(xo[0]), VADD(T7l, T7m), ovs, &(xo[0]));
                ST(&(xo[WS(os, 16)]), VADD(T7h, T7k), ovs, &(xo[0]));
                ST(&(xo[WS(os, 32)]), VSUB(T7l, T7m), ovs, &(xo[0]));
            }
            {
                V T76, T7a, T73, T7b, T6Z, T72;
                T76 = VSUB(T74, T75);
                T7a = VSUB(T78, T79);
                T6Z = VSUB(T6X, T6Y);
                T72 = VSUB(T70, T71);
                T73 = VMUL(LDK(KP707106781), VSUB(T6Z, T72));
                T7b = VMUL(LDK(KP707106781), VADD(T6Z, T72));
                {
                    V T77, T7c, T7d, T7e;
                    T77 = VBYI(VSUB(T73, T76));
                    T7c = VSUB(T7a, T7b);
                    ST(&(xo[WS(os, 24)]), VADD(T77, T7c), ovs, &(xo[0]));
                    ST(&(xo[WS(os, 40)]), VSUB(T7c, T77), ovs, &(xo[0]));
                    T7d = VBYI(VADD(T76, T73));
                    T7e = VADD(T7a, T7b);
                    ST(&(xo[WS(os, 8)]), VADD(T7d, T7e), ovs, &(xo[0]));
                    ST(&(xo[WS(os, 56)]), VSUB(T7e, T7d), ovs, &(xo[0]));
                }
            }
            {
                V T6C, T6S, T6I, T6P, T6r, T6Q, T6L, T6T, T6y, T6H;
                T6y = VMUL(LDK(KP707106781), VSUB(T6u, T6x));
                T6C = VSUB(T6y, T6B);
                T6S = VADD(T6B, T6y);
                T6H = VMUL(LDK(KP707106781), VADD(T6u, T6x));
                T6I = VSUB(T6G, T6H);
                T6P = VADD(T6G, T6H);
                {
                    V T6j, T6q, T6J, T6K;
                    T6j = VFNMS(LDK(KP382683432), T6i, VMUL(LDK(KP923879532), T6f));
                    T6q = VFMA(LDK(KP923879532), T6m, VMUL(LDK(KP382683432), T6p));
                    T6r = VSUB(T6j, T6q);
                    T6Q = VADD(T6j, T6q);
                    T6J = VFMA(LDK(KP382683432), T6f, VMUL(LDK(KP923879532), T6i));
                    T6K = VFNMS(LDK(KP382683432), T6m, VMUL(LDK(KP923879532), T6p));
                    T6L = VSUB(T6J, T6K);
                    T6T = VADD(T6J, T6K);
                }
                {
                    V T6D, T6M, T6V, T6W;
                    T6D = VBYI(VSUB(T6r, T6C));
                    T6M = VSUB(T6I, T6L);
                    ST(&(xo[WS(os, 20)]), VADD(T6D, T6M), ovs, &(xo[0]));
                    ST(&(xo[WS(os, 44)]), VSUB(T6M, T6D), ovs, &(xo[0]));
                    T6V = VSUB(T6P, T6Q);
                    T6W = VBYI(VSUB(T6T, T6S));
                    ST(&(xo[WS(os, 36)]), VSUB(T6V, T6W), ovs, &(xo[0]));
                    ST(&(xo[WS(os, 28)]), VADD(T6V, T6W), ovs, &(xo[0]));
                }
                {
                    V T6N, T6O, T6R, T6U;
                    T6N = VBYI(VADD(T6C, T6r));
                    T6O = VADD(T6I, T6L);
                    ST(&(xo[WS(os, 12)]), VADD(T6N, T6O), ovs, &(xo[0]));
                    ST(&(xo[WS(os, 52)]), VSUB(T6O, T6N), ovs, &(xo[0]));
                    T6R = VADD(T6P, T6Q);
                    T6U = VBYI(VADD(T6S, T6T));
                    ST(&(xo[WS(os, 60)]), VSUB(T6R, T6U), ovs, &(xo[0]));
                    ST(&(xo[WS(os, 4)]), VADD(T6R, T6U), ovs, &(xo[0]));
                }
            }
            {
                V T5N, T68, T61, T69, T5U, T65, T5Y, T66;
                {
                    V T5L, T5M, T5Z, T60;
                    T5L = VADD(T4p, T4w);
                    T5M = VADD(T5o, T5p);
                    T5N = VSUB(T5L, T5M);
                    T68 = VADD(T5L, T5M);
                    T5Z = VFNMS(LDK(KP195090322), T5O, VMUL(LDK(KP980785280), T5P));
                    T60 = VFMA(LDK(KP195090322), T5R, VMUL(LDK(KP980785280), T5S));
                    T61 = VSUB(T5Z, T60);
                    T69 = VADD(T5Z, T60);
                }
                {
                    V T5Q, T5T, T5W, T5X;
                    T5Q = VFMA(LDK(KP980785280), T5O, VMUL(LDK(KP195090322), T5P));
                    T5T = VFNMS(LDK(KP195090322), T5S, VMUL(LDK(KP980785280), T5R));
                    T5U = VSUB(T5Q, T5T);
                    T65 = VADD(T5Q, T5T);
                    T5W = VADD(T4E, T4L);
                    T5X = VADD(T5u, T5r);
                    T5Y = VSUB(T5W, T5X);
                    T66 = VADD(T5X, T5W);
                }
                {
                    V T5V, T62, T6b, T6c;
                    T5V = VADD(T5N, T5U);
                    T62 = VBYI(VADD(T5Y, T61));
                    ST(&(xo[WS(os, 50)]), VSUB(T5V, T62), ovs, &(xo[0]));
                    ST(&(xo[WS(os, 14)]), VADD(T5V, T62), ovs, &(xo[0]));
                    T6b = VBYI(VADD(T66, T65));
                    T6c = VADD(T68, T69);
                    ST(&(xo[WS(os, 2)]), VADD(T6b, T6c), ovs, &(xo[0]));
                    ST(&(xo[WS(os, 62)]), VSUB(T6c, T6b), ovs, &(xo[0]));
                }
                {
                    V T63, T64, T67, T6a;
                    T63 = VSUB(T5N, T5U);
                    T64 = VBYI(VSUB(T61, T5Y));
                    ST(&(xo[WS(os, 46)]), VSUB(T63, T64), ovs, &(xo[0]));
                    ST(&(xo[WS(os, 18)]), VADD(T63, T64), ovs, &(xo[0]));
                    T67 = VBYI(VSUB(T65, T66));
                    T6a = VSUB(T68, T69);
                    ST(&(xo[WS(os, 30)]), VADD(T67, T6a), ovs, &(xo[0]));
                    ST(&(xo[WS(os, 34)]), VSUB(T6a, T67), ovs, &(xo[0]));
                }
            }
            {
                V T11, T2C, T2v, T2D, T2e, T2z, T2s, T2A;
                {
                    V Tr, T10, T2t, T2u;
                    Tr = VSUB(Tb, Tq);
                    T10 = VSUB(TI, TZ);
                    T11 = VSUB(Tr, T10);
                    T2C = VADD(Tr, T10);
                    T2t = VFNMS(LDK(KP471396736), T1s, VMUL(LDK(KP881921264), T1B));
                    T2u = VFMA(LDK(KP471396736), T23, VMUL(LDK(KP881921264), T2c));
                    T2v = VSUB(T2t, T2u);
                    T2D = VADD(T2t, T2u);
                }
                {
                    V T1C, T2d, T2i, T2r;
                    T1C = VFMA(LDK(KP881921264), T1s, VMUL(LDK(KP471396736), T1B));
                    T2d = VFNMS(LDK(KP471396736), T2c, VMUL(LDK(KP881921264), T23));
                    T2e = VSUB(T1C, T2d);
                    T2z = VADD(T1C, T2d);
                    T2i = VSUB(T2g, T2h);
                    T2r = VSUB(T2l, T2q);
                    T2s = VSUB(T2i, T2r);
                    T2A = VADD(T2r, T2i);
                }
                {
                    V T2f, T2w, T2F, T2G;
                    T2f = VADD(T11, T2e);
                    T2w = VBYI(VADD(T2s, T2v));
                    ST(&(xo[WS(os, 53)]), VSUB(T2f, T2w), ovs, &(xo[WS(os, 1)]));
                    ST(&(xo[WS(os, 11)]), VADD(T2f, T2w), ovs, &(xo[WS(os, 1)]));
                    T2F = VBYI(VADD(T2A, T2z));
                    T2G = VADD(T2C, T2D);
                    ST(&(xo[WS(os, 5)]), VADD(T2F, T2G), ovs, &(xo[WS(os, 1)]));
                    ST(&(xo[WS(os, 59)]), VSUB(T2G, T2F), ovs, &(xo[WS(os, 1)]));
                }
                {
                    V T2x, T2y, T2B, T2E;
                    T2x = VSUB(T11, T2e);
                    T2y = VBYI(VSUB(T2v, T2s));
                    ST(&(xo[WS(os, 43)]), VSUB(T2x, T2y), ovs, &(xo[WS(os, 1)]));
                    ST(&(xo[WS(os, 21)]), VADD(T2x, T2y), ovs, &(xo[WS(os, 1)]));
                    T2B = VBYI(VSUB(T2z, T2A));
                    T2E = VSUB(T2C, T2D);
                    ST(&(xo[WS(os, 27)]), VADD(T2B, T2E), ovs, &(xo[WS(os, 1)]));
                    ST(&(xo[WS(os, 37)]), VSUB(T2E, T2B), ovs, &(xo[WS(os, 1)]));
                }
            }
            {
                V T3n, T3O, T3J, T3R, T3y, T3Q, T3G, T3N;
                {
                    V T3f, T3m, T3H, T3I;
                    T3f = VFNMS(LDK(KP098017140), T3e, VMUL(LDK(KP995184726), T3b));
                    T3m = VFMA(LDK(KP995184726), T3i, VMUL(LDK(KP098017140), T3l));
                    T3n = VSUB(T3f, T3m);
                    T3O = VADD(T3f, T3m);
                    T3H = VFMA(LDK(KP098017140), T3b, VMUL(LDK(KP995184726), T3e));
                    T3I = VFNMS(LDK(KP098017140), T3i, VMUL(LDK(KP995184726), T3l));
                    T3J = VSUB(T3H, T3I);
                    T3R = VADD(T3H, T3I);
                }
                {
                    V T3u, T3x, T3C, T3F;
                    T3u = VADD(T3q, T3t);
                    T3x = VADD(T3v, T3w);
                    T3y = VSUB(T3u, T3x);
                    T3Q = VADD(T3x, T3u);
                    T3C = VADD(T3A, T3B);
                    T3F = VADD(T3D, T3E);
                    T3G = VSUB(T3C, T3F);
                    T3N = VADD(T3C, T3F);
                }
                {
                    V T3z, T3K, T3T, T3U;
                    T3z = VBYI(VSUB(T3n, T3y));
                    T3K = VSUB(T3G, T3J);
                    ST(&(xo[WS(os, 17)]), VADD(T3z, T3K), ovs, &(xo[WS(os, 1)]));
                    ST(&(xo[WS(os, 47)]), VSUB(T3K, T3z), ovs, &(xo[WS(os, 1)]));
                    T3T = VSUB(T3N, T3O);
                    T3U = VBYI(VSUB(T3R, T3Q));
                    ST(&(xo[WS(os, 33)]), VSUB(T3T, T3U), ovs, &(xo[WS(os, 1)]));
                    ST(&(xo[WS(os, 31)]), VADD(T3T, T3U), ovs, &(xo[WS(os, 1)]));
                }
                {
                    V T3L, T3M, T3P, T3S;
                    T3L = VBYI(VADD(T3y, T3n));
                    T3M = VADD(T3G, T3J);
                    ST(&(xo[WS(os, 15)]), VADD(T3L, T3M), ovs, &(xo[WS(os, 1)]));
                    ST(&(xo[WS(os, 49)]), VSUB(T3M, T3L), ovs, &(xo[WS(os, 1)]));
                    T3P = VADD(T3N, T3O);
                    T3S = VBYI(VADD(T3Q, T3R));
                    ST(&(xo[WS(os, 63)]), VSUB(T3P, T3S), ovs, &(xo[WS(os, 1)]));
                    ST(&(xo[WS(os, 1)]), VADD(T3P, T3S), ovs, &(xo[WS(os, 1)]));
                }
            }
            {
                V T4N, T5G, T5z, T5H, T5m, T5D, T5w, T5E;
                {
                    V T4x, T4M, T5x, T5y;
                    T4x = VSUB(T4p, T4w);
                    T4M = VSUB(T4E, T4L);
                    T4N = VSUB(T4x, T4M);
                    T5G = VADD(T4x, T4M);
                    T5x = VFNMS(LDK(KP555570233), T4Y, VMUL(LDK(KP831469612), T53));
                    T5y = VFMA(LDK(KP555570233), T5f, VMUL(LDK(KP831469612), T5k));
                    T5z = VSUB(T5x, T5y);
                    T5H = VADD(T5x, T5y);
                }
                {
                    V T54, T5l, T5q, T5v;
                    T54 = VFMA(LDK(KP831469612), T4Y, VMUL(LDK(KP555570233), T53));
                    T5l = VFNMS(LDK(KP555570233), T5k, VMUL(LDK(KP831469612), T5f));
                    T5m = VSUB(T54, T5l);
                    T5D = VADD(T54, T5l);
                    T5q = VSUB(T5o, T5p);
                    T5v = VSUB(T5r, T5u);
                    T5w = VSUB(T5q, T5v);
                    T5E = VADD(T5v, T5q);
                }
                {
                    V T5n, T5A, T5J, T5K;
                    T5n = VADD(T4N, T5m);
                    T5A = VBYI(VADD(T5w, T5z));
                    ST(&(xo[WS(os, 54)]), VSUB(T5n, T5A), ovs, &(xo[0]));
                    ST(&(xo[WS(os, 10)]), VADD(T5n, T5A), ovs, &(xo[0]));
                    T5J = VBYI(VADD(T5E, T5D));
                    T5K = VADD(T5G, T5H);
                    ST(&(xo[WS(os, 6)]), VADD(T5J, T5K), ovs, &(xo[0]));
                    ST(&(xo[WS(os, 58)]), VSUB(T5K, T5J), ovs, &(xo[0]));
                }
                {
                    V T5B, T5C, T5F, T5I;
                    T5B = VSUB(T4N, T5m);
                    T5C = VBYI(VSUB(T5z, T5w));
                    ST(&(xo[WS(os, 42)]), VSUB(T5B, T5C), ovs, &(xo[0]));
                    ST(&(xo[WS(os, 22)]), VADD(T5B, T5C), ovs, &(xo[0]));
                    T5F = VBYI(VSUB(T5D, T5E));
                    T5I = VSUB(T5G, T5H);
                    ST(&(xo[WS(os, 26)]), VADD(T5F, T5I), ovs, &(xo[0]));
                    ST(&(xo[WS(os, 38)]), VSUB(T5I, T5F), ovs, &(xo[0]));
                }
            }
            {
                V T2J, T34, T2X, T35, T2Q, T31, T2U, T32;
                {
                    V T2H, T2I, T2V, T2W;
                    T2H = VADD(Tb, Tq);
                    T2I = VADD(T2g, T2h);
                    T2J = VSUB(T2H, T2I);
                    T34 = VADD(T2H, T2I);
                    T2V = VFNMS(LDK(KP290284677), T2K, VMUL(LDK(KP956940335), T2L));
                    T2W = VFMA(LDK(KP290284677), T2N, VMUL(LDK(KP956940335), T2O));
                    T2X = VSUB(T2V, T2W);
                    T35 = VADD(T2V, T2W);
                }
                {
                    V T2M, T2P, T2S, T2T;
                    T2M = VFMA(LDK(KP956940335), T2K, VMUL(LDK(KP290284677), T2L));
                    T2P = VFNMS(LDK(KP290284677), T2O, VMUL(LDK(KP956940335), T2N));
                    T2Q = VSUB(T2M, T2P);
                    T31 = VADD(T2M, T2P);
                    T2S = VADD(TI, TZ);
                    T2T = VADD(T2q, T2l);
                    T2U = VSUB(T2S, T2T);
                    T32 = VADD(T2T, T2S);
                }
                {
                    V T2R, T2Y, T37, T38;
                    T2R = VADD(T2J, T2Q);
                    T2Y = VBYI(VADD(T2U, T2X));
                    ST(&(xo[WS(os, 51)]), VSUB(T2R, T2Y), ovs, &(xo[WS(os, 1)]));
                    ST(&(xo[WS(os, 13)]), VADD(T2R, T2Y), ovs, &(xo[WS(os, 1)]));
                    T37 = VBYI(VADD(T32, T31));
                    T38 = VADD(T34, T35);
                    ST(&(xo[WS(os, 3)]), VADD(T37, T38), ovs, &(xo[WS(os, 1)]));
                    ST(&(xo[WS(os, 61)]), VSUB(T38, T37), ovs, &(xo[WS(os, 1)]));
                }
                {
                    V T2Z, T30, T33, T36;
                    T2Z = VSUB(T2J, T2Q);
                    T30 = VBYI(VSUB(T2X, T2U));
                    ST(&(xo[WS(os, 45)]), VSUB(T2Z, T30), ovs, &(xo[WS(os, 1)]));
                    ST(&(xo[WS(os, 19)]), VADD(T2Z, T30), ovs, &(xo[WS(os, 1)]));
                    T33 = VBYI(VSUB(T31, T32));
                    T36 = VSUB(T34, T35);
                    ST(&(xo[WS(os, 29)]), VADD(T33, T36), ovs, &(xo[WS(os, 1)]));
                    ST(&(xo[WS(os, 35)]), VSUB(T36, T33), ovs, &(xo[WS(os, 1)]));
                }
            }
            {
                V T41, T4g, T4b, T4j, T44, T4i, T48, T4f;
                {
                    V T3X, T40, T49, T4a;
                    T3X = VFNMS(LDK(KP634393284), T3W, VMUL(LDK(KP773010453), T3V));
                    T40 = VFMA(LDK(KP773010453), T3Y, VMUL(LDK(KP634393284), T3Z));
                    T41 = VSUB(T3X, T40);
                    T4g = VADD(T3X, T40);
                    T49 = VFMA(LDK(KP634393284), T3V, VMUL(LDK(KP773010453), T3W));
                    T4a = VFNMS(LDK(KP634393284), T3Y, VMUL(LDK(KP773010453), T3Z));
                    T4b = VSUB(T49, T4a);
                    T4j = VADD(T49, T4a);
                }
                {
                    V T42, T43, T46, T47;
                    T42 = VSUB(T3D, T3E);
                    T43 = VSUB(T3w, T3v);
                    T44 = VSUB(T42, T43);
                    T4i = VADD(T43, T42);
                    T46 = VSUB(T3A, T3B);
                    T47 = VSUB(T3q, T3t);
                    T48 = VSUB(T46, T47);
                    T4f = VADD(T46, T47);
                }
                {
                    V T45, T4c, T4l, T4m;
                    T45 = VBYI(VSUB(T41, T44));
                    T4c = VSUB(T48, T4b);
                    ST(&(xo[WS(os, 23)]), VADD(T45, T4c), ovs, &(xo[WS(os, 1)]));
                    ST(&(xo[WS(os, 41)]), VSUB(T4c, T45), ovs, &(xo[WS(os, 1)]));
                    T4l = VSUB(T4f, T4g);
                    T4m = VBYI(VSUB(T4j, T4i));
                    ST(&(xo[WS(os, 39)]), VSUB(T4l, T4m), ovs, &(xo[WS(os, 1)]));
                    ST(&(xo[WS(os, 25)]), VADD(T4l, T4m), ovs, &(xo[WS(os, 1)]));
                }
                {
                    V T4d, T4e, T4h, T4k;
                    T4d = VBYI(VADD(T44, T41));
                    T4e = VADD(T48, T4b);
                    ST(&(xo[WS(os, 9)]), VADD(T4d, T4e), ovs, &(xo[WS(os, 1)]));
                    ST(&(xo[WS(os, 55)]), VSUB(T4e, T4d), ovs, &(xo[WS(os, 1)]));
                    T4h = VADD(T4f, T4g);
                    T4k = VBYI(VADD(T4i, T4j));
                    ST(&(xo[WS(os, 57)]), VSUB(T4h, T4k), ovs, &(xo[WS(os, 1)]));
                    ST(&(xo[WS(os, 7)]), VADD(T4h, T4k), ovs, &(xo[WS(os, 1)]));
                }
            }
        }
    }
    VLEAVE();
}
// rdft/scalar/r2cb/r2cb_64.c
__device__ void r2cb_64(R *R0, R *R1, R *Cr, R *Ci, stride rs, stride csr, stride csi, INT v,
                        INT ivs, INT ovs) {
    DK(KP1_268786568, +1.268786568327290996430343226450986741351374190);
    DK(KP1_546020906, +1.546020906725473921621813219516939601942082586);
    DK(KP196034280, +0.196034280659121203988391127777283691722273346);
    DK(KP1_990369453, +1.990369453344393772489673906218959843150949737);
    DK(KP942793473, +0.942793473651995297112775251810508755314920638);
    DK(KP1_763842528, +1.763842528696710059425513727320776699016885241);
    DK(KP580569354, +0.580569354508924735272384751634790549382952557);
    DK(KP1_913880671, +1.913880671464417729871595773960539938965698411);
    DK(KP1_111140466, +1.111140466039204449485661627897065748749874382);
    DK(KP1_662939224, +1.662939224605090474157576755235811513477121624);
    DK(KP390180644, +0.390180644032256535696569736954044481855383236);
    DK(KP1_961570560, +1.961570560806460898252364472268478073947867462);
    DK(KP923879532, +0.923879532511286756128183189396788286822416626);
    DK(KP382683432, +0.382683432365089771728459984030398866761344562);
    DK(KP707106781, +0.707106781186547524400844362104849039284835938);
    DK(KP765366864, +0.765366864730179543456919968060797733522689125);
    DK(KP1_847759065, +1.847759065022573512256366378793576573644833252);
    DK(KP1_414213562, +1.414213562373095048801688724209698078569671875);
    DK(KP2_000000000, +2.000000000000000000000000000000000000000000000);
    {
        INT i;
        for (i = v; i > 0; i = i - 1, R0 = R0 + ovs, R1 = R1 + ovs, Cr = Cr + ivs, Ci = Ci + ivs,
            MAKE_VOLATILE_STRIDE(256, rs), MAKE_VOLATILE_STRIDE(256, csr),
            MAKE_VOLATILE_STRIDE(256, csi)) {
            E Ta, T2S, T18, T2u, T3F, T4V, T5l, T61, Th, T2T, T1h, T2v, T3M, T4W, T5o;
            E T62, T3Q, T5q, T5u, T44, Tp, Tw, T2V, T2W, T2X, T2Y, T3X, T5t, T1r, T2x;
            E T41, T5r, T1A, T2y, T4a, T5y, T5N, T4H, TN, T31, T4E, T5z, T39, T3q, T1L;
            E T2B, T4h, T5M, T2h, T2F, T12, T36, T5D, T5J, T5G, T5K, T1U, T26, T23, T27;
            E T4p, T4z, T4w, T4A, T34, T3r;
            {
                E T5, T3A, T3, T3y, T9, T3C, T17, T3D, T6, T14;
                {
                    E T4, T3z, T1, T2;
                    T4 = Cr[WS(csr, 16)];
                    T5 = KP2_000000000 * T4;
                    T3z = Ci[WS(csi, 16)];
                    T3A = KP2_000000000 * T3z;
                    T1 = Cr[0];
                    T2 = Cr[WS(csr, 32)];
                    T3 = T1 + T2;
                    T3y = T1 - T2;
                    {
                        E T7, T8, T15, T16;
                        T7 = Cr[WS(csr, 8)];
                        T8 = Cr[WS(csr, 24)];
                        T9 = KP2_000000000 * (T7 + T8);
                        T3C = T7 - T8;
                        T15 = Ci[WS(csi, 8)];
                        T16 = Ci[WS(csi, 24)];
                        T17 = KP2_000000000 * (T15 - T16);
                        T3D = T15 + T16;
                    }
                }
                T6 = T3 + T5;
                Ta = T6 + T9;
                T2S = T6 - T9;
                T14 = T3 - T5;
                T18 = T14 - T17;
                T2u = T14 + T17;
                {
                    E T3B, T3E, T5j, T5k;
                    T3B = T3y - T3A;
                    T3E = KP1_414213562 * (T3C - T3D);
                    T3F = T3B + T3E;
                    T4V = T3B - T3E;
                    T5j = T3y + T3A;
                    T5k = KP1_414213562 * (T3C + T3D);
                    T5l = T5j - T5k;
                    T61 = T5j + T5k;
                }
            }
            {
                E Td, T3G, T1c, T3K, Tg, T3J, T1f, T3H, T19, T1g;
                {
                    E Tb, Tc, T1a, T1b;
                    Tb = Cr[WS(csr, 4)];
                    Tc = Cr[WS(csr, 28)];
                    Td = Tb + Tc;
                    T3G = Tb - Tc;
                    T1a = Ci[WS(csi, 4)];
                    T1b = Ci[WS(csi, 28)];
                    T1c = T1a - T1b;
                    T3K = T1a + T1b;
                }
                {
                    E Te, Tf, T1d, T1e;
                    Te = Cr[WS(csr, 20)];
                    Tf = Cr[WS(csr, 12)];
                    Tg = Te + Tf;
                    T3J = Te - Tf;
                    T1d = Ci[WS(csi, 20)];
                    T1e = Ci[WS(csi, 12)];
                    T1f = T1d - T1e;
                    T3H = T1d + T1e;
                }
                Th = KP2_000000000 * (Td + Tg);
                T2T = KP2_000000000 * (T1f + T1c);
                T19 = Td - Tg;
                T1g = T1c - T1f;
                T1h = KP1_414213562 * (T19 - T1g);
                T2v = KP1_414213562 * (T19 + T1g);
                {
                    E T3I, T3L, T5m, T5n;
                    T3I = T3G - T3H;
                    T3L = T3J + T3K;
                    T3M = FNMS(KP765366864, T3L, KP1_847759065 * T3I);
                    T4W = FMA(KP765366864, T3I, KP1_847759065 * T3L);
                    T5m = T3G + T3H;
                    T5n = T3K - T3J;
                    T5o = FNMS(KP1_847759065, T5n, KP765366864 * T5m);
                    T62 = FMA(KP1_847759065, T5m, KP765366864 * T5n);
                }
            }
            {
                E Tl, T3O, T1v, T43, To, T42, T1y, T3P, Ts, T3R, T1p, T3S, Tv, T3U, T1m;
                E T3V;
                {
                    E Tj, Tk, T1t, T1u;
                    Tj = Cr[WS(csr, 2)];
                    Tk = Cr[WS(csr, 30)];
                    Tl = Tj + Tk;
                    T3O = Tj - Tk;
                    T1t = Ci[WS(csi, 2)];
                    T1u = Ci[WS(csi, 30)];
                    T1v = T1t - T1u;
                    T43 = T1t + T1u;
                }
                {
                    E Tm, Tn, T1w, T1x;
                    Tm = Cr[WS(csr, 18)];
                    Tn = Cr[WS(csr, 14)];
                    To = Tm + Tn;
                    T42 = Tm - Tn;
                    T1w = Ci[WS(csi, 18)];
                    T1x = Ci[WS(csi, 14)];
                    T1y = T1w - T1x;
                    T3P = T1w + T1x;
                }
                {
                    E Tq, Tr, T1n, T1o;
                    Tq = Cr[WS(csr, 10)];
                    Tr = Cr[WS(csr, 22)];
                    Ts = Tq + Tr;
                    T3R = Tq - Tr;
                    T1n = Ci[WS(csi, 10)];
                    T1o = Ci[WS(csi, 22)];
                    T1p = T1n - T1o;
                    T3S = T1n + T1o;
                }
                {
                    E Tt, Tu, T1k, T1l;
                    Tt = Cr[WS(csr, 6)];
                    Tu = Cr[WS(csr, 26)];
                    Tv = Tt + Tu;
                    T3U = Tt - Tu;
                    T1k = Ci[WS(csi, 26)];
                    T1l = Ci[WS(csi, 6)];
                    T1m = T1k - T1l;
                    T3V = T1l + T1k;
                }
                T3Q = T3O - T3P;
                T5q = T3O + T3P;
                T5u = T43 - T42;
                T44 = T42 + T43;
                Tp = Tl + To;
                Tw = Ts + Tv;
                T2V = Tp - Tw;
                {
                    E T3T, T3W, T1j, T1q;
                    T2W = T1y + T1v;
                    T2X = T1p + T1m;
                    T2Y = T2W - T2X;
                    T3T = T3R - T3S;
                    T3W = T3U - T3V;
                    T3X = KP707106781 * (T3T + T3W);
                    T5t = KP707106781 * (T3T - T3W);
                    T1j = Tl - To;
                    T1q = T1m - T1p;
                    T1r = T1j + T1q;
                    T2x = T1j - T1q;
                    {
                        E T3Z, T40, T1s, T1z;
                        T3Z = T3R + T3S;
                        T40 = T3U + T3V;
                        T41 = KP707106781 * (T3Z - T40);
                        T5r = KP707106781 * (T3Z + T40);
                        T1s = Ts - Tv;
                        T1z = T1v - T1y;
                        T1A = T1s + T1z;
                        T2y = T1z - T1s;
                    }
                }
            }
            {
                E TB, T48, T2c, T4G, TE, T4F, T2f, T49, TI, T4b, T1J, T4c, TL, T4e, T1G;
                E T4f;
                {
                    E Tz, TA, T2a, T2b;
                    Tz = Cr[WS(csr, 1)];
                    TA = Cr[WS(csr, 31)];
                    TB = Tz + TA;
                    T48 = Tz - TA;
                    T2a = Ci[WS(csi, 1)];
                    T2b = Ci[WS(csi, 31)];
                    T2c = T2a - T2b;
                    T4G = T2a + T2b;
                }
                {
                    E TC, TD, T2d, T2e;
                    TC = Cr[WS(csr, 17)];
                    TD = Cr[WS(csr, 15)];
                    TE = TC + TD;
                    T4F = TC - TD;
                    T2d = Ci[WS(csi, 17)];
                    T2e = Ci[WS(csi, 15)];
                    T2f = T2d - T2e;
                    T49 = T2d + T2e;
                }
                {
                    E TG, TH, T1H, T1I;
                    TG = Cr[WS(csr, 9)];
                    TH = Cr[WS(csr, 23)];
                    TI = TG + TH;
                    T4b = TG - TH;
                    T1H = Ci[WS(csi, 9)];
                    T1I = Ci[WS(csi, 23)];
                    T1J = T1H - T1I;
                    T4c = T1H + T1I;
                }
                {
                    E TJ, TK, T1E, T1F;
                    TJ = Cr[WS(csr, 7)];
                    TK = Cr[WS(csr, 25)];
                    TL = TJ + TK;
                    T4e = TJ - TK;
                    T1E = Ci[WS(csi, 25)];
                    T1F = Ci[WS(csi, 7)];
                    T1G = T1E - T1F;
                    T4f = T1F + T1E;
                }
                {
                    E TF, TM, T1D, T1K;
                    T4a = T48 - T49;
                    T5y = T48 + T49;
                    T5N = T4G - T4F;
                    T4H = T4F + T4G;
                    TF = TB + TE;
                    TM = TI + TL;
                    TN = TF + TM;
                    T31 = TF - TM;
                    {
                        E T4C, T4D, T37, T38;
                        T4C = T4b + T4c;
                        T4D = T4e + T4f;
                        T4E = KP707106781 * (T4C - T4D);
                        T5z = KP707106781 * (T4C + T4D);
                        T37 = T2f + T2c;
                        T38 = T1J + T1G;
                        T39 = T37 - T38;
                        T3q = T38 + T37;
                    }
                    T1D = TB - TE;
                    T1K = T1G - T1J;
                    T1L = T1D + T1K;
                    T2B = T1D - T1K;
                    {
                        E T4d, T4g, T29, T2g;
                        T4d = T4b - T4c;
                        T4g = T4e - T4f;
                        T4h = KP707106781 * (T4d + T4g);
                        T5M = KP707106781 * (T4d - T4g);
                        T29 = TI - TL;
                        T2g = T2c - T2f;
                        T2h = T29 + T2g;
                        T2F = T2g - T29;
                    }
                }
            }
            {
                E TQ, T4j, T1P, T4n, TT, T4m, T1S, T4k, TX, T4q, T1Y, T4u, T10, T4t, T21;
                E T4r;
                {
                    E TO, TP, T1N, T1O;
                    TO = Cr[WS(csr, 5)];
                    TP = Cr[WS(csr, 27)];
                    TQ = TO + TP;
                    T4j = TO - TP;
                    T1N = Ci[WS(csi, 5)];
                    T1O = Ci[WS(csi, 27)];
                    T1P = T1N - T1O;
                    T4n = T1N + T1O;
                }
                {
                    E TR, TS, T1Q, T1R;
                    TR = Cr[WS(csr, 21)];
                    TS = Cr[WS(csr, 11)];
                    TT = TR + TS;
                    T4m = TR - TS;
                    T1Q = Ci[WS(csi, 21)];
                    T1R = Ci[WS(csi, 11)];
                    T1S = T1Q - T1R;
                    T4k = T1Q + T1R;
                }
                {
                    E TV, TW, T1W, T1X;
                    TV = Cr[WS(csr, 3)];
                    TW = Cr[WS(csr, 29)];
                    TX = TV + TW;
                    T4q = TV - TW;
                    T1W = Ci[WS(csi, 29)];
                    T1X = Ci[WS(csi, 3)];
                    T1Y = T1W - T1X;
                    T4u = T1X + T1W;
                }
                {
                    E TY, TZ, T1Z, T20;
                    TY = Cr[WS(csr, 13)];
                    TZ = Cr[WS(csr, 19)];
                    T10 = TY + TZ;
                    T4t = TY - TZ;
                    T1Z = Ci[WS(csi, 13)];
                    T20 = Ci[WS(csi, 19)];
                    T21 = T1Z - T20;
                    T4r = T1Z + T20;
                }
                {
                    E TU, T11, T5B, T5C;
                    TU = TQ + TT;
                    T11 = TX + T10;
                    T12 = TU + T11;
                    T36 = TU - T11;
                    T5B = T4j + T4k;
                    T5C = T4n - T4m;
                    T5D = FNMS(KP923879532, T5C, KP382683432 * T5B);
                    T5J = FMA(KP923879532, T5B, KP382683432 * T5C);
                }
                {
                    E T5E, T5F, T1M, T1T;
                    T5E = T4q + T4r;
                    T5F = T4t + T4u;
                    T5G = FNMS(KP923879532, T5F, KP382683432 * T5E);
                    T5K = FMA(KP923879532, T5E, KP382683432 * T5F);
                    T1M = TQ - TT;
                    T1T = T1P - T1S;
                    T1U = T1M - T1T;
                    T26 = T1M + T1T;
                }
                {
                    E T1V, T22, T4l, T4o;
                    T1V = TX - T10;
                    T22 = T1Y - T21;
                    T23 = T1V + T22;
                    T27 = T22 - T1V;
                    T4l = T4j - T4k;
                    T4o = T4m + T4n;
                    T4p = FNMS(KP382683432, T4o, KP923879532 * T4l);
                    T4z = FMA(KP382683432, T4l, KP923879532 * T4o);
                }
                {
                    E T4s, T4v, T32, T33;
                    T4s = T4q - T4r;
                    T4v = T4t - T4u;
                    T4w = FMA(KP923879532, T4s, KP382683432 * T4v);
                    T4A = FNMS(KP382683432, T4s, KP923879532 * T4v);
                    T32 = T21 + T1Y;
                    T33 = T1S + T1P;
                    T34 = T32 - T33;
                    T3r = T33 + T32;
                }
            }
            {
                E T13, T3x, Ty, T3w, Ti, Tx;
                T13 = KP2_000000000 * (TN + T12);
                T3x = KP2_000000000 * (T3r + T3q);
                Ti = Ta + Th;
                Tx = KP2_000000000 * (Tp + Tw);
                Ty = Ti + Tx;
                T3w = Ti - Tx;
                R0[WS(rs, 16)] = Ty - T13;
                R0[WS(rs, 24)] = T3w + T3x;
                R0[0] = Ty + T13;
                R0[WS(rs, 8)] = T3w - T3x;
            }
            {
                E T3g, T3k, T3j, T3l;
                {
                    E T3e, T3f, T3h, T3i;
                    T3e = T2S + T2T;
                    T3f = KP1_414213562 * (T2V + T2Y);
                    T3g = T3e - T3f;
                    T3k = T3e + T3f;
                    T3h = T31 - T34;
                    T3i = T39 - T36;
                    T3j = FNMS(KP1_847759065, T3i, KP765366864 * T3h);
                    T3l = FMA(KP1_847759065, T3h, KP765366864 * T3i);
                }
                R0[WS(rs, 22)] = T3g - T3j;
                R0[WS(rs, 30)] = T3k + T3l;
                R0[WS(rs, 6)] = T3g + T3j;
                R0[WS(rs, 14)] = T3k - T3l;
            }
            {
                E T3o, T3u, T3t, T3v;
                {
                    E T3m, T3n, T3p, T3s;
                    T3m = Ta - Th;
                    T3n = KP2_000000000 * (T2X + T2W);
                    T3o = T3m - T3n;
                    T3u = T3m + T3n;
                    T3p = TN - T12;
                    T3s = T3q - T3r;
                    T3t = KP1_414213562 * (T3p - T3s);
                    T3v = KP1_414213562 * (T3p + T3s);
                }
                R0[WS(rs, 20)] = T3o - T3t;
                R0[WS(rs, 28)] = T3u + T3v;
                R0[WS(rs, 4)] = T3o + T3t;
                R0[WS(rs, 12)] = T3u - T3v;
            }
            {
                E T30, T3c, T3b, T3d;
                {
                    E T2U, T2Z, T35, T3a;
                    T2U = T2S - T2T;
                    T2Z = KP1_414213562 * (T2V - T2Y);
                    T30 = T2U + T2Z;
                    T3c = T2U - T2Z;
                    T35 = T31 + T34;
                    T3a = T36 + T39;
                    T3b = FNMS(KP765366864, T3a, KP1_847759065 * T35);
                    T3d = FMA(KP765366864, T35, KP1_847759065 * T3a);
                }
                R0[WS(rs, 18)] = T30 - T3b;
                R0[WS(rs, 26)] = T3c + T3d;
                R0[WS(rs, 2)] = T30 + T3b;
                R0[WS(rs, 10)] = T3c - T3d;
            }
            {
                E T25, T2p, T2i, T2q, T1C, T2k, T2o, T2s, T24, T28;
                T24 = KP707106781 * (T1U + T23);
                T25 = T1L + T24;
                T2p = T1L - T24;
                T28 = KP707106781 * (T26 + T27);
                T2i = T28 + T2h;
                T2q = T2h - T28;
                {
                    E T1i, T1B, T2m, T2n;
                    T1i = T18 + T1h;
                    T1B = FNMS(KP765366864, T1A, KP1_847759065 * T1r);
                    T1C = T1i + T1B;
                    T2k = T1i - T1B;
                    T2m = T18 - T1h;
                    T2n = FMA(KP765366864, T1r, KP1_847759065 * T1A);
                    T2o = T2m - T2n;
                    T2s = T2m + T2n;
                }
                {
                    E T2j, T2t, T2l, T2r;
                    T2j = FNMS(KP390180644, T2i, KP1_961570560 * T25);
                    R0[WS(rs, 17)] = T1C - T2j;
                    R0[WS(rs, 1)] = T1C + T2j;
                    T2t = FMA(KP1_662939224, T2p, KP1_111140466 * T2q);
                    R0[WS(rs, 13)] = T2s - T2t;
                    R0[WS(rs, 29)] = T2s + T2t;
                    T2l = FMA(KP390180644, T25, KP1_961570560 * T2i);
                    R0[WS(rs, 9)] = T2k - T2l;
                    R0[WS(rs, 25)] = T2k + T2l;
                    T2r = FNMS(KP1_662939224, T2q, KP1_111140466 * T2p);
                    R0[WS(rs, 21)] = T2o - T2r;
                    R0[WS(rs, 5)] = T2o + T2r;
                }
            }
            {
                E T2D, T2N, T2G, T2O, T2A, T2I, T2M, T2Q, T2C, T2E;
                T2C = KP707106781 * (T27 - T26);
                T2D = T2B + T2C;
                T2N = T2B - T2C;
                T2E = KP707106781 * (T1U - T23);
                T2G = T2E + T2F;
                T2O = T2F - T2E;
                {
                    E T2w, T2z, T2K, T2L;
                    T2w = T2u - T2v;
                    T2z = FNMS(KP1_847759065, T2y, KP765366864 * T2x);
                    T2A = T2w + T2z;
                    T2I = T2w - T2z;
                    T2K = T2u + T2v;
                    T2L = FMA(KP1_847759065, T2x, KP765366864 * T2y);
                    T2M = T2K - T2L;
                    T2Q = T2K + T2L;
                }
                {
                    E T2H, T2R, T2J, T2P;
                    T2H = FNMS(KP1_111140466, T2G, KP1_662939224 * T2D);
                    R0[WS(rs, 19)] = T2A - T2H;
                    R0[WS(rs, 3)] = T2A + T2H;
                    T2R = FMA(KP1_961570560, T2N, KP390180644 * T2O);
                    R0[WS(rs, 15)] = T2Q - T2R;
                    R0[WS(rs, 31)] = T2Q + T2R;
                    T2J = FMA(KP1_111140466, T2D, KP1_662939224 * T2G);
                    R0[WS(rs, 11)] = T2I - T2J;
                    R0[WS(rs, 27)] = T2I + T2J;
                    T2P = FNMS(KP1_961570560, T2O, KP390180644 * T2N);
                    R0[WS(rs, 23)] = T2M - T2P;
                    R0[WS(rs, 7)] = T2M + T2P;
                }
            }
            {
                E T5p, T5T, T5w, T5U, T5I, T5W, T5P, T5X, T5s, T5v;
                T5p = T5l + T5o;
                T5T = T5l - T5o;
                T5s = T5q - T5r;
                T5v = T5t + T5u;
                T5w = FNMS(KP1_111140466, T5v, KP1_662939224 * T5s);
                T5U = FMA(KP1_111140466, T5s, KP1_662939224 * T5v);
                {
                    E T5A, T5H, T5L, T5O;
                    T5A = T5y - T5z;
                    T5H = T5D + T5G;
                    T5I = T5A + T5H;
                    T5W = T5A - T5H;
                    T5L = T5J - T5K;
                    T5O = T5M + T5N;
                    T5P = T5L + T5O;
                    T5X = T5O - T5L;
                }
                {
                    E T5x, T5Q, T5Z, T60;
                    T5x = T5p + T5w;
                    T5Q = FNMS(KP580569354, T5P, KP1_913880671 * T5I);
                    R1[WS(rs, 17)] = T5x - T5Q;
                    R1[WS(rs, 1)] = T5x + T5Q;
                    T5Z = T5T + T5U;
                    T60 = FMA(KP1_763842528, T5W, KP942793473 * T5X);
                    R1[WS(rs, 13)] = T5Z - T60;
                    R1[WS(rs, 29)] = T5Z + T60;
                }
                {
                    E T5R, T5S, T5V, T5Y;
                    T5R = T5p - T5w;
                    T5S = FMA(KP580569354, T5I, KP1_913880671 * T5P);
                    R1[WS(rs, 9)] = T5R - T5S;
                    R1[WS(rs, 25)] = T5R + T5S;
                    T5V = T5T - T5U;
                    T5Y = FNMS(KP1_763842528, T5X, KP942793473 * T5W);
                    R1[WS(rs, 21)] = T5V - T5Y;
                    R1[WS(rs, 5)] = T5V + T5Y;
                }
            }
            {
                E T3N, T4N, T46, T4O, T4y, T4Q, T4J, T4R, T3Y, T45;
                T3N = T3F + T3M;
                T4N = T3F - T3M;
                T3Y = T3Q + T3X;
                T45 = T41 + T44;
                T46 = FNMS(KP390180644, T45, KP1_961570560 * T3Y);
                T4O = FMA(KP390180644, T3Y, KP1_961570560 * T45);
                {
                    E T4i, T4x, T4B, T4I;
                    T4i = T4a + T4h;
                    T4x = T4p + T4w;
                    T4y = T4i + T4x;
                    T4Q = T4i - T4x;
                    T4B = T4z + T4A;
                    T4I = T4E + T4H;
                    T4J = T4B + T4I;
                    T4R = T4I - T4B;
                }
                {
                    E T47, T4K, T4T, T4U;
                    T47 = T3N + T46;
                    T4K = FNMS(KP196034280, T4J, KP1_990369453 * T4y);
                    R1[WS(rs, 16)] = T47 - T4K;
                    R1[0] = T47 + T4K;
                    T4T = T4N + T4O;
                    T4U = FMA(KP1_546020906, T4Q, KP1_268786568 * T4R);
                    R1[WS(rs, 12)] = T4T - T4U;
                    R1[WS(rs, 28)] = T4T + T4U;
                }
                {
                    E T4L, T4M, T4P, T4S;
                    T4L = T3N - T46;
                    T4M = FMA(KP196034280, T4y, KP1_990369453 * T4J);
                    R1[WS(rs, 8)] = T4L - T4M;
                    R1[WS(rs, 24)] = T4L + T4M;
                    T4P = T4N - T4O;
                    T4S = FNMS(KP1_546020906, T4R, KP1_268786568 * T4Q);
                    R1[WS(rs, 20)] = T4P - T4S;
                    R1[WS(rs, 4)] = T4P + T4S;
                }
            }
            {
                E T63, T6h, T66, T6i, T6a, T6k, T6d, T6l, T64, T65;
                T63 = T61 - T62;
                T6h = T61 + T62;
                T64 = T5q + T5r;
                T65 = T5u - T5t;
                T66 = FNMS(KP1_961570560, T65, KP390180644 * T64);
                T6i = FMA(KP1_961570560, T64, KP390180644 * T65);
                {
                    E T68, T69, T6b, T6c;
                    T68 = T5y + T5z;
                    T69 = T5J + T5K;
                    T6a = T68 - T69;
                    T6k = T68 + T69;
                    T6b = T5D - T5G;
                    T6c = T5N - T5M;
                    T6d = T6b + T6c;
                    T6l = T6c - T6b;
                }
                {
                    E T67, T6e, T6n, T6o;
                    T67 = T63 + T66;
                    T6e = FNMS(KP1_268786568, T6d, KP1_546020906 * T6a);
                    R1[WS(rs, 19)] = T67 - T6e;
                    R1[WS(rs, 3)] = T67 + T6e;
                    T6n = T6h + T6i;
                    T6o = FMA(KP1_990369453, T6k, KP196034280 * T6l);
                    R1[WS(rs, 15)] = T6n - T6o;
                    R1[WS(rs, 31)] = T6n + T6o;
                }
                {
                    E T6f, T6g, T6j, T6m;
                    T6f = T63 - T66;
                    T6g = FMA(KP1_268786568, T6a, KP1_546020906 * T6d);
                    R1[WS(rs, 11)] = T6f - T6g;
                    R1[WS(rs, 27)] = T6f + T6g;
                    T6j = T6h - T6i;
                    T6m = FNMS(KP1_990369453, T6l, KP196034280 * T6k);
                    R1[WS(rs, 23)] = T6j - T6m;
                    R1[WS(rs, 7)] = T6j + T6m;
                }
            }
            {
                E T4X, T5b, T50, T5c, T54, T5e, T57, T5f, T4Y, T4Z;
                T4X = T4V - T4W;
                T5b = T4V + T4W;
                T4Y = T3Q - T3X;
                T4Z = T44 - T41;
                T50 = FNMS(KP1_662939224, T4Z, KP1_111140466 * T4Y);
                T5c = FMA(KP1_662939224, T4Y, KP1_111140466 * T4Z);
                {
                    E T52, T53, T55, T56;
                    T52 = T4a - T4h;
                    T53 = T4A - T4z;
                    T54 = T52 + T53;
                    T5e = T52 - T53;
                    T55 = T4p - T4w;
                    T56 = T4H - T4E;
                    T57 = T55 + T56;
                    T5f = T56 - T55;
                }
                {
                    E T51, T58, T5h, T5i;
                    T51 = T4X + T50;
                    T58 = FNMS(KP942793473, T57, KP1_763842528 * T54);
                    R1[WS(rs, 18)] = T51 - T58;
                    R1[WS(rs, 2)] = T51 + T58;
                    T5h = T5b + T5c;
                    T5i = FMA(KP1_913880671, T5e, KP580569354 * T5f);
                    R1[WS(rs, 14)] = T5h - T5i;
                    R1[WS(rs, 30)] = T5h + T5i;
                }
                {
                    E T59, T5a, T5d, T5g;
                    T59 = T4X - T50;
                    T5a = FMA(KP942793473, T54, KP1_763842528 * T57);
                    R1[WS(rs, 10)] = T59 - T5a;
                    R1[WS(rs, 26)] = T59 + T5a;
                    T5d = T5b - T5c;
                    T5g = FNMS(KP1_913880671, T5f, KP580569354 * T5e);
                    R1[WS(rs, 22)] = T5d - T5g;
                    R1[WS(rs, 6)] = T5d + T5g;
                }
            }
        }
    }
}
