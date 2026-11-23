"""Lower a native FFTW cubic ESTIMATE plan to single-precision CUDA line kernels."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
import subprocess
import sys
import json

import cupy as cp
import numpy as np


class UnsupportedPlan(ValueError):
    """The reference plan uses arithmetic or a layout without a CUDA lowering."""


@dataclass(frozen=True)
class Node:
    op: str
    code: str
    children: tuple


def parse_tree(tree):
    # The threaded rdft2 printer emits an extra ')' before its child plans.
    tree = re.sub(r"(\(rdft2-thr-vrank[^\n()]*)\)\n", r"\1\n", tree)
    tokens = iter(re.findall(r'\(|\)|"[^"]+"|[^()\s]+', tree))

    def read():
        op = next(tokens)
        code = ""
        children = []
        for token in tokens:
            if token == ")":
                return Node(op, code, tuple(children))
            if token == "(":
                children.append(read())
            elif token.startswith('"'):
                code = token.strip('"')
            else:
                raise UnsupportedPlan(token)
        raise UnsupportedPlan("unterminated FFTW plan")

    if next(tokens) != "(":
        raise UnsupportedPlan("missing FFTW root")
    root = read()
    if next(tokens, None) is not None:
        raise UnsupportedPlan("unexpected trailing plan")
    return root


def signature(node):
    # Vectors and temporary copy buffers only reorder independent lines.
    node = unwrap(node)
    if node.op.startswith("rdft2-rdft-"):
        return (
            node.op.split("-x")[0].split("/")[0],
            tuple(signature(c) for c in node.children),
        )
    return (
        re.sub(r"-x\d+(?:/\d+-\d+)?", "", node.op),
        node.code,
        tuple(signature(c) for c in node.children),
    )


def unwrap(node):
    while True:
        if "-thr-vrank" in node.op:
            children = [unwrap(c) for c in node.children]
            # Batch counts may differ between workers. Compare arithmetic downstream.
            if len(set(signature(c) for c in children)) != 1:
                raise UnsupportedPlan("heterogeneous worker plans")
            node = children[0]
        elif node.op in (
            "dft-indirect-before",
            "dft-indirect-after",
        ) or node.op.startswith("indirect-transpose-"):
            arithmetic = [
                c
                for c in node.children
                if not c.op.startswith(("dft-r2hc-1", "dft-nop"))
            ]
            if len(arithmetic) != 1:
                raise UnsupportedPlan(node.op)
            node = arithmetic[0]
        elif re.match(r"dft-thr-ct-(dit|dif)-x\d+/", node.op):
            rotations = node.children[:-1]
            if len(set(signature(c) for c in rotations)) != 1:
                raise UnsupportedPlan("heterogeneous threaded butterfly")
            direction, radix = re.search(r"ct-(dit|dif)-x\d+/(\d+)", node.op).groups()
            node = Node(
                f"dft-ct-{direction}/{radix}", "", (rotations[0], node.children[-1])
            )
        elif "-vrank" in node.op and len(node.children) == 1:
            node = node.children[0]
        elif (
            node.op.startswith("rdft-buffered-")
            and len(node.children) == 3
            and node.children[-1].op == "rdft-nop"
        ):
            node = node.children[0]
        elif (
            node.op.startswith(("rdft2-buffered-", "dft-buffered-"))
            and len(node.children) == 3
        ):
            child, copy, tail = node.children
            if not copy.op.startswith("dft-r2hc-1") or (
                tail.op not in ("dft-nop", "rdft2-nop")
                and signature(tail) != signature(child)
            ):
                raise UnsupportedPlan("heterogeneous buffered batch")
            node = node.children[0]
        else:
            return node


class Compiler:
    def __init__(self, inverse):
        self.inverse = inverse
        self.codelets = set()
        self.functions = []
        self.weights = []

    def leaf(self, node, pattern):
        if not re.fullmatch(pattern, node.code):
            raise UnsupportedPlan(f"{node.op} {node.code}")
        name = node.code.removesuffix("_sse2")
        self.codelets.add(name)
        return name

    def function(self, body):
        name = f"step{len(self.functions)}"
        self.functions.append(
            f"DEV void {name}(float *a,float *b,const float *w){{\n{body}\n}}"
        )
        return name

    def twiddle(self, n, r, start):
        phase = np.arange(start, n // r)[:, None] * np.arange(1, r)[None, :]
        values = np.exp(2j * np.pi * phase / n).astype(np.complex64)
        values.real[(4 * phase % (2 * n)) == n] = 0
        values.imag[(2 * phase % n) == 0] = 0
        offset = sum(a.size for a in self.weights)
        self.weights.append(values.ravel().view(np.float32))
        return f"(w+{offset})"

    def coefficients(self, phases, n):
        phases = np.asarray(phases, dtype=np.int64) % n
        table = np.exp((2j * np.pi / n) * phases).astype(np.complex64)
        table.real[(4 * phases % (2 * n)) == n] = 0
        table.imag[(2 * phases % n) == 0] = 0
        offset = sum(item.size for item in self.weights)
        self.weights.append(table.ravel().view(np.float32))
        return f"(w+{offset})"

    def bluestein(self, node, n):
        nb = int(node.op.split("/")[1])
        backward = self.inverse
        # FFTW uses the same forward convolution plan for both directions.
        self.inverse = False
        try:
            child = self.complex(node.children[0], nb)
        finally:
            self.inverse = backward
        chirp = self.coefficients(np.arange(n, dtype=np.int64) ** 2, 2 * n)
        x = int(backward)
        y = 1 - x
        return self.function(
            f"""
            const float *c={chirp};
            float t[{2*nb}]={{}}, v[{2*nb}], h[{2*nb}];
            for(int k=0;k<{n};++k){{
                t[2*k]=c[2*k]/{nb}.0f; t[2*k+1]=c[2*k+1]/{nb}.0f;
                if(k)paircopy(t+2*({nb}-k),t+2*k);
            }}
            {child}(t,h,w);
            for(int k=0;k<{n};++k){{
                float xr=a[2*k+{x}],xi=a[2*k+{y}],wr=c[2*k],wi=c[2*k+1];
                t[2*k]=xr*wr+xi*wi; t[2*k+1]=xi*wr-xr*wi;
            }}
            for(int k={2*n};k<{2*nb};++k)t[k]=0;
            {child}(t,v,w);
            for(int k=0;k<{nb};++k){{
                float xr=v[2*k],xi=v[2*k+1],wr=h[2*k],wi=h[2*k+1];
                t[2*k]=xi*wr+xr*wi; t[2*k+1]=xr*wr-xi*wi;
            }}
            {child}(t,v,w);
            for(int k=0;k<{n};++k){{
                float xi=v[2*k],xr=v[2*k+1],wr=c[2*k],wi=c[2*k+1];
                b[2*k+{x}]=xr*wr+xi*wi; b[2*k+{y}]=xi*wr-xr*wi;
            }}"""
        )

    def hartley(self, node, n):
        if not node.op.startswith(f"dht-rader-{n}/"):
            raise UnsupportedPlan(node.op)
        pad = int(node.op.split("/")[1])
        backward = self.inverse
        self.inverse = False
        try:
            transforms = [self.halfcomplex(c, pad) for c in node.children]
        finally:
            self.inverse = backward
        first, second = transforms[:2]
        omega = transforms[-1]
        g = next(
            k for k in range(2, n) if len({pow(k, j, n) for j in range(n - 1)}) == n - 1
        )
        gi = pow(g, -1, n)
        phases = np.array([pow(gi, j, n) for j in range(n - 1)])
        angle = 2 * np.pi * phases / n
        coeff = np.zeros(pad, np.float32)
        coeff[: n - 1] = (np.cos(angle) + np.sin(angle)) / pad
        if pad > n - 1:
            coeff[-(n - 2) :] = coeff[1 : n - 1]
        offset = sum(a.size for a in self.weights)
        self.weights.append(coeff)
        prepare = f"""
            float t[{pad}]={{}},v[{pad}],h[{pad}];
            for(int k=0;k<{pad};++k)t[k]=w[{offset}+k];
            {omega}(t,h,w);
            for(int k=0,j=1;k<{n-1};++k,j=(long long)j*{g}%{n})t[k]=a[j];
            for(int k={n-1};k<{pad};++k)t[k]=0;
            {first}(t,v,w);
            b[0]=a[0]+v[0];t[0]=v[0]*h[0]+a[0];t[{pad//2}]=v[{pad//2}]*h[{pad//2}];
            for(int k=1;k<{pad//2};++k){{
                float re=h[k]*v[k]-h[{pad}-k]*v[{pad}-k];
                float im=h[k]*v[{pad}-k]+h[{pad}-k]*v[k];
                t[k]=re+im;t[{pad}-k]=re-im;
            }}
            {second}(t,v,w);b[1]=v[0];"""
        value = (
            f"v[k]+v[{pad}-k]"
            if pad > n - 1
            else f"(2*k<{pad}?v[k]+v[{pad}-k]:(2*k=={pad}?v[k]:v[{pad}-k]-v[k]))"
        )
        return self.function(
            prepare
            + f"for(int k=1,j={gi};k<{n-1};++k,j=(long long)j*{gi}%{n}) b[j]={value};"
        )

    def halfcomplex(self, node, n):
        """Emit a scalar halfcomplex plan with FFTW's reversed imaginary half."""
        node = unwrap(node)
        d = "b" if self.inverse else "f"
        if node.op.startswith(("r2hc-dht-", "hc2r-dht-")):
            sub = self.hartley(node.children[0], n)
            if self.inverse:
                return self.function(
                    f"float t[{n}];t[0]=a[0];for(int k=1;2*k<{n};++k){{t[k]=a[k]-a[{n}-k];t[{n}-k]=a[k]+a[{n}-k];}} {sub}(t,b,w);"
                )
            return self.function(
                f"{sub}(a,b,w);for(int k=1;2*k<{n};++k){{float x=0.5f*b[k],y=0.5f*b[{n}-k];b[k]=x+y;b[{n}-k]=y-x;}}"
            )
        if node.op.startswith(
            (
                "rdft-r2hc-direct-r2c-",
                "rdft-hc2r-direct-r2c-",
                "rdft-r2hc-directbuf/",
                "rdft-hc2r-directbuf/",
            )
        ):
            fn = self.leaf(node, rf"r2c{d}_{n}")
            real = "b" if self.inverse else "a"
            # Imaginary index zero is unused. Allocate its dummy slot inside scratch.
            return self.function(
                f"""
                float h[{n+1}]; float *r={real};
                {'for(int k=0;k<'+str(n)+';++k) h[k]=a[k];' if self.inverse else ''}
                {fn}(r,r+1,h,h+{n},2,1,-1,1,0,0);
                {'' if self.inverse else 'for(int k=0;k<'+str(n)+';++k) b[k]=h[k];'}"""
            )
        if node.op.startswith("rdft-generic-") and n % 2:
            m = n // 2
            weights = self.coefficients(
                np.arange(1, m + 1)[:, None] * np.arange(1, m + 1), n
            )
            real = "2*a[k]" if self.inverse else f"a[k]+a[{n}-k]"
            imag = f"2*a[{n}-k]" if self.inverse else f"a[{n}-k]-a[k]"
            output = (
                f"b[j]=rr-ii; b[{n}-j]=rr+ii;"
                if self.inverse
                else f"b[j]=rr; b[{n}-j]=ii;"
            )
            return self.function(
                f"""
                float xr[{m+1}],xi[{m+1}]; float dc=a[0];
                for(int k=1;k<={m};++k){{xr[k]={real};xi[k]={imag};dc+=xr[k];}}
                b[0]=dc;
                for(int j=1;j<={m};++j){{float rr=a[0],ii=0;
                    const float *h={weights}+2*(j-1)*{m};
                    for(int k=1;k<={m};++k){{rr+=xr[k]*h[2*k-2];ii+=xi[k]*h[2*k-1];}}
                    {output}
                }}"""
            )
        if not node.op.startswith(("rdft-ct-dit/", "rdft-ct-dif/")):
            raise UnsupportedPlan(node.op)
        r = int(node.op.split("/")[1])
        m = n // r
        butterfly, child = node.children
        zero, mid = butterfly.children
        generic = butterfly.op.startswith("hc2hc-generic-")
        if not generic and m % 2 and mid.op != "rdft-nop":
            raise UnsupportedPlan(mid.op)
        base = self.halfcomplex(zero, r)
        sub = self.halfcomplex(child, m)
        if generic:
            special = self.generic_hc_butterfly(butterfly, r, m)
        else:
            fn = self.leaf(butterfly, rf"h{d}2?_{r}")
            harmonics = codelet_catalog()[1].get(fn, range(1, r))
            weights = self.coefficients(
                np.arange(1, (m + 1) // 2)[:, None] * list(harmonics), n
            )
            special = f"""
                for(int k=0;k<{r};++k)x[k]=t[k*{m}];
                {base}(x,y,w);
                for(int k=0;k<{r};++k)t[k*{m}]=y[k];
                {fn}(t+1,t+{m-1},{weights},{m},1,{(m+1)//2},1);"""
        if not generic and m % 2 == 0:
            midfn = self.leaf(mid, rf"r2c{'bIII' if self.inverse else 'fII'}_{r}")
            special += f"""
                for(int j=0;j<{r};++j)x[j]=t[{m//2}+{m}*j];
                {midfn}({'y,y+1,x,x' if self.inverse else 'x,x+1,y,y'}+{r-1},2,1,-1,1,0,0);
                for(int j=0;j<{r};++j)t[{m//2}+{m}*j]=y[j];"""
        if self.inverse:
            body = f"""
                for(int k=0;k<{n};++k)t[k]=a[k];
                {special}
                for(int j=0;j<{r};++j){{
                    {sub}(t+j*{m},y,w);
                    for(int k=0;k<{m};++k)b[k*{r}+j]=y[k];
                }}"""
        else:
            body = f"""
                for(int j=0;j<{r};++j){{
                    for(int k=0;k<{m};++k)x[k]=a[k*{r}+j];
                    {sub}(x,t+j*{m},w);
                }}
                {special}
                for(int k=0;k<{n};++k)b[k]=t[k];"""
        return self.function(f"float t[{n}],x[{max(r,m)}],y[{max(r,m)}];" + body)

    def generic_complex(self, n):
        half = n // 2
        h = self.coefficients(
            np.arange(1, half + 1)[:, None] * np.arange(1, half + 1), n
        )
        r = int(self.inverse)
        i = 1 - r
        return self.function(
            f"""
        float sx=a[{r}],sy=a[{i}];
        for(int k=1;k<={half};++k){{
            sx+=a[2*k+{r}]+a[2*({n}-k)+{r}];
            sy+=a[2*k+{i}]+a[2*({n}-k)+{i}];
        }}
        b[{r}]=sx; b[{i}]=sy;
        for(int j=1;j<={half};++j){{
            float rr=a[{r}],ri=0,ir=a[{i}],ii=0;
            const float *t={h}+2*(j-1)*{half};
            for(int k=1;k<={half};++k){{
                float x=a[2*k+{r}], y=a[2*k+{i}], xx=a[2*({n}-k)+{r}],yy=a[2*({n}-k)+{i}];
                rr+=(x+xx)*t[2*k-2]; ir+=(y+yy)*t[2*k-2];
                ri+=(x-xx)*t[2*k-1]; ii+=(y-yy)*t[2*k-1];
            }}
            b[2*j+{r}]=rr+ii; b[2*j+{i}]=ir-ri;
            b[2*({n}-j)+{r}]=rr-ii; b[2*({n}-j)+{i}]=ir+ri;
        }}"""
        )

    def rader(self, node, n):
        m = n - 1
        if len(node.children) != 3:
            raise UnsupportedPlan("ambiguous Rader child layout")
        backward = self.inverse
        self.inverse = False
        try:
            first, second, omega = [self.complex(c, m) for c in node.children]
        finally:
            self.inverse = backward
        g = next(k for k in range(2, n) if len({pow(k, j, n) for j in range(m)}) == m)
        gi = pow(g, -1, n)
        phase = np.array([pow(gi, j, n) for j in range(m)])
        table = (np.exp((-2j * np.pi / n) * phase) / m).astype(np.complex64)
        offset = sum(x.size for x in self.weights)
        self.weights.append(table.view(np.float32))
        x = int(backward)
        y = 1 - x
        return self.function(
            f"""
            float t[{2*m}],v[{2*m}],h[{2*m}];
            for(int k=0;k<{2*m};++k)t[k]=w[{offset}+k];
            {omega}(t,h,w);
            for(int k=0,j=1;k<{m};++k,j=(long long)j*{g}%{n}){{
                t[2*k]=a[2*j+{x}];t[2*k+1]=a[2*j+{y}];
            }}
            {first}(t,v,w);
            b[{x}]=a[{x}]+v[0];b[{y}]=a[{y}]+v[1];
            for(int k=0;k<{m};++k){{
                float xr=v[2*k],xi=v[2*k+1],wr=h[2*k],wi=h[2*k+1];
                t[2*k]=wr*xr-wi*xi;t[2*k+1]=-(wr*xi+wi*xr);
            }}
            t[0]+=a[{x}];t[1]-=a[{y}];
            {second}(t,v,w);
            for(int k=0,j=1;k<{m};++k,j=(long long)j*{gi}%{n}){{
                b[2*j+{x}]=v[2*k];b[2*j+{y}]=-v[2*k+1];
            }}"""
        )

    def generic_hc_butterfly(self, node, r, m):
        zero, other = node.children
        dc = self.halfcomplex(zero, r)
        ac = self.halfcomplex(other, r)
        h = self.coefficients(np.arange(r)[:, None] * np.arange(1, (m + 1) // 2), m * r)
        # Port of FFTW rdft/hc2hc-generic.c, with independent lines packed
        # contiguously. Columns j and m-j represent a conjugate frequency pair.
        sign = 1 if self.inverse else -1
        twiddle = f"""
            for(int k=1;k<{r};++k)for(int j=1;2*j<{m};++j){{
                float *p=t+{m}*k;int a={m}-j;
                const float *w2={h}+2*(k*{m//2}+j-1);
                float xr=p[j],xi=p[a],wr=w2[0],wi={sign}*w2[1];
                p[j]=xr*wr-xi*wi;p[a]=xi*wr+xr*wi;
            }}"""
        transforms = f"""
            for(int j=0;j<{m};++j){{
                for(int k=0;k<{r};++k)x[k]=t[k*{m}+j];
                if(j==0) {dc}(x,y,w); else {ac}(x,y,w);
                for(int k=0;k<{r};++k)t[k*{m}+j]=y[k];
            }}"""
        swap = f"""
            for(int k=0;2*k<{r};++k)for(int j=1;2*j<{m};++j){{
                float *p=t+(k+1)*{m}-j,*q=t+({r}-k)*{m}-j;
                float v=*p;*p=*q;*q=v;
            }}"""
        reorder = f"""
            for(int k=1;2*k<{r};++k)for(int j=1;2*j<{m};++j){{
                float *p=t+k*{m},*q=t+({r}-k)*{m};int a={m}-j;
                float rp=p[j],im=q[a],rm=q[j],ip=p[a];"""
        if self.inverse:
            reorder += "rp*=0.5f;im*=0.5f;rm*=0.5f;ip*=0.5f;p[j]=rp+im;q[a]=im-rp;q[j]=rm+ip;p[a]=ip-rm;}"
            return swap + reorder + transforms + twiddle
        return (
            twiddle
            + transforms
            + reorder
            + "p[j]=rp-im;q[a]=rp+im;q[j]=rm-ip;p[a]=ip+rm;}"
            + swap
        )

    def generic_complex_ct(self, n, r, tw, child, dit):
        m = n // r
        radix = self.complex(tw.children[0], r)
        line = self.complex(child, m)
        h = self.coefficients(np.arange(r)[:, None] * np.arange(m), n)
        sign = 1 if self.inverse else -1
        rotate = f"""
            for(int j=1;j<{r};++j)for(int k=1;k<{m};++k){{
                int i=2*(j*{m}+k);float xr=t[i],xi=t[i+1],wr={h}[i],wi={sign}*{h}[i+1];
                t[i]=xr*wr-xi*wi;t[i+1]=xi*wr+xr*wi;
            }}"""
        across = f"""
            for(int k=0;k<{m};++k){{
                for(int j=0;j<{r};++j)paircopy(x+2*j,t+2*(j*{m}+k));
                {radix}(x,y,w);
                for(int j=0;j<{r};++j)paircopy(t+2*(j*{m}+k),y+2*j);
            }}"""
        along = f"""
            for(int j=0;j<{r};++j){{
                {line}(t+{2*m}*j,y,w);
                for(int k=0;k<{2*m};++k)t[{2*m}*j+k]=y[k];
            }}"""
        order = (along + rotate + across) if dit else (across + rotate + along)
        inp = f"k*{r}+j" if dit else f"j*{m}+k"
        out = f"j*{m}+k" if dit else f"k*{r}+j"
        return self.function(
            f"""
            float t[{2*n}],x[{2*max(m,r)}],y[{2*max(m,r)}];
            for(int j=0;j<{r};++j)for(int k=0;k<{m};++k)paircopy(t+2*(j*{m}+k),a+2*({inp}));
            {order}
            for(int j=0;j<{r};++j)for(int k=0;k<{m};++k)paircopy(b+2*({out}),t+2*(j*{m}+k));"""
        )

    def complex(self, node, n):
        node = unwrap(node)
        d = "b" if self.inverse else "f"
        if node.op == f"dft-generic-{n}" and n % 2:
            return self.generic_complex(n)
        if node.op.startswith("dft-rader-"):
            return self.rader(node, n)
        if node.op.startswith("dft-bluestein-"):
            return self.bluestein(node, n)
        if node.op.startswith(("dft-direct-", "dft-directbuf/")):
            if node.code == f"n1_{n}":
                fn = self.leaf(node, rf"n1_{n}")
                k = int(self.inverse)
                return self.function(f"{fn}(a+{k},a+{1-k},b+{k},b+{1-k},2,2,1,0,0);")
            fn = self.leaf(node, rf"n[12]{d}v_{n}_sse2")
            return self.function(f"{fn}(a,a,b,b,2,2,1,0,0);")
        if not node.op.startswith(("dft-ct-dit/", "dft-ct-dif/")):
            raise UnsupportedPlan(node.op)
        r = int(node.op.split("/")[1])
        m = n // r
        tw, child = node.children
        if tw.op.startswith("dftw-generic-"):
            return self.generic_complex_ct(
                n, r, tw, child, node.op.startswith("dft-ct-dit/")
            )
        sub = self.complex(child, m)
        if node.op.startswith("dft-ct-dit/"):
            fn = self.leaf(tw, rf"t[123][fb]u?v_{r}_sse2")
            harmonics = codelet_catalog()[1][fn]
            w = self.coefficients(np.arange(m)[:, None] * harmonics, n)
            body = f"""
            float x[{2*m}], y[{2*m}];
            for(int j=0;j<{r};++j){{
                for(int k=0;k<{m};++k) paircopy(x+2*k,a+2*(j+k*{r}));
                {sub}(x,y,w);
                for(int k=0;k<{m};++k) paircopy(b+2*(j*{m}+k),y+2*k);
            }}
            {fn}(b,b,{w},{2*m},0,{m},2);"""
        else:
            w = self.twiddle(n, r, 0)
            fn = self.leaf(tw, rf"q1{d}v_{r}_sse2")
            # A square DIF codelet transposes r transforms. Populate one independent
            # transform and extract its transposed column, one frequency at a time.
            body = f"""
            float tile[{2*r*r}], tmp[{2*n}], x[{2*m}], y[{2*m}];
            for(int k=0;k<{m};++k){{
                for(int i=0;i<{2*r*r};++i) tile[i]=0;
                for(int j=0;j<{r};++j) paircopy(tile+{2*r}*j,a+2*(j*{m}+k));
                {fn}(tile,tile,{w}+k*{2*(r-1)},{2*r},2,0,1,2);
                for(int j=0;j<{r};++j) paircopy(tmp+2*(j*{m}+k),tile+2*j);
            }}
            for(int j=0;j<{r};++j){{
                for(int k=0;k<{m};++k) paircopy(x+2*k,tmp+2*(j*{m}+k));
                {sub}(x,y,w);
                for(int k=0;k<{m};++k) paircopy(b+2*(k*{r}+j),y+2*k);
            }}"""
        if node.op.startswith("dft-ct-dit/") and fn[2] != d:
            # FFTW can bind the SIMD combine to a physically swapped complex pair.
            swap = f"for(int k=0;k<{n};++k){{float tmp=b[2*k];b[2*k]=b[2*k+1];b[2*k+1]=tmp;}}"
            combine = f"{fn}(b,b,{w},{2*m},0,{m},2);"
            body = body.replace(combine, swap + combine + swap)
        return self.function(body)

    def real(self, node, n):
        node = unwrap(node)
        d = "b" if self.inverse else "f"
        packed = 2 * (n // 2 + 1)
        if node.op.startswith(("rdft2-r2hc-direct-", "rdft2-hc2r-direct-")):
            fn = self.leaf(node, rf"r2c{d}_{n}")
            call = (
                f"{fn}(b,b+1,a,a+1,2,2,2,1,0,0);"
                if self.inverse
                else f"{fn}(a,a+1,b,b+1,2,2,2,1,0,0); b[1]=0;"
            )
            if not self.inverse and n % 2 == 0:
                call += f" b[{n+1}]=0;"
            return self.function(call)
        if node.op.startswith("rdft2-rdft-") and node.children[1].op == "rdft2-nop":
            sub = self.halfcomplex(node.children[0], n)
            if self.inverse:
                body = f"float h[{n}];h[0]=a[0];for(int k=1;2*k<{n};++k){{h[k]=a[2*k];h[{n}-k]=a[2*k+1];}}"
                if n % 2 == 0:
                    body += f"h[{n//2}]=a[{n}];"
                body += f"{sub}(h,b,w);"
            else:
                body = f"float h[{n}];{sub}(a,h,w);b[0]=h[0];b[1]=0;for(int k=1;2*k<{n};++k){{b[2*k]=h[k];b[2*k+1]=h[{n}-k];}}"
                if n % 2 == 0:
                    body += f"b[{n}]=h[{n//2}];b[{n+1}]=0;"
            return self.function(body)
        if not node.op.startswith(("rdft2-ct-dit/", "rdft2-ct-dif/")):
            raise UnsupportedPlan(node.op)
        r = int(node.op.split("/")[1])
        m = n // r
        tw, child = node.children
        if r % 2 or n % r:
            raise UnsupportedPlan(f"unpaired real butterfly {node.op}")
        butterfly = self.leaf(tw, rf"hc2c{d}dftv_{r}_sse2")
        sub = self.complex(child, m)
        zero, mid = tw.children
        base = self.real(zero, r)
        # Repack the strided DC column for the child real transform.
        special = f"""
        float x[{max(r+2,2*m)}], y[{max(r+2,2*m)}];
        for(int j=0;j<{(r//2+1) if self.inverse else r//2};++j) paircopy(x+2*j,t+{2*m}*j);
        {base}(x,y,w);
        for(int j=0;j<{r//2 if self.inverse else (r//2+1)};++j) paircopy(t+{2*m}*j,y+2*j);
        """
        w = self.twiddle(n, r, 1)
        special += (
            f"{butterfly}(t+2,t+3,t+{2*(m-1)},t+{2*(m-1)+1},{w},{2*m},1,{(m+1)//2},2);"
        )
        if m % 2 == 0:
            midfn = self.leaf(mid, rf"r2c{'bIII' if self.inverse else 'fII'}_{r}")
            special += (
                f"{{ float *h=t+{m}; {midfn}(h,h+1,h,h+1,{2*m},{2*m},{2*m},1,0,0); }}"
            )
        if self.inverse:
            body = f"""float t[{packed}]; for(int i=0;i<{packed};++i)t[i]=a[i];
            {special}
            for(int j=0;j<{r//2};++j){{
                {sub}(t+{2*m}*j,y,w);
                for(int k=0;k<{m};++k)paircopy(b+{r}*k+2*j,y+2*k);
            }}"""
        else:
            body = f"""float *t=b;
            for(int j=0;j<{r//2};++j){{
                float input[{2*m}];
                for(int k=0;k<{m};++k)paircopy(input+2*k,a+{r}*k+2*j);
                {sub}(input,t+{2*m}*j,w);
            }}
            {special}
            t[1]=t[{n+1}]=0;"""
        return self.function(body)

    def build(self, tree, n):
        node = unwrap(parse_tree(tree))
        if n == 1 and node.op in ("rdft2-r2hc-rank0", "rdft2-hc2r-rank0"):
            identity = self.function("paircopy(b,a);")
            real = self.function("b[0]=a[0];" if self.inverse else "b[0]=a[0]; b[1]=0;")
            steps = [real, identity, identity]
        else:
            steps = self.axis_steps(node, n)
        return self.kernel_source(steps, n)

    def axis_steps(self, node, n):
        if node.op != "rdft2-rank>=2/1" or len(node.children) != 2:
            raise UnsupportedPlan(node.op)
        pair = unwrap(node.children[0])
        if pair.op != "rdft2-rank>=2/1" or len(pair.children) != 2:
            raise UnsupportedPlan(pair.op)
        return [
            self.real(pair.children[0], n),
            self.complex(pair.children[1], n),
            self.complex(node.children[1], n),
        ]

    def kernel_source(self, steps, n):
        source = Path(__file__).resolve().with_name("fftw_gpu")
        parts = [
            (source / "lanes.cuh").read_text(),
            "DEV void paircopy(float *b,const float *a){b[0]=a[0];b[1]=a[1];}",
        ]
        catalog = codelet_catalog()[0]
        for key in sorted(self.codelets):
            if key not in catalog:
                raise UnsupportedPlan(key)
            parts.append(catalog[key])
        parts.extend(self.functions)
        for axis, step in zip((2, 1, 0), steps):
            ins = outs = 2 * n
            if axis == 2:
                ins, outs = n, 2 * (n // 2 + 1)
                if self.inverse:
                    ins, outs = outs, ins
            parts.append(
                f'extern "C" __global__ void axis{axis}(float *a,float *b,int rows,const float *w){{ int j=blockDim.x*blockIdx.x+threadIdx.x; if(j<rows) {step}(a+{ins}*(long long)j,b+{outs}*(long long)j,w); }}'
            )
        return "\n".join(parts), (
            np.concatenate(self.weights) if self.weights else np.zeros(1, np.float32)
        )


@lru_cache(maxsize=1)
def codelet_catalog():
    import gzip

    with gzip.open(
        Path(__file__).resolve().with_name("fftw_gpu") / "codelets.json.gz", "rt"
    ) as f:
        return json.load(f)


@lru_cache(maxsize=128)
def plans(n, threads):
    # FFTW wisdom/thread settings are process-global. Inspect a clean ESTIMATE
    # reference in a separate process without mutating the simulation's state.
    script = Path(__file__).resolve().with_name("inspect_fft_plans.py")
    return json.loads(
        subprocess.check_output(
            [sys.executable, str(script), str(n), "--threads", str(threads)], text=True
        )
    )


@cp.memoize(for_each_device=True)
def compile_plan(tree, n, inverse):
    """Compile the printed plan for the specified cubic layout and direction."""
    code, weights = Compiler(inverse).build(tree, n)
    mod = cp.RawModule(code=code, options=("--std=c++11", "--fmad=false"))
    return [mod.get_function(f"axis{i}") for i in range(3)], cp.asarray(weights)


def executable(n, inverse, threads):
    return compile_plan(plans(n, threads)[int(inverse)]["tree"], n, inverse)


def transform(a, n, inverse=False, threads=2, *, plan_tree=None):
    expected = (n, n, n // 2 + 1) if inverse else (n, n, n)
    if a.shape != expected or a.dtype != (cp.complex64 if inverse else cp.float32):
        raise ValueError("Unexpected FFT plan input layout or dtype")
    kernels, w = (
        executable(n, inverse, threads)
        if plan_tree is None
        else compile_plan(plan_tree, n, inverse)
    )
    for axis in (0, 1, 2) if inverse else (2, 1, 0):
        src = cp.ascontiguousarray(cp.moveaxis(a, axis, -1))
        size = (n if inverse else n // 2 + 1) if axis == 2 else n
        dst = cp.empty(
            (*src.shape[:-1], size),
            dtype=cp.float32 if axis == 2 and inverse else cp.complex64,
        )
        rows = src.size // src.shape[-1]
        kernels[axis](((rows + 63) // 64,), (64,), (src, dst, np.int32(rows), w))
        a = cp.moveaxis(dst, -1, axis)
    return a
