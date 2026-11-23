// Each segment is a complete LZF token stream. Concatenating the segments
// preserves the HDF5 LZF format because references only point backwards.
extern "C" __global__ void shuffle_chunks(const unsigned char *field,
    unsigned char *raw, int ny, int nz, int nx, int cx, int cy, int cz,
    int first, int bytes, int chunks_y, int chunks_z) {
    int i = blockDim.x*blockIdx.x+threadIdx.x;
    if(i>=bytes) return;
    int volume=cx*cy*cz, chunk=first+i/(4*volume);
    int j=i%volume, x=(chunk/chunks_z/chunks_y)*cx+j/(cy*cz);
    int y=(chunk/chunks_z%chunks_y)*cy+(j/cz)%cy;
    int z=(chunk%chunks_z)*cz+j%cz;
    raw[i]=(x<nx && y<ny && z<nz) ? field[4LL*((long long)(x*ny+y)*nz+z)+(i/volume)%4] : 0;
}

extern "C" __global__ void encode_segments(const unsigned char *raw,
    unsigned char *tokens, int *sizes, int segment, int stride, int count, int chunk_bytes) {
    int tid = blockDim.x*blockIdx.x+threadIdx.x;
    if(tid>=count) return;
    int nseg=(chunk_bytes+segment-1)/segment, within=(tid%nseg)*segment;
    int n=min(segment,chunk_bytes-within);
    const unsigned char *a=raw+(tid/nseg)*chunk_bytes+within;
    unsigned char *b=tokens+tid*stride;
    unsigned short last[512]={};
    int p=0,o=0,start=0;
    while(p<n) {
        int match=0,dist=0;
        if(p+2<n) {
            unsigned int h=(a[p]*251u+a[p+1])*251u+a[p+2];
            int slot=(h^(h>>9))&511, prev=(int)last[slot]-1;
            last[slot]=p+1;
            if(prev>=0 && a[prev]==a[p] && a[prev+1]==a[p+1] && a[prev+2]==a[p+2]) {
                match=3;
                while(match<264 && p+match<n && a[prev+match]==a[p+match]) ++match;
                dist=p-prev-1;
            }
        }
        if(match || p-start==32) {
            if(p>start) {
                b[o++]=p-start-1;
                for(int k=start;k<p;++k)b[o++]=a[k];
                start=p;
            }
        }
        if(match) {
            int len=match-2;
            b[o++]=(min(len,7)<<5)|(dist>>8);
            if(len>=7)b[o++]=len-7;
            b[o++]=dist&255;
            p+=match; start=p;
        } else ++p;
    }
    if(p>start) {
        b[o++]=p-start-1;
        for(int k=start;k<p;++k)b[o++]=a[k];
    }
    sizes[tid]=o;
}

extern "C" __global__ void pack_segments(const unsigned char *tokens,
    const int *offset,const int *sizes,unsigned char *packed,int stride,int count) {
    int s=blockIdx.x;
    if(s>=count)return;
    for(int i=threadIdx.x;i<sizes[s];i+=blockDim.x)
        packed[offset[s]+i]=tokens[s*stride+i];
}
