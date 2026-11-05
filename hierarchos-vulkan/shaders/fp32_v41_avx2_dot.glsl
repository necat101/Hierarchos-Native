// Independent AVX2 Linear/einsum corpus pins the six-row SGEMM panel;
// narrow two/three-column projections use an eight-lane unfused dot instead.
// Callers provide readonly float storage buffers named x and weight, and the
// device's exact FMA implementation. No tensor values select an arithmetic path.
float v41_avx2_projection_dot(uint xb, uint wb, uint rows, uint outputs, uint width, uint row) {
    if (outputs <= 3u && rows*outputs<=32u) {
        if(rows==1u) {
            precise float v[8];for(uint j=0u;j<8u;++j)v[j]=0.0;
            uint aligned=width/8u*8u;
            for(uint k=0u;k<aligned;++k)v[k&7u]=fma(x[xb+k],weight[wb+k],v[k&7u]);
            precise float a[4];for(uint j=0u;j<4u;++j)a[j]=v[j]+v[j+4u];
            // The four-wide epilogue runs after the vector is folded in half.
            for(uint k=aligned;k<width;++k)a[k&3u]=fma(x[xb+k],weight[wb+k],a[k&3u]);
            precise float even=a[0]+a[2],odd=a[1]+a[3];return even+odd;
        }
        if((rows&1u)!=0u && row==rows-1u) {
            precise float v[32];for(uint j=0u;j<32u;++j)v[j]=0.0;
            uint aligned=width/32u*32u;
            for(uint k=0u;k<aligned;++k)v[k&31u]=fma(x[xb+k],weight[wb+k],v[k&31u]);
            precise float banks[8];
            for(uint j=0u;j<8u;++j) {
                precise float first=v[j]+v[j+8u],second=v[j+16u]+v[j+24u];
                banks[j]=first+second;
            }
            precise float a[4];for(uint j=0u;j<4u;++j)a[j]=banks[j]+banks[j+4u];
            for(uint k=aligned;k<width;++k) {precise float product=x[xb+k]*weight[wb+k];a[k&3u]+=product;}
            precise float even=a[0]+a[2],odd=a[1]+a[3];return even+odd;
        }
        precise float a[8];
        for (uint lane=0u; lane<8u; ++lane) a[lane]=0.0;
        uint aligned=width/8u*8u;
        for (uint k=0u; k<aligned; ++k) {
            precise float product=x[xb+k]*weight[wb+k];
            a[k%8u]+=product;
        }
        precise float p0=a[0]+a[4], p1=a[1]+a[5], p2=a[2]+a[6], p3=a[3]+a[7];
        precise float result=(p0+p2)+(p1+p3);
        for (uint k=aligned; k<width; ++k) {
            precise float product=x[xb+k]*weight[wb+k]; result+=product;
        }
        return result;
    }
    uint tail=rows%6u;
    bool short_panel=tail>0u && tail<=3u && row>=rows-tail;
    precise float a=0.0, b=0.0;
    if (short_panel || outputs<=8u) {
        for (uint k=0u; k<width; k+=2u) {
            if (short_panel && tail<=2u) {
                precise float p=x[xb+k]*weight[wb+k]; a+=p;
                if (k+1u<width) { precise float q=x[xb+k+1u]*weight[wb+k+1u]; b+=q; }
            } else {
                a=fma(x[xb+k],weight[wb+k],a);
                if (k+1u<width) b=fma(x[xb+k+1u],weight[wb+k+1u],b);
            }
        }
    } else {
        for (uint k=0u; k<width; ++k) a=fma(x[xb+k],weight[wb+k],a);
    }
    return a+b;
}
