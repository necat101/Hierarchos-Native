// SumKernel's four interleaved AVX2 banks and four-level cascade, for Engram's
// materialized stream squares, key squares, or (stream * weight) * key product.
// Width 5120 crosses the cascade boundary; a flat 32-lane bank changes BF16
// residual rounding even when the projection is bit-exact.
#ifndef V41_ENGRAM_CUSTOM_TERM
float v41_engram_term(uint mode, uint sb, uint kb, uint wb, uint d) {
    if (mode == 0u) {
        precise float square = stream[sb+d] * stream[sb+d];
        return square;
    }
    if (mode == 1u) {
        precise float square = kv[kb+d] * kv[kb+d];
        return square;
    }
    precise float weighted = stream[sb+d] * weight[wb+d];
    precise float product = weighted * kv[kb+d];
    return product;
}
#endif

float v41_engram_sum(uint mode, uint sb, uint kb, uint wb, uint width) {
    precise float acc[128];
    for (uint i=0u; i<128u; ++i) acc[i]=0.0;
    uint vectors=width/8u, groups=vectors/4u;
    uint bits=groups>1u ? uint(findMSB(groups-1u))+1u : 0u;
    uint power=max(4u,bits/4u), step=1u<<power, mask=step-1u;
    for (uint group=0u; group<groups; ++group) {
        for (uint i=0u; i<32u; ++i)
            acc[i]+=v41_engram_term(mode,sb,kb,wb,group*32u+i);
        uint count=group+1u;
        if (count%step==0u) {
            for (uint level=1u; level<4u; ++level) {
                for (uint i=0u; i<32u; ++i) {
                    acc[level*32u+i]+=acc[(level-1u)*32u+i];
                    acc[(level-1u)*32u+i]=0.0;
                }
                if ((count & (mask<<(level*power)))!=0u) break;
            }
        }
    }
    for (uint level=1u; level<4u; ++level)
        for (uint i=0u; i<32u; ++i) acc[i]+=acc[level*32u+i];
    for (uint v=groups*4u; v<vectors; ++v)
        for (uint lane=0u; lane<8u; ++lane)
            acc[lane]+=v41_engram_term(mode,sb,kb,wb,v*8u+lane);
    precise float sum=0.0;
    for (uint d=vectors*8u; d<width; ++d)
        sum+=v41_engram_term(mode,sb,kb,wb,d);
    for (uint lane=0u; lane<8u; ++lane) {
        precise float p=acc[lane]+acc[8u+lane];
        p+=acc[16u+lane]; p+=acc[24u+lane]; sum+=p;
    }
    return sum;
}
