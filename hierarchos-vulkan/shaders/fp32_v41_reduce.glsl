// CPU PyTorch SumKernel.cpp uses AVX2 even on this AVX512 oracle: four
// interleaved eight-lane vectors and a four-level cascaded sum. Squares are
// rounded first. V41_REDUCE_INPUT names the declaring shader's input buffer.
float v41_sum_squares(uint base, uint width) {
    precise float acc[128];
    for (uint i=0u; i<128u; ++i) acc[i]=0.0;
    uint vectors=width/8u, groups=vectors/4u;
    uint bits=groups>1u ? uint(findMSB(groups-1u))+1u : 0u;
    uint power=max(4u,bits/4u), step=1u<<power, mask=step-1u;
    for (uint group=0u; group<groups; ++group) {
        for (uint i=0u; i<32u; ++i) {
            float value=V41_REDUCE_INPUT[base+group*32u+i];
            precise float square=value*value;
            acc[i]+=square;
        }
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
    // Leftover vectors contribute to the first interleaved bank after folding
    // the cascade, matching row_sum's partial_sums[0] tail.
    for (uint v=groups*4u; v<vectors; ++v) {
        for (uint lane=0u; lane<8u; ++lane) {
            float value=V41_REDUCE_INPUT[base+v*8u+lane];
            precise float square=value*value;
            acc[lane]+=square;
        }
    }
    precise float sum=0.0;
    for (uint col=vectors*8u; col<width; ++col) {
        float value=V41_REDUCE_INPUT[base+col];
        precise float square=value*value;
        sum+=square;
    }
    for (uint lane=0u; lane<8u; ++lane) {
        precise float p=acc[lane]+acc[8u+lane];
        p+=acc[16u+lane]; p+=acc[24u+lane]; sum+=p;
    }
    return sum;
}
