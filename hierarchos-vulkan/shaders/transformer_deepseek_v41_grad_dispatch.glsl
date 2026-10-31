// Guard padded workgroups before multiplying by 64: the padded 2-D grid can
// exceed u32 element addressing even when the logical tensor itself fits.
bool v41_grad_index(uint length, out uint index) {
    uint group = gl_WorkGroupID.x + gl_WorkGroupID.y * gl_NumWorkGroups.x;
    uint groups = length / 64u + uint(length % 64u != 0u);
    if (group >= groups) return false;
    index = group * 64u + gl_LocalInvocationID.x;
    return index < length;
}
