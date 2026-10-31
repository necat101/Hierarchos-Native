// All indices refer to separate batch-local window/compressed allocations.
uint slot_count() { return pc.window_topk + pc.compressed_topk; }
int slot_index(uint row, uint slot) {
    return slot < pc.window_topk
        ? window_indices[row * pc.window_topk + slot]
        : compressed_indices[row * pc.compressed_topk + slot - pc.window_topk];
}
bool slot_valid(uint slot, int index) {
    return index >= 0 && (slot < pc.window_topk
        ? uint(index) < pc.window_capacity
        : uint(index) >= pc.window_capacity && uint(index)-pc.window_capacity < pc.compressed_capacity);
}
#ifdef V41_GRAD_KV_VALUES
float slot_value(uint row, uint slot, uint column) {
    int index = slot_index(row, slot);
    if (!slot_valid(slot, index)) return 0.0;
    uint batch = row / pc.seq;
    return slot < pc.window_topk
        ? window_kv[(batch * pc.window_capacity + uint(index)) * pc.dim + column]
        : compressed_kv[(batch * pc.compressed_capacity + uint(index)-pc.window_capacity) * pc.dim + column];
}
#endif
