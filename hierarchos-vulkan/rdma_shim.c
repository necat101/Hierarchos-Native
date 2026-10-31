/* C shim exposing the static-inline ibverbs functions and internal field
 * accessors that bindgen cannot reach (ibv_qp/ibv_context are opaque in
 * the bindings). Compiled into librdma_shim.a by build.rs.
 */
#include <infiniband/verbs.h>
#include <stdint.h>

int shim_poll_cq(struct ibv_cq *cq, int num_entries, struct ibv_wc *wc)
{
	return ibv_poll_cq(cq, num_entries, wc);
}

int shim_post_send(struct ibv_qp *qp, struct ibv_send_wr *wr,
		   struct ibv_send_wr **bad_wr)
{
	return ibv_post_send(qp, wr, bad_wr);
}

int shim_post_recv(struct ibv_qp *qp, struct ibv_recv_wr *wr,
		   struct ibv_recv_wr **bad_wr)
{
	return ibv_post_recv(qp, wr, bad_wr);
}

uint32_t shim_qp_num(struct ibv_qp *qp) { return qp->qp_num; }

uint32_t shim_mr_lkey(struct ibv_mr *mr) { return mr->lkey; }

uint32_t shim_mr_rkey(struct ibv_mr *mr) { return mr->rkey; }
