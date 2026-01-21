import numpy as np

import deterministic


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.lp_buffer = []

    def add(self, s0, a, r, s1, done, index, fixline_id, ctx):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(
            (
                s0[None, :],
                a,
                r,
                s1[None, :],
                done,
                index,
                fixline_id,
                ctx.detach().numpy(),
            )
        )

    def sample(self, batch_size):
        s0, a, r, s1, done, index, fixline_id, ctx = zip(
            *deterministic.buffer_rng.choice(self.buffer, batch_size, replace=False)
        )
        return (
            np.concatenate(s0),
            a,
            r,
            np.concatenate(s1),
            done,
            index,
            fixline_id,
            ctx,
        )

    def lp_add(self, s0, a, r, ctx):
        if len(self.lp_buffer) >= self.capacity:
            self.lp_buffer.pop(0)
        self.lp_buffer.append((s0[None, :], a, r, ctx.detach().cpu())) # 修改

    # def lp_sample(self, batch_size):
    #     s0, a, r, ctx = zip(
    #         *deterministic.buffer_rng.choice(self.lp_buffer, batch_size, replace=False)
    #     )
    #     return np.concatenate(s0), a, r, ctx
    # 在 flowpipe/agent/replay.py 的 lp_sample 方法中添加调试
    def lp_sample(self, batch_size):
        print(f"DEBUG: lp_buffer length: {len(self.lp_buffer)}")
        if self.lp_buffer:
            print(f"DEBUG: First item type: {type(self.lp_buffer[0])}")
            if isinstance(self.lp_buffer[0], tuple):
                for i, elem in enumerate(self.lp_buffer[0]):
                    print(
                        f"DEBUG: Element {i} type: {type(elem)}, requires_grad: {getattr(elem, 'requires_grad', 'N/A')}")

        # s0, a, r, ctx = zip(
        #     *deterministic.buffer_rng.choice(self.lp_buffer, batch_size, replace=False)
        # )
        # return np.concatenate(s0), a, r, ctx
        idx = deterministic.buffer_rng.choice(len(self.lp_buffer), batch_size, replace=False)
        batch = [self.lp_buffer[i] for i in idx]
        s0, a, r, ctx = zip(*batch)
        return np.concatenate(s0), a, r, ctx

    def size(self):
        return len(self.buffer)

    def lp_size(self):
        return len(self.lp_buffer)
