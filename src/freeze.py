"""Universal-Transformer-style halting inside Bort's loop, as a forward hook on the last looped block (no change to arch.py).

After every loop iteration a rule decides, from the iteration's output state, who stops. Whatever has stopped keeps its state: the hook
returns the previous iteration's state in its place, from then on (UT: "once the per-symbol recurrent block halts, its state is simply copied
to the next step"). Frozen tokens are still run through the block, and the tokens that go on still attend to them, so the coda sees
exactly the mixed-depth states a real halting forward would give it. The loop runs for the full depth: the saving is not realised here, it is
counted: `loops[b]` is the iteration after which sequence b was done, i.e. the loop count a real implementation would stop at.

  mode "token":    the voters (the masked positions) halt individually; a sequence is done when all its voters have halted; the other
                   positions (non-voters) are never frozen before that. Once a sequence is done everything in it is frozen.
  mode "sequence": the rule gives one decision per sequence; when it fires, the whole sequence is frozen (this is exactly the fixed-depth
                   forward at that sequence's loop count).

`rule(i, previous, state)` -> bool [B, L] (token) or [B] (sequence): True = stop after iteration i. `previous` is None at i = 1."""
import torch
import torch.nn.functional as F

from halting import loop_bounds


class FreezeLoop:
    def __init__(self, student, voters, rule, mode="token", max_depth=8):
        if mode not in ("token", "sequence"):
            raise ValueError("mode is token or sequence")
        self.student, self.voters, self.rule, self.mode, self.max_depth = student, voters, rule, mode, max_depth
        batch = voters.shape[0]
        self.i, self.previous, self.handle = 0, None, None
        self.halted = torch.zeros_like(voters)
        self.exit = torch.full(voters.shape, max_depth, device=voters.device)  # the iteration after which each voter halted (max_depth: never)
        self.done = torch.zeros(batch, dtype=torch.bool, device=voters.device)
        self.loops = torch.full((batch,), max_depth, device=voters.device)

    def _hook(self, _module, _inputs, out):
        self.i += 1
        frozen = self.done[:, None] | self.halted
        if self.previous is not None and frozen.any():
            out = torch.where(frozen[..., None], self.previous, out)
        if self.i < self.max_depth:  # after the last iteration there is nothing left to decide
            want = self.rule(self.i, self.previous, out)
            if self.mode == "token":
                newly = want & self.voters & ~self.done[:, None] & ~self.halted
                self.exit = torch.where(newly, self.i, self.exit)
                self.halted = self.halted | newly
                finished = (self.halted | ~self.voters).all(1) & ~self.done
            else:
                finished = want & ~self.done
            self.loops = torch.where(finished, self.i, self.loops)
            self.done = self.done | finished
        self.previous = out
        return out

    def __enter__(self):
        self.handle = self.student.model.layers[loop_bounds(self.student)[1]].register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        self.handle.remove()


# ---------------------------------------------------------------- rules

def pool(state, voters):
    """Mean state over each sequence's voters -> [B, H]."""
    weights = voters.to(state.dtype)[..., None]
    return (state * weights).sum(1) / weights.sum(1).clamp_min(1)


def head_rule(weight, bias, threshold, mode, voters):
    """A trained head (raw-state weight [1, H], bias [1]): stop when its probability reaches the threshold. Sequence mode reads the pooled state."""
    def rule(i, previous, state):
        x = state.float() if mode == "token" else pool(state.float(), voters)
        return (x @ weight.squeeze(0).to(x.device) + bias.to(x.device)).sigmoid() >= threshold
    return rule


def change_rule(threshold, mode, voters):
    """Parameter-free: stop when the state barely moved (1 - cosine to the previous iteration < threshold). Sequence mode: mean over voters."""
    def rule(i, previous, state):
        if previous is None:
            return torch.zeros(state.shape[:2] if mode == "token" else state.shape[:1], dtype=torch.bool, device=state.device)
        change = 1 - F.cosine_similarity(state.float(), previous.float(), dim=-1)
        if mode == "token":
            return change < threshold
        weights = voters.to(change.dtype)
        return (change * weights).sum(1) / weights.sum(1).clamp_min(1) < threshold
    return rule


def never(i, previous, state):
    """The rule that never stops anything: FreezeLoop with it is the plain full-depth forward (what the tests check the hook against)."""
    return torch.zeros(state.shape[:2], dtype=torch.bool, device=state.device)
