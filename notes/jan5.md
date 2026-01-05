┌──────────────────────────────┐
│        Host / Control        │
│  (control flow, heuristics)  │
└──────────────┬───────────────┘
               │
     profiler / decision logic
               │
┌──────────────┴───────────────┐
│      Fine-Grained Compute     │
│      (CFU / CVXIF ops)        │
└──────────────┬───────────────┘
               │
┌──────────────┴───────────────┐
│    Coarse Algebraic Engine    │
│      (RedMulE kernels)        │
└──────────────────────────────┘




Looks like DLRM (Deep Learning Recommendation Model) is best suited for servers or data-centers.

About DLRMs, ChatGPT:
The key realization (this is the crux)

When you scale DLRM down to edge memory sizes, it collapses into “MLP + dot products”.

At that point:

You are no longer accelerating DLRM

You are accelerating generic dense primitives

So asking:

“Is DLRM useful on edge?”

The accurate answer is:

DLRM as a model → no
DLRM as a source of primitives → yes


Brainstorming with ChatGPT suggested to go ahead with Sparse GEMMs. It also gave ideas on doing graph analytics.
I checked on genomics. Looks like it is suitable, more so with RedMulE. Here is why:

| Criterion         | Genomics      | Tiny ML          |
| ----------------- | ------------- | ---------------- |
| Math structure    | Clear algebra | Mixed heuristics |
| Semiring-friendly | ✅ Yes         | ❌ Mostly ×/+     |
| Graph connection  | Strong        | Weak             |
| Edge relevance    | Real          | Sometimes forced |
| RedMulE fit       | Excellent     | Mixed            |
| AIE fit           | Also good     | Also good        |


ChatGPT: My clear recommendation (no hedging)

If this were my FCCM submission, given your background:

Primary application:
🧬 Genomics DP alignment (single kernel)

Core contribution:
Profiler-driven hybrid execution (CFU + RedMulE)

Comparison:
Algebraic engine vs programmable spatial array (AIE-style)

Explanation tool:
Constraint-aware roofline
