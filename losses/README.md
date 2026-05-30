# losses/

Auxiliary objectives used alongside the standard cross-entropy LM loss.

| File | Role |
|---|---|
| `load_balance.py` | Switch-Transformer load-balancing loss (Fedus et al. 2022): `L_lb = K · Σ_k f_k · p_k`, where `f_k` is the batch fraction dispatched to expert *k* (argmax, no gradient) and `p_k` is the mean router probability for *k* (with gradient). Combined as `L = L_CE + λ_lb · L_lb` with `λ_lb = 0.01` to prevent the router from collapsing onto a single expert. |
