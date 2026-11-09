"""Hierarchical Bayesian second-degree-moment inversion with BayesBay
(Magrini et al., 2025).

Model: the six Cholesky parameters x0..x5 (see ``moments``) with uniform
priors.  Data: apparent durations tau_c(s).  Forward model:
tau_c = 2 sqrt(G m2(x)).  The data-noise standard deviation is itself
sampled (hierarchical), uniform on [std_min, std_max].

Default settings: 20 chains, 1,000,000 iterations, 50,000 burn-in, every
100th sample kept (190,000 samples), starting states drawn from the priors
with seed 620 (Seo et al., 2026).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import SEED
from .io import Ensemble

PARAM_SPACE = "my_param_space"      # BayesBay names of the saved arrays
TARGET = "my_data"


@dataclass
class Priors:
    """Uniform prior bounds (vmin, vmax) and proposal widths of x0..x5.

    Defaults: x0 in [log10 0.01, log10 0.4] (half-duration sqrt(mu02) from
    0.01 to 0.4 s), x1, x2, x4 in [-1, 1], x3, x5 in [log10 0.01, log10 1]
    (moments in km and s).  Widen them for larger events.
    """
    bounds: list = field(default_factory=lambda: [
        (np.log10(0.01), np.log10(0.4)), (-1.0, 1.0), (-1.0, 1.0),
        (np.log10(0.01), np.log10(1.0)), (-1.0, 1.0), (np.log10(0.01), np.log10(1.0))])
    perturb_std: list = field(default_factory=lambda: [0.01, 0.01, 0.01, 0.1, 0.1, 0.1])


@dataclass
class Noise:
    """Hierarchical data-noise prior: std ~ U(std_min, std_max) [s]."""
    std_min: float = 0.0
    std_max: float = 0.05
    std_perturb_std: float = 0.005


def starting_states(priors, n_chains, seed=SEED):
    """Starting models drawn from the priors in the order of the original
    notebooks (chain by chain, x0..x5), so a run can be repeated exactly."""
    from bayesbay._state import ParameterSpaceState, State
    rng = np.random.RandomState(seed)
    states = []
    for _ in range(n_chains):
        values = {f"m{i}": np.array([rng.uniform(lo, hi)]) for i, (lo, hi) in enumerate(priors.bounds)}
        states.append(State(param_values={PARAM_SPACE: ParameterSpaceState(n_dimensions=1, param_values=values)}))
    return states


def build(G, tauc, priors=None, noise=None, n_chains=20, seed=SEED):
    """Assemble a ``bayesbay.BayesianInversion`` for one event."""
    import bayesbay as bb
    priors, noise = priors or Priors(), noise or Noise()
    G = np.asarray(G, float)
    params = [bb.prior.UniformPrior(name=f"m{i}", vmin=lo, vmax=hi, perturb_std=s)
              for i, ((lo, hi), s) in enumerate(zip(priors.bounds, priors.perturb_std))]
    space = bb.parameterization.ParameterSpace(name=PARAM_SPACE, n_dimensions=1, parameters=params)
    parameterization = bb.parameterization.Parameterization(space)

    keys = [f"m{i}" for i in range(6)]

    def forward(state):
        # the expression and array shapes of the original notebooks, so the
        # predictions are bit-identical to theirs; per call this is faster than
        # cholesky_to_m2, which is written for whole ensembles
        x0, x1, x2, x3, x4, x5 = [state[PARAM_SPACE][k] for k in keys]
        a0, a3 = 10 ** x0, 10 ** x3
        m2 = np.array([10 ** (2 * x0), a0 * x1, a0 * x2, x1 * x1 + 10 ** (2 * x3),
                       x1 * x2 + a3 * x4, x2 * x2 + x4 * x4 + 10 ** (2 * x5)])
        return 2 * np.sqrt(G @ m2)[:, 0]

    target = bb.likelihood.Target(name=TARGET, dobs=np.asarray(tauc, float),
                                  std_min=noise.std_min, std_max=noise.std_max,
                                  std_perturb_std=noise.std_perturb_std, noise_is_correlated=False)
    likelihood = bb.likelihood.LogLikelihood(targets=target, fwd_functions=forward)
    if seed is not None:
        # BayesBay draws the starting noise level from Python's ``random``
        # while building the chains; seed it so a build is repeatable.
        import random
        random.seed(seed)
    return bb.BayesianInversion(log_likelihood=likelihood, parameterization=parameterization,
                                n_chains=n_chains,
                                walkers_starting_states=starting_states(priors, n_chains, seed))


def _chain_seed(seed, index, ith_iteration):
    """Seed of one chain for one call of ``run``: seed + chain index for a new
    chain; a chain that is continued gets a stream that depends on the
    iteration it stopped at, so a second ``run`` does not repeat the first."""
    if ith_iteration == 0:
        return seed + index
    return int(np.random.SeedSequence([seed, index, ith_iteration]).generate_state(1)[0])


def _seeded_advance(index, chain, seed, **kwargs):
    import random

    from bayesbay._markov_chain import MarkovChain
    s = _chain_seed(seed, index, chain.ith_iteration)
    random.seed(s)
    np.random.seed(s % 2 ** 32)
    return MarkovChain.advance_chain(chain, **kwargs)


def _seeded_sampler(seed):
    """BayesBay's vanilla sampler with the random generators of each chain
    seeded (seed + chain index) inside its worker process.

    BayesBay draws proposals from Python's ``random`` module in each joblib
    worker, so without this two runs with identical settings differ.
    Seeding does not change the algorithm, only makes a run repeatable.
    """
    from functools import partial

    import joblib
    from bayesbay.samplers import VanillaSampler

    class SeededSampler(VanillaSampler):
        def advance_chain(self, n_iterations, burnin_iterations=0, save_every=100,
                          verbose=True, print_every=100):
            func = partial(_seeded_advance, seed=seed, n_iterations=n_iterations,
                           burnin_iterations=burnin_iterations, save_every=save_every,
                           verbose=verbose, print_every=print_every,
                           begin_iteration=self.begin_iteration, end_iteration=self.end_iteration)
            if self.parallel_config.get("n_jobs", 1) > 1:
                self._chains = joblib.Parallel(**self.parallel_config)(
                    joblib.delayed(func)(i, c) for i, c in enumerate(self.chains))
            else:
                self._chains = [func(i, c) for i, c in enumerate(self.chains)]
            self.on_end_advance_chain()
            return self.chains

    return SeededSampler()


def run(inversion, n_iterations=1_000_000, burnin_iterations=50_000, save_every=100,
        n_jobs=None, seed=SEED):
    """Run the chains (in parallel, one process per chain by default) and
    return (ensemble, per_chain) where per_chain is (n_chains, n_draws, 6).

    With ``seed`` (default 620) the run is repeatable; ``seed=None`` uses
    BayesBay's unseeded workers.
    """
    cfg = None if n_jobs is None else {"n_jobs": n_jobs}
    sampler = None if seed is None else _seeded_sampler(seed)
    inversion.run(sampler=sampler, n_iterations=n_iterations, burnin_iterations=burnin_iterations,
                  save_every=save_every, verbose=False, parallel_config=cfg)
    per_chain = []
    for chain in inversion.chains:
        r = inversion.get_results_from_chains(chains=chain)
        per_chain.append(np.column_stack([np.ravel(r[f"{PARAM_SPACE}.m{i}"]) for i in range(6)]))
    per_chain = np.stack(per_chain)
    res = inversion.get_results(concatenate_chains=True)
    x = np.column_stack([np.ravel(res[f"{PARAM_SPACE}.m{i}"]) for i in range(6)])
    std = np.ravel(res[f"{TARGET}.std"]) if f"{TARGET}.std" in res else None
    dpred = np.asarray(res[f"{TARGET}.dpred"]) if f"{TARGET}.dpred" in res else None
    return Ensemble(x, std, dpred), per_chain


def diagnostics(per_chain):
    """Gelman-Rubin R-hat and bulk/tail effective sample sizes of x0..x5
    (for example, require R-hat < 1.01 and ESS > 400)."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        import arviz
    post = {f"m{i}": per_chain[:, :, i] for i in range(per_chain.shape[2])}
    idata = arviz.from_dict(posterior=post)
    out = {}
    for name, ds in [("rhat", arviz.rhat(idata)), ("ess_bulk", arviz.ess(idata, method="bulk")),
                     ("ess_tail", arviz.ess(idata, method="tail"))]:
        out[name] = {k: float(ds[k]) for k in post}
    return out
