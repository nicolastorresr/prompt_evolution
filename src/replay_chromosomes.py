"""
replay_chromosomes.py
Deterministically replays the evolutionary run (same seed, same operators) using the
fitness values stored in score_records.csv instead of calling the LLM, in order to
recover the prompt text of every evaluated chromosome.

Run from the project root (needs src/ modules, data/prompts/seed_prompts.json and the
results folder of the run):
    python replay_chromosomes.py --results results/run_XXXX --seed 42

Validation: the replay must (i) evaluate exactly the chromosome IDs found in
score_records.csv and (ii) reproduce every row of generation_stats.csv. The script
aborts if either check fails.
"""
import argparse, json, random, sys
from pathlib import Path
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--results", required=True)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--seed-prompts", default="data/prompts/seed_prompts.json")
ap.add_argument("--out", default="chromosomes_replayed.json")
a = ap.parse_args()
sys.path.insert(0, "src")
from evolution import load_seed_prompts, initialise_population, population_diversity
from mutation import apply_mutation
from crossover import single_point_crossover
from selection import select_parents, elitism

res = Path(a.results)
rec = pd.read_csv(res / "score_records.csv")
gs = pd.read_csv(res / "generation_stats.csv")
cfg = json.load(open(res / "config.json"))
P, G = cfg["population_size"], cfg["generations"]
pm, pc, k, ne = cfg["mutation_rate"], cfg["crossover_rate"], cfg["tournament_size"], cfg["n_elite"]
fit = rec.groupby("chromosome_id").bas.mean().to_dict()

rng = random.Random(a.seed)
pop = initialise_population(load_seed_prompts(a.seed_prompts), P, rng)
allc, evaluated = {}, set()

def ev(inds, gen):
    for c in inds:
        c.fitness = fit.get(c.chromosome_id, float("nan")); c.generation = gen
        allc[c.chromosome_id] = c; evaluated.add(c.chromosome_id)

def stats(gen, pop):
    f = [c.fitness for c in pop]
    best = max(pop, key=lambda c: c.fitness or 0.0)
    return (max(f), sum(f) / len(f), min(f), population_diversity(pop), best.chromosome_id)

ev(pop, 0); st = [stats(0, pop)]
for gen in range(1, G + 1):
    el = elitism(pop, ne); nxt = list(el)
    while len(nxt) < P:
        par = select_parents(pop, 2, k, rng)
        x, y = single_point_crossover(par[0], par[1], pc, rng)
        x = apply_mutation(x, pm, rng); y = apply_mutation(y, pm, rng)
        nxt.append(x)
        if len(nxt) < P: nxt.append(y)
    new = nxt[ne:]; ev(new, gen); pop = el + new; st.append(stats(gen, pop))

ok = evaluated == set(rec.chromosome_id)
for s, r in zip(st, gs.itertuples()):
    ok &= (abs(s[0]-r.best_fitness) < 1e-9 and abs(s[1]-r.avg_fitness) < 1e-9 and
           abs(s[2]-r.worst_fitness) < 1e-9 and abs(s[3]-r.diversity) < 1e-9 and s[4] == r.best_chromosome_id)
if not ok:
    sys.exit("Replay does NOT match the logs (different Python version or modified code?). Aborting.")
json.dump({i: {"text": c.assemble(), "fitness": fit.get(i), "generation": c.generation,
               "parents": c.parent_ids} for i, c in allc.items() if i in set(rec.chromosome_id)},
          open(a.out, "w"), indent=1)
print("Replay validated; wrote", a.out)
