"""
Genetic Algorithm (GA) — minimal, GUI-free implementation.

Merged and cleaned up from two duplicate lab implementations:
  - algos/2_sem/lab4.ipynb   (weighted/roulette-style selection with
    inverted-fitness weights, arithmetic crossover, gaussian mutation,
    hyperparameter grid search)
  - algos/3 sem/lab4/main.py (tkinter GUI version: roulette-wheel selection
    on 1/(1+f) fitness, arithmetic-mean crossover, uniform-random mutation)

Both minimize the same toy objective f(x1, x2) = 4*(x1-5)^2 + (x2-6)^2 to
make the merge straightforward; swap in any objective via `fitness_fn`.

Note: the notebook version's `if name == "main":` (no dunders — a real bug
that means the block never executes as a script guard) is fixed here to the
correct `if __name__ == "__main__":`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

Individual = list[float]


@dataclass
class GAResult:
    best: Individual
    best_fitness: float
    history: list[float] = field(default_factory=list)  # best fitness per generation


def default_objective(x: Individual) -> float:
    """Toy objective to minimize: 4*(x1-5)^2 + (x2-6)^2. Global min at (5, 6)."""
    x1, x2 = x
    return 4 * (x1 - 5) ** 2 + (x2 - 6) ** 2


def initialize_population(pop_size: int, num_genes: int, low: float, high: float) -> list[Individual]:
    return [[random.uniform(low, high) for _ in range(num_genes)] for _ in range(pop_size)]


def roulette_select(population: list[Individual], fitness_fn, k: int) -> list[Individual]:
    """Fitness-proportionate (roulette-wheel) selection for a MINIMIZATION
    objective: convert cost -> fitness via 1/(1+cost) so lower cost gets a
    higher chance of being picked, then sample proportionally.
    """
    scores = [1.0 / (1.0 + fitness_fn(ind)) for ind in population]
    total = sum(scores)
    weights = [s / total for s in scores]
    return random.choices(population, weights=weights, k=k)


def arithmetic_crossover(parent1: Individual, parent2: Individual, rate: float = 0.7) -> tuple[Individual, Individual]:
    """With probability `rate`, blend genes with a random alpha; otherwise
    pass a parent through unchanged (clone). Produces two children.
    """
    if random.random() < rate:
        alpha = random.random()
        child1 = [alpha * g1 + (1 - alpha) * g2 for g1, g2 in zip(parent1, parent2)]
        child2 = [alpha * g2 + (1 - alpha) * g1 for g1, g2 in zip(parent1, parent2)]
        return child1, child2
    return list(parent1), list(parent2)


def gaussian_mutate(individual: Individual, rate: float, low: float, high: float, sigma: float = 1.0) -> Individual:
    """Mutate each gene independently with probability `rate` by adding
    gaussian noise, clamped back into [low, high].
    """
    out = []
    for gene in individual:
        if random.random() < rate:
            gene = gene + random.gauss(0, sigma)
            gene = min(max(gene, low), high)
        out.append(gene)
    return out


def run_genetic_algorithm(
    pop_size: int = 50,
    num_genes: int = 2,
    low: float = -10,
    high: float = 10,
    generations: int = 100,
    crossover_rate: float = 0.7,
    mutation_rate: float = 0.1,
    fitness_fn=default_objective,
) -> GAResult:
    """Run a standard generational GA: selection -> crossover -> mutation,
    replacing the whole population each generation (no elitism, matching
    the source labs).
    """
    population = initialize_population(pop_size, num_genes, low, high)
    history: list[float] = []

    for _ in range(generations):
        new_population: list[Individual] = []
        while len(new_population) < pop_size:
            parent1, parent2 = roulette_select(population, fitness_fn, k=2)
            child1, child2 = arithmetic_crossover(parent1, parent2, crossover_rate)
            new_population.append(gaussian_mutate(child1, mutation_rate, low, high))
            new_population.append(gaussian_mutate(child2, mutation_rate, low, high))
        population = new_population[:pop_size]

        best = min(population, key=fitness_fn)
        history.append(fitness_fn(best))

    best = min(population, key=fitness_fn)
    return GAResult(best=best, best_fitness=fitness_fn(best), history=history)


if __name__ == "__main__":
    result = run_genetic_algorithm()
    print("best individual:", result.best)
    print("best fitness:", result.best_fitness)
    print("fitness history (first 5 / last 5):", result.history[:5], result.history[-5:])
