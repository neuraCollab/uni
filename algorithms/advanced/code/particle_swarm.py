"""
Particle Swarm Optimization (PSO) — minimal, GUI/plotting-free implementation.

Merged and cleaned up from two duplicate lab implementations:
  - algos/2_sem/lab5.ipynb    (fixed inertia weight)
  - algos/3 sem/lab5/main.py  (tkinter + matplotlib GUI version, with linear
    inertia decay from inertia_start to inertia_end over the run, plus a
    velocity-magnitude clamp)

Both minimize the same toy objective f(x1, x2) = 4*(x1-5)^2 + (x2-6)^2.
This merge keeps the more complete GUI version's features (inertia decay,
velocity clamping) since they strictly generalize the plain notebook
version (set inertia_start == inertia_end to recover a fixed inertia).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


@dataclass
class Particle:
    position: list[float]
    velocity: list[float]
    best_position: list[float]
    best_value: float


@dataclass
class PSOResult:
    best_position: list[float]
    best_value: float
    history: list[float] = field(default_factory=list)  # global best value per iteration


def default_objective(pos: list[float]) -> float:
    x, y = pos
    return 4 * (x - 5) ** 2 + (y - 6) ** 2


def _vec_sub(a: list[float], b: list[float]) -> list[float]:
    return [ai - bi for ai, bi in zip(a, b)]


def _vec_add(*vs: list[float]) -> list[float]:
    return [sum(components) for components in zip(*vs)]


def _vec_scale(v: list[float], s: float) -> list[float]:
    return [s * vi for vi in v]


def _clip(v: list[float], low: float, high: float) -> list[float]:
    return [min(max(vi, low), high) for vi in v]


def _norm(v: list[float]) -> float:
    return math.sqrt(sum(vi * vi for vi in v))


def _create_particle(dim: int, low: float, high: float, v_limit: float, objective_fn) -> Particle:
    position = [random.uniform(low, high) for _ in range(dim)]
    velocity = [random.uniform(-v_limit, v_limit) for _ in range(dim)]
    value = objective_fn(position)
    return Particle(position=position, velocity=velocity, best_position=list(position), best_value=value)


def run_particle_swarm(
    num_particles: int = 30,
    dim: int = 2,
    low: float = -10,
    high: float = 10,
    iterations: int = 100,
    cognitive_coeff: float = 1.5,   # pull toward the particle's own best
    social_coeff: float = 1.5,      # pull toward the swarm's global best
    inertia_start: float = 0.9,
    inertia_end: float = 0.4,
    velocity_limit: float = 20.0,
    objective_fn=default_objective,
) -> PSOResult:
    """Run PSO with linearly decaying inertia (starts exploratory, ends
    exploitative) and a velocity-magnitude clamp so particles don't fly
    off to infinity.
    """
    particles = [_create_particle(dim, low, high, velocity_limit, objective_fn) for _ in range(num_particles)]

    global_best_position = min(particles, key=lambda p: p.best_value).best_position
    global_best_value = min(p.best_value for p in particles)

    history: list[float] = []

    for it in range(iterations):
        inertia = max(
            inertia_start - it * (inertia_start - inertia_end) / max(iterations, 1),
            inertia_end,
        )

        for particle in particles:
            r1, r2 = random.random(), random.random()
            cognitive = _vec_scale(_vec_sub(particle.best_position, particle.position), cognitive_coeff * r1)
            social = _vec_scale(_vec_sub(global_best_position, particle.position), social_coeff * r2)
            new_velocity = _vec_add(_vec_scale(particle.velocity, inertia), cognitive, social)

            speed = _norm(new_velocity)
            if speed > velocity_limit:
                new_velocity = _vec_scale(new_velocity, velocity_limit / speed)
            particle.velocity = new_velocity

            particle.position = _clip(_vec_add(particle.position, particle.velocity), low, high)

            value = objective_fn(particle.position)
            if value < particle.best_value:
                particle.best_position = list(particle.position)
                particle.best_value = value

            if value < global_best_value:
                global_best_position = list(particle.position)
                global_best_value = value

        history.append(global_best_value)

    return PSOResult(best_position=global_best_position, best_value=global_best_value, history=history)


if __name__ == "__main__":
    result = run_particle_swarm()
    print("best position:", result.best_position)
    print("best value:", result.best_value)
    print("history (first 5 / last 5):", result.history[:5], result.history[-5:])
