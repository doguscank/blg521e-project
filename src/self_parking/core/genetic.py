"""Genetic algorithm primitives."""

from __future__ import annotations

import random
from collections.abc import Callable

from self_parking.core.probability import weighted_random
from self_parking.core.types import Gene, Generation, Genome, Percentage, Probability

FitnessFunction = Callable[[Genome], float]


def genome_string_to_genome(genome_string: str) -> Genome:
    return [1 if token.strip() == "1" else 0 for token in genome_string.split()]


def create_genome(length: int, rng: random.Random) -> Genome:
    return [0 if rng.random() < 0.5 else 1 for _ in range(length)]


def create_generation(generation_size: int, genome_length: int, rng: random.Random) -> Generation:
    return [create_genome(genome_length, rng) for _ in range(generation_size)]


def mutate(genome: Genome, mutation_probability: Probability, rng: random.Random) -> Genome:
    mutated = list(genome)
    for gene_index, gene in enumerate(mutated):
        mutated_gene: Gene = 1 if gene == 0 else 0
        mutated[gene_index] = mutated_gene if rng.random() < mutation_probability else gene
    return mutated


def mate(
    father: Genome,
    mother: Genome,
    mutation_probability: Probability,
    rng: random.Random,
) -> tuple[Genome, Genome]:
    if len(father) != len(mother):
        raise ValueError("Cannot mate different species")

    first_child: Genome = []
    second_child: Genome = []

    for gene_index in range(len(father)):
        first_child.append(father[gene_index] if rng.random() < 0.5 else mother[gene_index])
        second_child.append(father[gene_index] if rng.random() < 0.5 else mother[gene_index])

    return (
        mutate(first_child, mutation_probability, rng),
        mutate(second_child, mutation_probability, rng),
    )


def select(
    generation: Generation,
    fitness: FitnessFunction,
    mutation_probability: Probability,
    long_living_champions_percentage: Percentage,
    rng: random.Random,
) -> Generation:
    """Select and produce the next generation."""
    if not generation:
        raise ValueError("Generation must not be empty")

    old_generation = [list(genome) for genome in generation]
    old_generation.sort(key=fitness, reverse=True)

    new_generation: Generation = []
    long_livers_count = int(long_living_champions_percentage * len(old_generation) / 100)

    if long_livers_count > 0:
        for champion in old_generation[:long_livers_count]:
            new_generation.append(list(champion))

    fitness_per_old_genome = [fitness(genome) for genome in old_generation]

    while len(new_generation) < len(generation):
        father, father_idx = weighted_random(old_generation, fitness_per_old_genome, rng)
        mother, mother_idx = weighted_random(old_generation, fitness_per_old_genome, rng)

        while father_idx == mother_idx:
            mother, mother_idx = weighted_random(old_generation, fitness_per_old_genome, rng)

        first_child, second_child = mate(father, mother, mutation_probability, rng)
        new_generation.append(first_child)
        if len(new_generation) < len(generation):
            new_generation.append(second_child)

    return new_generation
