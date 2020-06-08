import numpy as np
from typing import List
from .individual import Individual

class Population:
    def __init__(self, individuals: List[Individual]):
        self.individuals = individuals
        
    @property
    def num_individuals(self) -> int:
        return len(self.individuals)

    @num_individuals.setter
    def num_individuals(self, val) -> None:
        raise Exception("cannot set the number of individuals, you must change 'Population.individuals' instead")

    @property
    def num_gens(self) -> int:
        return len(self.individuals)

    @num_gens.setter
    def num_gens(self, val):
        raise Exception("Cannot set the number of genes. You must change 'Population.individuals' instead")

    @property
    def average_fitness(self):
        return (sum(individual.fitness for individual in self.individuals) / float(self.num_individuals))

    @average_fitness.setter
    def average_fitness(self, val):
        raise Exception("Cannot set the average fitness. This is read-only property")

    @property
    def fittest_individual(self) -> Individual:
        return max(self.individuals, key=lambda individual: individual.fitness)

    @fittest_individual.setter
    def fittest_individual(self, val):
        raise Exception("Cannot set fittest individual. This is read-only property")

    def calculate_fitness(self):
        for individual in self.individuals:
            individual.calculate_fitness()
    
    def get_fitness_std(self) -> float:
        return np.std(np.array([individual.fitness for individual in self.individuals]))

