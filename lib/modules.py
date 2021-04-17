import random
from math import pow, log2, cos, pi, sin
import logging
from operator import attrgetter
from copy import deepcopy, copy
import numpy
import time
import csv
from time import time
import numba
from models import Individual

@numba.jit(nopython=True, fastmath=True)
def random_real(range_a,  range_b,  precision):
    prec = pow(10, precision)
    return numpy.round(random.randrange(range_a * prec, (range_b) * prec + 1)/prec, precision)


@numba.jit(nopython=True, fastmath=True)
def power_of_2(range_a,  range_b,  precision):
    return int(numpy.rint(numpy.log2(((range_b - range_a) * (1/pow(10, -precision)) + 1))))

@numba.jit(nopython=True, fastmath=True)
def real_to_int(real,  range_a,  range_b,  power):
    return int(numpy.rint((1/(range_b-range_a)) * (real - range_a) * ((pow(2, power)-1))))


@numba.jit(nopython=True, fastmath=True)
def bin_to_int(binary):
    out = 0
    for bit in binary:
        out = (out << 1) | bit
    return out


def int_to_bin(integer, power):
    return numpy.asarray([numpy.int(n) for n in bin(integer)[2:].zfill(power)], dtype=numpy.int)


@numba.jit(nopython=True, fastmath=True)
def int_to_real(integer,  range_a,  range_b, precision, power):
    return numpy.round(range_a + ((range_b - range_a) * integer)/(pow(2, power)-1), precision)


@numba.jit(nopython=True, fastmath=True)
def func(real):
    return numpy.mod(real, 1) * (cos(20.0 * pi * real) - sin(real))


def get_individual(range_a,  range_b, precision, power):
    real = random_real(range_a, range_b, precision)
    int_from_real = real_to_int(real, range_a, range_b, power)
    return int_to_bin(int_from_real, power)

@numba.jit(nopython=True, fastmath=True, parallel=True)
def mutation(bins, new_bin, new_fxs, power, generations_number):
    for bit in numpy.arange(power):
        new_bin[bit] = bins
        new_bin[bit, bit] = 1 - new_bin[bit, bit]
        new_fxs[bit] = func(new_bin[bit, bit])
        #individuals[bit] = Individual(new_bin[bit], func(new_bin[bit, bit]))
        #individuals[bit].binary = new_bin[bit]
        #individuals[bit].fx = func(new_bin[bit, bit])


def evolution(range_a, range_b, precision, tau, generations_number, save_file=True):
    power = power_of_2(range_a, range_b, precision)
    reals = numpy.empty(generations_number, dtype=numpy.double)
    bins = numpy.empty((generations_number, power), dtype=numpy.int)
    fxs = numpy.empty(generations_number, dtype=numpy.double)
    best = numpy.empty(power, dtype=numpy.int)
    new_bin = numpy.empty((power, power), dtype=numpy.int)
    new_fxs = numpy.empty(generations_number, dtype=numpy.double)
    individuals = numpy.empty(power, dtype=object)

    best = get_individual(range_a, range_b, precision, power)
    bins[0] = best
    
    reals[0] = int_to_real(bin_to_int(bins[0,]), range_a, range_b, precision, power)
    fxs[0] = func(reals[0])

    for i in numpy.arange(1, generations_number):
        mutation(bins[i], new_bin, new_fxs, power, generations_number)
        for bit in numpy.arange(power):
            individuals[bit] = Individual(new_bin[bit], new_fxs[bit])

    # print(bins[0])
    #for bit in numpy.arange(power):
    #    new[bit] = bins[0]
    #    new[bit, bit] = 1 - new[bit, bit]
        # individual = Individual(new[bit, bit], func(new[bit, bit]))
        # print(new[bit])

    
            # individual = Individual(new[bit, bit], func(new[bit, bit]))

            # new[bit] = bins[i]
            # new[bit, bit] = 1 - new[bit, bit]

        # best = get_individual(range_a, range_b, precision, power)
        # bins[i,] = best
        # reals[i] = int_to_real(bin_to_int(bins[i,]), range_a, range_b, precision, power)
        # fxs = func(reals[i], precision)
'''
    for i in numpy.arange(1, generations_number):
        print(bins[i-1])
        for bit in numpy.arange(power):
            new[bit] = bins[i-1]
            new[bit] = bins[i, bit] ^ 1
            print(new[bit])

        # best = get_individual(range_a, range_b, precision, power)
        # bins[i,] = best
        # reals[i] = int_to_real(bin_to_int(bins[i,]), range_a, range_b, precision, power)
        # fxs = func(reals[i], precision)


    # table[table[:,9].argsort()]

    print(bins[generations_number-1])



cdef numpy.ndarray[int] get_individual(double range_a, double range_b, int precision, int power):
    cdef double real = random_real(range_a, range_b, precision)
    cdef int int_from_real = real_to_int(real, range_a, range_b, power)
    cdef numpy.ndarray[int] binary = int_to_bin(int_from_real, power)
    cdef int int_from_bin = bin_to_int(binary)
    cdef double real_from_int = int_to_real(
        int_from_bin, range_a, range_b, precision, power)

    return Individual(
        real=real,
        int_from_real=int_from_real,
        binary=binary,
        int_from_bin=int_from_bin,
        real_from_int=real_from_int,
        fx=func(real, precision))

cdef Individual get_individual_from_binary(str binary, double range_a, double range_b, int precision, int power):
    cdef int int_from_bin = bin_to_int(binary)
    cdef double real_from_int = int_to_real(
        int_from_bin, range_a, range_b, precision, power)

    return Individual(
        real=real_from_int,
        int_from_real=int_from_bin,
        binary=binary,
        int_from_bin=int_from_bin,
        real_from_int=real_from_int,
        fx=func(real_from_int, precision))

cdef numpy.ndarray[object] get_individuals_array(double range_a, double range_b, int precision, int population_size, int power):
    cdef numpy.ndarray[object] individuals = numpy.empty(population_size, dtype=Individual)

    cdef int i
    for i in range(population_size):
        individuals[i] = get_individual(range_a, range_b, precision, power)

    return individuals


cdef numpy.ndarray[object, ndim=1] selection_of_individuals(individuals, int precision):
    cdef int len_individuals = len(individuals)
    cdef double fx_min = min(individual.fx for individual in individuals)

    cdef double precision_var = pow(10, -precision)
    cdef int i

    for i in range(0, len_individuals):
        individuals[i].gx = individuals[i].fx - fx_min + precision_var

    cdef double sum_gx = numpy.sum([individual.gx for individual in individuals])
    cdef numpy.ndarray[object, ndim=1] selected_individuals = numpy.empty(len_individuals, dtype=Individual)

    individuals[0].px = individuals[0].gx / sum_gx
    individuals[0].qx = individuals[0].px
 
    for i in range(1, len_individuals):
        individuals[i].qx = individuals[i].gx / sum_gx + individuals[i-1].qx
        individuals[i].px = individuals[i].gx / sum_gx

    for i in range(0, len_individuals):
        individuals[i].random = random.random()
        selected_individuals[i] = copy(individuals[numpy.searchsorted(
            individuals, individuals[i].random, side='right')])

    return selected_individuals


cdef void crossover(numpy.ndarray[object, ndim=1] individuals, double crossover_probability, int power):
    parents = []
    cdef int i
    cdef Individual individual

    for individual in individuals:
        if random.random() <= crossover_probability:
            individual.is_parent = True
            parents.append(individual)
        else:
            individual.cross_population = individual.binary

    cdef int len_parents = len(parents)

    if len_parents > 1:
        if len_parents % 2 == 0:
            for i in range(0, len_parents, 2):
                crossover_of_individuals(parents[i], parents[i+1], power)
        else:
            for i in range(0, len_parents-1, 2):
                crossover_of_individuals(parents[i], parents[i+1], power)
            crossover_of_individuals(parents[0], parents[len_parents-1], power)
    elif len_parents == 1:
        parents[0].is_parent = False
        parents[0].cross_population = parents[0].binary


cdef void crossover_of_individuals(Individual individual_1, Individual individual_2, int power):
    cdef int crossover_point = random.randrange(1, power)
    if individual_1.crossover_points:
        individual_1.crossover_points += ", "
        individual_1.crossover_points += str(crossover_point)
    else:
        individual_1.crossover_points += str(crossover_point)
        individual_1.child_binary = individual_1.binary[:crossover_point] + \
            individual_2.binary[crossover_point:]
        individual_1.cross_population = individual_1.child_binary
    individual_2.crossover_points += str(crossover_point)
    individual_2.child_binary = individual_2.binary[:crossover_point] + \
        individual_1.binary[crossover_point:]
    individual_2.cross_population = individual_2.child_binary


cdef void mutation(numpy.ndarray[object, ndim=1] individuals, double mutation_probability, int power):
    cdef int i
    for individual in individuals:
        individual.mutant_population = individual.cross_population
        for i in range(0, power):
            if random.random() <= mutation_probability:
                individual.mutant_population = individual.mutant_population[:i] + (
                    str(1 - int(individual.mutant_population[i]))) + individual.mutant_population[i+1:]
                if individual.mutation_points:
                    individual.mutation_points += ", "
                    individual.mutation_points += str(i+1)
                else:
                    individual.mutation_points += str(i+1)


def evolution(range_a, range_b, precision, population_size, generations_number, crossover_probability, mutation_probability, elite_number, save_file=True):
    generations = numpy.empty(generations_number, dtype=object)
    population = numpy.empty(population_size, dtype=object)

    cdef int power = power_of_2(range_a, range_b, precision)

    cdef numpy.ndarray[object, ndim=1] individuals = get_individuals_array(
        range_a, range_b, precision, population_size, power)
    
    cdef Individual elite
    if elite_number:
        elite = copy(max(individuals, key=attrgetter('fx')))
    
    cdef numpy.ndarray[object, ndim=1] selected_individuals = selection_of_individuals(
        individuals, precision)

    crossover(selected_individuals, crossover_probability, power)

    mutation(
        selected_individuals, mutation_probability, power)

    cdef int i
    for i in range(0, population_size):
        population[i] = get_individual_from_binary(
            selected_individuals[i].mutant_population, range_a, range_b, precision, power)

    generation = Generation(numpy.empty(population_size, dtype=Individual))

    if elite_number:
        if not any(individual.real == elite.real for individual in population):
            index = random.randrange(0, population_size)
            if population[index].fx < elite.fx:
                population[index] = elite

    generation.individuals = population
    generation.fmin = min(
        individual.fx for individual in generation.individuals)
    generation.fmax = max(
        individual.fx for individual in generation.individuals)
    generation.favg = sum(
        individual.fx for individual in generation.individuals) / population_size
    generations[0] = generation

    get_generations(generations, generations_number, range_a, range_b, precision,
                    population_size, power, crossover_probability, mutation_probability, elite_number)

    return generations


cdef void get_generations(Generation[:] generations, int generations_number, double range_a, double range_b, int precision, int population_size, int power, double crossover_probability, double mutation_probability, int elite_number):
    cdef int gereration_number
    cdef int i
    cdef Individual elite
    cdef numpy.ndarray[object, ndim=1] selected_individuals

    for gereration_number in range(1, generations_number):
        if elite_number:
            elite = copy(
                max(generations[gereration_number - 1].individuals, key=attrgetter('fx')))
                
        selected_individuals = selection_of_individuals(
            generations[gereration_number - 1].individuals, precision)

        crossover(selected_individuals, crossover_probability, power)

        mutation(
            selected_individuals, mutation_probability, power)

        generation = Generation(numpy.empty(population_size, dtype=Individual))

        for i in range(0, population_size):
            generation.individuals[i] = get_individual_from_binary(
                selected_individuals[i].mutant_population, range_a, range_b, precision, power)

        if elite_number:
            if not any(individual.real == elite.real for individual in generation.individuals):
                index = random.randrange(0, population_size)
                if generation.individuals[index].fx < elite.fx:
                    generation.individuals[index] = elite

        generation.fmin = min(
            individual.fx for individual in generation.individuals)
        generation.fmax = max(
            individual.fx for individual in generation.individuals)
        generation.favg = numpy.sum([
            individual.fx for individual in generation.individuals]) / population_size

        generations[gereration_number] = generation

'''
'''
def test(tests_number, precision):
    tests = []

    for generations_number in range(50, 151, 10):
        print(generations_number)
        for population_size in range(30, 81, 5):
            for crossover_probability in numpy.around(numpy.arange(0.5, 0.91, 0.05), 2):
                    generations = []
                    
                    for test in range(0, tests_number):
                        mutation_probability = 0.0001
                        generations.append(evolution(-4.0, 12.0, precision, population_size, generations_number,
                                                 float(crossover_probability), float(mutation_probability), 1, False))
                    tests.append(Test(generations_number, population_size, crossover_probability, mutation_probability, sum(
                        generation[-1].favg for generation in generations)/tests_number, max(generation[-1].fmax for generation in generations)))
                    
                    for mutation_probability in numpy.around(numpy.arange(0.0005, 0.0101, 0.0005), 4):
                        generations = []
                        for test in range(0, tests_number):
                            generations.append(evolution(-4.0, 12.0, precision, population_size, generations_number,
                                                         float(crossover_probability), float(mutation_probability), 1, False))
                        tests.append(Test(generations_number, population_size, crossover_probability, mutation_probability, sum(
                            generation[-1].favg for generation in generations)/tests_number, max(generation[-1].fmax for generation in generations)))

    return tests
'''
