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
from lib.models import Individual

@numba.jit(nopython=True, fastmath=True)
def random_real(range_a,  range_b,  precision):
    prec = pow(10, precision)
    return numpy.round(random.randrange(range_a * prec, (range_b) * prec + 1)/prec, precision)


@numba.jit(nopython=True, fastmath=True)
def power_of_2(range_a,  range_b,  precision):
    return int(numpy.rint(numpy.log2(((range_b - range_a) * (1/pow(10, -precision)) + 1))))

@numba.jit(fastmath=True)
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

@numba.jit(nopython=True, fastmath=True)
def new_individuals(bins, new_bins, new_fxs, range_a,  range_b, precision, power, generations_number):
    for bit in numpy.arange(power):
        new_bins[bit] = bins
        new_bins[bit, bit] = 1 - new_bins[bit, bit]
        new_fxs[bit] = func(int_to_real(bin_to_int(new_bins[bit]), range_a,  range_b, precision, power))


@numba.jit(nopython=True, fastmath=True)
def mutation(bins, individuals, power, tau):
    for bit in numpy.arange(1, power + 1):
        r = random.random()
        t = 1/pow(bit, tau)
        if r <= t:
            bins[individuals[bit-1]] = 1 - bins[individuals[bit-1]]

#@numba.jit(forceobj=True)
def get_evolution(individuals, bins, reals, fxs, best_fxs, new_bins, new_fxs, best, range_a, range_b, precision, power, tau, generations_number):
    for i in numpy.arange(1, generations_number):
        bins[i] = bins[i-1]
        new_individuals(bins[i], new_bins, new_fxs, range_a, range_b, precision, power, generations_number)
        for bit in numpy.arange(power):
            individuals[bit] = Individual(new_bins[bit], new_fxs[bit], bit)

        individuals = sorted(individuals)
        individuals_bins = numpy.array([individual.id for individual in individuals], dtype=numpy.int)
        mutation(bins[i], individuals_bins, power, tau)

        reals[i] = int_to_real(bin_to_int(bins[i]), range_a, range_b, precision, power)
        fxs[i] = func(reals[i])

        if fxs[i] > best.fx:
            best.fx = fxs[i]
            best.real = reals[i]
            best.binary = bins[i]

        best_fxs[i] = best.fx

        new_bins = numpy.empty((power, power), dtype=numpy.int)
        new_fxs = numpy.empty(power, dtype=numpy.double)
        individuals = numpy.empty(power, dtype=numpy.object)



def evolution(range_a, range_b, precision, tau, generations_number, save_file=True):
    power = power_of_2(range_a, range_b, precision)
    reals = numpy.empty(generations_number, dtype=numpy.double)
    bins = numpy.empty((generations_number, power), dtype=numpy.int)
    fxs = numpy.empty(generations_number, dtype=numpy.double)
    best_fxs = numpy.empty(generations_number, dtype=numpy.double)
    new_bins = numpy.empty((power, power), dtype=numpy.int)
    new_fxs = numpy.empty(power, dtype=numpy.double)
    individuals = numpy.empty(power, dtype=numpy.object)

    bins[0] = get_individual(range_a, range_b, precision, power)
    
    new_individuals(bins[0], new_bins, new_fxs, range_a, range_b, precision, power, generations_number)

    for bit in numpy.arange(power):
        individuals[bit] = Individual(new_bins[bit], new_fxs[bit], bit + 1)

    individuals = sorted(individuals)
    individuals_bins = numpy.array([individual.id for individual in individuals], dtype=numpy.int)
    mutation(bins[0], individuals_bins, power, tau)
    reals[0] = int_to_real(bin_to_int(bins[0]), range_a, range_b, precision, power)
    fxs[0] = func(reals[0])
    
    best = Individual(bins[0], fxs[0], 0, reals[0])
    best_fxs[0] = best.fx

    get_evolution(individuals, bins, reals, fxs, best_fxs, new_bins, new_fxs, best, range_a, range_b, precision, power, tau, generations_number)

    return best, fxs, best_fxs

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
