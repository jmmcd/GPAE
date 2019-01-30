#!/usr/bin/env python

import random
import numpy as np
import math
import sys
import time

"""Late-acceptance hill-climbing (LAHC) is a stochastic hill-climbing
algorithm with a history mechanism, proposed by Bykov
[http://www.cs.nott.ac.uk/~yxb/LAHC/LAHC-TR.pdf].

(LAHC is not to be confused with an acceptance to the GECCO
late-breaking papers track.)

The history mechanism is very simple but in some domains it seems to
provide a remarkable performance improvement compared to hill-climbing
itself and other heuristics. The big advantage is its simplicity:
fewer parameters to tune compared to many alternative methods.

In standard stochastic hill-climbing, we accept a move to a new
proposed point (created by mutation) if that point is as good as or
better than the current point.

In LAHC, we accept the move if the new point is as good as or better
than that we encountered L steps ago. L is the only new parameter
compared to hill-climbing: it stands for history length.

Step-counting hill-climbing (SCHC)
[http://link.springer.com/article/10.1007/s10951-016-0469-x] is a
variant, proposed by Bykov as an improvement on LAHC. Although less
"natural" it may be slightly simpler to tune again. In SCHC, we
maintain a threshold cost value. We accept moves which are better than
that. We update it once every L steps to the current cost value.

There are also two SCHC variants: instead of counting all steps, we
can count only accepted, or only improving moves.

TODO Implement convergence-detection and restarts.

TODO Implement one short run to detect T/Lc, then choose n based on that
(see SCHC paper above).
"""



"""
This is the LAHC pseudo-code from Bykov and Burke.

Produce an initial solution s
Calculate initial cost function C(s)
Specify Lfa
For all k in {0...Lfa-1} f_k := C(s)
First iteration I=0;
Do until a chosen stopping condition
    Construct a candidate solution s*
    Calculate its cost function C(s*)
    v := I mod Lfa
    If C(s*)<=fv or C(s*)<=C(s)
    Then accept the candidate (s:=s*)
    Else reject the candidate (s:=s)
    Insert the current cost into the fitness array fv:=C(s)
    Increment the iteration number I:=I+1

"""

# Lfa is history length
# n is the budget of evaluations
# C is the cost function
# init is a function that creates an initial individual
# perturb is a mutation function
# mapping is an optional function which maps a genome to a phenotype: if not supplied, no mapping is done. The cost function runs on the phenotype, but init and perturb work on the genotype.
def LAHC(Lfa, n, C, init, perturb, mapping=None):
    if mapping is None:
        mapping = lambda x: x
    s = init() # solution
    sm = mapping(s) # solution mapped
    Cs = C(sm) # cost of solution
    best = s # best solution
    bestm = sm # best solution mapped
    Cbest = Cs # cost of best solution
    #f = Cs * np.ones(Lfa) # If Lfa is large, an array will be more efficient than a list
    f = [Cs] * Lfa
    for I in range(n):
        s_ = perturb(s)
        sm_ = mapping(s_)
        Cs_ = C(sm_)
        if Cs_ < Cbest: # update best-ever, only if *better* than previous best. Note best, bestm, Cbest don't affect the algorithm.
            Cbest = Cs_
            bestm = sm_
            best = s_
        v = I % Lfa
        if Cs_ <= f[v] or Cs_ <= Cs: # accept the candidate
            s = s_
            sm = sm_
            Cs = Cs_
        else:
            pass # reject the candidate
        f[v] = Cs

        # print stats
        # if I % 100 == 0:
        #     print(I, best, bestm, Cbest)
        print(f)

    return best, bestm, Cbest


"""
This is the SCHC pseudo-code from Bykov and Petrovic.

Produce an initial solution s
Calculate an initial cost function C(s)
Initial cost bound Bc := C(s)
Initial counter nc := 0
Specify Lc
Do until a chosen stopping condition
    Construct a candidate solution s*
    Calculate the candidate cost function C(s*)
    If C(s*) < Bc or C(s*) <= C(s)
        Then accept the candidate s := s*
        Else reject the candidate s := s
    Increment the counter nc := nc + 1
    If nc >= Lc
        Then update the bound Bc := C(s)
        reset the counter nc := 0

Two alternative counting methods (start at the first If):

SCHC-acp counts only accepted moves:

    If C(s*) < Bc or C(s*) <= C(s)
        Then accept the candidate s := s*
             increment the counter nc := nc + 1
        Else reject the candidate s := s
    If nc >= Lc
        Then update the bound Bc := C(s)
             reset the counter nc := 0

SCHC-imp counts only improving moves:

    If C(s*) < C(s)
        Then increment the counter nc := nc + 1
    If C(s*) < Bc or C(s*) <= C(s)
        Then accept the candidate s := s*
        Else reject the candidate s := s
    If nc >= Lc
        Then update the bound Bc := C(s)
             reset the counter nc := 0
"""

# Using Lfa instead of Lc as the name for the history length to stay
# consistent with LAHC
def SCHC(Lfa, n, C, init, perturb, mapping=None, count_method="all", outfile=None, Ctest=None):
    start_time = time.time()

    if mapping is None:
        mapping = lambda x: x
    s = init() # solution
    sm = mapping(s) # solution mapped
    Cs = C(sm) # cost of solution
    Bc = Cs # initial cost bound
    nc = 0 # initial counter
    best = s # best solution
    bestm = sm # best solution mapped
    Cbest = Cs # cost of best solution
    first_it = True
    for I in range(n-1): # 1 initial solution plus n-1 iterations
        s_ = perturb(s)
        sm_ = mapping(s_)
        Cs_ = C(sm_)
        if Cs_ < Cbest or first_it: # update best-ever, only on first iteration or if *better* than previous best
            Cbest = Cs_
            bestm = sm_
            best = s_
            if Ctest:
                Ctestv = Ctest(sm_)
            else:
                Ctestv = 0.0
            stats = "%5d %.3f %.3f %.1f" % (I, Cbest, Ctestv, (time.time() - start_time))
            if outfile:
                outfile.write(stats + "\n")
            else:
                print(stats)
            first_it = False

        if count_method == "all": # we count all iterations (moves)
            nc += 1 # increment the counter
        elif count_method == "acp": # we count accepted moves only
            if Cs_ < Bc or Cs_ <= Cs:
                nc += 1 # increment the counter
        elif count_method == "imp": # we count improving moves only
            if Cs_ < Cs:
                nc += 1 # increment the counter

        # NB this stanza must be after the nc += 1 because here we will
        # overwrite Cs_
        if Cs_ < Bc or Cs_ <= Cs: # accept the candidate
            s = s_
            sm = sm_
            Cs = Cs_
        else:
            pass # reject the candidate

        if nc >= Lfa:
            Bc = Cs # update the bound
            nc = 0 # reset the counter

        # print stats
        # if I % 1000 == 0:
        #     print(I, best, bestm, Cbest)

    stats = "%5d %.3f %.3f %.1f" % (I, Cbest, Ctestv, (time.time() - start_time))
    if outfile:
        outfile.write(stats + "\n")
    else:
        print(stats)
    return best, bestm, Cbest


def default_fitness_wrapper(f, default=1e10):
    """Wrap a fitness function with a try/except giving a default
    fitness. The default 1e10 is often suitable for minimisation
    problems.

    """
    def g(x):
        try:
            return f(x)
        except:
            return default
    g.func_name = f.__name__ + "_wrapped"
    return g

def rastrigin(x):
    """F5 Rastrigin's function: multimodal, symmetric, separable"""
    fitness = 10.0 * len(x)
    fitness += sum(xi**2 - 10 * math.cos(2*math.pi*xi) for xi in x)
    return fitness

def cubic(x):
    return x[0] ** 3 - x[1] ** 3

def init(d):
    return [np.random.normal(0.0, 1.0) for i in range(d)]

def mutate(x):
    x = x[:]
    i = random.randrange(len(x))
    x[i] = not x[i]
    return x

def perturb(x, s):
    return normal_with_vector_of_centres(x, s)

def normal_with_vector_of_centres(x, s):
    return [np.random.normal(xi, s) for xi in x]

problems = {
    "rastrigin": {
        "n": 100,
        "Lfa": 5,
        "C": default_fitness_wrapper(rastrigin),
        "init": lambda: init(10),
        "perturb": lambda x: perturb(x, 1.0),
        "mapping": None
    },
    "test_ga": {
        "n": 10000,
        "Lfa": 100, # using Lfa=1 is the best for a onemax problem!
        "C": default_fitness_wrapper(lambda x: x),
        "init": lambda: [random.choice((False, True)) for i in range(50)],
        "perturb": mutate,
        "mapping": sum
    },
}

def main():
    #x, xm, xf = LAHC(**problems["rastrigin"])
    #x, xm, xf = SCHC(count_method="imp", **problems["rastrigin"])
    x, xm, xf = LAHC(**problems["test_ga"])
    print(x, xm, xf)

if __name__ == "__main__":
    random.seed(sys.argv[1])
    main()
