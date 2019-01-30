#!/usr/bin/env python3

# This code accompanies the paper
# "Why is autoencoding difficult for genetic programming?"
# McDermott, EuroGP 2019

# A basic Linear GP implementation -- a bit like Brameier and Banzhaf
# but no if-statements or loops.
#
# See eg http://lgp.readthedocs.io/en/latest/lgp/representation.html
# for explanation of a similar system.
#
# We can use this linear GP system for symbolic regression including
# with multiple outputs. And so we can put two instances of a linear
# GP system together to use it for autoencoding.

from itertools import product
import sys, os, random, copy, time

import numpy as np
numpy_error_settings = np.seterr(under='ignore', over='raise')

from sklearn.linear_model import LinearRegression
from sklearn import decomposition

from operators import operators, operators_keys
from la_sc_hc import SCHC
from process_data import load_data, normalize_data, read_pair_data



def mark_effective_parts(prog, n_in, n_calc, n_out):
    """Given a program prog, mark which instructions actually
    have any effect."""    

    def _find_idx_of_prev_write(idx, reg):
        for i in range(idx-1, -1, -1):
            lhs, op, *args = prog[i]
            if lhs == reg:
                return i
        return None            

    def _mark_effective_parts(idx):
        lhs, op, *args = prog[idx]
        # print(idx, ":", prog[idx])
        arity = operators[op][1]
        # we mark True and recurse only if not already marked True,
        # to avoid repeating...
        if not marks[idx]: 
            marks[idx] = True
            # print(marks)
            for arg in args[:arity]:
                prev_write = _find_idx_of_prev_write(idx, arg)
                if prev_write is not None:
                    _mark_effective_parts(prev_write)

    # for i, instr in enumerate(prog):
    #     print(i, ":", instr)

    # print("")
    marks = [False for instr in prog]
    out_regs = list(range(n_in + n_calc, n_in + n_calc + n_out))
    # print(out_regs)
    
    for idx in range(len(prog)-1, -1, -1):
        lhs, op, *args = prog[idx]
        if lhs in out_regs:
            # print("starting at ", idx)
            _mark_effective_parts(idx)
    return marks



def evaluate_lgp_prog(prog, X, n_calc, n_out, use_marks=False):
    """Given a program prog and inputs X, evaluate the program
    to produce outputs."""
    
    n_in = X.shape[1]
    # calculation registers hold constants initially
    calc_regs = np.ones((X.shape[0], n_calc)) * np.linspace(0, 1, n_calc)
    # output registers hold zeros initially
    out_regs = np.zeros((X.shape[0], n_out))
    # put input (X), calculation, and output registers all together
    r = np.concatenate((X, calc_regs, out_regs), axis=1)

    if use_marks:
        # start = time.time()
        marks = mark_effective_parts(prog, n_in, n_calc, n_out)
        # end = time.time()
        # print("marking", end-start)
    # else:
    #     marks = [True for _ in prog]
    # print(sum(marks))
    
    # start = time.time()
    for i, (lhs, op, *args) in enumerate(prog): # for each instruction
        if use_marks and not marks[i]: continue
        f, arity = operators[op] # get the function in this instruction
        r[:, lhs] = f(*[r[:, argi] for argi in args[:arity]]) # apply the function
    # end = time.time()
    # print("executing", end-start)
    return r[:, -n_out:]


def initialise(n_regs, n_instrs):
    """Make a random program."""
    return [
        [random.randrange(n_regs), random.choice(operators_keys),
         random.randrange(n_regs), random.randrange(n_regs)]
        for _ in range(n_instrs)
        ]

def initialise_pair(n_regs0, n_regs1, n_instrs):
    """Make a random individual where one individual is a pair of programs."""
    return (initialise(n_regs0, n_instrs),
            initialise(n_regs1, n_instrs))

def mutate(p, n_regs):
    """Mutate a program."""
    p = copy.deepcopy(p) # our search algo assumes the operator doesn't change the original
    i = random.randrange(len(p))
    j = random.randrange(4)
    if j == 1:
        # mutate the op
        p[i][j] = random.choice(operators_keys)
    else:
        # mutate a register index
        p[i][j] = random.randrange(n_regs)
    return p

def mutate_pair(pair, n_regs0, n_regs1):
    """Mutate an individual where one individual is a pair of programs."""
    # mutate both
    # return (mutate(pair[0], n_regs0), mutate(pair[1], n_regs1))
    # or mutate just one:
    if random.randrange(2):
        return (mutate(pair[0], n_regs0), pair[1])
    else:
        return (pair[0], mutate(pair[1], n_regs1))

def MSE(y, yhat):
    """Our objective function is the mean (over each output) of the mean
    squared error."""
    return np.mean([_MSE(y[i], yhat[i]) for i in range(len(y))])
    
def _MSE(y, yhat):
    """Plain-old mean square error."""
    return np.mean(np.square(y - yhat))


def test():
    """This is a test, but really the purpose is to show an example program."""
    prog = [
        [0, "+", 0, 2], # r0 = r0 + r2
        [6, "*", 1, 0], # r6 = r1 * r0
        [5, "sin", 1, 3]  # r5 = sin r1 # ignore last operand since sin has arity 1
    ]
    X = np.array([[1, 1], [2, 2.]])
    n_calc = 3
    n_out = 2
    print(evaluate_lgp_prog(prog, X, n_calc, n_out))


def SR(X, y, Xtest, ytest, n_calc, n_instrs, L, n, outfile):
    """Symbolic regression on data (X, y) with test data (Xtest, ytest).
    We allow n fitness evaluations and a step-counting parameter L."""

    n_in = X.shape[1]
    n_out = y.shape[1]

    n_regs = n_in + n_calc + n_out

    C = lambda prog: MSE(y, evaluate_lgp_prog(prog, X, n_calc, n_out))
    Ctest = lambda prog: MSE(ytest, evaluate_lgp_prog(prog, Xtest, n_calc, n_out))
    
    init = lambda: initialise(n_regs=n_regs, n_instrs=n_instrs)
    perturb = lambda p: mutate(p, n_regs=n_regs)
    
    best, _, Cbest = SCHC(L, n, C, init, perturb, mapping=None, count_method="all", Ctest=Ctest, outfile=outfile)

def AE(X, Xtest, n_calc, n_hidden, n_instrs, L, n, outfile):
    """Auto-encoder on data X with test data Xtest. We allow n fitness evaluations
    and step-counting parameter L."""

    n_in = X.shape[1]
    n_out = n_in # AE

    n_regs0 = n_in + n_calc + n_hidden
    n_regs1 = n_hidden + n_calc + n_out
    
    C = lambda pair: MSE(X, evaluate_lgp_prog(pair[1], evaluate_lgp_prog(pair[0], X, n_calc, n_hidden), n_calc, n_out)) # TODO: KL loss
    Ctest = lambda pair: MSE(Xtest, evaluate_lgp_prog(pair[1], evaluate_lgp_prog(pair[0], Xtest, n_calc, n_hidden), n_calc, n_out)) # TODO: KL loss
    
    init = lambda: initialise_pair(n_regs0=n_regs0, n_regs1=n_regs1, n_instrs=n_instrs)
    perturb = lambda pair: mutate_pair(pair, n_regs0=n_regs0, n_regs1=n_regs1)
    
    best, _, Cbest = SCHC(L, n, C, init, perturb, mapping=None, count_method="all", Ctest=Ctest, outfile=outfile)
    

def test_marks():
    """Our evaluate_lgp_prog() function marks effective code before
    execution, and then when executing, doesn't run the ineffective
    code, to try to save time. This function is for checking that the
    marking is correct: no difference in output between a version that
    does this versus a version that does. And the good news is that
    it is correct.

    However, this function also tests whether the marks actually save
    time, and the answer is no. Here is I think a typical result (I
    haven't done proper averages):

    use_marks = True:
    marking 0.02069091796875
    executing 0.021139860153198242

    use_marks = False
    executing 0.031243085861206055    

    That is the execution itself does take longer with False,
    but the marking takes almost as long as the execution,
    so summed together it's worse!

    """
    
    X, _ = generate_AE_data()
    n_in = X.shape[1]
    n_calc = 5
    n_out = 2
    n_instrs = 3000
    
    for i in range(1000):
        # generate a random prog, and evaluate it with and without the
        # effective code marking scheme.
        p = initialise(n_in + n_calc + n_out, n_instrs)
        start = time.time()
        y0 = evaluate_lgp_prog(p, X, n_calc, n_out, use_marks=True)
        end = time.time()
        elapsed_marks_True = end-start
        start = time.time()
        y1 = evaluate_lgp_prog(p, X, n_calc, n_out, use_marks=False)
        end = time.time()
        elapsed_marks_False = end-start
        print(elapsed_marks_True < elapsed_marks_False)
        if not np.allclose(y0, y1):
            print("A test failed!")
            print(i)
            print(p)
            marks = mark_effective_parts(p, n_in, n_calc, n_out)
            print(marks)
            print(y0)
            print(y1)
            break
    else:
        print("All tests pass")
            
    
def LR(Xtrain, ytrain, Xtest, ytest):
    """Linear regression using sklearn."""
    LR = LinearRegression()
    LR.fit(Xtrain, ytrain)
    yhat = LR.predict(Xtest)
    print(MSE(ytest, yhat))
    
def generate_multiple_fake_SR_targets(X, Xtest):
    assert X.shape == (130, 9) # this is hard-coded for GLASS dataset
    # output 4 columns for the GLASS dataset as simple transformations
    # of the input
    def f(X):
        y = np.array([
            X[:, 0]**2,
            X[:, 1] - X[:, 2]**3,
            np.sqrt(X[:, 3] * X[:, 4] * X[:, 5]),
            np.sin(X[:, 6] + X[:, 7] / (1 + np.abs(X[:, 8])))
        ]).T
        return y
    return X, f(X), Xtest, f(Xtest)

def generate_identical_fake_SR_targets(X, Xtest):
    assert X.shape == (130, 9) # this is hard-coded for GLASS dataset
    # output 4 columns for the GLASS dataset as simple transformations
    # of the input
    def f(X):
        y = np.array([
            np.sin(X[:, 6] + X[:, 7] / (1 + np.abs(X[:, 8]))),
            np.sin(X[:, 6] + X[:, 7] / (1 + np.abs(X[:, 8]))),
            np.sin(X[:, 6] + X[:, 7] / (1 + np.abs(X[:, 8]))),
            np.sin(X[:, 6] + X[:, 7] / (1 + np.abs(X[:, 8])))
        ]).T
        return y
    return X, f(X), Xtest, f(Xtest)

def generate_single_fake_SR_target(X, Xtest):
    assert X.shape == (130, 9) # this is hard-coded for GLASS dataset
    # output 4 columns for the GLASS dataset as simple transformations
    # of the input
    def f(X):
        y = np.array([
            np.sin(X[:, 6] + X[:, 7] / (1 + np.abs(X[:, 8])))
        ]).T
        return y
    return X, f(X), Xtest, f(Xtest)

def generate_AE_data():
    # generate some data: _X is random, but X has structure
    _X = np.random.random((100000, 5))
    X = np.stack((_X[:, 2], _X[:, 0] * _X[:, 1], _X[:, 2] * _X[:, 3], _X[:, 2]), axis=1)
    Xtrain, Xtest = X[:80], X[80:]
    return Xtrain, Xtest

if __name__ == "__main__":
    s = int(sys.argv[1])
    random.seed(s); np.random.seed(s)
    # prob = "test"
    # prob = "test_marks"
    # prob = "AE"
    # prob = "AE_experiment"
    # prob = "AE_GLASS_experiment"
    # prob = "SR_experiment_multi_fake"
    # prob = "SR_experiment_single_fake"
    prob = "SR_experiment_multi_identical_fake"

    if prob == "test":
        test()
        
    elif prob == "test_marks":
        test_marks()
            
    elif prob == "SR":
        SR(*read_pair_data("data/Vladislavleva4_train.txt", "data/Vladislavleva4_test.txt"))

    elif prob == "SR_baseline":
        LR(*read_pair_data("data/Vladislavleva4_train.txt", "data/Vladislavleva4_test.txt"))
    
    elif prob == "AE_experiment":
        
        data_filename = "GLASS" # 9 dimensions
        n_hidden = 4 # as used by Loi Van Cao for GLASS
        outdir = "../results/" + "AE_experiment_" + data_filename + "/"

        X, Xtest, ytest = load_data(data_filename)
        X, Xtest = normalize_data(X, Xtest, "maxabs")

        pca = decomposition.PCA(n_components=n_hidden)
        pca.fit(X)
        Z = pca.transform(X)
        Xhat = pca.inverse_transform(Z)
        print("train", MSE(X, Xhat))
        Ztest = pca.transform(Xtest)
        Xtesthat = pca.inverse_transform(Ztest)        
        print("test", MSE(Xtest, Xtesthat))
        #sys.exit()

        
        
        n = 100000
        n_calcs = [50]
        n_instrss = [50]
        Ls = [1000]
        
        for n_calc in n_calcs:
            for n_instrs in n_instrss:
                for L in Ls:
                    outfilename = outdir + "/%d_%d_%d_%d_%d_%d.dat" % (
                        n_calc,
                        n_instrs,
                        L,
                        n_hidden,
                        n,
                        s)
                    print(outfilename)
                    os.makedirs(outdir, exist_ok=True)
                    outfile = open(outfilename, "w")
                    AE(X, Xtest, n_calc, n_hidden, n_instrs, L, n, outfile)

    elif prob.startswith("SR_experiment"):
        
        data_filename = "GLASS"
        outdir = "../results/" + "SR_experiment_" + data_filename + "/"

        X, Xtest, ytest = load_data(data_filename)
        X, Xtest = normalize_data(X, Xtest, "maxabs")

        if prob == "SR_experiment_multi_fake":
            X, y, Xtest, ytest = generate_multiple_fake_SR_targets(X, Xtest)
        elif prob == "SR_experiment_multi_identical_fake":
            X, y, Xtest, ytest = generate_identical_fake_SR_targets(X, Xtest)
        elif prob == "SR_experiment_single_fake":
            X, y, Xtest, ytest = generate_single_fake_SR_target(X, Xtest)
        else:
            raise ValueError()

        print("baseline: if we predicted mean of training data at the output")
        Xhat = np.ones_like(y) * np.mean(y, axis=0)
        print("train", MSE(y, Xhat))
        Xhat = np.ones_like(ytest) * np.mean(y, axis=0)
        print("test", MSE(ytest, Xhat))

        # sys.exit()

        
        n = 100
        n_calcs = [5, 50, 500]
        n_instrss = [5, 50, 500]
        Ls = [10, 100, 1000]
        n_hidden = 0 # a dummy value; we put it in the filename for compatibility with AE
        
        for n_calc in n_calcs:
            for n_instrs in n_instrss:
                for L in Ls:
                    outfilename = outdir + "/%d_%d_%d_%d_%d_%d.dat" % (
                        n_calc,
                        n_instrs,
                        L,
                        n_hidden,
                        n,
                        s)
                    print(outfilename)
                    os.makedirs(outdir, exist_ok=True)
                    outfile = open(outfilename, "w")
                    SR(X, y, Xtest, ytest, n_calc, n_instrs, L, n, outfile)
                    
