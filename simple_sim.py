import numpy as np
from numpy.random import geometric, exponential, random_sample
from decimal import Decimal
from math import log, e, floor, sin
from matplotlib import pyplot as plt
from copy import deepcopy
import scipy.fftpack

BLOCK_INTERVAL = 600000 #miliseconds
INITIAL_SAMPLE = 2020
RATE = 1*10**18

class Block:
    def __init__(self, time, difficulty = None, hashrate = None):
        assert(difficulty>0)
        self.time = int(time)
        self.difficulty = difficulty # actually work
        self.hashrate = hashrate

    @classmethod
    def mine_with_hashrate(cls, difficulty, hashrate):
        probability = hashrate / difficulty # per unit time ~1/600
        time = geometric(probability)
        return cls(time, difficulty, hashrate)


class Chain(list):
    def __init__(self, d, h):
        assert( len(h) == len(d))
        for i in range(len(d)):
            self.append(Block.mine_with_hashrate(d[i],h[i]))

    def timestamps(self):
        '''Returns a list of global time for each block'''
        s = 0
        return [int(s) for b in self if ( s := s+b.time )]

    def chainwork(self, index):
        h = [b.difficulty for b in self[:index]]
        return sum(h)

    def mine(self, hashrate_list, DAA):
        '''Adds blocks with hashrates on the list using DAA'''
        for r in hashrate_list:
            difficulty = DAA(self)
            self.append(Block.mine_with_hashrate(difficulty, r))

    def plot(self, options):
        options = options.split()
        def moving_average(data, window_width):
            cumsum_vec = np.cumsum(np.insert(data, 0, 0))
            return (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
        data = []
        if 'difficulty' in options:
            data = [b.difficulty for b in self[INITIAL_SAMPLE:]]
        if 'time' in options:
            data = [b.time for b in self]
        if 'average' in options:
            data = moving_average(data, 144)
        if 'spectrum' in options:
            data = [np.abs(d - (BLOCK_INTERVAL* RATE)) for d in data]
            data = scipy.fftpack.rfft(data)
        #plt.scatter(self.timestamps(), data, s = 5)
        plt.plot(data)
        plt.show()


def bch_daa(chain, delta):
    ts = chain.timestamps()
    tb_new = np.median([ts[-1], ts[-2], ts[-3]])
    tb_old = np.median([ts[delta-1], ts[delta-2], ts[delta-3]])
    index_new, index_old = ts.index(tb_new), ts.index(tb_old)
    tb_delta = tb_new - tb_old
    low_delta = (-delta/2 * BLOCK_INTERVAL)
    up_delta = (-delta*2 * BLOCK_INTERVAL)
    tb_delta = low_delta if tb_delta < low_delta else tb_new - tb_old
    tb_delta = up_delta if tb_delta > up_delta else tb_new - tb_old
    work_delta = chain.chainwork(index_new)-chain.chainwork(index_old)
    projected_work = (work_delta * BLOCK_INTERVAL) // tb_delta
    return projected_work # I skip calculating target because it's just a simulation


def legacy_daa(chain):
    if len(chain)%2015 != 0:
        return chain[-1].difficulty
    else:
        ts = chain.timestamps()
        c = (ts[-1] - ts[-2016])/1209600
        return chain[-1].difficulty / c


def tripple_daa(chain):
    a = bch_daa(chain, -144)
    b = bch_daa(chain, -200)
    c = bch_daa(chain, -100)
    return (0.2*a+0.3*b+0.5*c)


d = [RATE* BLOCK_INTERVAL]*INITIAL_SAMPLE
h = [RATE ]*INITIAL_SAMPLE

chain = Chain(d,h)

hashrates = [RATE]*144*30
bch_daa_144 = lambda r: bch_daa(r,-144)
chain.mine(hashrates, bch_daa_144)
#chain.mine(hashrates,tripple_daa)
chain.plot('difficulty average')



