import time
from numpy import zeros, array, hstack
from abc import ABCMeta, abstractmethod

from ..utils.config import make_configuration
from ..utils.observer import Observable
from ..utils import rand_bounds
from .testcase import Lattice

class Environment(Observable):
    """ Abstract class to define environments.

        When defining your sub-environment, you should specify whether they could be forked and run in different processes through the use_process class variable. By default, it is set to False to guarantee that the code will work.

    """
    __metaclass__ = ABCMeta
    use_process = False

    def __init__(self, m_mins, m_maxs, s_mins, s_maxs):
        """
        :param numpy.array m_mins, m_maxs, s_mins, s_maxs: bounds of the motor (m) and sensory (s) spaces

        """
        Observable.__init__(self)

        self.conf = make_configuration(m_mins, m_maxs, s_mins, s_maxs)

    def one_update(self, m_ag, log=True):
        m_env = self.compute_motor_command(m_ag)
        s = self.compute_sensori_effect(m_env)

        if log:
            self.emit('motor', m_env)
            self.emit('sensori', s)

        return s

    def update(self, m_ag, reset=True, log=True):
        """ Computes sensorimotor values from motor orders.

        :param numpy.array m_ag: a motor command with shape (self.conf.m_ndims, ) or a set of n motor commands of shape (n, self.conf.m_ndims)

        :param bool log: emit the motor and sensory values for logging purpose (default: True).

        :returns: an array of shape (self.conf.ndims, ) or (n, self.conf.ndims) according to the shape of the m_ag parameter, containing the motor values (which can be different from m_ag, e.g. bounded according to self.conf.m_bounds) and the corresponding sensory values.

        .. note:: self.conf.ndims = self.conf.m_ndims + self.conf.s_ndims is the dimensionality of the sensorimotor space (dim of the motor space + dim of the sensory space).
        """

        if reset:
            self.reset()
        if len(array(m_ag).shape) == 1:
            s = self.one_update(m_ag, log)
        else:
            s = []
            for m in m_ag:
                s.append(self.one_update(m, log))
            s = array(s)
        return s

    @abstractmethod
    def reset(self):
        """ reset environment before update """
        return

    @abstractmethod
    def compute_motor_command(self, ag_state):
        return

    @abstractmethod
    def compute_sensori_effect(self):
        return

    def random_motors(self, n=1):
        return rand_bounds(self.conf.bounds[:, self.conf.m_dims], n)

    def dataset(self, orders):
        return array([hstack((m, self.update(m))) for m in orders])

    def uniform_sensor(self, n_cases=100):
        n_random_motor = n_cases * 100
        m_feats = tuple(range(-self.conf.m_ndims, 0))
        s_feats = tuple(range(self.conf.s_ndims))
        m_bounds = tuple([(self.conf.m_mins[d], self.conf.m_maxs[d])
                          for d in self.conf.m_dims])

        ms_uniform_motor = self.dataset(self.random_motors(n_random_motor))
        resolution = max(2, int((1.3*n_cases)**(1.0/len(s_feats))))
        observations = []
        for ms in ms_uniform_motor:
            observations.append((ms[self.conf.m_dims], ms[self.conf.s_dims]))
        lattice = Lattice(s_feats, observations, res=resolution)
        result = []
        for ms in list(lattice.grid.values()):
            result.append(ms[1])
        return array(result)

    @abstractmethod
    def plot(self, ax, m, s, **kwargs):
        return

    @abstractmethod
    def test_case(self, n_samples):
        return