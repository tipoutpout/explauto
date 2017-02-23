import numpy as np
from numpy import array, pi, sqrt, cos, sin, linspace, zeros, random
from ..environment import Environment
from ...utils import bounds_min_max


def forward(angles, lengths):
    """ Link object as defined by the standard DH representation.

    :param list angles: angles of each joint

    :param list lengths: length of each segment

    :returns: a tuple (x, y) of the end-effector position

    .. warning:: angles and lengths should be the same size.
    """
    x, y = joint_positions(angles, lengths)
    return x[-1], y[-1]


def joint_positions(angles, lengths, unit='rad'):
    """ Link object as defined by the standard DH representation.

    :param list angles: angles of each joint

    :param list lengths: length of each segment

    :returns: x positions of each joint, y positions of each joints, except the first one wich is fixed at (0, 0)

    .. warning:: angles and lengths should be the same size.
    """
    if len(angles) != len(lengths):
        raise ValueError('angles and lengths must be the same size!')

    if unit == 'rad':
        a = np.array(angles)
    elif unit == 'std':
        a = np.pi * np.array(angles)
    else:
        raise NotImplementedError
     
    a = np.cumsum(a)
    return np.cumsum(np.cos(a)*lengths), np.cumsum(np.sin(a)*lengths)


def lengths(n_dofs, ratio):
    l = np.ones(n_dofs)
    for i in range(1, n_dofs):
        l[i] = l[i-1] / ratio
    return l / sum(l)


class SimpleArmEnvironment(Environment):
    use_process = True

    def __init__(self, m_mins, m_maxs, s_mins, s_maxs,
                 length_ratio, noise):
        Environment.__init__(self, m_mins, m_maxs, s_mins, s_maxs)

        self.length_ratio = length_ratio
        self.noise = noise

        self.lengths = lengths(self.conf.m_ndims, self.length_ratio)

    def compute_motor_command(self, joint_pos_ag):
        return bounds_min_max(joint_pos_ag, self.conf.m_mins, self.conf.m_maxs)

    def compute_sensori_effect(self, joint_pos_env):
        hand_pos = np.array(forward(joint_pos_env, self.lengths))
        hand_pos += self.noise * np.random.randn(*hand_pos.shape)
        return hand_pos

    def reset(self):
        return NotImplementedError

    def plot(self, ax, m, s, **kwargs_plot):
        self.plot_arm(ax, m, **kwargs_plot)

    def plot_arm(self, ax, m, **kwargs_plot):
        x, y = joint_positions(m, self.lengths)
        x, y = [np.hstack((0., a)) for a in x, y]
        ax.plot(x, y, 'grey', lw=2, **kwargs_plot)
        ax.plot(x[0], y[0], 'ok', ms=6)
        ax.plot(x[-1], y[-1], 'sk', ms=6)
        ax.axis([self.conf.s_mins[0], self.conf.s_maxs[0], self.conf.s_mins[1], self.conf.s_maxs[1]])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

    def test_case(self, n_samples=200):
        # tests = zeros((n_samples, 2))
        # #FIXME low_dimensional
        # if config_str in ('high_dimensional', 'high_dim_high_s_range'):
        #     i = 0
        #     for r, theta in array([1., 2*pi]) * random.rand(n_samples, 2) + array([0., -pi]):
        #         tests[i, :] = sqrt(r) * array([cos(theta), sin(theta)])
        #         i += 1
        #     return tests
        #
        # else:
        #     env = environment(**configurations[config_str])
        #     env.noise = 0.
        return self.uniform_sensor(n_samples)
