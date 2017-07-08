from explauto.utils import *

from explauto.environment.simple_arm import SimpleArmEnvironment
from numpy import array, pi, random,absolute,power,vectorize,remainder

class InhomogeneousArmEnvironment(SimpleArmEnvironment):
    def __init__(self, s_mins, s_maxs, length_ratio, noise,nb_motor):
        m_mins=array([-pi] * nb_motor)
        m_maxs=array([pi] * nb_motor)
        super(InhomogeneousArmEnvironment, self).__init__(m_mins, m_maxs, s_mins, s_maxs, length_ratio, noise)

    def compute_motor_command(self, joint_pos_ag):
        temp=apply_gaussian_noise_vec( joint_pos_ag)
        # return remainder(temp+pi,2*pi)-pi
        return temp

def aply_gaussian_noise(motor_position):
    if motor_position==0: return 0.
    return motor_position + random.normal(0,power(absolute(motor_position),5)/400)
apply_gaussian_noise_vec=vectorize(aply_gaussian_noise)

if __name__ == '__main__':
    from random import random as rand
    import matplotlib.pyplot as plt
    plt.axis([-pi,pi,-4*pi,4*pi])
    environment = InhomogeneousArmEnvironment(array([-1, -1.]), array([1., 1.]), 1., 0., 1)
    for _ in range(1000):
        x=rand()*2*pi-pi
        plt.plot(x,environment.compute_motor_command([x])[0],'.')
    plt.show()
