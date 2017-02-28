from explauto.environment.simple_arm import SimpleArmEnvironment
from explauto.utils import inside,rand_bounds
import numpy as np


class SimpleArmWithCloud(SimpleArmEnvironment):
    """This  extends the SimpleArmEnvironment class by adding a cloud on the sensitive space where they is no feeback
    If the arm is behind the cloud, the position of the arm is unknow and a random value on the cloud is returned"""
    def __init__(self, m_mins, m_maxs, s_mins, s_maxs, length_ratio, noise, cloud_min, cloud_max):
        """
        :param m_mins: An numpy array containing the minimum value taken by all motor dimentions
        :param m_maxs: An numpy array containing the maximum value taken by all motor dimentions
        :param s_mins: An numpy array containing the minimum value taken by all sensitives dimentions
        :param s_maxs: An numpy array containing the maximum value taken by all sensitives dimentions
        :param length_ratio: length ratio from one segment to the following one
        :param noise: gaussian noise added in the sensor space
        :param cloud_min: An numpy array containing the minimum corner of the cloud
        :param cloud_max: An numpy array containing the maximum corner of the cloud
        """
        SimpleArmEnvironment.__init__(self,m_mins, m_maxs, s_mins, s_maxs, length_ratio, noise)
        self.cloudAreaMin = cloud_min
        self.cloudAreaMax = cloud_max

    def compute_sensori_effect(self, joint_pos_env):
        effect = SimpleArmEnvironment.compute_sensori_effect(self,joint_pos_env)
        if inside(effect,self.cloudAreaMin,self.cloudAreaMax):
            effect =rand_bounds(np.vstack( (self.cloudAreaMin,self.cloudAreaMax) ))[0]
            return effect
            # return self.cloudAreaMin
        else:
            return effect


if __name__ == '__main__':
    from explauto.experiment import ExperimentPool
    from explauto.environment.simple_arm import *

    config = configurations["mid_dimensional"].copy()
    config['cloud_min'] = array([-0.5, -1.])
    config['cloud_max'] = array([1., 1.])
    xps = ExperimentPool.from_settings_product(environments=[SimpleArmWithCloud(**config)],
                                               babblings=['motor'],
                                               interest_models=[('random', 'default'),
                                                                ('discretized_progress', 'default'),
                                                                ('tree', 'default')],
                                               sensorimotor_models=[('nearest_neighbor', 'default')],
                                               # evaluate_at=[200, 300, 500],
                                               evaluate_at=[200, 300, 500, 1000],
                                               same_testcases=True)
    logs = xps.run()