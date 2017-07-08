from explauto.sensorimotor_model.non_parametric import NonParametric,sensorimotor_models

from explauto.interest_model.discrete_progress import DiscretizedProgress, discretized_progress_config


from explauto.experiment import ExperimentPool
from explauto.environment.simple_arm import *
from explauto.interest_model.random import RandomInterest,random_config
from explauto.interest_model.tree import InterestTree,tree_config

from explauto.environment.simple_arm.inhomogeneous_arm import InhomogeneousArmEnvironment
xps = ExperimentPool.from_settings_product(environments=[InhomogeneousArmEnvironment(array([-1, -1.]), array([1., 1.]),1.,0.,5)],
                                           babblings=['goal'],
                                           interest_models=[
                                                # (RandomInterest,random_config["default"]),
                                                #             (DiscretizedProgress,discretized_progress_config['default']),
                                                              (InterestTree,tree_config['default'])
                                           ],
                                           sensorimotor_models=[(NonParametric,sensorimotor_models["nearest_neighbor"]["default"])],
                                           # evaluate_at=[200, 300, 500,1000,1500,2000,3000,4000,5000,8000],
                                           evaluate_at=[10,20,30,40,50,100],
                                           # evaluate_at=[200, 300, 500,1000,1500,2000,3000,4000,5000,6000],
                                           # evaluate_at=[200, 300, 500,1000,1500,2000,3000,4000,5000,8000,10000,12000,14000,16000,18000,20000],
                                           same_testcases=True)
logs = xps.run()

#
# environment=InhomogeneousArmEnvironment(array([-1, -1.]), array([1., 1.]),1.,0.,5)
# babblings=NonParametric(environment.conf,**sensorimotor_models["nearest_neighbor"]["default"])
# random = RandomInterest(environment.conf, environment.conf.m_dims,**random_config["default"])
# testingsMotorPositions=environment.random_motors(100)
# result=
# print random.sample()
# print "oto"
#
# def evaluate(learning)