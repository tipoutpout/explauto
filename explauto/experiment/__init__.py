from collections import namedtuple



class Settings(namedtuple('Settings', ('environment',
                                       'babbling_mode',
                                       'interest_model_class',
                                       'interest_model_config',
                                       'sensorimotor_model_class',
                                       'sensorimotor_model_config',
                                       'context_mode'))):
    @property
    def default_testcases(self):
        return self.environment.test_case()


def make_settings(environment,
                  babbling_mode,
                  interest_model_class, sensorimotor_model_class,
                  interest_model_config='default',
                  sensorimotor_model_config='default',
                  context_mode=None):

    return Settings(environment,
                    babbling_mode,
                    interest_model_class, interest_model_config,
                    sensorimotor_model_class, sensorimotor_model_config, context_mode)

from .experiment import Experiment
from .pool import ExperimentPool
from .log import ExperimentLog
