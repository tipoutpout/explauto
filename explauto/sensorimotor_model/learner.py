
import numpy as np

from . import inverse
from . import forward


class Learner(object):
    def __init__(self, Mfeats, Sfeats, Mbounds, fwd=forward.LWLRForwardModel, inv= inverse.BFGSInverseModel, **kwargs):
        """

            :arg Mfeats:  the motor features (tuple of int)
            :arg Sfeats:  the sensory features (tuple of int, element-wise distinct from Mfeats)
            :arg Mbounds: the boundary describing legal motor values.
            :arg fwd:     string representing the kind of forward model to instanciate.
            :arg inv:     string representing the kind of inverse model to instanciate.
        """
        self.Mfeats  = Mfeats
        self.Sfeats  = Sfeats
        self.Mbounds = Mbounds
        fmodel = fwd(len(Mfeats), len(Sfeats), **kwargs)
        self.imodel = inv(dim_x=len(Mfeats), dim_y=len(Sfeats), fmodel=fmodel, constraints=Mbounds, **kwargs)

    # Interface

    def add_xy(self, x, y):
        """ Add an motor order/sensory effect pair to the model

            :arg x:  an input (order) vector compatible with self.Mfeats.
            :arg y:  a output (effect) vector compatible with self.Sfeats.
        """
        self.imodel.add_xy(self._pre_x(x), self._pre_y(y))
                
    def add_xy_batch(self, x_list, y_list): self.imodel.fmodel.add_xy_batch(x_list, y_list)

    def infer_order(self, goal, **kwargs):
        """Infer an order in order to obtain an effect close to goal, given the
        observation data of the model.

        :arg goal:  a goal vector compatible with self.Sfeats
        :rtype:  same as goal (if list, tuple, or pandas.Series, else tuple)
        """
        x = self.imodel.infer_x(np.array(self._pre_y(goal)), **kwargs)[0]
        return self._post_x(x, goal)

    def predict_effect(self, order, **kwargs):
        """Predict the effect of a goal.

        :arg order:  an order vector compatible with self.Mfeats
        :rtype:  same as order (if list, tuple, or pandas.Series, else tuple)
        """
        y = self.imodel.fmodel.predict_y(np.array(self._pre_x(order)), **kwargs)
        return self._post_y(y, order)

    # Pre and post treatment

    def _pre_x(self, x):
        assert len(x) == len(self.Mfeats)
        return tuple(x)

    def _pre_y(self, y):
        assert len(y) == len(self.Sfeats)
        return tuple(y)

    def _post_x(self, result, input_object):
        return tuple(result)

    def _post_y(self, result, input_object):
        return tuple(result)
