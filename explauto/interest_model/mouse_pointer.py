from numpy import array

from .interest_model import InterestModel
from ..ioExplauto.mouse_pointer import MousePointer as MP


class MousePointer(InterestModel):
    def __init__(self, conf, expl_dims, width, height):
        InterestModel.__init__(self, expl_dims)
        self.pointer = MP(width, height)

    def sample(self):
        return array(self.pointer.xy)

    def update(xy, ms):
        pass

mopurse_pointer_config = {'default': {'width': 320,'height': 240}}
