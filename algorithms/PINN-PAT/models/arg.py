from models.activation import Activation
from enum import Enum
class MEDIUM(Enum):
    A = "A: Homogeneous and unlossy"
    B = "B: Heterogeneous and unlossy"
    C = "C: Homogeneous and lossy"
    D = "D: Heterogeneous and lossy"

class WAVEFIELD_FIT_TYPE(Enum):
    FORWARD = "Forward"
    INVERSE = "Inverse"

class Arg:
    def __init__(
        self,
        fit_type: WAVEFIELD_FIT_TYPE=None,
        medium: MEDIUM=None,
        in_channels: int=None,
        out_channels: int=None,
        n_hidden: int=None,
        l_layers: int=None,
        activation: Activation=None,

        t_min: float=None,
        t_max: float=None,
        n_t: int=None,

        pde_batch_size: int=None,
        t0_batch_size: int=None,
        bc_batch_size: int=None,
        p0_batch_size: int=None,

        pde_weight: float=None,
        dpdt0_weight: float=None,

        lr_start: float=None,
        lr_end: float=None,

        num_iteration: int=None,

        save_filepath: str=None,
        load_filepath: str=None,
        sensor_filepath: str=None,
        p0_filepath: str=None
    ):
        self.medium = medium if medium is not None else MEDIUM.A
        self.fit_type = fit_type if fit_type is not None else WAVEFIELD_FIT_TYPE.INVERSE
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_hidden = n_hidden
        self.l_layers = l_layers
        self.activation = activation

        self.t_min = t_min
        self.t_max = t_max
        self.n_t = n_t

        self.pde_batch_size = pde_batch_size
        self.t0_batch_size = t0_batch_size
        self.bc_batch_size = bc_batch_size
        self.p0_batch_size = p0_batch_size

        self.pde_weight = pde_weight
        self.dpdt0_weight = dpdt0_weight

        self.num_iteration = num_iteration
        self.lr_start = lr_start
        self.lr_end = lr_end

        self.save_filepath = save_filepath
        self.load_filepath = load_filepath
        self.sensor_filepath = sensor_filepath
        self.p0_filepath = p0_filepath


        

    def to_dict(self):
        res = dict(self.__dict__)
        res["activation"] = self.activation._value_
        res["medium"] = self.medium._value_
        res["fit_type"] = self.fit_type._value_
        return res

    @classmethod
    def from_dict(cls, dt: dict) -> "Arg":
        obj = cls()
        for k, v in dt.items():
            if k == "activation":
                obj.__dict__[k] = Activation(v)
            elif k == "medium":
                obj.__dict__[k] = MEDIUM(v)
            elif k == "fit_type":
                obj.__dict__[k] = WAVEFIELD_FIT_TYPE(v)
            else:
                obj.__dict__[k] = v
        return obj