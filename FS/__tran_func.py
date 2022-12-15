import numpy as np
class tran_func():
    
    @staticmethod
    def l_trans(val):
        return val

    @staticmethod
    def sl_trans(val):
        return 1 / (1+np.exp(-2 * val))    