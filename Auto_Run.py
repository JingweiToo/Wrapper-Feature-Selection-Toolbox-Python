import pandas as pd
import numpy as np
import yaml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from FS.pso import jfs as jfs_pso
from FS.ga import jfs as jfs_ga
from FS.de import jfs as jfs_de
from FS.ba import jfs as jfs_ba
from FS.cs import jfs as jfs_cs
from FS.fa import jfs as jfs_fa
from FS.fpa import jfs as jfs_fpa
from FS.sca import jfs as jfs_sca
from FS.woa import jfs as jfs_woa


class DataPipeline(object):
    """
    Data Model with Pipeline to Process and Feature Select.
    """
    def __init__(self, _ori_data_path:str, _train_test_ratio:float, _str_read_meta_para:str) -> None:
        """
        para1:_ori_data_path:Read_RawData_Path(CSV_Format) => Convert to DataFrame 
        para2:_train_test_ratio:Split DataSet Ratio
        """
        self._ori_data_path = _ori_data_path
        self._train_test_ratio = _train_test_ratio
        self._str_read_meta_para = _str_read_meta_para

        self.data = None
        self.feat = None
        self.label = None

        self.fold = None # Contain Splited Train/Test Data
        self.meta_para = None

        self.sf = None # Select_Feature 
        self.fmdl = None # Feature Model

    def _load_ori_file(self, _ori_path) -> None:
        self.data = pd.read_csv(_ori_path)
        self.data  = self.data.values
        #All Feature
        self.feat  = np.asarray(self.data[:, 0:-1]) 
        #Predict Y Value
        self.label = np.asarray(self.data[:, -1])

    def _data_split(self, ratio:float) -> None:
        xtrain, xtest, ytrain, ytest = train_test_split(self.feat, self.label, test_size=ratio, stratify=self.label)
        self.fold = {'xt':xtrain, 'yt':ytrain, 'xv':xtest, 'yv':ytest}

    def _read_algo_parameter(self, _str_read_meta_para:str) ->None:
        with open('algo_para.yaml', 'r') as _str_read_meta_para:
            self.meta_para = yaml.full_load(_str_read_meta_para)['meta_para']
    
    def _build_fmdl(self, algo_name):
        interface_fmdl = {'pso': jfs_pso,'ga':jfs_ga, 'de':jfs_de, 'ba':jfs_ba, 'cs':jfs_cs, 'fa': jfs_fa, 'fpa':jfs_fpa, 'sca':jfs_sca, 'woa':jfs_woa}
        return interface_fmdl[algo_name]   

    def _process_feature_select(self, strMetaName:str, listMetaOpts:list) -> None:
        # perform feature selection
        ## Append Fold Data Into OPTS
        #print(self._build_fmdl('ga'))
        listMetaOpts['fold'] = self.fold
        #self.fmdl = jfs_ga(self.feat, self.label, listMetaOpts)
        self.fmdl = self._build_fmdl(strMetaName)(self.feat, self.label, listMetaOpts)
        self.sf   = self.fmdl['sf']
    
    def _data_feature_select(self, strMetaName:str, listMetaOpts:list)->None:
        # model with selected features
        num_train = np.size(self.fold['xt'], 0)
        num_valid = np.size(self.fold['xv'], 0)
        x_train   = self.fold['xt'][:, self.sf]
        y_train   = self.fold['yt'].reshape(num_train)  # Solve bug
        x_valid   = self.fold['xv'][:, self.sf]
        y_valid   = self.fold['yv'].reshape(num_valid)  # Solve bug

        mdl       = KNeighborsClassifier(n_neighbors = listMetaOpts['k']) 
        mdl.fit(x_train, y_train)

        # accuracy
        y_pred    = mdl.predict(x_valid)
        Acc       = np.sum(y_valid == y_pred)  / num_valid
        print("Accuracy:", 100 * Acc)

        # number of selected features
        num_feat = self.fmdl['nf']
        print("Feature Size:", num_feat)

    def _plot_converege(self, strMetaName:str, listMetaOpts:list)-> None:
        '''
        plot convergence
        '''
        curve   = self.fmdl['c']
        curve   = curve.reshape(np.size(curve,1))
        x       = np.arange(0, listMetaOpts['T'], 1.0) + 1.0
        fig, ax = plt.subplots()
        ax.plot(x, curve, 'o-')
        ax.set_xlabel('Number of Iterations')
        ax.set_ylabel('Fitness')
        ax.set_title(strMetaName)
        ax.grid()
        plt.show()
    
    def proceed(self)->None:
        '''
        Execute Function
        '''
        self._load_ori_file(self._ori_data_path)
        self._data_split(self._train_test_ratio)
        self._read_algo_parameter(self._str_read_meta_para)

        for meta in self.meta_para:
            print(meta['name'])
            print(meta['opts'])
            self._process_feature_select(meta['name'], meta['opts'])
            self._data_feature_select(meta['name'], meta['opts'])
            self._plot_converege(meta['name'], meta['opts'])

def main():
    str_read_file_path = './ionosphere.csv'
    str_read_meta_para = './algo_para.yaml'
    float_split_ratio = 0.3

    auto_run = DataPipeline(str_read_file_path, float_split_ratio, str_read_meta_para)
    auto_run.proceed()

if __name__ == '__main__':
    main()
    