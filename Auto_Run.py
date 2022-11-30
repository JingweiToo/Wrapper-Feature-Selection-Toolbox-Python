import pandas as pd
import numpy as np
import yaml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from FS.pso import jfs as jfs_pso


class DataPipeline(object):
    """
    Data Model with Pipeline to Process and Feature Select.
    """
    def __init__(self, _ori_data_path:str, _train_test_ratio:float, _str_read_meta_para:str) -> None:
        """
        para1:_ori_data_path:Read_RawData_Path(CSV_Format) => Convert to DataFrame 
        para2:_train_test_ratio:Split DataSet Ratio
        """
        self.data = None
        self.feat = None
        self.label = None

        self.fold = None # Contain Splited Train/Test Data
        self.meta_para = None

        self._load_ori_file(_ori_data_path)
        self._data_split(_train_test_ratio)
        self._read_algo_parameter(_str_read_meta_para)
        self._process_feature_select()

    def _load_ori_file(self, _ori_path) -> None:
        self.data = pd.read_csv(_ori_path)
        self.data  = self.data.values
        #All Feature
        self.feat  = np.asarray(self.data[:, 0:-1]) 
        #Predict Y Value
        self.label = np.asarray(self.data[:, -1])
        #print(self.data)

    def _data_split(self, ratio:float) -> None:
        xtrain, xtest, ytrain, ytest = train_test_split(self.feat, self.label, test_size=ratio, stratify=self.label)
        self.fold = {'xt':xtrain, 'yt':ytrain, 'xv':xtest, 'yv':ytest}

    def _read_algo_parameter(self, _str_read_meta_para) ->None:
        with open('algo_para.yaml', 'r') as _str_read_meta_para:
            self.meta_para = yaml.full_load(_str_read_meta_para)['meta_para']
            print(self.meta_para)
    
    def _process_feature_select(self) -> None:
        # perform feature selection
        ## Append Fold Data Into OPTS~~~~~~~~~~~~~~~ 
        fmdl = jfs_pso(self.feat, self.label, self.meta_para[0]['opts'])
        sf   = fmdl['sf']

def main():
    str_read_file_path = './ionosphere.csv'
    str_read_meta_para = './algo_para.yaml'
    float_split_ratio = 0.3

    auto_run = DataPipeline(str_read_file_path, float_split_ratio, str_read_meta_para)


if __name__ == '__main__':
    main()
    