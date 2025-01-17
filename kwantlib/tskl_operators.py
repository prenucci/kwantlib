import pandas as pd 
from sklearn.linear_model import ElasticNet

try:
    from tskit_learn.timeseriesmodel import ExpandingModel

    class tskl_Operator:
        @staticmethod   
        def infer(
            target: pd.DataFrame | pd.Series, features:pd.DataFrame, 
            model:object = ElasticNet(), train_every_n_steps:int = 30, 
            lookahead_steps:int = 0,
        ) -> pd.DataFrame:
            """ 
            Infer the target from the features using a model trained in expanding window for a given frequency of retraining
            """
            assert hasattr(model, 'fit') and hasattr(model, 'predict'), 'model should have fit and predict methods'
            assert isinstance(target, pd.DataFrame) or isinstance(target, pd.Series), 'target should be a pd.DataFrame or pd.Series'
            model_ts = ExpandingModel(model, train_every_n_steps, train_every_n_steps, lookahead_steps)
            return model_ts.fit_predict(features, target.shift(lookahead_steps), skipna=True)

        @staticmethod   
        def cluster(signal:pd.DataFrame) -> pd.DataFrame: 
            """ 
            Cluster the signal df using a clustering algorithm
            """
            raise NotImplementedError('cluster')
            return 
    
except Exception:
    class tskl_Operator:
        @staticmethod   
        def infer(*args, **kwargs) -> pd.DataFrame:
            raise NotImplementedError('tskit_learn is necessary for this operation. pip install git+https://github.com/Cnernc/tskit_learn.git')                      
        @staticmethod   
        def cluster(*args, **kwargs) -> pd.DataFrame: 
            raise NotImplementedError('tskit_learn is necessary for this operation. pip install git+https://github.com/Cnernc/tskit_learn.git')

def monkey_patch_tskl_operators(): 
    pd.Series.infer = tskl_Operator.infer
    pd.DataFrame.infer = tskl_Operator.infer
    pd.DataFrame.cluster = tskl_Operator.cluster
