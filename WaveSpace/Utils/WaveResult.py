import pandas as pd


class waveResult:
    def __init__(self) -> None:
        self._simInfo = None
        self._log = None
        initData = {'WaveEvent': [0,0], 'waveDuration':[0,0]}
        self._result = self.create_data_frame(initData)
    
    def set_sim_info(self,simInfo):
        self.simInfo= simInfo
    
    def set_log(self,log):
        self._log = log
    
    def set_result(self,result):
        self._result = result

    def create_data_frame(data):
        return pd.DataFrame(data)