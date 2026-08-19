import pandas as pd


class waveResult:
    def __init__(self) -> None:
        """
        Returns
        -------
        None
        """
        self._simInfo = None
        self._log = None
        initData = {'WaveEvent': [0,0], 'waveDuration':[0,0]}
        self._result = self.create_data_frame(initData)
    
    def set_sim_info(self,simInfo):
        """
        Parameters
        ----------
        simInfo : object

        Returns
        -------
        None
        """
        self.simInfo= simInfo
    
    def set_log(self,log):
        """
        Parameters
        ----------
        log : object

        Returns
        -------
        None
        """
        self._log = log
    
    def set_result(self,result):
        """
        Parameters
        ----------
        result : pandas.DataFrame

        Returns
        -------
        None
        """
        self._result = result

    def create_data_frame(data):
        """
        Parameters
        ----------
        data : dict

        Returns
        -------
        pandas.DataFrame
        """
        return pd.DataFrame(data)