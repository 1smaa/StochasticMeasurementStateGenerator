import numpy as np
from typing import List,Callable

from GeneratorException import GeneratorException

class LinearGenerator(object):
    def __init__(self,seed: int,n: int,A: np.ndarray,P: np.ndarray,H: np.ndarray,S: np.ndarray) -> None:
        '''
        seed: random generator seed for reproducibility
        n: number of elements of the measurement matrix
        A: state transition matrix
        P: variance vector
        H: observation matrix
        S: initial state estimated matrix
        '''
        np.random.seed(seed)
        self.__n=n
        self.__A=A
        self.__P=P
        self.__H=H
        self.__S=S
        #Check for matrixes dimensions validity
        self.__check()
        
    def __check(self) -> None:
        x=self.__P.shape
        if(x[0]!=self.__n): raise GeneratorException("Variance vector should be of the same length as the measurement vector.")
        x,y=self.__A.shape
        x1=self.__S.shape[0]
        if(x1!=x or x1!=y): raise GeneratorException("State Transition Matrix should be nxn where n is the dimension of the state estimate vector.")
        x,y=self.__H.shape
        if(x!=x1 or y!=self.__n): raise GeneratorException("Observation matrix shaped incorrectly.")
    
    def generate(self) -> np.ndarray:
        '''
        Generates a new array of random data
        '''
        # Estimate the next state by multiplying the state transition matrix and the previous state
        self.__S=self.__A@self.__S
        # Map the state estimate to the measurement matrix
        m=self.__S@self.__H
        # Map the covariances to the variance of the single measurement
        return np.random.normal(m,np.sqrt(self.__P),m.shape)
    
    @property
    def state_estimate(self) -> np.ndarray:
        '''
        Returns the current state estimate on which the generator is working
        '''
        return self.__S
    
class NonLinearGenerator(LinearGenerator):
    def __init__(self,seed: int,n: int,A: Callable[[np.ndarray],np.ndarray],P: np.ndarray,H: np.ndarray,S: np.ndarray) -> None:
        '''
        seed: random generator seed for reproducibility
        n: number of elements of the measurement matrix
        A: state transition function
        P: variance vector
        H: observation matrix
        S: initial state estimated matrix
        '''
        np.random.seed(seed)
        self.__n=n
        self.__A=A
        self.__P=P
        self.__H=H
        self.__S=S
        #Check for matrixes dimensions validity
        self.__check()
        
    def __check(self) -> None:
        x=self.__P.shape
        if(x[0]!=self.__n): raise GeneratorException("Variance vector should be of the same length as the measurement vector.")
        x1=self.__S.shape[0]
        x,y=self.__H.shape
        if(x!=x1 or y!=self.__n): raise GeneratorException("Observation matrix shaped incorrectly.")
    
    def generate(self) -> np.ndarray:
        '''
        Generates a new array of random data
        '''
        # Estimate the next state by multiplying the state transition matrix and the previous state
        self.__S=self.__A(self.__S)
        # Map the state estimate to the measurement matrix
        m=self.__S@self.__H
        # Map the covariances to the variance of the single measurement
        return np.random.normal(m,np.sqrt(self.__P),m.shape)