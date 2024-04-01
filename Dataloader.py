import pandas as pd
default_path="E:\Dataset\RLSolver_data_result\data\\tsplib\\att48"
import numpy as np
import torch
import networkx as nx
class TSPDataLoader():
    def __init__(self,path=default_path):
        self.path=path
        self.tsp_suffix='.tsp'
        self.tsp_opt_suffix='.opt.tour'
        self.df = pd.read_csv(self.path+self.tsp_suffix, sep=' ', skiprows=6, header=None)
        self.answer=pd.read_csv(self.path+self.tsp_opt_suffix, sep=' ', skiprows=5, header=None)
    def LoadTSP(self):
        # 载入数据
        city = np.array(self.df[0][0:len(self.df) - 1])  # 最后一行为EOF，不读入
        city_name = city.tolist()
        initial_solution=list(map(float,city_name))
        #print('city name:', city_name)
        city_x = np.array(self.df[1][0:len(self.df) - 1])
        city_y = np.array(self.df[2][0:len(self.df) - 1])
        city_location = list(zip(city_x, city_y))
        #print('city loc:', city_location)
        city_location = torch.tensor(city_location, dtype=float)
        initial_solution=torch.tensor([initial_solution],dtype=float)
        return initial_solution.T,city_location

    def GetSolution(self):
        city = np.array(self.answer[0][0:len(self.answer) - 2])  # 最后一行为EOF，不读入
        city_name = city.tolist()
        _solution = list(map(float, city_name))
        _solution = torch.tensor([_solution], dtype=float)
        return _solution.T

default_MIS_path="E:\Dataset\RLSolver_data_result\data\syn_ER\erdos_renyi_100_ID0.txt"
class MISDataLoader():
    def __init__(self,path=default_MIS_path):
        self.path=path
        self.df = pd.read_csv(self.path, sep=' ', skiprows=0, header=None)

    def LoadMIS(self):
        num_nodes=self.df[0][0]
        city_x = np.array(self.df[0][1:len(self.df)])
        city_y = np.array(self.df[1][1:len(self.df)])
        edges = list(zip(city_x, city_y))
        return num_nodes,edges

    def LoadMISFromPickle(self,path):
        g=nx.readwrite.read_gpickle(path)
        edges=g.edges
        num_nodes=np.max(edges)+1
        return num_nodes,edges
if __name__=='__main__':
    dataloader=MISDataLoader()
    num_nodes,answer=dataloader.LoadMIS()
    print(num_nodes,answer)



