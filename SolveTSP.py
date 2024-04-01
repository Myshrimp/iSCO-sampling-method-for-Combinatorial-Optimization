import torch
import numpy as np
import torch_geometric.nn as gnn
import torch.nn as nn
from tqdm import trange
import matplotlib.pyplot as plt
from ImportTSP import TSPDataLoader
class TSP():
    def __init__(self):
        self.K=4
        self.nodes=[]
        self.size=0
        self.t=10
        self.distances=[]
        self.solutions=[]
        self.minimum_distance=99000
        self.minimum_value_pointer=-1
        self.matrix=None
        self.knn_graph=None
    def change(self,x):
        #2-OPT
        solution_new=x.clone()
        knn_graph=self.knn_graph
        #print('(TSP:change)knn_graph:',knn_graph)
        knn_indices=[]
        while True:
            loc1 = int(np.ceil(np.random.rand() * (self.nodes.shape[0] - 1)))
            loc2 = int(np.ceil(np.random.rand() * (self.nodes.shape[0] - 1)))
            if loc1!=loc2:
                break
        #print('(TSP:change)loc1:',loc1)
        for i in range(self.K):
            knn_indices.append(int(knn_graph[0][(loc1+1) * self.K-1-i]))
        #print('(TSP:change)knn_indices',knn_indices)
        if np.random.rand()<self.K/(self.K+1):
            loc2=np.random.choice(knn_indices)
        #print('(TSP:change)loc2:',loc2)
        solution_new=solution_new.detach().numpy()
        temp=solution_new[0][loc1]
        solution_new[0][loc1]=solution_new[0][loc2]
        solution_new[0][loc2]=temp
        solution_new=torch.tensor(solution_new,dtype=float)
        #print('(TSP:change)solution_new:',solution_new)
        value_new=self.get_total_distance(solution_new)
        if value_new <self.get_total_distance(x):
            #print('(TSP:change)better solution,new distance:',self.get_total_distance(solution_new))
            self.distances.append(value_new)
            if value_new < self.minimum_distance:
                self.minimum_distance=value_new
                self.minimum_value_pointer+=1
                self.solutions.append(solution_new)
            return solution_new
        else:
            if np.random.rand()<np.exp((self.energy(solution_new).detach().numpy()-self.energy(x).detach().numpy())/self.t):
                #print('(TSP:change)high energy')
                return solution_new
            else:
                #print('(TSP:change)low energy')
                return x


    def energy(self,x):
        return -1 * self.get_total_distance(x)
    def get_distance_matrix(self):
        p_dist = nn.PairwiseDistance(p=2)
        size=int(self.nodes.shape[0])
        matrix=torch.zeros((size,size),dtype=float)
        for i in trange(size):
            for j in range(size):
                matrix[i][j]=p_dist(self.nodes[i],self.nodes[j])
        return matrix

    def get_total_distance(self,x):
        x_copy=x.T.clone()
        indices=x_copy.long() #convert to long type that can serve as index in gather function
        indices=indices.detach().numpy()
        value=0
        mat=self.matrix.detach().numpy()
        for i in range(self.size-1):
            value+=mat[indices[i][0]][indices[i+1][0]]
        value+=mat[indices[0][0]][indices[self.size-1][0]]
        return torch.tensor(value)

    def init(self,nodes):
        self.nodes=nodes
        self.size=nodes.shape[0]
        print('self.size:',self.size)
        self.matrix=self.get_distance_matrix()
        self.knn_graph=gnn.knn_graph(self.nodes, k=self.K)
    def get_best_solution(self):
        return self.solutions[self.minimum_value_pointer]
# a=np.load('E:\Projects\GitProject\DIMES\DIMES\TSP\data\\test-500-coords.npy')
# a=torch.tensor(a[0])
# print(a.shape)
# knns=gnn.knn_graph(a,k=10)
# print(knns.shape)
def Draw_TSP(nodes):
    plt.scatter(nodes[:,0],nodes[:,1])
    plt.show()
def solve_TSP():
    m = 500
    n = 100
    t = 10
    #nodes = [[1, 3], [2, 6], [-1, 5], [4, 1], [8, 3], [9, -1], [6, 9], [2, 8]]
    nodes=np.load('E:\Projects\GitProject\DIMES\DIMES\TSP\data\\test-500-coords.npy')
    nodes=nodes[0]
    z=[]
    use_dataloader=True
    if use_dataloader:
        loader = TSPDataLoader()
        x, nodes = loader.LoadTSP()
        x=x.T
        x -= 1
        real_best_solution = loader.GetSolution()
        real_best_solution -= 1
    else:
        nodes = np.load('E:\Projects\GitProject\DIMES\DIMES\TSP\data\\test-500-coords.npy')
        nodes = nodes[0]
        x = [[i] for i in range(len(nodes))]
        x = torch.tensor(x, dtype=float)
        nodes = torch.tensor(nodes, dtype=float)
    model = TSP()
    model.init(nodes)
    model.t=t
    initial_t=model.t
    new_x=x
    for i in range(m):
        for j in range(n):
            new_x=model.change(new_x)
        z.append(new_x)
        model.t=initial_t*(1-(i+1)/m)
        print("current temperature:",model.t)
    best_value=np.min(model.distances)
    print('Minimum distance:',best_value)

    best=model.get_best_solution()
    print('Identical route:',best)
    print('Correct minimum trip:', model.get_total_distance(real_best_solution))
    print('Distance between answer:', best_value - model.get_total_distance(real_best_solution))
    Draw_TSP(nodes)
solve_TSP()