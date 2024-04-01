import torch
import numpy as np
import torch_geometric.nn as gnn
import torch.nn as nn
from tqdm import trange
import random
class MIS(nn.Module):
    def __init__(self,adj_matrix):
        super().__init__()
        self.adjacency_matrix=adj_matrix
        self.size=int(self.adjacency_matrix.shape[0])
        self.initial_x=0
        self.current_x=torch.zeros((1,self.size),dtype=float)
        self.Lambda=1.0001
        self.t=15000
    def energy(self,x):
        #return -x.sum()+self.Lambda * (x.T @ self.adjacency_matrix @ x)/2
        return -x.sum(dim=1) + self.Lambda * (x @ self.adjacency_matrix @ x.T).diag() / 2

    def forward(self,x):
        #print(torch.exp((x.sum(dim=1) - self.Lambda * (x @ self.adjacency_matrix @ x.T).diag() / 2)/self.t))
        #return -x.sum()+self.Lambda * (x.T @ self.adjacency_matrix @ x)/2
        return torch.exp((x.sum(dim=1) - self.Lambda * (x @ self.adjacency_matrix @ x.T).diag() / 2)/self.t)
    def change(self,x):
        value_change=(1-2*x)*(-1+self.Lambda*(self.adjacency_matrix@x))
        return value_change.T
    def post_process(self,x):
        x_copy=x.clone()
        for i in range(self.size):
            if x_copy[i][0] == 1:
                is_invalid = self.adjacency_matrix[i] @ x_copy
                if is_invalid >= 0.5:
                    x_copy[i][0] = 0
        return x_copy

    # def flip_state(self,x,idx):
    #     xc=x.clone().detach().numpy()
    #     xc[idx][0]=1-xc[idx][0]
    #     return torch.tensor(xc) # size[num_nodes,1]

    def flip_state(self,cur_x,b_idx,index):
        cur_x[b_idx,index]=1-cur_x[b_idx,index]
        return cur_x
    def init_state(self):
        return self.current_x
    def trace(self,x):
        return 1
        pass
class TSP():
    def __init__(self,nodes):
        self.K=20
        self.nodes=nodes
        self.size = nodes.shape[0]
        self.matrix = self.get_distance_matrix()
        self.t=10
        self.distances=[]
        self.solutions=[]
        self.minimum_distance=99000
        self.minimum_value_pointer=-1
        self.current_x=0
        self.scalar=0.0002
        self.knn_graph=gnn.knn_graph(self.nodes,k=self.K)
        self.changeValue=[]
    def flip_state(self,x,idx):#input size:[1,num_nodes]
        #2-OPT
        solution_new=x.clone()

        #print('(TSP:change)knn_graph:',knn_graph)
        knn_indices=[]
        while True:
            loc1 = int(np.ceil(np.random.rand() * (self.nodes.shape[0] - 1)))
            loc2 = idx.detach().numpy()
            loc2=loc2[0][0]
            if loc1!=loc2:
                break
        #print('(TSP:change)loc1:',loc1)
        # for i in range(self.K):
        #     knn_indices.append(int(self.knn_graph[0][(loc1+1) * self.K-1-i]))
        #print('(TSP:change)knn_indices',knn_indices)
        lower_bound=int(self.knn_graph[0][loc1 * self.K])
        upper_bound=int(self.knn_graph[0][(loc1 + 1) * self.K - 1])
        if np.random.rand()<self.K/(self.K+1):
            #loc2=np.random.choice(knn_indices)
            if lower_bound< upper_bound:
                loc2 = random.randint(lower_bound, upper_bound)
            else:
                loc2 = random.randint(upper_bound,lower_bound)

        #print('(TSP:change)loc2:',loc2)
        solution_new=solution_new.detach().numpy()
        temp=solution_new[loc1][0]
        loc1_=loc1+1 if loc1+1 < self.size else 0
        solution_new[loc1][0]=solution_new[loc1_][0]
        solution_new[loc1_][0]=temp

        temp = solution_new[loc2][0]
        loc2_ = loc2 + 1 if loc2 + 1 < self.size else 0
        solution_new[loc2][0] = solution_new[loc2_][0]
        solution_new[loc2_][0] = temp
        solution_new=torch.tensor(solution_new,dtype=float)
        #print('(TSP:change)solution_new:',solution_new)
        value_new=self.get_total_distance(solution_new)
        self.distances.append(value_new)
        self.solutions.append(solution_new)
        print('flipped loc:',loc1,loc2)
        return solution_new #size[5,1]

    def energy(self,x):
        return -1 * self.get_total_distance(x)*self.scalar
    def change(self,x):
        prob=torch.tensor(self.changeValue,dtype=float)
        return prob.T
    def get_distance_matrix(self):
        p_dist = nn.PairwiseDistance(p=2)
        size=int(self.nodes.shape[0])
        matrix=torch.zeros((size,size),dtype=float)
        for i in trange(size):
            for j in range(size):
                matrix[i][j]=p_dist(self.nodes[i],self.nodes[j]).requires_grad_()
                #print("(TSP:get_distance_matrix)matrix[i][j]:",matrix[i][j])
        return matrix

    def init_state(self):
        return self.current_x
    def trace(self,x):
        return 1
        pass
    def get_total_distance(self,x):
        x_copy=x.clone()
        indices=x_copy.long() #convert to long type that can serve as index in gather function
        indices=indices.detach().numpy()
        value=0
        mat=self.matrix.detach().numpy()
        self.changeValue=[]
        for i in range(self.size-1):
            value+=mat[indices[i][0]][indices[i+1][0]]
            self.changeValue.append([mat[indices[i][0]][indices[i+1][0]]])
        value+=mat[indices[0][0]][indices[self.size-1][0]]
        self.changeValue.append([mat[indices[0][0]][indices[self.size-1][0]]])
        #print(self.changeValue)
        return torch.tensor(value)

    def init(self,nodes):
        self.nodes=nodes
        self.size=nodes.shape[0]
        self.matrix=self.get_distance_matrix()
    def get_best_solution(self):
        return self.solutions[self.minimum_value_pointer]

class Maxcut(nn.Module):
    def __init__(self,edges,num_nodes):
        super().__init__()
        self.edges=edges
        self.size=num_nodes
        self.matrix=None
        self.x=None
        self.bsize=5
        self.weight_matrix=self.get_weight_matrix(self.edges)
    def init_state(self):
        x=[[1 for j in range(self.size)] for i in range(self.bsize)]
        return torch.tensor(x,dtype=float)

    def flip_state(self,x,b_idx,idx):
        #print('(Maxcut)idx:',b_idx,idx)
        x[b_idx,idx]=-x[b_idx,idx]
        return x

    def forward(self,x):
        weight=torch.tensor(self.weight_matrix,dtype=float)
        # value=0
        # for edge in self.edges:
        #     value+=(1-(2*x[:,edge[0]]) * (2*x[:,edge[1]]))/2
        # print('(Maxcut):value',value)
        #print('x:',x)
        mat=x@weight
        value=mat @ mat.T
        value=-value.diag()
        #print('values:',value)
        return value

    def get_weight_matrix(self,edges):
        adjacency_matrix=[[0 for i in range(self.size)] for j in range(self.size)]
        for loc in edges:
            adjacency_matrix[loc[0]][loc[1]] = 1
            adjacency_matrix[loc[1]][loc[0]] = 1
        return adjacency_matrix



