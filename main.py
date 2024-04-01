import time
import torch
import numpy as np
from MH_sampler import PathAuxiliarySampler,MSASampler
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import trange,tqdm
from Models import TSP,MIS,Maxcut
from ImportTSP import TSPDataLoader,MISDataLoader
from PAFS import PathAuxiliaryFastSampler
log_g=lambda x:x/(x+1)
#sampler=PathAuxiliarySampler(log_g=log_g)
sampler=MSASampler(1)
def Draw_TSP(nodes):
    plt.scatter(nodes[:,0],nodes[:,1])
    plt.show()
def solve_TSP():
    m =1
    n =1
    t = 10
    use_dataloader=False
    #nodes = [[1, 3], [2, 6], [-1, 5], [4, 1], [8, 3], [9, -1], [6, 9], [2, 8]]
    if use_dataloader:
        loader=TSPDataLoader()
        x,nodes=loader.LoadTSP()
        x-=1
        real_best_solution=loader.GetSolution()
        real_best_solution-=1
    else:
        nodes=np.load('E:\Projects\GitProject\DIMES\DIMES\TSP\data\\test-500-coords.npy')
        nodes=nodes[0]
        sols=np.load('E:\Projects\GitProject\DIMES\DIMES\TSP\data\\test-500-costs.npy')
        sols=sols[0]
        x=[[i] for i in range(len(nodes))]
        x = torch.tensor(x, dtype=float)
        nodes = torch.tensor(nodes, dtype=float)
    model = TSP(nodes)
    model.t=1
    initial_t=model.t
    new_x=x
    sampler.x=x
    model.current_x=x
    sampler.R=1
    sampler.size=model.size
    model.scalar=1
    model.K=5
    z = []

    trange_m=trange(m)
    for i in trange_m:
        trange_m.set_description('Train progress')
        # for j in range(n):
        #     new_x=sampler.step(model,model.current_x)
        #     sampler.update_U()
        #     if model.energy(new_x)<model.energy(model.current_x):
        #         model.current_x=new_x
        #     else:
        #         if np.random.rand()<np.exp(-(model.energy(new_x)-model.energy(model.current_x)).detach().numpy()/model.t):
        #             model.current_x=new_x
        log_p,_trace,elapse,succ=sampler.sample(model)
        if model.energy(model.current_x) < model.energy(sampler.x):
            model.current_x = sampler.x
            model.distances.append(model.get_total_distance(model.current_x))
        else:
            if np.random.rand() < np.exp(
                    (model.energy(sampler.x)).detach().numpy() / model.t):
                model.current_x = sampler.x
        #print('model.x:',model.current_x)
        sampler.update_U(succ/1000)
        print('succ:',succ)
        z.append(new_x)
        model.t=initial_t*(1-(i+1)/m)
        #print("current temperature:",model.t)
    best_value=np.min(model.distances)
    print('Minimum distance:',best_value)

    best=model.get_best_solution()
    #print('Identical route:',best)
    best=best.detach().numpy()
    # print('Correct best path:',real_best_solution)
    # print('Correct minimum trip:',model.get_total_distance(real_best_solution))
    # print('Distance between answer:',best_value-model.get_total_distance(real_best_solution))
    #Draw_TSP(nodes)


def solve_MIS():
    er_p=0.2
    m=300
    n=100
    mis=0
    show_graph=False
    use_dataloader=True
    if use_dataloader==False:
        seed = 324
        num_nodes = 100
        g=nx.erdos_renyi_graph(num_nodes,er_p,seed=seed)
        edges=g.edges
    else:
        dataloader=MISDataLoader(path="E:\Dataset\RLSolver_data_result\data\syn_ER\erdos_renyi_500_ID3.txt")
        num_nodes,edges=dataloader.LoadMISFromPickle("E:\Dataset\er_test\ER_700_800_0.15_0.gpickle")
        g=nx.from_edgelist(edges)
        print('edges:',edges)
        print('num_nodes:',num_nodes)
    adjacency_matrix = [[0 for i in range(num_nodes)] for j in range(num_nodes)]
    x = [[1] for i in range(num_nodes)]
    for loc in edges:
        adjacency_matrix[loc[0]-1][loc[1]-1]=1
        adjacency_matrix[loc[1]-1][loc[0]-1]=1
    adjacency_matrix=torch.tensor(adjacency_matrix,dtype=float)
    x=torch.tensor(x,dtype=float)
    print(x.shape)
    model=MIS(adjacency_matrix)
    initial_t =model.t
    next_x=x.clone()
    model.current_x=next_x.clone()
    main_training_loop=trange(m)
    main_training_loop.set_description('Training progress')
    sampler.R=1
    sampler.x=x
    for i in main_training_loop:
        #for j in range(n):
            # next_x=sampler.step(t=model.current_x,model=model)
            #
            # if model.energy(next_x)<model.energy(model.current_x):
            #     model.current_x=next_x
            # else:
            #     if np.random.rand()<np.exp(model.energy(next_x).detach().numpy()/model.t)[0][0]:
            #         model.current_x=next_x
        logp, _trace, elapsed, succ = sampler.sample(model)
        sampler.update_U(succ / 1000)
        if model.energy(model.current_x)>model.energy(sampler.x):
            model.current_x = sampler.x
        else:
            if np.random.rand()<np.exp(-model.energy(sampler.x).detach().numpy()/model.t):
                model.current_x=sampler.x
        next_x=model.post_process(model.current_x)
        model.t=initial_t*(1-(i+1)/m)
        independent=next_x.sum().detach().numpy()
        if mis<independent:
            mis=independent
            print('energy:',model.energy(next_x))
    print('Maximum Independent Set:',mis)
    if show_graph:
        nx.draw(g)
        plt.show()
    return mis

def solve_Maxcut():
    #edges=[(1,2),(3,4)]
    dataloader=MISDataLoader()
    num_nodes,edges=dataloader.LoadMISFromPickle("E:\Dataset\er_test\ER_700_800_0.15_0.gpickle")
    # num_nodes=5
    # edges=[(0,1),(0,3),(1,2),(2,4),(3,4)]
    pafs=PathAuxiliaryFastSampler(log_g=log_g)
    model=Maxcut(edges,num_nodes)
    x=model.init_state()
    result=x.clone()
    for i in trange(50):
        for j in range(100):
            result=pafs.step(result,model)
            #print('avg_accs:',pafs.avg_accs)
            pafs.U=np.clip(pafs.U + 0.001 * (pafs.avg_accs - 0.574), a_min=1, a_max=model.size)
    print(model(result))


def solve_MIS_new():
    dataloader = MISDataLoader(path="E:\Dataset\RLSolver_data_result\data\syn_ER\erdos_renyi_500_ID3.txt")
    num_nodes, edges = dataloader.LoadMISFromPickle("E:\Dataset\er_test\ER_700_800_0.15_0.gpickle")
    g = nx.from_edgelist(edges)
    bsize=128
    pafs = PathAuxiliaryFastSampler(log_g=log_g)
    adjacency_matrix = [[0 for i in range(num_nodes)] for j in range(num_nodes)]
    x = [[1 for i in range(num_nodes)] for j in range(bsize)]
    for loc in edges:
        adjacency_matrix[loc[0]][loc[1]] = 1
        adjacency_matrix[loc[1]][loc[0]] = 1
    adjacency_matrix=torch.tensor(adjacency_matrix,dtype=float)
    model = MIS(adjacency_matrix)
    model.current_x=torch.tensor(x,dtype=float)
    x = model.init_state()
    result = x.clone()
    initial_t=model.t
    m=50
    n=100
    for i in trange(m):
        for j in range(n):
            result = pafs.step(result, model)
            #print('avg_accs:',pafs.avg_accs)
            pafs.U = np.clip(pafs.U + 0.001 * (pafs.avg_accs - 0.574), a_min=1, a_max=model.size)
        model.t=initial_t * (1-(i+1)/m)
    for sol in result:
        print('result:',model.post_process(sol.reshape(num_nodes,1)).sum())
    print(model(result))

solve_MIS_new()