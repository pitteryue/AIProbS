from packages_management import *
import parameters_management
import math
import recbole.model.general_recommender as RecBole_general_benchmarks
from recbole.config.configurator import Config
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger
from logging import getLogger
import recbole.data.dataset.dataset
from recbole.data.interaction import Interaction
from recbole.data.utils import load_split_dataloaders, get_dataloader
from recbole.data import dataloader
from DhcExtension import *
from MD_DHC_Recbole import *


def Data_By_Recbole_General(RecBole_args):
    ## Recbole_general_data includes fields [user_id, item_id, rating, timestamp]
    RecBole_general_data = RecBole_General_Data(RecBole_args.task, RecBole_args.dataset_name, RecBole_args.distribution,RecBole_args.save)
    train_data, valid_data, test_data = RecBole_general_data.generate_train_valid_test()

    return train_data, valid_data, test_data

def Evaluation_Metrics_to_CSV(this_round, RecBole_args, model_name, test_result):
    this_result_dict = {'model_name': model_name,
                        'this_round': [this_round],
                        'recall@10': test_result['recall@10'],
                        'mrr@10': test_result['mrr@10'],
                        'ndcg@10': test_result['ndcg@10'],
                        'hit@10': test_result['hit@10'],
                        'precision@10': test_result['precision@10']}

    this_result_pd = pd.DataFrame(this_result_dict, columns=['model_name', 'this_round', 'recall@10', 'mrr@10', 'ndcg@10', 'hit@10','precision@10'])

    this_result_pd.to_csv(RecBole_args.csv_addr, mode = 'a', header = False, index = False)

def Uniform_Stabilities(Stabilities):
    max_convergence_steps = 0
    for i in range(len(Stabilities)):
        if len(Stabilities[i]) > max_convergence_steps:
            max_convergence_steps = len(Stabilities[i])

    for i in range(len(Stabilities)):
        this_convergence_steps = len(Stabilities[i])
        coreness = Stabilities[i][this_convergence_steps - 1]
        while this_convergence_steps < max_convergence_steps:
            Stabilities[i].append(coreness)
            this_convergence_steps += 1

    stabilities_matrix = np.array(Stabilities)
    return stabilities_matrix

def Dhc(train_data_matrix):
    dhc_extension = DHC_extension()
    dhc_features = []

    neighborhood = []
    for i in range(train_data_matrix.shape[1]):
        posi = np.where(train_data_matrix[i, :] > 0)[0].tolist()
        neighborhood.append(posi)

    this_H = list(np.sum(train_data_matrix, axis=1))
    this_H = np.array(this_H)
    dhc_features.append(this_H)
    this_convergence = this_H

    while True:
        this_H = dhc_extension.UpdataDistribution(neighborhood, this_H)
        if (this_H == this_convergence).all():
            break
        else:
            dhc_features.append(this_H)
            this_convergence = this_H

    return dhc_features

def To_Networkx(graph_DF):
    graph_DF['user_id'] = 'u' + graph_DF['user_id'].astype(str)
    graph_DF['item_id'] = 'i' + graph_DF['item_id'].astype(str)

    graph_nx = nx.Graph()
    edges = [tuple(x) for x in graph_DF[['user_id', 'item_id']].values.tolist()]
    graph_nx.add_nodes_from(graph_DF['user_id'].unique(), bipartite = 0, label = 'user_id')
    graph_nx.add_nodes_from(graph_DF['item_id'].unique(), bipartite = 1, label = 'item_id')
    for row in edges:
        graph_nx.add_edge(row[0], row[1])

    return graph_nx

def Create_Dhc_Features(train_data_DF):
    ## create networkx bipartite graph.
    train_data_nx = To_Networkx(train_data_DF)

    ## generate dhc features for every node.
    train_data_matrix = nx.to_numpy_array(train_data_nx)
    features_dhc_list = Dhc(train_data_matrix)
    features_dhc_matrix = np.array(features_dhc_list)
    dim_dhc = features_dhc_matrix.shape[0]

    ## correspond dhc features to networkx nodes.
    features_dhc_dict = {}
    i = 0
    for node_id in list(train_data_nx.nodes()):
        features_dhc_dict[node_id] = features_dhc_matrix[:,i]
        i += 1
    nx.set_node_attributes(train_data_nx, features_dhc_dict, 'dhc_features')

    return train_data_nx, dim_dhc

def Measure_Similarity_Dhc(train_data_DF,  max_user, max_item,  method):
    ## Create dhc features.
    train_data_nx_with_features_dhc, dim_dhc = Create_Dhc_Features(train_data_DF)
    dhc_features_dict = nx.get_node_attributes(train_data_nx_with_features_dhc, 'dhc_features')

    ## Generate user_dhc_features_matrix and item_dhc_features_matrix.
    user_dhc_features_matrix = np.zeros((max_user + 1, dim_dhc))
    item_dhc_features_matrix = np.zeros((max_item + 1, dim_dhc))
    valid_id = list(dhc_features_dict.keys())
    for i in range(user_dhc_features_matrix.shape[0]):
        user_id = 'u' + str(i)
        if user_id in valid_id:
            user_dhc_features_matrix[i, :] = dhc_features_dict[user_id]
    for j in range(item_dhc_features_matrix.shape[0]):
        item_id = 'i' + str(j)
        if item_id in valid_id:
            item_dhc_features_matrix[j, :] = dhc_features_dict[item_id]

    if method == "dot_product":
        user_item_similarity_dhc = np.dot(user_dhc_features_matrix, item_dhc_features_matrix.transpose())

    if method == "cosine":
        dot_product = np.dot(user_dhc_features_matrix, item_dhc_features_matrix.transpose())
        xi = np.zeros((max_user + 1,1))
        yj = np.zeros((1,max_item + 1))
        for i in range(max_user + 1):
            xi[i,0] = math.sqrt(sum(pow(user_dhc_features_matrix[i,:] ,2)))
        for j in range(max_item + 1):
            yj[0,j] = math.sqrt(sum(pow(item_dhc_features_matrix[j,:] ,2)))
        divider = np.dot(xi,yj)
        user_item_similarity_dhc = np.divide(dot_product,divider + 0.000000000000000000000001)

    if method == "ED":
        dot_product = np.dot(user_dhc_features_matrix, item_dhc_features_matrix.transpose())
        A = np.zeros((max_user + 1, max_item + 1))
        A_column = np.linalg.norm(user_dhc_features_matrix, ord = 2, axis = 1, keepdims = False)
        # axis = 1 by rows
        for j in range(max_item + 1):
            A[:,j] = A_column
        B = np.zeros((max_user + 1, max_item + 1))
        B_row = np.linalg.norm(item_dhc_features_matrix, ord = 2, axis = 1, keepdims = False)
        for i in range(max_user + 1):
            B[i,:] = B_row
        user_item_similarity_dhc = np.sqrt(abs(-2 * dot_product + A + B))

    if method == "Cov":
        A_avg = np.zeros((max_user + 1, dim_dhc))
        A_avg_column = user_dhc_features_matrix.mean(axis = 1)
        for j in range(dim_dhc):
            A_avg[:,j] = A_avg_column
        B_avg = np.zeros((max_item + 1, dim_dhc))
        B_avg_column = item_dhc_features_matrix.mean(axis = 1)
        for j in range(dim_dhc):
            B_avg[:,j] = B_avg_column
        user_item_similarity_dhc = np.dot(user_dhc_features_matrix - A_avg, (item_dhc_features_matrix - B_avg).transpose())/dim_dhc

    if method == "Pearson":
        A_avg = np.zeros((max_user + 1, dim_dhc))
        A_avg_column = user_dhc_features_matrix.mean(axis=1)
        for j in range(dim_dhc):
            A_avg[:, j] = A_avg_column
        B_avg = np.zeros((max_item + 1, dim_dhc))
        B_avg_column = item_dhc_features_matrix.mean(axis=1)
        for j in range(dim_dhc):
            B_avg[:, j] = B_avg_column
        xi = np.zeros((max_user + 1, 1))
        yj = np.zeros((1, max_item + 1))
        xi[:,0] = np.linalg.norm(user_dhc_features_matrix - A_avg, ord = 2, axis = 1, keepdims = False)
        yj[0,:] = np.linalg.norm(item_dhc_features_matrix - B_avg, ord = 2, axis = 1, keepdims = False)
        user_item_similarity_dhc = np.divide(np.dot(user_dhc_features_matrix - A_avg, (item_dhc_features_matrix - B_avg).transpose()), np.dot(xi,yj) + 0.000000000000000000000001)

    return user_item_similarity_dhc

### driver
if __name__=='__main__':

    ## Configurations.
    dataset = "ml-100k/ml-100k "
    this_data_name = 'ml-100k'
    RecBole_args = parameters_management.RecBole_parameter_parser()
    config = Config('Pop', RecBole_args.dataset_name)
    config['epochs'] = 1
    # Since the AIProbS is a non-machine learning-based model, 'epochs' is set to 1.

    ## Implement the 30 realizations.
    for this_round in range(30):
        ## The AIProbS model with different similarity measurements and operations of normalization and appropriation.
        for similarity_measurement in ['cosine']:
            for normalize_method in ['max-min + mean']:
            # methods include ['dot_product', 'cosine', 'ED', 'Cov', 'Pearson'].
            # normalize_methods include ['None', 'mean', 'max-min', 'max-min + mean'].
                addr = dataset + str(this_round) + ".pth"
                train_data, valid_data, test_data = load_split_dataloaders(addr)
                train_data_DF = pd.DataFrame(train_data.dataset.inter_feat.interaction)
                valid_data_DF = pd.DataFrame(valid_data.dataset.inter_feat.interaction)
                test_data_DF = pd.DataFrame(test_data.dataset.inter_feat.interaction)
                max_user = max(max(train_data_DF['user_id']), max(valid_data_DF['user_id']), max(test_data_DF['user_id']))
                max_item = max(max(train_data_DF['item_id']), max(valid_data_DF['item_id']), max(test_data_DF['item_id']))

                user_item_similarity_matrix = Measure_Similarity_Dhc(train_data_DF, max_user, max_item, similarity_measurement)
                # Generate the user-item similarity matrix from the DHC features.

                init_logger(config)
                logger = getLogger()
                model = MD_DHC(config, train_data.dataset, user_item_similarity_matrix, normalize_method)
                trainer = Trainer(config, model)
                logger.info(model)
                best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)
                test_result = trainer.evaluate(test_data)
                addr =  'MD_DHC(' + str(similarity_measurement) + ' + '+ str(normalize_method) + ')'
                Evaluation_Metrics_to_CSV(this_round, RecBole_args, addr, test_result)