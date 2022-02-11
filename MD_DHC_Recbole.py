import numpy as np
import scipy.sparse as sp
import torch
import copy
import pandas as pd
from sklearn.preprocessing import normalize
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import InputType, ModelType


class ComputeSimilarity:
    def __init__(self, train_rating_matrix, user_item_similarity_matrix, normalize_method):
        self.train_rating_matrix = train_rating_matrix.todense()
        self.user_item_similarity_matrix = user_item_similarity_matrix
        self.n_rows, self.n_columns = self.train_rating_matrix.shape
        self.train_rating_matrix = pd.DataFrame(self.train_rating_matrix)
        #self.train_rating_matrix = self.train_rating_matrix.loc[~(self.train_rating_matrix == 0).all(axis=1)]
        #self.train_rating_matrix = self.train_rating_matrix.loc[:, (self.train_rating_matrix != 0).any(axis=0)]
        #self.train_rating_matrix['0'] = [0 for _ in range(self.train_rating_matrix.shape[0])]
        #self.train_rating_matrix.loc[0] = [0 for _ in range(self.train_rating_matrix.shape[1])]
        self.train_rating_matrix_T = self.train_rating_matrix.transpose()
        self.normalize_method = normalize_method

    def compute_similarity(self):
        normalized_similarity_matrix_item = self.user_item_similarity_matrix
        normalized_similarity_matrix_item = np.multiply(normalized_similarity_matrix_item, np.array(self.train_rating_matrix))
        normalized_similarity_matrix_user = self.user_item_similarity_matrix
        normalized_similarity_matrix_user = np.multiply(normalized_similarity_matrix_user,np.array(self.train_rating_matrix))

        if self.normalize_method == 'mean':
            for j in range(normalized_similarity_matrix_item.shape[1]):
                if sum(normalized_similarity_matrix_item[:,j]) != 0:
                    normalized_similarity_matrix_item[:,j] = normalized_similarity_matrix_item[:,j]/sum(normalized_similarity_matrix_item[:,j])
            for i in range(normalized_similarity_matrix_user.shape[0]):
                if sum(normalized_similarity_matrix_user[i,:]) != 0:
                    normalized_similarity_matrix_user[i,:] = normalized_similarity_matrix_user[i,:]/sum(normalized_similarity_matrix_user[i,:])

        if self.normalize_method == 'max-min' or 'max-min + mean':
            for j in range(normalized_similarity_matrix_item.shape[1]):
                max = normalized_similarity_matrix_item[:,j].max()
                min = normalized_similarity_matrix_item[:,j].min()
                if max - min != 0:
                    normalized_similarity_matrix_item[:, j] = (normalized_similarity_matrix_item[:,j] - min)/(max - min)
            for i in range(normalized_similarity_matrix_user.shape[0]):
                max = normalized_similarity_matrix_user[i,:].max()
                min = normalized_similarity_matrix_user[i,:].min()
                if max - min != 0:
                    normalized_similarity_matrix_user[i,:] = (normalized_similarity_matrix_user[i,:] - min)/(max - min)

            if self.normalize_method == 'max-min + mean':
                for j in range(normalized_similarity_matrix_item.shape[1]):
                    if sum(normalized_similarity_matrix_item[:,j]) != 0:
                        normalized_similarity_matrix_item[:,j] = normalized_similarity_matrix_item[:,j]/sum(normalized_similarity_matrix_item[:,j])
                for i in range(normalized_similarity_matrix_user.shape[0]):
                    if sum(normalized_similarity_matrix_user[i,:]) != 0:
                        normalized_similarity_matrix_user[i,:] = normalized_similarity_matrix_user[i,:]/sum(normalized_similarity_matrix_user[i,:])

        estimated_rating_matrix_np = np.dot(np.dot(np.array(self.train_rating_matrix), normalized_similarity_matrix_item.transpose()),normalized_similarity_matrix_user)
        estimated_rating_matrix = pd.DataFrame(estimated_rating_matrix_np, index=self.train_rating_matrix.index.values,columns=self.train_rating_matrix.columns.values)

        estimated_rating_matrix_sp = sp.csc_matrix(estimated_rating_matrix.values)

        return estimated_rating_matrix_sp

class MD_DHC(GeneralRecommender):

    input_type = InputType.POINTWISE

    def __init__(self, config, dataset, user_item_similarity_matrix, normalize_method):
        super(MD_DHC, self).__init__(config, dataset)

        self.interaction_matrix = dataset.inter_matrix(form = 'csr').astype(np.float32)
        self.user_item_similarity_matrix = user_item_similarity_matrix
        shape = self.interaction_matrix.shape
        assert self.n_users == shape[0] and self.n_items == shape[1]
        self.pred_mat = ComputeSimilarity(self.interaction_matrix, self.user_item_similarity_matrix, normalize_method).compute_similarity()
        self.fake_loss = torch.nn.Parameter(torch.zeros(1))
        self.other_parameter_name = ['pred_mat']

    def forward(self, user, item):
        pass

    def calculate_loss(self, interaction):
        return torch.nn.Parameter(torch.zeros(1))

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user = user.cpu().numpy().astype(int)
        item = item.cpu().numpy().astype(int)
        result = []

        for index in range(len(user)):
            uid = user[index]
            iid = item[index]
            score = self.pred_mat[uid, iid]
            result.append(score)
        result = torch.from_numpy(np.array(result)).to(self.device)

        return result

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user = user.cpu().numpy()

        score = self.pred_mat[user, :].toarray().flatten()
        result = torch.from_numpy(score).to(self.device)

        return result