import argparse

def RecBole_parameter_parser():

    parser = argparse.ArgumentParser(description="Parameter settings for RecBole")

    parser.add_argument("--task", type = str, default = 'BPR',
                        help = "four tasks, using the values of keys to indicate: "
                               " (1) General [.INTER]: "
                               "    (a) Traditional: Pop, ItemKNN "
                               "    (b) GNN: DGCF, GCMC, LightGCN, NGCF"
                               "    (c) Classical-DNN: DMF, NeuMF"
                               "    (d) AutoEncoder: CDAE, MacridVAE, MultiDAE, MultiVAE"
                               "    (e) Classical-Others: SpectralCF, BPR, LINE (FISM)"
                               "    (f) + Attention: (NAIS)"
                               "    (g) + Convolution: ConvNCF"
                               " (2) Context-aware [.INTER .USER .ITEM]: AFM, AutoInt, DCN, DeepFM, DSSM, FFM, FM, FNN, FwFM, LR, NFM, PNN, WideDeep, xDeepFM"
                               " (3) Knowledge-based [.INTER, .KG, .LINK]: CFKG, CKE, KGAT, KGCN, KGNNLS, KTUP, MKR, RippleNet"
                               " (4) Sequential [.INTER]: BERT4Rec, Caser, DIN, FDSA, FOSSIL, FPMC, GCSAN, GRU4Rec, GRU4RecF, GRU4RecKG, HGN, HRM, KSR, NARM, NextItNet, NPE, RepeatNet, S3Rec, SASRec, SASRecF, SHAN, SRGNN, STAMP, TransRec")
    """
    General: (a) load_col: inter: [user_id, item_id]
              (b) eval_args: group_by: user
              (c) neg_sampling: uniform: 1
    Context-aware: (a) load_col: inter: [inter_feature1, inter_feature2], item: [item_feature1, item_feature2], user: [user_feature1, user_feature2]
                    (b) eval_args: group_by: None, mode: labeled;
                    (c) metrics: ['AUC', 'LogLoss']
                    (d) valid_metric: AUC
    Knowledge-based: (a) load_col: inter: [user_id, item_id], kg: [head_id, relation_id, tail_id], link: [item_id, entity_id]
    Sequential:  (a) load_col: inter: [user_id, item_id, timestamp]
                  (b) MAX_ITEM_LIST_LENGTH: 50
    """

    parser.add_argument("--dataset_name", type = str, default = 'ml-100k',
                        help = "see RecBole website for lists and 'recbole/properties/dataset/url.yaml' or '.../kg_url.yaml' for correct entries, say,"
                               "(1) ratings: "
                               "   (a) ml-100k ([1,5], timestamp): .INTER, .ITEM, .USER, .KG, .LINK"
                               "   (b) ml-1m ([1,5], timestamp): .INTER, .ITEM, .USER"
                               "   (c) jester ([-10,10]): .INTER"
                               "   (d) yelp ([1,5]): .INTER, .ITEM, .USER"
                               "   (e) epinions ([1,5]): .INTER"
                               "   (f) book-crossing ([0,10]): .INTER, .ITEM, .USER"
                               "   (g) douban ([0,5]): .INTER"
                               "(2) clicks: "
                               "   (a) lastfm (weight:float (i.e., click counts), tag_value:token_seq): .INTER, .ITEM (but actually it's about artists)"
                               "   (b) pinterest: .INTER" )
    """
    (a) 'yelp' and 'lasfm' miss the item_id FILED
    (b) on 'Jester' and 'Douban', cannot sample negative items for users interacted with all items, set `user_inter_num_interval` to filter them
    (c) to insure ratings and timestamps in '.INTER' to not be missed when loading, add following lines into config_dict:
               'load_col': {'inter': ['user_id', 'item_id', 'rating', 'timestamp']},
               'USER_ID_FIELD': 'user_id',
               'ITEM_ID_FIELD': 'item_id',
               'RATING_FIELD': 'rating',
               'TIME_FIELD': 'timestamp'
        while this method could block the load of other atomic files like '.KG' and stuff
    """

    parser.add_argument("--csv_addr", type= str, default = "results_ml-100k.csv", help = "save multiple models' evaluation results on the same dataset to the common .csv")

    parser.add_argument("--save", type = bool, default = True, help = "save split dataset")

    parser.add_argument("--distribution", type = str, default = 'uniform', help = "distribution methods include uniform and popularity")

    return parser.parse_args()
    # necessary!

def my_parameter_parser():

    parser = argparse.ArgumentParser(description = "Run my model")

    return parser.parse_args()