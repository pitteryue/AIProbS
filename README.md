# Determinable and interpretable network representation for link prediction [Python code]
As an intuitive description of complex physical, social, or brain systems, complex networks have fascinated scientists for decades. Recently, to abstract a network’s topological and dynamical attributes, network representation has been a prevalent technique, which can map a network or substructures (like nodes) into a low-dimensional vector space. Since its mainstream methods are mostly based on machine learning, a black box of an input-output data fitting mechanism, the learned vector’s dimension is indeterminable and the elements are not interpreted. Although massive efforts to cope with this issue have included, say, automated machine learning by computer scientists and learning theory by mathematicians, the root causes still remain unresolved. Consequently, enterprises need to spend enormous computing resources to work out a set of model hyperparameters that can bring good performance, and business personnel still finds difficulties in explaining the learned vector’s practical meaning. Given that, from a physical perspective, this article proposes two determinable and interpretable node representation methods. To evaluate their effectiveness and generalization, this article proposes Adaptive and Interpretable ProbS (AIProbS), a network-based model that can utilize node representations for link prediction. Experimental results showed that the AIProbS can reach state-of-the-art precision beyond baseline models on some small data whose distribution of training and test sets is usually not unified enough for machine learning methods to perform well. Besides, it can make a good trade-off with machine learning methods on precision, determinacy (or robustness), and interpretability. In practice, this work contributes to industrial companies without enough computing resources but who pursue good results based on small data during their early stage of development and who require high interpretability to better understand and carry out their business.

Article link: https://doi.org/10.1038/s41598-022-21607-4

**How to use the Python codes**

As for data sets, the three files that the **30 MovieLens-1M Realizations** at the root directory and the **30 MovieLens-100K Realizations** and **30 LastFM Realizations** both at the directory <i>''/Data''</i> involve all the items used in this article's experiments.

In order to run the source codes based on either of them, one is expected to follow the three steps completing some modifications based on the source codes:

(1) In the first place, <code><i>default = 'ml-100k'</i></code> of <code><i>parser.add_argument()</i></code> in the file **parameters_management.py** should be specified by the name of a currently used data set, including 'ml-100k', 'ml-1m' and 'lastfm'.

(2) Next, <code><i>this_data_name = 'ml-100k'</i></code> of <code><i>if \__name__=='\__main\__':</i></code> in the file **main.py** should be specified by the name of a currently used data set, as above.

(3) Finally, the input file address can be modified in <code><i>dataset = "ml-100k/ml-100k "</i></code> of <code><i>if \__name\__=='\__main\__':</i></code> in the file **main.py**, and the output file address can be modified in <code><i> default = "results_ml-100k.csv"</i></code> of <code><i> parser.add_argument()</i></code> in the file **parameters_management.py**.

One can access the codes of DHC-E algorithm used in this project from https://github.com/HW-HaoWang/DHC-E.
