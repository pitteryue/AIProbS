# Interdisciplinary Studies Are All You Need [Python code]
In making great strides to the machine learning-based era in artificial intelligence, are we really making much progress? It appears that researchers generally face the dichotomies on making a trade-off between the precision and scalability of machine learning-based frameworks and the explainability and hyperparameter-free of conventional ones. Faced with this situation, interdisciplinary studies seem to be a promising future. In terms of the field of recommender systems, a significant application of artificial intelligence, this article reveals the flaw of ProbS (a prevalent conventional recommendation framework) in self-adaptive perception towards different recommendation scenarios. To make up for the flaw, this article proposes two nodal feature generation methods for representing users and items on the advice of the essential thought from machine learning-based recommendation frameworks, which are hyperparameter-free, explainable and scenario-oriented. Based on the ProbS framework and the generated features, this article proposes the **A**daptive and **I**nterpretable **P**robS (**AIProbS**) model, a recommendation model with self-adaptive perception, quantification and control abilities and with precise, hyperparameter-free and explainable instincts. To evaluate its performance on diverse real recommendation scenarios, this article designs control experiments, revealing that the AIProbS can achieve state-of-the-art performance on model precision, compared with baseline models of both conventional and machine learning-based frameworks.

**How to use the Python codes**

As for data sets, the three files that the **30 MovieLens-1M Realizations** in the root directory and the **30 MovieLens-1M Realizations** and **30 MovieLens-1M Realizations** both in the directory <i>''/Data''</i> involve all the items used in this article's experiments.

In order to run the codes based on either of them, one is expected to follow the three steps complete some modifications:

(1) In the first place, <code><i>default = 'ml-100k'</i></code> of <code><i>parser.add_argument()</i></code> in the file **parameters_management.py** should be specified by the name of currently used data set, including 'ml-100k', 'ml-1m' and 'lastfm'.

(2) Next, <code><i>this_data_name = 'ml-100k'</i></code> of <code><i>if \__name__=='\__main\__':</i></code> in the file **main.py** should be specified by the name of the currently used data set, as above.

(3) Finally, the input file address can be modified in <code><i>dataset = "ml-100k/ml-100k "</i></code> of <code><i>if \__name\__=='\__main\__':</i></code> in the file **main.py**, and the output file address can be modified in <code><i> default = "results_ml-100k.csv"</i></code> of <code><i> parser.add_argument()</i></code> in the file **parameters_management.py**.
