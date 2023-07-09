import random

from sklearn.decomposition import NMF
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
import warnings
warnings.filterwarnings("ignore")


def display_topics(model, feature_names, num_top_words, topic_names=None):
    '''
    Given an NMF model, feature_names, and number of top words, print topic number
    and its top feature names, up to specified number of top words.
    '''
    for ix, topic in enumerate(model.components_):
        #print topic, topic number, and top words
        if not topic_names or not topic_names[ix]:
            print("\nTopic ", ix)
        else:
            print("\nTopic: '",topic_names[ix],"'")
        print(", ".join([feature_names[i] \
             for i in topic.argsort()[:-num_top_words - 1:-1]]))

data = pd.read_csv("median_frequencies_by_id.csv",index_col=None)
ids = data['id']
data.drop([data.columns[0], data.columns[1], 'id', 'Num_Clones'], axis=1, inplace=True)
codons = data.columns.to_list()


res = pd.DataFrame()
for k in tqdm.tqdm(range(2,20), total=20):
    for loss in ['frobenius']:
        for s in ['cd', 'mu']:
            for init_state in ['nndsvd', 'random', 'nndsvda']:
                if (s == 'cd' and loss == 'kullback-leibler') or (init_state == 'nndsvd' and s == 'mu'):
                    print("cd + KL or nndsvd + mu")
                else:
                    for i in range(50):
                        model = NMF(n_components=k,
                                    init=init_state,
                                    random_state=random.randint(1, 9000),
                                    tol=1e-4,
                                    max_iter=1000,
                                    solver=s,
                                    beta_loss=loss #kullback-leibler,frobenius
                                    )
                        model.fit_transform(data)
                        tmp = {'Error': model.reconstruction_err_, 'K':k, 'loss': loss, 'init_state': init_state}
                        res = pd.concat([res, pd.DataFrame([tmp])])


plt.figure(figsize=[15,10])
sns.boxplot(x=res['K'], y=res['Error'], hue=res['init_state'])
plt.show()
W = model.fit_transform(data)
H = model.components_
display_topics(model, codons, 10)