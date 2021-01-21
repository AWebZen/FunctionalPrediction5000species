#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 14:19:32 2020

@author: adele
"""
import cPickle as cpk
from os.path import join as joinP
from collections import Counter
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

from utils_general import train_test_proportions, get_pathways_from_nodes, \
    is_enz



def from_CM_get_y(m, labels):
    """From confusion matrix, deduce fake y_true y_pred"""
    y_true = []
    y_pred = []
    for tr, line in enumerate(m):
        y_true += [labels[tr]] * np.sum(line)
        for col, val in enumerate(line):
            y_pred += [labels[col]] * val
    return y_true, y_pred


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          cmap=plt.cm.YlGnBu):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    np.set_printoptions(precision=2)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           ylabel='True',
           # fontsize=16,
           xlabel='Predicted')

    # Loop over data dimensions and create text annotations.
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > thresh else "black")

    plt.xlim(-0.5, len(np.unique(classes))-0.5)
    plt.ylim(len(np.unique(classes))-0.5, -0.5)
    plt.xlabel("Predicted", fontsize=16)
    plt.ylabel('True', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    return ax


def mask_kmer(i_list, kmer_action, *args):
    """
    I invert values in k-mer to see effect on prediciton. 0 become 1 and 1 become 0.
    Otherwise, put to 0 or to 1 all values.

    Parameters
    ----------
    i_list : list
        list of column indexes to remove.
    kmer_action : str
        Among "invert", "zeros", "ones". Will invert values or put them to 0 or to 1.
    *args : 
        List of X with columns to mask.

    Returns
    -------
    list of modified arrays

    """
    assert kmer_action in ["invert", "zeros", "ones"], "Invalid k-mer action."
    masked_arrays = []
    for X in args:
        assert all([val in [0, 1] for val in np.unique(X)]), "Non 0/1 binary array"
        if kmer_action == "invert":
            X[:, i_list] = np.abs(X[:, i_list] - 1)
        elif kmer_action == "zeros":
            X[:, i_list] = 0
        elif kmer_action == "ones":
            X[:, i_list] = 1
        masked_arrays.append(X)
    return masked_arrays

def save_csv_pred(y, df, col_list, fname):
     df_pred = df.loc[:, col_list]
     df_pred["randomF_prediction"] = y
     df_pred.to_csv(fname, index=False)

class RandomForestClassifierClass:
    def __init__(self, matrix, classes, weights, ntrees, kwargs):
        self.matrix = matrix
        self.classes = classes
        self.class_weights = weights
        self.ntrees = ntrees
        self.kwargs = kwargs
        
    def train_test_validation(self):
        self.data, self.data_v, self.y, self.y_v, _, _ = train_test_proportions(.9, self.matrix.T, self.classes)
        self.X_train, self.X_test, self.y_tr, self.y_t, _, _ = train_test_proportions(.66, self.data, self.y)
        
    def rf_model_fit(self):
        self.clf = RandomForestClassifier(class_weight=self.class_weights,
                             n_estimators=self.ntrees, **self.kwargs)
        self.clf.fit(self.X_train, self.y_tr)
        
    def predict(self, data, y=False, text=""):
        y_pred = self.clf.predict(data)
        if np.any(y) != False:
            print text, sum(y_pred == y)/float(len(y))
        return y_pred
    
    def predict_test(self, y=True):
        if np.any(y) != False:
            y = self.y_t
        self.y_pred = self.predict(self.X_test, y, text="Accuracy testing dataset:")
        return self.y_pred
        
    def predict_valid(self, y=True):
        if np.any(y) != False:
            y = self.y_v
        self.y_pred_v = self.predict(self.data_v, y, text="Accuracy validation dataset:")
        return self.y_pred_v

    def mask_analysis(self, step=50, brk=False, acc_print=True, plot=True):
        """
        K-mer analysis, inverting matrix values in *step*-size sliding window, and plotting (and printing) the resulting
        accuracy

        step - sliding window size
        brk - Defaults to False. Stop sliding when accuracy inferior to brk.

        return accuracies of each mask
        """
        if plot:
            plt.figure(figsize=(10,7))
            plt.axhline(sum(self.y_pred_v == self.y_v)/float(len(self.y_v)), c="r")

        accs = []
        for i in range(0, self.matrix.T.shape[1]):
            i_list = [i+n for n in range(step) if i+n < self.matrix.T.shape[1]]
            X_v_k = mask_kmer(i_list, "invert", deepcopy(self.data_v))[0]
            y_pred2 = self.clf.predict(X_v_k)
            if plot:
                plt.scatter(i, sum(y_pred2 == self.y_v)/float(len(self.y_v)), c="k")
            if acc_print:
                print i, step, sum(y_pred2 == self.y_v)/float(len(self.y_v))
            accs.append(sum(y_pred2 == self.y_v)/float(len(self.y_v)))
            if brk and sum(y_pred2 == self.y_v)/float(len(self.y_v)) < brk:
                print Counter([tuple(coupl)  # Wrong class - correct one numbers
                         for coupl in np.vstack((y_pred2[y_pred2 != self.y_v],
                                                 self.y_t[y_pred2 != self.y_v])).T
                         ])
                break
        if plot:
            plt.ylabel("Accuracy")
            plt.xlabel("Position of first compound in mask")
            plt.title("Mask size: {}".format(step))
            plt.plot()
        return accs



if __name__ == '__main__':
    backup_dir = "backup_cpkl/"
    med_name = "PsychroMesoThermHyperMedium"
    #The following files are generated by scope_kegg_prk.py
    simplified_matrix = cpk.load(open(joinP(backup_dir, "simplified_matrix_scope{}_prk.cpk".format(med_name)), "rb"))
    nodes_simplified = cpk.load(open(joinP(backup_dir, "simplified_nodes_scope{}_prk.cpk".format(med_name)), "rb"))


    df_species = pd.read_csv("species_metadata.csv", index_col=0)
    df_species.loc[df_species.sp_codes.isnull(), "sp_codes"] = "nan"  # Otherwise interpreted as NaN


    # =============================================================================
    #
    #                   TEMPERATURE CLASS PREDICTION
    #
    # =============================================================================


    matrix_temp = simplified_matrix[:, df_species.temp_range_deduced.notnull()]
    classes = df_species.temp_range_deduced[df_species.temp_range_deduced.notnull()].values

    accuracy_v = []
    gini = []
    mat = []
    mask_min50 = []
    f1_scores_temp = []
    depths = []
    cv_n = 300
    for cv in xrange(cv_n): #cross validation
        print cv
        # Random 300 mesophiles so as to balance classes: index of matrix without temp null values
        no_meso = sorted(list(np.where(classes != "mesophilic")[0]) +
                         list(np.random.choice(np.where(classes == "mesophilic")[0],
                                               300, replace=False)
                              ))
        matrix_T300 = matrix_temp[:, no_meso]
        classesT300 = classes[no_meso]
        tclass300RF = RandomForestClassifierClass(matrix=matrix_T300,
                                                  classes=classesT300,
                                                  weights={"mesophilic":300./782,
                                                   "thermophilic":188./782,
                                                   "hyperthermophilic":76./782,
                                                   "psychrophilic":218./782},
                                                  ntrees=1000,
                                                  kwargs={})

        # Split dataset
        tclass300RF.train_test_validation()

        # Fit on train dataset
        tclass300RF.rf_model_fit()
        # Test dataset prediciton
        tclass300RF.predict_test()
        # True validation - independent testing set
        tclass300RF.predict_valid()
        accuracy_v.append(sum(tclass300RF.y_pred_v == tclass300RF.y_v) / float(len(tclass300RF.y_v)))
        f1_scores_temp.append(
            f1_score(tclass300RF.y_v, tclass300RF.y_pred_v,
                     labels=["hyperthermophilic", "thermophilic", "mesophilic", "psychrophilic"],
                     average="micro"))

        #Real as lines, preds as cols
        m = confusion_matrix(tclass300RF.y_v, tclass300RF.y_pred_v,
                               labels=["hyperthermophilic", "thermophilic", "mesophilic", "psychrophilic"],)
        mat.append(m)


        #Get depth
        max_dpth = max([dectree.tree_.max_depth for dectree in tclass300RF.clf.estimators_])
        depths.append(max_dpth)


        # Feature importances
        gini_50_nodes = nodes_simplified[np.argsort(tclass300RF.clf.feature_importances_)[::-1]][:50]
        gini.append(gini_50_nodes)

    print "Accuracy:", np.mean(accuracy_v), np.std(accuracy_v)
    print "F1-score:", np.mean(f1_scores_temp), np.std(f1_scores_temp)

    # Plot cross-validation boxplot of accuracies per class
    m_diag = [np.diag(m) for m in mat]
    plt.figure(figsize=(10, 7))
    sns.boxplot(data=(np.array(m_diag) / np.sum(mat[0], axis=1).astype(float)),
                palette=sns.color_palette(["#d82f00", "goldenrod", "lightgreen", "#41a5b7", ]))
    plt.xticks(range(4), ["HT", "T", "M", "P"], fontsize=16)
    plt.ylabel("Accuracy", fontsize=18)
    plt.xlabel("Temperature class", fontsize=18)
    plt.yticks(fontsize=16)
    plt.ylim(0, 1.05)

    # Mean matrix o
    m_mean = deepcopy(mat[0])
    for m in mat[1:]:
        m_mean += m
    print m_mean / float(cv_n)
    norm = (m_mean / float(cv_n)).astype('float') / (m_mean / float(cv_n)).sum(axis=1)[:, np.newaxis]

    plot_confusion_matrix(norm, ["HT", "T", "M", "P"],
                          normalize=False,
                          cmap=plt.cm.YlGnBu)

    # Most important nodes for model prediction, and pathways to which they belong
    c_gini = Counter(np.concatenate(gini)) #Counter of nodes all
    c, pathways = get_pathways_from_nodes(c_gini.keys()) #pathways= equivalence node-pathways, c= counter of pathways in values of pathways dict (not weigthed by apparition of nodes)
    pathways_g = [] #all nodes transformed by its equivalence
    for n in np.concatenate(gini):
        n = n.split("_")[0]
        pathways_g += pathways[n]
    c_all = Counter(np.array(pathways_g)[:, 1]) #Counter of pathways all (weighted by the repetitions of nodes)
    c_mean = Counter({key: v / float(cv_n) for key, v in c_all.items()})
    pathways_lists = []
    c_std = {ky:0 for ky in c_mean.keys()}
    for crssv in gini:
        pathways_lists.append([])
        for nod in crssv:
            nod = nod.split("_")[0]
            pathways_lists[-1] += [p[1] for p in pathways[nod]]
    for pthway in c_mean.keys():
        for crssv in pathways_lists:
            c_cv = Counter(crssv)
            c_std[pthway] += (c_cv[pthway] - c_mean[pthway])**2
    for pthway in c_mean.keys():
        c_std[pthway] = np.sqrt(c_std[pthway]/float(cv_n))

    #Histogram index nodes per number of models
    index_nodes = {ky:list(nodes_simplified).index(ky) for ky in c_gini.keys()}
    c_gini_idx = {index_nodes[ky]:v for ky, v in c_gini.items()}
    plt.figure(figsize=(10, 7))
    plt.bar(range(len(c_gini_idx)), np.array(sorted(c_gini_idx.values(), reverse=True)) * 100./cv_n) #plots the number of model counts per union of 50 most important nodes for all models.
    plt.xlabel("Nodes", fontsize=16)
    plt.ylabel("Percentage of models with node in 50 most important", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlim(0, 177)
    plt.tight_layout()

    #Mean of tree max depths
    print np.mean(depths)

    #Species x 17 nodes amongst union of "50 most important nodes" present in 90% of models matrix
    nods17 = ['1.1.1.38_0','1.1.1.38_1', '1.1.1.40_0', '1.1.1.40_1', '1.4.1.4', '1.8.1.7', '3.5.1.1', '4.3.1.1',
                 '4.3.3.6_1', '6.3.2.2', '6.3.2.3', '7.3.2.6', 'C00127', 'C00208', 'C00385', 'C00669', 'C20679']
    mat_17 = np.genfromtxt("matrix_speciesX17impNodes.csv")
    #add column for 1.1.1.38_0, same as 1.1.1.38_1
    mat_17 = np.hstack((mat_17[:,0].reshape(5610, 1), mat_17))
    HT_i = np.where(df_species.temp_range_deduced == "hyperthermophilic")[0]
    T_i = np.where(df_species.temp_range_deduced == "thermophilic")[0]
    M_i = np.where(df_species.temp_range_deduced == "mesophilic")[0]
    P_i = np.where(df_species.temp_range_deduced == "psychrophilic")[0]
    sort_for_fig = np.argsort(np.sum(mat_17[HT_i, :], axis=0)/float(len(HT_i)) +
                              np.sum(mat_17[T_i, :], axis=0)/float(len(T_i)) -
                              np.sum(mat_17[M_i, :], axis=0)/float(len(M_i)) -
                              np.sum(mat_17[P_i, :], axis=0)/float(len(P_i))
                              )[::-1]
    x = np.arange(17)  # the label locations
    width = 0.8  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/4. - width/8., (np.sum(mat_17[HT_i, :], axis=0)/float(len(HT_i)))[sort_for_fig], width/4., label="HT", color="#d82f00")
    rects2 = ax.bar(x - width/8., (np.sum(mat_17[T_i, :], axis=0)/float(len(T_i)))[sort_for_fig], width/4., label='T', color="goldenrod")
    rects2 = ax.bar(x + width/8., (np.sum(mat_17[M_i, :], axis=0)/float(len(M_i)))[sort_for_fig], width/4., label='M', color="lightgreen")
    rects2 = ax.bar(x + width/4. + width/8., (np.sum(mat_17[P_i, :], axis=0)/float(len(P_i)))[sort_for_fig], width/4., label='P', color="#41a5b7")
    ax.set_xticks(x)
    print list(np.array(nods17)[sort_for_fig]) #Tick labels manually modified because _0 was after _1
    ax.set_xticklabels(['4.3.3.6_1', '7.3.2.6', 'C20679', '1.1.1.38_0', '1.1.1.38_1',
                        '4.3.1.1', '3.5.1.1', '1.4.1.4', 'C00208', '6.3.2.3', '1.8.1.7',
                        'C00385', '1.1.1.40_0', '1.1.1.40_1', '6.3.2.2', 'C00669', 'C00127'], fontsize=12)
    ax.set_yticklabels(["0", "0.2", "0.4", "0.6", "0.8"], fontsize=12)
    ax.legend(fontsize=14)
    ax.set_ylabel("Proportion of species", fontsize=16)
    ax.set_xlabel("17 common important nodes", fontsize=16)
    fig.tight_layout()



    # =============================================================================
    # DIFFERENTIAL GENOMICS RF TCLASS
    # =============================================================================
    all_nodes = cpk.load(open("backup_cpkl/all_nodes_union_prk.cpk", "rb"))
    enzs_only_db = np.array(sorted([n for n in all_nodes if is_enz(n)]))
    indx_to_remove = [i for i, enz in enumerate(enzs_only_db) if not enz.endswith("_0") and not len(enz.split("_")) == 1]
    gene_mat = np.load("enzs_per_sp.npy") #sp x enzymes
    gene_mat_temp_doubles = gene_mat[df_species.temp_range_deduced.notnull(), :].T
    # DIFFERENTIAL GENOMICS FOR GENES WITH ENZYMES (no _1, _2, ...)
    # Our graphs consider reactions (appended by _0, _1 if multiple reactions per enzyme).
    # Here, enzymes are considered, so when duplicates (_1, _2) are removed
    gene_mat_temp_nodoubles = np.delete(gene_mat_temp_doubles, indx_to_remove, 0)
    enzs_only_nodouble = enzs_only_db[[i for i in range(len(enzs_only_db)) if i not in indx_to_remove]]

    gene_mat_temp_ = gene_mat_temp_nodoubles
    enzs_only = enzs_only_nodouble


    accuracy_v_DG = []
    gini_DG = []
    mat_DG = []
    f1_scores_DG = []
    cv_n = 300
    depths_DG = []
    if300_DG = True
    for cv in xrange(cv_n): #cross validation
        print cv
        if if300_DG:
            no_meso = sorted(list(np.where(classes != "mesophilic")[0]) +
                             list(np.random.choice(np.where(classes == "mesophilic")[0],
                                                   300, replace=False)
                                  ))
            gene_mat_temp = gene_mat_temp_[:, no_meso]
            classes_DG = classes[no_meso]
            w = {"mesophilic":300./782,
               "thermophilic":188./782,
               "hyperthermophilic":76./782,
               "psychrophilic":218./782}
        else:
            classes_DG = classes
            w = {"mesophilic":2910./3392,
                   "thermophilic":188./3392,
                   "hyperthermophilic":76./3392,
                   "psychrophilic":218./3392}
            gene_mat_temp = gene_mat_temp_
        tempclassRF_DG = RandomForestClassifierClass(matrix=gene_mat_temp,
                                                  classes=classes_DG,
                                                  weights=w,
                                                  ntrees=1000,
                                                  kwargs={})

        # Split dataset
        tempclassRF_DG.train_test_validation()

        # Fit on train dataset
        tempclassRF_DG.rf_model_fit()
        # Test dataset prediciton
        tempclassRF_DG.predict_test()
        # True validation - independent testing set
        tempclassRF_DG.predict_valid()
        accuracy_v_DG.append(sum(tempclassRF_DG.y_pred_v == tempclassRF_DG.y_v)/float(len(tempclassRF_DG.y_v)))
        f1_scores_DG.append(
            f1_score(tempclassRF_DG.y_v, tempclassRF_DG.y_pred_v,
                     labels=["hyperthermophilic", "thermophilic", "mesophilic", "psychrophilic"],
                     average="micro"))

        #Real as lines, preds as cols
        m = confusion_matrix(tempclassRF_DG.y_v, tempclassRF_DG.y_pred_v, labels=["hyperthermophilic", "thermophilic", "mesophilic", "psychrophilic"])
        mat_DG.append(m)

        # Feature importances
        gini_50_nodes = enzs_only[np.argsort(tempclassRF_DG.clf.feature_importances_)[::-1]][:50]
        gini_DG.append(gini_50_nodes)

        # Get depth
        max_dpth = max([dectree.tree_.max_depth for dectree in tempclassRF_DG.clf.estimators_])
        depths_DG.append(max_dpth)

    # Plot cross-validation boxplot of accuracies per class
    m_diag_DG = [np.diag(m) for m in mat_DG]
    plt.figure(figsize=(10, 7))
    sns.boxplot(data=(np.array(m_diag_DG) / np.sum(mat_DG[0], axis=1).astype(float)),
                palette=sns.color_palette(["#d82f00", "goldenrod", "lightgreen", "#41a5b7", ]))
    plt.xticks(range(4), ["HT", "T", "M", "P"], fontsize=16)
    plt.ylabel("Accuracy", fontsize=18)
    plt.xlabel("Temperature class", fontsize=18)
    plt.yticks(fontsize=16)
    plt.ylim(0, 1.05)

    print np.mean(accuracy_v_DG)
    print np.std(accuracy_v_DG)

    print np.mean(f1_scores_DG)
    print np.std(f1_scores_DG)

    # Mean matrix o
    m_mean_DG = deepcopy(mat_DG[0])
    for m in mat_DG[1:]:
        m_mean_DG += m
    print m_mean_DG / float(cv_n)
    # for row in range(4):
    #     print ((m_mean_DG / float(cv_n))[row, :]/np.sum(mat_DG[0], axis=1)[row]*100)
    norm_DG = (m_mean_DG / float(cv_n)).astype('float') / (m_mean_DG / float(cv_n)).sum(axis=1)[:, np.newaxis]
    plot_confusion_matrix(norm_DG, ["HT", "T", "M", "P"],
                          normalize=False,
                          cmap=plt.cm.YlGnBu)

    # Most important nodes for model prediction, and pathways to which they belong
    c_gini_DG = Counter(np.concatenate(gini_DG))
    c_DG, pathways_DG = get_pathways_from_nodes(c_gini_DG.keys())
    pathways_g_DG = []
    for n in np.concatenate(gini_DG):
        n = n.split("_")[0]
        pathways_g_DG += pathways_DG[n]
    c_all_DG = Counter(np.array(pathways_g_DG)[:, 1])
    c_mean_DG = Counter({key: v / float(cv_n) for key, v in c_all_DG.items()})

    # Histogram index nodes per number of models
    index_nodes_DG = {ky: list(enzs_only).index(ky) for ky in c_gini_DG.keys()}
    c_gini_idx_DG = {index_nodes_DG[ky]: v for ky, v in c_gini_DG.items()}
    plt.figure(figsize=(10, 7))
    plt.bar(range(len(c_gini_idx_DG)), np.array(sorted(c_gini_idx_DG.values(),
                                                    reverse=True)) * 100. / cv_n)  # plots the number of model counts per union of 50 most important nodes for all models.
    plt.xlabel("Nodes", fontsize=16)
    plt.ylabel("Percentage of models with node in 50 most important", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()

    # Mean of tree max depths
    print np.mean(depths_DG)


    # =============================================================================
    #
    #                   SIMPLIFIED HABITAT PREDICTION
    #
    # =============================================================================


    matrix_hab = simplified_matrix[:, df_species.habMix.notnull()]
    classesHmix = df_species.habMix[df_species.habMix.notnull()].values

    accuracy_v_habMix = []
    gini_habMix = []
    mat_habMix = []
    f1_scores_habMix = []
    depths_habMix = []
    cv_n = 300
    for cv in xrange(cv_n): #cross validation
        print cv
        habitatRF = RandomForestClassifierClass(matrix=matrix_hab,
                                                classes=classesHmix,
                                                weights={"Environment": 0.33,
                                                         "Symbiont": 0.47,
                                                         "Mixed":0.20,
                                                         },
                                                ntrees=1000,
                                                kwargs={})

        # Split dataset
        habitatRF.train_test_validation()

        # Fit on train dataset
        habitatRF.rf_model_fit()
        # Test dataset prediciton
        habitatRF.predict_test()
        # True validation - independent testing set
        habitatRF.predict_valid()
        accuracy_v_habMix.append(sum(habitatRF.y_pred_v == habitatRF.y_v) / float(len(habitatRF.y_v)))
        f1_scores_habMix.append(f1_score(habitatRF.y_v, habitatRF.y_pred_v, labels=["Environment", "Symbiont", "Mixed"],
                                         average="micro"))


        # Feature importances
        gini_50_nodes = nodes_simplified[np.argsort(habitatRF.clf.feature_importances_)[::-1]][:50]
        gini_habMix.append(gini_50_nodes)

        #Get depth
        max_dpth = max([dectree.tree_.max_depth for dectree in habitatRF.clf.estimators_])
        depths_habMix.append(max_dpth)

    print np.mean(accuracy_v_habMix)
    print np.std(accuracy_v_habMix)

    print np.mean(f1_scores_habMix)
    print np.std(f1_scores_habMix)

    # Plot cross-validation boxplot of accuracies per class
    m_diag_habMix = [np.diag(m) for m in mat_habMix]
    plt.figure(figsize=(10, 7))
    sns.boxplot(data=(np.array(m_diag_habMix) / np.sum(mat_habMix[0], axis=1).astype(float)),
                palette=sns.color_palette(["lightgreen", "#d82f00", "goldenrod", ]))
    plt.xticks(range(3), ["Environment", "Symbiont", "Mixed"], fontsize=18)
    plt.ylabel("Accuracy", fontsize=20)
    plt.xlabel("Habitat", fontsize=20)
    plt.yticks(fontsize=18)
    plt.ylim(0, 1.05)


    # Mean matrix o
    m_mean_habMix = deepcopy(mat_habMix[0])
    for m in mat_habMix[1:]:
        m_mean_habMix += m
    print m_mean_habMix / float(cv_n)
    norm_habMix = (m_mean_habMix / float(cv_n)).astype('float') / (m_mean_habMix / float(cv_n)).sum(axis=1)[:, np.newaxis]
    plot_confusion_matrix(norm_habMix, ["Environment", "Symbiont", "Mixed"],
                          normalize=False,
                          cmap=plt.cm.YlGnBu)


    c_gini_habMix = Counter(np.concatenate(gini_habMix))
    c, pathways = get_pathways_from_nodes(c_gini_habMix.keys())
    pathways_g_habMix = []
    for n in np.concatenate(gini_habMix):
        n = n.split("_")[0]
        pathways_g_habMix += pathways[n]
    c_all_habMix = Counter(np.array(pathways_g_habMix)[:, 1])
    c_mean_habMix = Counter({key: v / float(cv_n) for key, v in c_all_habMix.items()})


    #Histogram index nodes per number of models
    index_nodes_habMix = {ky:list(nodes_simplified).index(ky) for ky in c_gini_habMix.keys()}
    c_gini_idx_habMix = {index_nodes_habMix[ky]:v for ky, v in c_gini_habMix.items()}
    plt.figure(figsize=(10,7))
    plt.bar(range(len(c_gini_idx_habMix)), np.array(sorted(c_gini_idx_habMix.values(), reverse=True)) * 100./cv_n) #plots the number of model counts per union of 50 most important nodes for all models.
    plt.xlabel("Nodes", fontsize=16)
    plt.ylabel("Percentage of models with node in 50 most important", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlim(0, 177)
    plt.tight_layout()

    #Mean of tree max depths
    print np.mean(depths_habMix)


    # =============================================================================
    #
    #                   OXYGEN TOLERANCE PREDICTION
    #
    # =============================================================================


    matrix_oxygenSimpl = simplified_matrix[:, df_species.oxySimpl.notnull()]
    classesOsimpl = df_species.oxySimpl[df_species.oxySimpl.notnull()].values


    accuracy_v_oxySimp = []
    gini_oxySimp = []
    mat_oxySimp = []
    f1_scores_oxySimp = []
    depths_oxySimp = []
    cv_n = 300
    for cv in xrange(cv_n):  # cross validation
        print cv
        oxySimp = RandomForestClassifierClass(matrix=matrix_oxygenSimpl,
                                          classes=classesOsimpl,
                                          weights={"Aerobe": 917./2231,
                                                   "Facultative": 782/2231.,
                                                   "Anaerobe": 532./2231,
                                                   },
                                          ntrees=1000,
                                          kwargs={})

        # Split dataset
        oxySimp.train_test_validation()

        # Fit on train dataset
        oxySimp.rf_model_fit()
        # Test dataset prediciton
        oxySimp.predict_test()
        # True validation - independent testing set
        oxySimp.predict_valid()
        accuracy_v_oxySimp.append(sum(oxySimp.y_pred_v == oxySimp.y_v) / float(len(oxySimp.y_v)))
        f1_scores_oxySimp.append(f1_score(oxySimp.y_v, oxySimp.y_pred_v,
                                          labels=['Aerobe', 'Facultative', 'Anaerobe'],
                                          average="micro"))

        # Feature importances
        gini_50_nodes = nodes_simplified[np.argsort(oxySimp.clf.feature_importances_)[::-1]][:50]
        gini_oxySimp.append(gini_50_nodes)

        # Get depth
        max_dpth = max([dectree.tree_.max_depth for dectree in oxySimp.clf.estimators_])
        depths_oxySimp.append(max_dpth)

        # Real as lines, preds as cols
        m = confusion_matrix(oxySimp.y_v, oxySimp.y_pred_v,
                             labels=['Aerobe', 'Facultative',
                                     'Anaerobe'])
        mat_oxySimp.append(m)

    print np.mean(f1_scores_oxySimp)
    print np.std(f1_scores_oxySimp)

    print np.mean(accuracy_v_oxySimp)
    print np.std(accuracy_v_oxySimp)

    # Plot cross-validation boxplot of accuracies per class
    m_diag_oxySimp = [np.diag(m) for m in mat_oxySimp]
    plt.figure(figsize=(10, 7))
    sns.boxplot(data=(np.array(m_diag_oxySimp) / np.sum(mat_oxySimp[0], axis=1).astype(float)), )
    # palette=sns.color_palette(["lightgreen", "#d82f00", "goldenrod", ]))
    plt.xticks(range(3), ['Aerobe', 'Facultative',
                          'Anaerobe'], fontsize=16)
    plt.ylabel("Accuracy", fontsize=20)
    plt.xlabel("Oxygen Tolerance", fontsize=20)
    plt.yticks(fontsize=18)
    plt.ylim(0, 1.05)

    # Mean matrix o
    m_mean_oxySimp = deepcopy(mat_oxySimp[0])
    for m in mat_oxySimp[1:]:
        m_mean_oxySimp += m
    print m_mean_oxySimp / float(cv_n)
    norm_oxySimp = (m_mean_oxySimp / float(cv_n)).astype('float') / (m_mean_oxySimp / float(cv_n)).sum(axis=1)[:, np.newaxis]
    plot_confusion_matrix(norm_oxySimp, ["Aerobe", "Facultative", "Anaerobe"],
                          normalize=False,
                          cmap=plt.cm.YlGnBu)

    c_gini_oxySimp = Counter(np.concatenate(gini_oxySimp))
    c, pathways = get_pathways_from_nodes(c_gini_oxySimp.keys())
    pathways_g_oxySimp = []
    for n in np.concatenate(gini_oxySimp):
        n = n.split("_")[0]
        pathways_g_oxySimp += pathways[n]
    c_all_oxySimp = Counter(np.array(pathways_g_oxySimp)[:, 1])
    c_mean_oxySimp = Counter({key: v / float(cv_n) for key, v in c_all_oxySimp.items()})

    # Histogram index nodes per number of models
    index_nodes_oxySimp = {ky: list(nodes_simplified).index(ky) for ky in c_gini_oxySimp.keys()}
    c_gini_idx_oxySimp = {index_nodes_oxySimp[ky]: v for ky, v in c_gini_oxySimp.items()}
    plt.figure(figsize=(10,7))
    plt.bar(range(len(c_gini_idx_oxySimp)), np.array(sorted(c_gini_idx_oxySimp.values(), reverse=True)) * 100. / cv_n)  # plots the number of model counts per union of 50 most important nodes for all models.
    plt.xlabel("Nodes", fontsize=16)
    plt.ylabel("Percentage of models with node in 50 most important", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlim(0,177)
    plt.tight_layout()

    # Mean of tree max depths
    print np.mean(depths_oxySimp)