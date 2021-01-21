#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 09:45:33 2019

@author: adele
"""
import logging
import collections
import sys
if sys.version_info[0] < 3:
    pythonv = 2
else:
    pythonv = 3

import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import linregress, kruskal, bartlett, f_oneway
from scipy.sparse import issparse
from sklearn.manifold import TSNE
if pythonv == 2:
    from bioservices import KEGG

    k = KEGG()
    k.settings.TIMEOUT = 1000  # Changing timeout


#Setting logging preferences
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



MESO_MEDIUM = [ #letort c et al, 2001 : https://mediadb.systemsbiology.net/defined_media/media/322/
"C00568", #4-Aminobenzoate
"C00147", #Adenine
"C00041", #Alanine
"C01342", "C00158", #Ammonium citrate
"C00062", #Arginine
"C00072", #Ascorbate
"C00152", #Asparagine
"C00049", #Aspartate
"C00120", #Biotin
"C08130", "C00076", "C00698", #Calcium chloride anhydrous, Ca, Cl
"C00864", #Calcium pantothenate
"C00175", #Cobalt chloride: Cb
"C00070", "C00059", #Cupric sulfate: Cu, SO42-
"C00097", #Cysteine
"C00740", "C00065", #DL-Serine
"C06420", "C00082", #DL-Tyrosine
"C00855", "C00073", #DL-methionine
"C14818", "C14819",  #Ferrous chloride
"C00504", #Folate
"C00064", #Glutamine
"C00037", #Glycine
"C00242", #Guanine
"C00135", #Histidine
"C00294", "C00081", "C00104",#Inosine, Inosine TP, IDP
"C00407", #Isoleucine
"C00025", #L-Glutamate
"C00243", #Lactose
"C00123", #Leucine
"C00725", #Lipoate
"C00047", #Lysine
"C00305", #Magnesium chloride
"C00034", "C19610", "C19611", #Mn, Mn2+, Mn3+ :Manganese sulfate
"C00253", #Nicotinate
"C00295", #Orotate
"C00079", #Phenylalanine
"C13197", #Potassium dibasic phosphate
"C00238", "C00009", #Potassium dihydrogen phosphate
"C00148", #Proline
"C00534", #Pyridoxamine HCl
"C00314", #Pyridoxine HCl
"C00255", #Riboflavin
"C01330", "C00033", #Sodium acetate
"C00378", #Thiamine HCl
"C00188", #Threonine
"C00214", #Thymidine
"C00078", #Tryptophan
"C00106", #Uracil
"C00183", #Valine
"C05776", #Vitamin B12
"C00385", #Xanthine
"C00038", #Zinc sulfate
]


PSYCHROMED = [ #Maria-Paz Cortes https://www.frontiersin.org/articles/10.3389/fmicb.2017.02462/full
    "C00407", # L-Isoleucine
    "C00123", # L-Leucine
    "C00183", # L-Valine
    "C00073", # L-Met
    "C00135", # L-His
    "C00062", # L-Arg
    "C00097", # L-Cys
    "C00037", # Gly
    "C00041", # L-Ala
    "C00049", # L-Asp
    "C00025", # L-Glu
    "C00064", # L-Gln
    "C00079", # L-Phe
    "C00148", # L-Pro
    "C00065", # L-Ser
    "C00188", # L-Threonine
    "C00082", # L-Tyrosine
    "C00314", # Pyridoxine
    "C00378", # Thiamine
    "C00864", # D-pantothenate
    "C14819", # Fe3+
    "C00122", # Fumarate
    "C00149", "C00497", # (L/D)-Malate
    "C00042", # Succinate
    "C00026", # Ketoglutarate (2-Oxoglutarate; Oxoglutaric acid; 2-Ketoglutaric acid; alpha-Ketoglutaric acid)
    "C00031", # D-Glu
    "C00124", # D-Gal
    "C00051", # Glutathione
    "C00116", # Glycerol
    "C01342", # Ammonium
    "C00147", # Adenine
    ]


HYPERTHERM_MED = [ #https://mediadb.systemsbiology.net/defined_media/media/382/ Rinker kd et al, 2000 Thermotoga maritima
"C00568", # 4-Aminobenzoate
"C01342", # Ammonium chloride
"C00120", # Biotin
"C12486", # Boric acid
"C08130", "C00698", # Calcium chloride anhydrous
"C00076", "C00864", # Calcium pantothenate
"C00504", # Folate
"C00725", # Lipoate
"C07755", "C00305", # Magnesium chloride
"C00208", # Maltose
"C00253", # Nicotinate
"C00238", # Potassium chloride: K
# "C13197", # Potassium dibasic phosphate
"C08219", "C01382", # Potassium iodide: K iodide, iodine
"C00314", # Pyridoxine HCl
"C11178", # Resazurin
"C00255", # Riboflavin
"C01330", "C01324", # Sodium bromide
# "", # Sodium chloride
"C00059", # Sodium sulfate
# "", # Sodium sulfide
"C20679", # Sodium tungstate
"C13884", # Strontium chloride
"C00378", # Thiamine HCl
"C05776", # Vitamin B12
]


THERM_MED = [ #https://mediadb.systemsbiology.net/defined_media/media/227/ Suzuki et al, 2001 Hydrogenobacter thermophilus TK-6
"C01342", "C00698", # Ammonium chloride
"C12486", # Boric acid
"C00076", # Calcium chloride anhydrous
 # Cupric chloride: Cu2+, Cl-
"C00070", "C00059",  # Cupric sulfate
"C07755", "C00305", # Magnesium sulfate
"C00034", "C19610", "C19611", #Mn, Mn2+, Mn3+ :Manganese sulfate
"C00150", #Molybdenum trioxide
"C14818", # Ferrous sulfate
"C00238", "C00009", "C13197", # Potassium dibasic phosphate
"C08219", # Potassium dihydrogen phosphate
"C01330", # Sodium chloride
"C00038", #Zinc sulfate
]



def train_test_proportions(freq, X, y):
    """
    Split data into training (freq proportion) and testing (1-freq) datasets. Different classes will be represented in
    proportions from y, and observations will be shuffled.

    freq - percentage of data to go to train, between 0 and 1
    X - data
    y - classes
    """
    assert freq < 1 and 0 < freq, "freq must be a proportion"
    c = collections.Counter(y)
    idx = {k:np.where(np.array(y) == k)[0] for k in c.keys()}
    train_idx = []
    test_idx = []
    for k in c.keys():
        n_to_choose_train = int(c[k] * freq + 0.5) # + 0.5 hack to round number to closest int (for positives) like in statistics
        train_idx.extend(list(np.random.choice(idx[k], n_to_choose_train, replace=False)))
        test_idx.extend([x for x in idx[k] if x not in train_idx])
    if len(train_idx) == 0 or len(test_idx) == 0:
        print("Frequence too high or two low, unable to form 2 groups")
        raise SystemExit()
    y_train = np.array(y)[train_idx]
    y_test = np.array(y)[test_idx]
    if len(y) == X.shape[0]:
        X_train = X[train_idx, :]
        X_test = X[test_idx, :]
    else:
        X_train = X[:, train_idx]
        X_test = X[:, test_idx]
    return X_train, X_test, y_train, y_test, train_idx, test_idx


def is_enz (node):
    """Tells if node is an enzyme (EC) or not"""
    split = node.split(".")
    if len(split) == 4 :#and np.all(np.array([sp.isdigit() for sp in split]) == True) :
        return True
    else:
        return False

if pythonv == 2:
    def get_domain(codes):
        """From list of KEGG codes, get organism domain"""
        domain = []
        orgs = k.list("organism")
        for code in codes:
            hit = orgs.find("\t"+code+"\t")
            if hit == -1:
                logger.error("Did not find species %s" %code)
                domain.append(None)
            else:
                dom = orgs[hit:].split("\n")[0].split("\t")[3].split(";")[1]
                domain.append(dom)
        return domain


    def get_pathways_from_nodes(nodes):
        """
        Get KEGG pathways associated to nodes in list. Nodes are compound codes and EC codes.
        :param nodes: list of node names
        :return:

        """
        global k
        nodes = [nod.split("_")[0] for nod in nodes]  # Remove _0, _1, _2 when multiple reactions per enzyme
        pathways = {}
        # pathways_ec = {}
        for p, cpd in enumerate(nodes):
            print(cpd)
            ok = False
            txt = k.get("ec:" + cpd)
            try:
                int(txt)
                txt = k.get("cpd:" + cpd)
                try:
                    int(txt)
                except ValueError:
                    ok = True
                    txt = txt.split("\n")
            except ValueError:
                ok = True
                txt = txt.split("\n")
            if ok:
                pathways[cpd] = _parse_get_pathway(txt)
                # pathways_ec[nodes[p]] = _parse_get_pathway(txt)
            else:
                print(cpd, "did not work")
        try:
            c = collections.Counter(np.concatenate(pathways.values())[:, 1])
        except ValueError:
            pathways_without = {k: v for k, v in pathways.items() if len(v) > 0}
            c = collections.Counter(np.concatenate(pathways_without.values())[:, 1])

        # del c['Metabolic pathways']
        return c, pathways  # , pathways_ec


def from_y_to_dict(y):
    """From vector of values, build dict where keys are the values and dict values
    are indexes of vector with such values"""
    return {k:np.where(y == k)[0] for k in np.unique(y)}

def common_nodes (nodes1, nodes2) :
    """
    Returns list of the intersection of two sets/lists of nodes
    """
    nodes1 = set(nodes1)
    nodes2 = set(nodes2)
    return nodes1 & nodes2


def invert_ko_to_ec_dict(ko_dict):
    """
    Inverts KO to EC dict (ko_ec_dict.cpkl in backup_cpkl) so as to have a
    {ec : set of kos} list, only with the KOs that interest us (that are in ko_dict).
    """
    inverted_dict = collections.defaultdict(set)
    for ko, ecs in ko_dict.items():
        if ecs == "":
            continue
        ecs = ecs.strip(" ").split()
        for ec in ecs:
            inverted_dict[ec].add(ko)
    return inverted_dict


def jaccard_index(y1, y2):
    sum_array = np.array([y1, y2]).sum(axis=0)
    if np.sum(sum_array) == 0:
        return 0
    jacc_ind = np.sum(sum_array == 2)/float(np.sum(sum_array != 0))
    return jacc_ind


def sim_matrix(X):
    """
    For a species x other thing (nodes) matrix, build a species x species
    jaccard index similarity matrix of species vectors.
    """
    sim = np.zeros((len(X), len(X)))
    for i, pati in enumerate(X):
        for j, pati2 in enumerate(X):
            if j > i:
                break
            if issparse(pati):
                pati = pati.todense()
            if issparse(pati2):
                pati2 = pati2.todense()
            sim[i,j] = jaccard_index(pati, pati2)
    return sim + sim.T - np.diag(sim.diagonal()) #Symmetrizing triangular matrix


def fitting_tsne(X, n_comp, perp, metric='euclidean'):
    """Fit t-SNE"""
    tsne = TSNE(n_components=n_comp, perplexity=perp, metric=metric)
    X_tsne = tsne.fit_transform(X)
    return X_tsne

def plot_tsne(X_tsne, t_idx, perp):
    """Plot t-SNE"""
    t1_idx, t2_idx, t3_idx, t4_idx = t_idx
    colours = ["r", "g", "b", "m"]

    plt.figure()
    plt.scatter(X_tsne[t1_idx, 0], X_tsne[t1_idx, 1], c=colours[0], label="T0")
    plt.scatter(X_tsne[t2_idx, 0], X_tsne[t2_idx, 1], c=colours[1], label="T1")
    plt.scatter(X_tsne[t3_idx, 0], X_tsne[t3_idx, 1], c=colours[2], label="T2")
    plt.scatter(X_tsne[t4_idx, 0], X_tsne[t4_idx, 1], c=colours[3], label="T3")
    plt.legend()
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.title("Perplexity:"+str(perp))
    plt.show()

def _parse_get_pathway(txt):
    """txt - list of lines of the entry"""
    assert type(txt) == list, "Takes list as input"
    pathways = []
    i = 0
    line = txt[i].rstrip("\n")
    while i+1 < len(txt) and not line.startswith("PATHWAY"):
        i += 1
        line = txt[i].rstrip("\n") #Skip lines until PATHWAY field reached
    if line.startswith("PATHWAY"): #PATHWAY field exists
        while line.startswith("PATHWAY") or line[0] == " ":  #while still in PATHWAY field
            line = line.lstrip("PATHWAY ")
            line_sp = line.split("  ")
            if len(line_sp) == 2:
                pathways.append(tuple(line_sp))
            else:
                logger.error("Uh oh, something wrong with the parsing : %s" %line)
            i += 1
            line = txt[i].rstrip("\n")
    if len(pathways) == 0:
        logger.error("No pathway for entry.")
    return pathways


def split_in_n(n, y):
    """
    Split my instances into n groups, with each class in its right proportion being represented in each
    split.

    /!\ If there are too few instances of each class to distribute among splits, the splits will be imbalanced.


    Parameters
    ----------
    n : int
        Number of splits.
    y : array-like
        Class array.

    Returns
    -------
    splits : list
        List of instances indices split into n lists.
    """
    idx = {k:  # Each class is shuffled and split into n groups
        np.array_split(
            np.random.permutation(
                np.where(np.array(y) == k)[0]
            ),
            n)
        for k in np.unique(y)}
    splits = [[] for _ in xrange(n)]
    for i in xrange(n):
        for k in idx.keys():
            splits[i].extend(list(idx[k][i]))
    return splits


def crossvalidation_splits_n(n, y):
    """
    For each group in splits, will be considered testing group, the others will
    be fused and considered training group. Will thus have n (train, test) data
    splits

    Parameters
    ----------
    n : int
        Number of splits.
    y : array-like
        Class array.

    Returns
    -------
    train_test : iterator
        Iterator of (train indices, test indices) couples

    """
    split = split_in_n(n, y)
    train_test = []
    for test in split:
        train = []
        for tr in split:
            if tr is not test:
                train += tr
        train_test.append((train, test))
    return iter(train_test)




def reorder_matrix (m, d) :
    """
    Reorder similarity matrix : put species in same cluster together.

    INPUT:
    m - similarity matrix
    d - medoid dictionary : {medoid : [list of species index in cluster]}

    OUTPUT :
    m in new order
    new_order - order of species indexes in matrix
    """
    new_order = []
    for i, med_class in enumerate(d.values()):
        new_order.append(med_class)
    return m[np.concatenate(new_order), :], new_order



def plot_similitude_matrix_clusters(similitude, d, hba1c="no_fig", 
                                    parameters_txt_xlim=[-6,-7],
                                    wratio=8,
                                    xbar=0.92,
                                    label_hba1c="HbA1c",
                                    xlab_acc="Nodes in scope",
                                    ylab_acc="Patients",
                                    figsiz=(20,13),
                                    xlim_hba1c=[],
                                    colormap="viridis",
                                    subset_sp=[],
                                    hba1c_barplot_width=15,
                                    colorbar=True):
    """
    Heatmap of common node similarity by pair of species, with species in the order
    of clusters (contrasts to the other heatmap in the order of temperature classes).
    A colored line at the top gives the cluster

    Careful! Cluster order may vary (may not be the same as temperature classes
    even when there is a correspondance) : the cluster more or less corresponding
    to the first temperature class may not be the first one (and therefore won't
    be the first one in plot neither)
    """
    if subset_sp:
        d2 = {}
        for k in d.keys():
            d2[k] = np.array([sp for sp in d[k] if sp in subset_sp])
    else:
        d2 = d
    sim, new_order = reorder_matrix (similitude, d2)
    colors = ["k", "mediumseagreen",  "dodgerblue", "crimson", "lightgrey",
              "yellow", "peru", "magenta", "b", "darkorchid", "brown",
              "olive", "wheat", "purple", "cadetblue", "pink", "red", "grey",
              "turquoise", "lime", "orange", "salmon", "cyan", "g", "hotpink",
              "tan", "lavender", "teal", "darkorange", "seagreen"]
#    ratio = sim.shape[1]/float(sim.shape[0])
#    if ratio < 0.8:
#        figsiz = ()
#    elif ratio > 1.2
    fig = plt.figure(figsize=figsiz)

    if hba1c != "no_fig":
        gs = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[wratio,1],
                               wspace=0.0, hspace=0.0,) #8,1
    else:
        gs = gridspec.GridSpec(nrows=1, ncols=1)

    #First subplot
    ax0 = plt.subplot(gs[0])
    im = ax0.imshow(sim, vmin=0, vmax=1, cmap=colormap)

    print (im.axes.get_position(), im.axes.get_position(original=True))

    ax0.set_xlabel(xlab_acc)
    for tick in ax0.xaxis.get_major_ticks():
        tick.label1.set_horizontalalignment('right')
    ax0.set_ylabel(ylab_acc)
    ax0.set_ylim(-0.5, sim.shape[0])
    plt.subplots_adjust(left=0.07, right=0.9, top=0.98, bottom=.07)

    length = 0
    pos = []
#    plt.title("Accessible nodes vs Hba1c; WITH CLUSTERS, patients at T0, bypass, MB+ATOX+LG, 9.9")
    for i, med_class in enumerate(new_order):
        pos.append((2*length+len(med_class)-1)/2.)
        ax0.plot([-1, -1], [length, length+len(med_class)-1], lw = 4, c=colors[i])
        ax0.text(parameters_txt_xlim[0], (2*length+len(med_class)-1)/2., "Cl."+str(i),
                 fontsize = 10, fontweight="bold", color=colors[i]) #DMEM -5 #Complete -10
        if i < len(new_order) - 1:
            ax0.plot([-1, sim.shape[1]], [length+len(med_class)-0.5, length+len(med_class)-0.5], c="white")
        length += len(med_class)
    ax0.set_xlim(parameters_txt_xlim[1], sim.shape[1]) #DMEM -6 #Complete -12


    if hba1c != "no_fig":
        hba1c = np.array(hba1c)
        ax1 = plt.subplot(gs[1], sharey=ax0)
        ax1.barh(width=[np.mean(hba1c[order]) for order in new_order], y=pos,
                    height=[hba1c_barplot_width for _ in new_order],
                 xerr=[np.std(hba1c[order]) for order in new_order], capsize=5)
        # ax1.boxplot([hba1c[order] for order in new_order], positions=pos,
        #             widths=[hba1c_barplot_width for _ in new_order], vert=False, manage_xticks=False)
        # ax1.scatter(hba1c[np.concatenate(new_order)], range(len(np.concatenate(new_order))))#, c=hba1c[np.concatenate(new_order)])
        ax1.set_ylim(-0.5, sim.shape[0])
        if xlim_hba1c:
            ax1.set_xlim(*xlim_hba1c)
        ax1.set_xlabel(label_hba1c)
        ax1.axes.get_yaxis().set_visible(False)
    #Colorbar params: [left, bottom, width, height]
#    gs.tight_layout(fig, rect=[0, 0, 0.95, 1])
    position = ax0.axes.get_position()
    print (position, position.y0)
    if figsiz[0] < 5:
        k = .05
    elif figsiz[0] < 10:
        k = .02
    else:
        k = .01
    if colorbar:
        cbax = fig.add_axes([xbar, position.y0, k, position.y1-position.y0]) #DMEM [0.95, 0.075, 0.01, 0.85] #Complete [0.95, 0.25, 0.01, 0.5]
        fig.colorbar(im, cax=cbax)
    fig.subplots_adjust(wspace=.0)

def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

class ParseKeggEntry:
    def __init__(self, txt_entry):
        assert type(txt_entry) != int, "Did not get a KEGG entry: int type error message from bioservices"
        list_ent = txt_entry.split("\n")
        self.entry_list = list_ent[1:]
        entry = filter(lambda s: s != "", list_ent[0].split("    "))
        self.entry_code = entry[1].strip(" ")
        self._type = entry[2].strip(" ")
        self.parse_categories()

    def parse_categories(self):
        self.entry = collections.defaultdict(list)
        self.entry["ENTRY"] = [self.entry_code, self._type]
        i = 0
        line = self.entry_list[i]
        while line[:2] != "//":
            if line[0] != " ": #Fetch category
                cat = line.split(" ")[0]
                assert cat.isupper(), "This is not a category name: " + cat
                value = line[len(cat):].lstrip(" ")
                if is_float(value):
                    value = float(value)
                self.entry[cat].append(value)
            else: #Fill category
                value = line.lstrip(" ")
                if is_float(value):
                    value = float(value)
                self.entry[cat].append(value)
            i += 1
            line = self.entry_list[i]

    def parse_column_categories(self, categories):
        """For categories with columns separated by a double space, a list of
        each column values will be created."""
        for cat in categories:
            if not cat in self.entry.keys():
                continue
            values = [[val] for val in self.entry[cat][0].split("  ")]
            for val in self.entry[cat][1:]:
                for i, col_val in enumerate(val.split("  ")):
                    values[i].append(col_val)
            self.entry[cat] = values


class KeggPathway(ParseKeggEntry):
    #Non exhaustive
    def __init__(self, txt_entry):
        ParseKeggEntry.__init__(self, txt_entry)
        self.parse_column_categories(["PATHWAY_MAP", "MODULE", "COMPOUND", "REL_PATHWAY", "DISEASE", "ORTHOLOGY"])

