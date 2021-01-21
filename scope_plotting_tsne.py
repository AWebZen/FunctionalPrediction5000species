import pickle as pk
from os.path import exists, join as joinP
from copy import deepcopy

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import manhattan_distances

from utils_general import fitting_tsne, sim_matrix


def plot_ppty(df_info_val, X_tsne, colours, alph=1, intermediate=False, unique_vals=False, plotNull=False):
    """Plot t-SNE by colouring with df_info_val (metadata). Can choose to plot None values or not (plotNull)."""
    values = deepcopy(df_info_val.values)
    values[values == None] = "None"
    values[values != values] = "None"
    if not unique_vals:
        unique_vals = sorted(list(set(values)))
    if not plotNull:
        if "None" in unique_vals:
            unique_vals = list(unique_vals)
            unique_vals.remove("None")
    j = 0
    for i, val in enumerate(unique_vals):
        indexes = np.where(values == val)[0]
        if val == "None":
            plt.scatter(X_tsne[indexes, 0], X_tsne[indexes, 1], c="#e4e5e5", label=val, alpha=alph)
        else:
            plt.scatter(X_tsne[indexes, 0], X_tsne[indexes, 1], c=colours[j], label=val, alpha=alph)
            j += 1
        if intermediate:
            plt.savefig("{}.png".format(val))

colours = ["#3fbfbc", "#cf532c", "#7362d1", "#60b443", "#b854be", "#b1b135", "#6171bb", "#d88d2e", "#5fa1d8",
           "#d24157", "#55bb83", "#d24992", "#3f7b43", "#c189d2", "#96ad64", "#9c4b6d", "#796d26", "#e2869d",
           "#d09f60", "#b56348"]
colours2 = ["#974355", "#44c758", "#9f5ad6", "#92bb30", "#5c52c7", "#7ac954", "#456ae1", "#b7b83c", "#6553b0",
            "#4a9e33", "#cf3a9d", "#45c37c", "#da6fd8", "#387725", "#96449d", "#7ea03f", "#6f87eb", "#daa530",
            "#396db5", "#dd812f", "#4e9ad3", "#c64720", "#55c6ea", "#d9374a", "#42cdb7", "#e4427b", "#509858",
            "#de69ad", "#36723e", "#bb86da", "#737c24", "#7460a5", "#a28e32", "#9e9ee1", "#535e0f", "#aa2e63",
            "#6ebf92", "#b74545", "#39afb6", "#ec7059", "#379577", "#d16480", "#206e54", "#d68ec0", "#505e25",
            "#a15d90", "#a7bc74", "#5c6a9f", "#a65d22", "#8a4768", "#d6ac66", "#ee90a1", "#758348", "#c7726c",
            "#746015", "#e89871", "#8e651e", "#944d32", "#896f3b", "#b78050"]

colours3 = ["blue", "#f32a30", "#de81d3", "#98cd28", "#278300","#fb2376", "#6adc90","#45124e", "#0082ad",
            "#7c4f00", "#5a8fff","#dbc672", "#bd9eff","#ffa988", "#535254","#fdb0cd", "#b20085", "#54d3ff",] #red b90a06, red i like #fc4d45 violeta 8a335b, blue 0049c3


backup_dir = "backup_cpkl/"
med_name = "PsychroMesoThermHyperMedium"

#Get species metadata
df_species = pd.read_csv("species_metadata.csv", index_col=0)
df_species.loc[df_species.sp_codes.isnull(), "sp_codes"] = "nan"  # Otherwise interpreted as NaN

#Get the scope matrix by running scope_kegg_prk.py :
simplified_matrix = pk.load(open(joinP(backup_dir, "simplified_matrix_scope{}_prk.cpk".format(med_name)), "rb"), encoding='latin1')

plotNull = True #Whether to plot species without metadata or not
perp = 40 #t-SNE perplexity
distance_tsne = "jaccard" #Distance to use for t-SNE


#Evaluating distance matrix
if distance_tsne == "jaccard":
    if exists("backup_cpkl/jaccard_simi.pk"):
        similarity = pk.load(open("backup_cpkl/jaccard_simi.pk", "rb"))
    else:
        similarity = sim_matrix(simplified_matrix.T) #takes some time
    dist = 1 - similarity
elif distance_tsne == "manhattan":
    dist = manhattan_distances(simplified_matrix.T)

#Fitting t-SNE
if exists(joinP(backup_dir, "X_tsne_jaccardDist_perp40.cpk")):
    #t-SNE
    X_tsne = pk.load(open(joinP(backup_dir, "X_tsne_jaccardDist_perp40.cpk"), "rb"))
else:
    print ("Recalculating X_tsne")
    X_tsne = fitting_tsne(dist, n_comp=2, perp=perp, metric="precomputed")


#Plotting t-SNE by colouring species with metadata classes
for var, cols, alpha, varname, unique_v in [(df_species.clades, colours2, 1, "clades", False),
                                  (df_species.temp_range_deduced, ["lightgreen", "#41a5b7", "goldenrod", "#d82f00",],
                                        1, "tempClass",
                                        ["None", 'mesophilic',
                                         'psychrophilic',
                                         'thermophilic',
                                         'hyperthermophilic']),
                                  (df_species.habMix, ['#d82f00', 'lightgreen','#4164aa'],
                                        1, "habMix",
                                        ['None', "Environment", "Symbiont", "Mixed"]),
                                  (df_species.oxySimpl, ['#d82f00', 'lightgreen', '#4164aa'],
                                             .8, "oxygenSimp",
                                             ['None', 'Aerobe', 'Facultative', 'Anaerobe']),
                                  ]:
    plt.figure(figsize=(17,10))
    plt.ylim(min(X_tsne[:, 1]) - 10, max(X_tsne[:, 1]) + 140)
    plot_ppty(var, X_tsne, colours=cols, alph=alpha, intermediate=False, unique_vals=unique_v, plotNull=plotNull)
    plt.legend(loc="best", ncol=5, prop={'size': 22})
    if plotNull:
        plt.savefig("tSNE_scope{}_{}_perp{}_{}TSNE.pdf".format(med_name, varname, perp, distance_tsne), bbox_inches='tight')
    else:
        plt.savefig("tSNE_scope{}_{}_perp{}_{}TSNE_noNone.pdf".format(med_name, varname, perp, distance_tsne),
                    bbox_inches='tight')

