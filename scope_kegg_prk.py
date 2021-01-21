#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 09:24:00 2019

@author: adele
"""

import logging
import os
from os.path import join as joinP
import cPickle as cpk
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from bioservices import KEGG
import pandas as pd

from utils_objet import GraphClass
import utils_general

# Setting logging preferences
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

k = KEGG()


if __name__ == '__main__':
    medium = list(set(utils_general.PSYCHROMED).union(set(utils_general.MESO_MEDIUM)).union(
        set(utils_general.HYPERTHERM_MED)).union(set(utils_general.THERM_MED)))
    med_name = "PsychroMesoThermHyperMedium"
    save = True

    # LOADING IMPORTANT VARIABLES
    work_dir = "graph_species/" # working directory : directory where we can find our graphs
    backup_dir = "backup_cpkl/"
    if not os.path.exists(work_dir):
        logger.error("Directory with graphs %s could not be found" %work_dir)
        raise SystemExit()

    if not os.path.exists(backup_dir):
        logger.error("Backup directory backup_cpkl/ could not be found")
        raise SystemExit()

    df_species = pd.read_csv("species_metadata.csv", index_col=0)
    df_species.loc[df_species.sp_codes.isnull(), "sp_codes"] = "nan"  # Otherwise interpreted as NaN

    #either get backup or generate scope
    if os.path.exists(joinP(backup_dir, "simplified_matrix_scope{}_prk.cpk".format(med_name))) and\
        os.path.exists(joinP(backup_dir, "simplified_nodes_scope{}_prk.cpk".format(med_name))):
        simplified_matrix = cpk.load(open(joinP(backup_dir, "simplified_matrix_scope{}_prk.cpk".format(med_name)), "rb"))
        nodes_simplified = cpk.load(open(joinP(backup_dir, "simplified_nodes_scope{}_prk.cpk".format(med_name)), "rb"))
    else:
        all_scope = []
        all_all_nodes = []

        for i, spec in enumerate(df_species.sp_names.values):

            print i, spec
            g = GraphClass("reaction graph")

            #Get species directory name
            if os.path.exists(work_dir + spec.replace("/", ".").replace(" ", "_")
                                      + "_" + df_species.iloc[i, 0]): #directory
                directory = work_dir + spec.replace("/", ".").replace(" ", "_") + "_" + df_species.iloc[i, 0]
            else:
                logger.error("Directory does not exist!")
                raise SystemExit()

            #Load GraphML metabolic network
            if os.path.exists(directory+"/metabolites_reaction_" + df_species.sp_codes.values[i] + ".graphml"):
                g.load_graph_graphml(directory+"/metabolites_reaction_" + df_species.sp_codes.values[i] + ".graphml")
            else:
                logger.error("Graph not found")

            if len(g.nodes()) == 0:
                logger.error("Uh oh, empty graph!")

            #Get scope
            try:
                scope_prk = g.scope(inputs=medium)
            except SystemExit:
                logger.error("Species will have no scope")
                all_scope.append([])
                continue
            sco_true = [n for n, s in scope_prk.items() if s == "Accessible"]
            print len(sco_true), "/", len(scope_prk), "nodes in scope"
            sco_nodes = g.get_full_names(sco_true)

            all_scope.append(sco_nodes)
            all_all_nodes.append(g.get_full_names(g.nodes()))

        c_nodes = Counter(np.concatenate(all_scope))
        all_nodes_sco = list(np.array(c_nodes.most_common(len(c_nodes)))[:,0])

        presence_sco_nodes = np.zeros((len(all_nodes_sco), len(all_scope)))
        for i, sco in enumerate(all_scope):
            for nod in sco:
                idx = all_nodes_sco.index(nod)
                presence_sco_nodes[idx, i] = 1

        print presence_sco_nodes.shape

        # cpk.dump([presence_sco_nodes, all_nodes_sco, df_species.sp_names.values],
        #          open("backup_cpkl/completescopematrix_allPrk_matrix_nodes_spnames.pk", "wb"))

        plt.matshow(presence_sco_nodes.T)
        plt.xlabel("Nodes")
        plt.ylabel("Species")
        frame1 = plt.gca()
        frame1.axes.get_xaxis().set_ticks([])
        # plt.title("Nodes in scope per species, {}, pathway networks".format(med_name))

        simplified_matrix = presence_sco_nodes[presence_sco_nodes.sum(axis=1) != presence_sco_nodes.shape[1],:]
        nodes_simplified = np.array(all_nodes_sco)[presence_sco_nodes.sum(axis=1) != presence_sco_nodes.shape[1]] #abscisse of simplified_matrix
        if save:
            cpk.dump(simplified_matrix, open(joinP(backup_dir, "simplified_matrix_scope{}_prk.cpk".format(med_name)), "wb"))
            cpk.dump(nodes_simplified, open(joinP(backup_dir, "simplified_nodes_scope{}_prk.cpk".format(med_name)), "wb"))