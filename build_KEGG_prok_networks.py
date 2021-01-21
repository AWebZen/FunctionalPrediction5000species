#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 09:07:07 2019

@author: adele
"""
import os
from os.path import join as joinP
import logging
import cPickle as cpk

import pandas as pd

from utils_objet import MetabolicGraph


#Setting logging preferences
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    backup_dir = "backup_cpkl/"
    cdna_dir = "genomes_cdna/"
    org_dir = "graph_species/"
    backup_namesf = "names_codes_prk_kegg_br.cpk"
    through_bri = True
    
    df_species = pd.read_csv("species_metadata.csv", index_col=0)
    df_species.loc[df_species.sp_codes.isnull(), "sp_codes"] = "nan" #Interpreted as NaN

    if not os.path.exists(org_dir): 
        os.makedirs(org_dir)
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    
    valid_species_index = []    
    for i, code in enumerate(df_species.sp_codes):
        old_dir = None
        
        print "\n", i, df_species.iloc[i,[0, 1]].values
            
        try :
            obj = MetabolicGraph(df_species.iloc[i, 1], "", #no need for fasta for brite reconstruction
                                 code=code, work_dir=org_dir, 
                                 through_brite=through_bri)
        except SystemExit, IOError:
            continue
            
        try:
            obj.directory = joinP(org_dir, 
                                  df_species.iloc[i, 1].replace("/", ".").replace(" ", "_") 
                                  + "_" + df_species.iloc[i, 0])
            
            
            obj.get_reaction_graph(gname="metabolites_reaction_" + code + ".graphml", 
                                   pklname="metabolites_reactions_graph_" + code + ".cpkl",
                                   dir_ec="EC_global/", rn_dir="reaction_files/",
                                   through_brite=through_bri)
            obj.build_reaction_graph(filtr=False, 
                                     gname="metabolites_reaction_" + code + ".graphml", 
                                     pklname="metabolites_reactions_graph_" + code + ".cpkl")
            valid_species_index.append(i)
        except (SystemExit, IOError, TypeError): #Problem when constructing
            os.system("rm -rf " + obj.directory) #remove directory
            logger.error("Species %s %s will not be handled" %(df_species.iloc[i, 1], code))
            continue
        
        cpk.dump(df_species.iloc[valid_species_index, :], open(joinP(backup_dir, backup_namesf), "wb"))
            
