"""
Utility functions for all our codes.

"""
import os
from os.path import join as joinP
import logging
import cPickle as cpk
import collections
import re

from bioservices import KEGG
from Bio import SeqIO
from Bio.KEGG import Enzyme
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csgraph

from utils_general import is_enz

#Setting logging preferences
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphClass:
    """
    Class for the different attributes and properties of the graphs from the MetabolicGraph class
    """
    def __init__ (self, name) :
        """
        name - name/type of graph, to appear in plot titles, etc
        """
        self.graph = nx.DiGraph()
        self.node_name_equivalence = {} #global name equivalence dictionary, node ids as keys and names as values
        self.name = name

    def nodes(self):
        """Shortcut for node list"""
        return self.graph.nodes


    def remove_nodes(self, perc) :
        """
        Removes random perc % (freq, so 0 <= perc <= 1) nodes from graph. (check
        robustness)

        perc - proportion of random nodes to remove from graph
        """
        assert perc <= 1 and perc >= 0, "Wrong value for percentage"
        rmv_nodes = np.random.choice(self.graph.nodes(), int(perc*len(self.graph.nodes())), replace = False)
        self.graph.remove_nodes_from(rmv_nodes)


    def remove_ecs(self, perc) :
        """
        Removes random perc % (freq, so 0 <= perc <= 1) ECs from graph. (check
        robustness), which is +/- equivalent to a random % of edges for substrate
        graph.

        perc - proportion of random edges to remove from graph
        """
        assert perc <= 1 and perc >= 0, "Wrong value for percentage"
        enz_nodes = [nod for nod in self.nodes() if is_enz(nod)]
        assert len(enz_nodes) > 0, "No enzyme codes in nodes! Is this a reaction graph?"
        rmv_edges = np.random.choice(enz_nodes,
                                     int(perc*len(enz_nodes)),
                                     replace = False)
        print len(enz_nodes)
        self.graph.remove_nodes_from(rmv_edges)


    def get_full_names(self, ids):
        """
        Gives the list of names of a list of node labels from a GraphClass graph.
        Keeps KEGG compound code (CXXXXX) code as is.

        INPUT:
        ids - group of node names from g

        OUTPUT:
        names - list of names
        """
        if not np.all(np.in1d(list(ids), list(self.graph.nodes)) == True):
            logger.error("At least one wrong compound name for inputs")
            raise SystemExit()
        names = []
        for cpd in ids :
            if len(cpd) == 6 and cpd[1:6].isdigit():
                names.append(cpd)
            elif is_enz(cpd):
                names.append(cpd)
            else:
                names.append(self.node_name_equivalence[cpd])
        return names


    def scope(self, inputs=False, added_inp=[]):
        """
        Determines metabolic scope.

        /!\ Only for directed reaction graphs!!

        INPUT:
        inputs - list of input compounds of the network. If none given, the inputs
            will be the reaction graph nodes with in-degrees of 0.
        added_inp - inputs to add to the ones detected when inputs = False. Not compatible
                    with inputs, if you want to add compounds to inputs add them manually.

        OUTPUT:
        accessibility - dict, scope: says if node is accessible or not for the set of inputs.
        (steps - +/- hierarchy steps to take from input to output)
        """

        def test_inputs (input_cmp) :
            """ Tests if inputs exist in graph, else only selects compounds in graph,
            otherwise throws an error """
            if not np.all(np.in1d(list(input_cmp), list(self.graph.nodes)) == True):
                length = len(input_cmp)
                logger.warning("At least one wrong compound name for inputs, will be removed")
                input_cmp = list(np.array(input_cmp)[np.in1d(list(input_cmp), list(self.graph.nodes))])
                logger.warning("%d/%d kept from added inputs" %(len(input_cmp), length))
                if len(input_cmp) < 1 :
                    logger.error("Not enough inputs")
                    raise SystemExit()
            return input_cmp

        assert len(self.graph.nodes()) > 0, "Graph needs to be built or loaded"
        assert nx.is_directed(self.graph), "Needs a directed graph!"
        assert any(map(is_enz, self.graph.nodes())), "Only implemented for reaction graphs"


        #Source compounds (inputs) from which accessibility will be measured
        if inputs:
            inputs = test_inputs(inputs)
        else: #Inputs deduced from graph ("complete" input compounds)
            in_array = np.array(list(self.graph.in_degree)) #reaction graph's in-degrees
            input_cpd = in_array[np.where(in_array[:,1] == '0')[0],0] #out nodes of in/out graph (sources)
            if added_inp :
                added_inp = test_inputs(added_inp)
            inputs = set(list(input_cpd) + added_inp) #+ ["C00028", "C00073"]

        accessibility = dict.fromkeys(self.graph.nodes, "Non accessible")

        #Initialisation: all inputs are accessible
        for i in inputs:
            accessibility[i] = "Accessible"
            if is_enz(i):
                #if it is an enzyme it is not actually the real input (substrates filtered).
                sccssors = list(self.graph.successors(i))
                for succ in sccssors:
                    accessibility[succ] = "Accessible"

        #Accessible nodes from each input
        for inp in inputs:
            for _, t in nx.bfs_edges(self.graph, inp):
                if not is_enz(t):
                    continue
                preds = list(self.graph.predecessors(t))
                access_preds = np.array([accessibility[pr] for pr in preds])
                if np.all(access_preds == "Accessible"):# or accessibility[t] == "Accessible":
                    accessibility[t] = "Accessible"
#                    if is_enz(t) :
                    sccssors = list(self.graph.successors(t))
                    for succ in sccssors :
                        accessibility[succ] = "Accessible"
#                else :
#                    accessibility[t] = "Non accessible"
        return accessibility



    def load_graph_graphml(self, fname):
        """
        Loads graph from existing graphml file and node name equivalence file.

        INPUT:
        fname - graph file path
        fname_equiv - input file path (cpickle or csv) for node name equivalence file.
        cpkl -  boolean if cpickle file for node name equivalence file.
        """
        if os.path.exists(fname) :
            self.graph = nx.read_graphml(fname)
            self.node_name_equivalence = nx.get_node_attributes(self.graph, "id")
        else:
            logger.error("File {} does not exist!".format(fname))


    def load_graph_cpkl(self, fname):
        """
        Loads graph from existing cPickle binary file and node name equivalence file.

        INPUT:
        fname - file path
        fname_equiv - input file path (cpickle or csv) for node name equivalence file.
        cpkl -  boolean if cpickle file for node name equivalence file.
        """
        if os.path.exists(fname) :
            self.graph = cpk.load(open(fname, "rb"))
            self.node_name_equivalence = nx.get_node_attributes(self.graph, "id")
        else:
            logger.error("File {} does not exist!".format(fname))



class MetabolicGraph:
    """
    Builds metabolic graphs thanks to the KEGG database for a given species.
    """
    k = KEGG() #Bioservices' KEGG interface
    k.settings.TIMEOUT = 1000000 #Changing timeout
    ubi_metab = ["C00001", "C00002", "C00008", "C00003", "C00004", "C00005",
                     "C00006",  "C00011",  "C00014", "C00059", "C00342", "C00009",
                     "C00013", "C00080"] #C00006 - NADP added

    def __init__(self, organism_name, fasta_name,
                 code="", KO=False, merged=False, work_dir="./",
                 through_brite=False):
        """
        To build hybrid species, we need a list of fasta files instead of just
        the name of one, and a list of codes instead of just one.

        organism_name - organism name given as input, space separated (or list)
        fasta_name - file name/path for fasta file. Needs to be a cDNA fasta file with a "gene:", "gene="
                or "locus_tag=" field in the description of each sequence, such as the ones from Ensembl DB.
                Can be a list of fasta names for artificial hybrid species.
        code - if KEGG organism 3-letter code is already known, it can be given.
                Else it will be deduced from organism name.
                If hybrid species, must be a list of already known codes.
        KO - boolean. If we build graphs through KO or not. Defaults to False.
            Instead of True, can put a KO dictionary with KOs as keys and ECs
            as values to quicken code.
        merged - boolean if we want to create an object of merged graphs.
        work_dir - working directory where the species directory will be located. Defaults to current dir.
        through_brite - do not need to go through genes for reconstruction
        """
        if type(organism_name) == list:
            organism_name = "_".joinP(organism_name)

        self.organism_name = organism_name
        self.directory = joinP(work_dir, organism_name.replace(" ", "_")) #directory - name of the directory with KEGG gene files
        self.code = code #KEGG organism code
        self.number_genes = 0 #Number of genes kept (valid)
        self.valid_ecs = [] #list of one or multiple EC codes as strings
        self.enzs_parsed = [] #List of KEGG enzyme entry file parsers
        self.genes = []
        self.multiple = [] #List of starting indexes of the genes in other species
        self.KO = KO

        self.reaction_graph = GraphClass("reaction graph") #reaction graph, filtered
        self.unfiltered_reaction_graph = GraphClass("unfiltered reaction graph") #unfiltered reaction graph (all metabolites)
        self.pathway_reaction_graph = GraphClass("pathway reaction graph") #reaction graph, filtered
        self.pathway_unfiltered_reaction_graph = GraphClass("pathway unfiltered reaction graph") #unfiltered reaction graph (all metabolites)


        if through_brite:
            if self.code == "":
                logger.error("Need a species code")
                raise SystemExit()

        if not KO and not merged and not through_brite: #If not building graphs with KO, need organism code and fastas
            if type(fasta_name) == list: #Hybrid species
                assert type(self.code) == list and len(self.code) > 1, "Missing multiple codes as list"
                for i, fname in enumerate(fasta_name) :
                    self.multiple.append(len(self.genes))
                    self.load_data_file(fname) #Get gene names
            else:
                self.load_data_file(fasta_name) #Get gene names

            if self.code == "" : #Find organism code
                self.findandtest_organism()
            elif type(self.code) == str or type(self.code) == unicode or type(self.code) == np.unicode or type(self.code) == np.unicode_ :
                self.test_code() #Testing gene name - organism code correspondance (tests a single gene)
            else :
                self.test_multiple_codes()


    def load_data_file(self, fname):
        """
        Loads fasta file using Bio parser and returns genes names

        INPUT:
        fname - file name/path for fasta file. Needs to be a cDNA fasta file with a "gene:"
                field in the description of each sequence, such as the ones from Ensembl DB.
                Also supports cDNA fasta files with [locus_tag=XXXXXX] field, such as the
                fastas from Genbank.
        """
        seqs = [s for s in SeqIO.parse(fname, "fasta")]
        self._genes = []
        for seq in seqs :
            descr = seq.description
            i_gen = descr.find("gene:") #Find gene name field
            i_gen2 = descr.find("locus_tag=") #For fasta from Genbank
            i_gen3 = descr.find("gene=") #For fasta from Genbank
            if i_gen != -1 :
                gene = descr[i_gen+5:].split()[0]
                self.genes.append(gene)
            elif i_gen2 != -1:
                gene = descr[i_gen2+10:].split("]")[0]
                self.genes.append(gene)
                if i_gen3 != -1: #Two possibilities from Genbank present
                    gene = descr[i_gen3+5:].split("]")[0]
                    self._genes.append(gene)
            elif i_gen3 != -1: #Last priority
                gene = descr[i_gen3+5:].split("]")[0]
                self.genes.append(gene)
        if len(self.genes) != len(seqs):
            if len(self.genes) <= int(.5*len(seqs)):
                logger.error("Could not find enough gene names. Is field 'gene:'/'gene='/'locus_tag=' present in your fasta descriptions?")
                raise SystemExit()
            else :
                logger.warning("Not all gene names found.")


    def test_code(self) :
        """
        Tests if 3-letter KEGG species code works with the set of gene names
        (tests only one) from the fasta file (if there is a correspondance in KEGG).
        """
        if type(MetabolicGraph.k.get(self.code + ":" + self.genes[3])) == type(1) :
            if len(self._genes) == 0 or type(MetabolicGraph.k.get(self.code + ":" + self._genes[3])) == type(1) : #Priority to locus_tag= rather than gene=
                logger.error("Uh oh! 3-letter KEGG species code does not work with fasta file genes...")
                raise SystemExit()
            else:
                self.genes = self._genes


    def test_multiple_codes(self) :
        """
        Tests if 3-letter KEGG species code works with the set of gene names
        (tests only one) from the fasta file (if there is a correspondance in KEGG).
        """
        for i, code in enumerate(self.code) :
            if type(MetabolicGraph.k.get(code + ":" + self.genes[self.multiple[i]])) == type(1) :
                logger.error("Uh oh! 3-letter KEGG species code does not work with fasta file genes for species %d!" %(i+1))
                raise SystemExit()


    def get_organism(self, org_name):
        """
        Finds the KEGG organism code name through the organism name. Tests hits found.

        INPUT:
        org_name - name of organism or parts of it, space separated

        OUTPUT:
        code - KEGG organism ID/code or None, if not found
        """
        org_list = MetabolicGraph.k.lookfor_organism(org_name)
        if len(org_list) > 0: #Found possible organism hits
            for org in org_list: #Test hits
                code = org.split()[1]
                txt = MetabolicGraph.k.get(code + ":" + self.genes[3])
                try: #If organism code works, keep code
                    int(txt)
                except ValueError:
                    self.code = code
                    return code
        return None


    def findandtest_organism(self, work_dir = "./"):
        """
        Finds the KEGG organism code name through the organism name. If not found, tests
        with parts of the name as query. If not found, asks the user for a new name.
        Raises an error if no code name found at the end.

        INPUT:
        work_dir - working directory where the species directory will be located.
                   Defaults to current dir.

        OUTPUT:
        code - KEGG organism ID/code or Error, if not found
        """
        logger.info("Looking for organism code in KEGG...")
        code = self.get_organism(self.organism_name)
        if code == None:
            org_name_list = self.organism_name.split()
            org_name_list.append(org_name_list.pop(0)) #reshuffling : putting the first element of the name (usually the genus) as the last one to test, as it probably has a lot more hits
            logger.info("No hits for whole organism name, testing with parts of name...")
            for name in org_name_list: #Test parts of name
                code = self.get_organism(name)
                if code != None:
                    break
            if code == None:
                new_name = raw_input("Organism name " + self.organism_name + " was not found in KEGG, write another name for it (enter S to stop) : ")
                if new_name.lower() == "s" :
                    logger.error("Uh oh! Organism name not found in KEGG database!")
                    raise SystemExit()
                else:
                    self.organism_name = new_name #Updating attributes
                    self.directory = joinP(work_dir, self.organism_name.replace(" ", "_"))
                    code = self.findandtest_organism()
        if code != None:
            self.code = code
            logger.info("Organism code found!")


    def get_kegg_genes(self) :
        """
        Downloads KEGG gene files into org_name directory.
        """
        logger.info("Fetching KEGG gene entries...")
        count = 0
        if type(self.code) == str or type(self.code) == unicode or type(self.code) == np.unicode or type(self.code) == np.unicode_ :
            code = [self.code]
        i_cod = 0
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        for i_gen, gene in enumerate(self.genes) :
            if not os.path.exists(joinP(self.directory, gene + "_gene.txt")) : #Download missing gene files
                if i_gen in self.multiple :
                    i_cod = self.multiple.index(i_gen)
                txt = MetabolicGraph.k.get(code[i_cod] + ":" + gene)
                try:
                    int(txt)
                    count += 1
                    if count > 0.5 * len(self.genes) :
                        break
                except ValueError:
                    open(joinP(self.directory, gene + "_gene.txt"), "w").write(txt)
        if count > 0.5 * len(self.genes) :
            logger.error("Not enough gene hits in KEGG database!")
            raise SystemExit()
        elif count != 0:
            logger.warning("No hits in the gene KEGG database for %d genes" %count)
        self.number_genes = len(self.genes) - count


    def _extract_ECs_brite(self):
        brite = MetabolicGraph.k.get("br:"+self.code+"00001")
        ec_re = "\[EC:[0-9\. -][0-9\. -].*\]"
        all_ec = [" " + ec.strip("[]EC:") for ec in re.findall(ec_re, brite)]
        if len(all_ec) < 20:
            logger.error("Uh oh, low hit number for ECs in BRITE entry!")
            raise SystemExit()
        return all_ec


    def shortcut_get_ECs(self, dir_ec):
        logger.info("Fetching KEGG enzyme entries...")

        all_ECs = [] #List of ECs with hits in KEGG db
        if not os.path.exists(dir_ec):
            logger.info("Creating EC directory")
            os.makedirs(dir_ec)
        if not os.path.exists(self.directory) or not os.path.exists(joinP(self.directory, "backups/")):
            os.makedirs(joinP(self.directory, "backups/"))

        #Check if shortcut exists (if user has already run function once, and EC list has been saved)
        if os.path.exists(joinP(self.directory, "backups/valid_EC_list.cpkl")) :
            all_ECs = cpk.load(open(joinP(self.directory, "backups/valid_EC_list.cpkl"), "rb"))
            for valid_ecs in all_ECs: #valid ECs taken from one of the gene files
                name = valid_ecs[1:].replace(" ", "_")
                if not os.path.exists(joinP(dir_ec, "ec_" + name + ".txt")) : #get missing EC files
                    txt = MetabolicGraph.k.get("ec:" + valid_ecs)
                    try:
                        int(txt)
                    except ValueError:
                        open(joinP(dir_ec, "ec_" + name + ".txt"), "w").write(txt)
        else: #Complete download
            all_ecs = self._extract_ECs_brite()
            for ECs in all_ecs:
                name = ECs[1:].replace(" ", "_")
                if not os.path.exists(joinP(dir_ec, "ec_" + name + ".txt")) : #If not first time dowloading, will check only non valid ECs
                    txt = MetabolicGraph.k.get("ec:" + ECs)
                    try:
                        int(txt)
                    except ValueError:
                        all_ECs.append(ECs)
                        open(joinP(dir_ec, "ec_" + name + ".txt"), "w").write(txt)
                else:
                    if ECs not in all_ECs :
                        all_ECs.append(ECs)
            cpk.dump(all_ECs, open(joinP(self.directory, "backups/valid_EC_list.cpkl"), "wb"))
        self.valid_ecs = all_ECs
        self.dir_ec = dir_ec



    def get_ECs(self, dir_ec):
        """
        Extracts ECs for all gene files in our directory and downloads KEGG enzyme
        entries in directory/EC_files/ directory.
        """

        def extract_ec_number(fname):
            """
            Extracts EC and KO number(s) (if found) from the orthology field of a KEGG entry for a given gene.

            INPUT:
            fname - name/path of a KEGG gene database file (downloaded with get_kegg_genes() or otherwise)

            OUTPUT:
            KO - list of KO ids retrieved in ORTHOLOGY field
            ec_all - string of space-separated EC numbers, empty string otherwise
            """
            iOF = open(fname, "r")
            line = iOF.readline()
            KO = []
            ec_all = ""
            while line != "" and not line.startswith("ORTHO"):
                line = iOF.readline() #Skip lines until ORTHOLOGY field reached
            if line.startswith("ORTHO"): #ORTHOLOGY field exists
                while line[0] == " " or line[0:5] == "ORTHO":  #while still in ORTHOLOGY field
                    line = line.lstrip("ORTHOLOGY ") #Any of these characters are stripped from the beginning of str, order does not matter
                    ll = line.split()
                    if ll[0][1:].isdigit() and line[0] == "K" and len(ll[0]) == 6 : #Check presence of KO id
                         KO.append(ll[0])
                    i_ec = line.find("EC")
                    if i_ec != -1: #There should be at least one EC
                        ec = line[i_ec+3:].split("]")[0] #Cropping first 3 characters ("EC:") and last ] of [EC:XXXXXXX] field
                        ECs = ec.split() #List of ECs
                        for EC in ECs:
                            if EC.find(".") != -1 : #EC confirmed
                                if EC not in ec_all :
                                    ec_all += " " + EC
                    line = iOF.readline()
            iOF.close()
            return KO, ec_all


        logger.info("Fetching KEGG enzyme entries...")

        all_ECs = [] #List of ECs with hits in KEGG db
        gene_files = os.listdir(self.directory)
        if not os.path.exists(dir_ec):
            logger.info("Creating EC directory")
            os.makedirs(dir_ec)
        if not os.path.exists(self.directory) or not os.path.exists(joinP(self.directory, "backups/")):
            os.makedirs(joinP(self.directory, "backups/"))

        #Check if shortcut exists (if user has already run function once, and EC list has been saved)
        if os.path.exists(joinP(self.directory, "backups/valid_EC_list.cpkl")) :
            all_ECs = cpk.load(open(joinP(self.directory, "backups/valid_EC_list.cpkl"), "rb"))
            for valid_ecs in all_ECs: #valid ECs taken from one of the gene files
                name = valid_ecs[1:].replace(" ", "_")
                if not os.path.exists(joinP(dir_ec, "ec_" + name + ".txt")) : #get missing EC files
                    txt = MetabolicGraph.k.get("ec:" + valid_ecs)
                    try:
                        int(txt)
                    except ValueError:
                        open(joinP(dir_ec, "ec_" + name + ".txt"), "w").write(txt)
        else: #Complete download. Possible improvement (?) : with bash check if number of ec_ files in EC_files/ is the same as 'grep -l "EC:" '+ self.directory + '/*|wc' ??
            for fi in gene_files :
                if fi.endswith("_gene.txt") :
                    fname = joinP(self.directory, fi)
                    KO, ECs = extract_ec_number(fname)
                    if len(ECs) > 0 :
                        name = ECs[1:].replace(" ", "_") #We don't gain much time since we parse every gene file...
                        if not os.path.exists(joinP(dir_ec, "ec_" + name + ".txt")) : #If not first time dowloading, will check only non valid ECs
                            txt = MetabolicGraph.k.get("ec:" + ECs)
                            try:
                                int(txt)
                            except ValueError:
                                all_ECs.append(ECs)
                                open(joinP(dir_ec, "ec_" + name + ".txt"), "w").write(txt)
                        else:
                            if ECs not in all_ECs :
                                all_ECs.append(ECs)
            cpk.dump(all_ECs, open(joinP(self.directory, "backups/valid_EC_list.cpkl"), "wb"))
        self.valid_ecs = all_ECs
        self.dir_ec = dir_ec



    def get_ecs_from_KOs(self, KO_list, dir_ec):
        def extract_ec_number_KO(ko, ko_dict):
            """
            Extracts EC (if found) from the definition field of a KEGG entry for a KO.

            INPUT:
            ko - Kegg Orthology (KO) code name, string
            ko_dict - boolean or dict of KO keys and their associated ECs as values

            OUTPUT:
            ec_all - string of space-separated EC numbers, empty string otherwise
            """
            try:
                if ko in ko_dict.keys():
                    return ko_dict[ko]
            except TypeError :
                pass
            txt = MetabolicGraph.k.get("ko:"+ko)
            try :
                int(txt)
                return ""
            except ValueError:
                txt = txt.split("\n")
                ec_all = ""
                i = 0
                line = txt[i]
                while line != "" and not line.startswith("DEFINITION"):
                    i += 1
                    line = txt[i] #Skip lines until DEFINITION field reached
                if line.startswith("DEFINITION"): #DEFINITION field exists
                    while line[0] == " " or line[0:5] == "DEFIN":  #while still in DEFINITION field
                        line = line.lstrip("DEFINITION ") #Any of these characters are stripped from the beginning of str, order does not matter
                        i_ec = line.find("EC:")
                        if i_ec != -1: #There should be at least one EC
                            ec = line[i_ec+3:].split("]")[0] #Cropping first 3 characters ("EC:") and last ] of [EC:XXXXXXX] field
                            ECs = ec.split() #List of ECs
                            for EC in ECs:
                                if EC.find(".") != -1 : #EC confirmed
                                    if EC not in ec_all :
                                        ec_all += " " + EC
                        i += 1
                        line = txt[i]
            return ec_all


        logger.info("Fetching KEGG enzyme entries...")
        all_ECs = [] #List of ECs with hits in KEGG db

        if not os.path.exists(dir_ec):
            logger.error("{} directory given in command does not exist! Check path (current one: {})".format(dir_ec, os.getcwd()))
            raise SystemExit()
        if not os.path.exists(self.directory) or not os.path.exists(joinP(self.directory, "backups/")):
            os.makedirs(joinP(self.directory, "backups/"))

        #Check if shortcut exists (if user has already run function once, and EC list has been saved)
        if os.path.exists(joinP(self.directory, "backups/valid_EC_list.cpkl")) :
            logger.info("Found a copy of the list of enzymes. Taking it as list of ECs...")
            all_ECs = cpk.load(open(joinP(self.directory, "backups/valid_EC_list.cpkl"), "rb"))
            for valid_ecs in all_ECs: #valid ECs taken from one of the gene files
                name = valid_ecs[1:].replace(" ", "_")
                if not os.path.exists(joinP(dir_ec, "ec_" + name + ".txt")) : #get missing EC files in global EC directory
                    logger.info("Fetching undownloaded EC files: {}".format(valid_ecs))
                    txt = MetabolicGraph.k.get("ec:" + valid_ecs)
                    try:
                        int(txt)
                    except ValueError:
                        open(joinP(dir_ec, "ec_" + name + ".txt"), "w").write(txt)

        else: #Complete download
            for ko in KO_list :
                ECs = extract_ec_number_KO(ko, self.KO)
                if len(ECs) > 0 :
                    name = ECs[1:].replace(" ", "_") #We don't gain much time since we parse every gene file...
                    if not os.path.exists(joinP(dir_ec, "ec_" + name + ".txt")) : #If not first time dowloading, will check only non valid ECs
                        txt = MetabolicGraph.k.get("ec:" + ECs)
                        try:
                            int(txt)
                        except ValueError:
                            all_ECs.append(ECs)
                            open(joinP(dir_ec, "ec_" + name + ".txt"), "w").write(txt)
                    else:
                        if ECs not in all_ECs :
                            all_ECs.append(ECs)
            cpk.dump(all_ECs, open(joinP(self.directory, "backups/valid_EC_list.cpkl"), "wb"))
        self.valid_ecs = all_ECs
        self.dir_ec = dir_ec


    def parse_enzymes(self) :
        """
        Retrieves all KEGG enzyme records with Biopython parser. Saves them as cpickle
        object for backup.

        OUTPUT :
        enzs - list of enzyme records
        """
        enzs = []
        logger.info("Parsing enzymes...")


        if os.path.exists(joinP(self.directory, "EC_files")):
            if os.path.exists(joinP(self.directory, "EC_files/enzs_parser_backup.cpkl")): #Gains only a few seconds...
                enzs = cpk.load(open(joinP(self.directory, "EC_files/enzs_parser_backup.cpkl"), "rb"))
            else:
                for fi in sorted(os.listdir(joinP(self.directory, "EC_files"))):
                    if fi.startswith("ec_"):
                        enzs += list(Enzyme.parse(open(joinP(self.directory, "EC_files/", fi))))
        else:
            try:
                if not os.path.exists(self.dir_ec):
                    logger.error("<{}> global EC directory does not exist! Check path (current one: {})".format(self.dir_ec, os.getcwd()))
                    raise SystemExit()
            except AttributeError:
                logger.error("self.dir_ec does not exist. Run get_ecs_from_KOs or get_ECs")
                raise SystemExit()
            if not self.valid_ecs and not os.path.exists(joinP(self.directory, "backups/valid_EC_list.cpkl")):
                logger.error("Run get_ecs_from_KOs or get_ECs")
                raise SystemExit()
            if os.path.exists(joinP(self.directory, "backups/enzs_parser_backup.cpkl")): #Gains only a few seconds...
                enzs = cpk.load(open(joinP(self.directory, "backups/enzs_parser_backup.cpkl"), "rb"))
            else:
                for ecs in sorted(self.valid_ecs):
                    name = ecs[1:].replace(" ", "_")
                    fi = joinP(self.dir_ec, "ec_" + name + ".txt")
                    try:
                        enzs += list(Enzyme.parse(open(fi)))
                    except IOError:
                        logger.error("<{}> file does not exist".format(fi))
                        raise SystemExit()
        return enzs


    def get_substrates_products(self, e, filtr, graphe):
        """
        Finds unique substrate and products node ids and updates name equivalence dictionary.
        May filter following compounds : water, ATP, ADP, NAD, NADH, NADPH, carbon dioxide,
        ammonia, sulfate, thioredoxin, (ortho) phosphate (P), pyrophosphate (PPi), H+ and NADP.

        Will consider as different compounds the metabolites that also appear in compounds that are
        actually a list, or slightly different name versions of same metabolite.

        INPUT:
        e - KEGG enzyme/reaction entry parser (Biopython)
        filtr - boolean. If True, filters list of ubiquitous metabolites.
        graphe - determines to which graph these compounds need to be added

        OUPUT:
        subs - list of substrate node ids for given reaction, each being 10-char long
        prod - list of product node ids for given reaction, each being 10-char long
        """

        def extract_compound(comp) :
            """
            Extracts compound code or first 10 characters if code is not present.

            INPUT:
            comp - string of compound

            OUTPUT:
            compound code or 10 first compound characters
            i_cpd - -1 when no compound code
            """
            i_cpd = comp.find('CPD:')
            if i_cpd == -1:
                return comp[:10].upper(), i_cpd #+/- random 10-char code
            else:
                return comp[i_cpd+4:].split("]")[0], i_cpd #CPD code


        ubi_metab = ["C00001", "C00002", "C00008", "C00003", "C00004", "C00005",
                     "C00006",  "C00011",  "C00014", "C00059", "C00342", "C00009",
                     "C00013", "C00080"] #C00006 - NADP added


        subs = [] #Substrate node ids
        prod = [] #Product node ids


        for s in e.substrate :
            sub, i_cpd = extract_compound(s)
            if filtr :
                if sub in ubi_metab :
                    continue
            if s not in graphe.node_name_equivalence.values(): #Check if substrate exists in our equivalence dictionary
                i = 0
                while sub in graphe.node_name_equivalence.keys() and i_cpd == -1 : #Check if by bad luck our random compound node id exists in dictionary. Compound code should be unique.
                    if s[i*10+10:] != "" :
                        sub, i_cpd = extract_compound(s[i*10+10:]) #Find new compound node id in name
                    else :
                        sub += str(i) #add number if no unique compound node id can be found
                    i += 1
                graphe.node_name_equivalence[sub] = s
            else:
                sub = [k for k,name in graphe.node_name_equivalence.items() if name == s][0]
            subs.append(sub)


        for p in e.product :
            prd, i_cpd = extract_compound(p)
            if filtr :
                if prd in ubi_metab :
                    continue
            if p not in graphe.node_name_equivalence.values(): #Check if product exists in our equivalence dictionary
                i = 0
                while prd in graphe.node_name_equivalence.keys() and i_cpd == -1 : #Check if by bad luck our random compound node id exists
                    if p[i*10+10:] != "" :
                        prd, i_cpd = extract_compound(p[i*10+10:]) #Find new compound node id
                    else :
                        prd += str(i)
                    i += 1
                graphe.node_name_equivalence[prd] = p
            else:
                prd = [k for k,name in graphe.node_name_equivalence.items() if name == p][0]
            prod.append(prd)


        return subs, prod

    @staticmethod
    def _get_reaction_code(e):
        """Finds KEGG reaction code(s) from reaction field in enzyme entry"""
        rct = e.reaction
        if len(rct) == 0:
            return []
        rct_codes = []
        for r in rct:
            i = r.find("[RN:")
            if i != -1:
                codes = r[i+4:].split("]")[0].split()
                for code in codes:
                    if code[0] == "R" and len(code) == 6 and code[1:].isdigit():
                        rct_codes.append(code)
        return rct_codes


    def _get_rn_from_ec(self, ec_fi, rns):
        """Assign reaction(s) to EC in dict, from EC file"""
        p_enzs = list(Enzyme.parse(open(joinP(self.dir_ec, ec_fi))))
        for e in p_enzs:
            rcts = self._get_reaction_code(e)
            self.ec_rn[e.entry] = rcts
            rns = rns.union(set(rcts))
        return rns


    def _get_all_reactions(self):
        """For all ECs in EC directory, get RN codes"""
        if not os.path.exists(self.dir_ec) or len(os.listdir(self.dir_ec)) == 0:
            logger.error("Empty EC directory (%s), get EC entries first, or correct directory path" %self.dir_ec)
            raise SystemExit()

        ecs = os.listdir(self.dir_ec)

        if os.path.exists("backup_cpkl/ec_to_rn.cpkl"):
            self.ec_rn = cpk.load(open("backup_cpkl/ec_to_rn.cpkl", "rb"))
            rns = set()
            for rn in self.ec_rn.values():
                rns = rns.union(set(rn))

            #Checking self ECs are among keys, adding them
            if not hasattr(self , "valid_ecs"): #Do we have the list of ECs?
                if os.path.exists(joinP(self.directory, "backups/valid_EC_list.cpkl")) :
                    self.valid_ecs = cpk.load(open(joinP(self.directory, "backups/valid_EC_list.cpkl"), "rb"))
                else:
                    logger.error("Uh oh, no valid EC list when checking reactions.")
                    raise SystemExit()

            for ecs_v in self.valid_ecs: #checking presence in dict
                indiv_ECs = [ec for ec in ecs_v.split()[1:] if "-" not in ec]
                if not np.all(np.in1d(indiv_ECs, self.ec_rn.keys())):
                    ec_fi = "ec_" + ecs_v[1:].replace(" ", "_") + ".txt"
                    print ecs_v
                    if not os.path.exists(joinP(self.dir_ec, ec_fi)) :
                        logger.error("Uh oh, EC not downloaded, something wrong... have ECs been retrieved? %s" %ec_fi)
                    else:
                        rns = self._get_rn_from_ec(ec_fi, rns)
                    cpk.dump(self.ec_rn, open("backup_cpkl/ec_to_rn.cpkl", "wb"))
            return rns

        self.ec_rn = {}
        rns = set()
        for ec_fi in ecs:
            if ec_fi.startswith("ec_"):
                rns = self._get_rn_from_ec(ec_fi, rns)
#                    try:
#                        self.cpds[e.entry] = e.name[0] #takes the first out of all synonymous enzyme names
#                    except AttributeError:
#                        self.cpds = {e.entry:e.name[0]}
        cpk.dump(self.ec_rn, open("backup_cpkl/ec_to_rn.cpkl", "wb"))
        return rns



    def _parse_RN_entry(self, rn, txt_list):
        try:
            self.rcts[rn] = {"substrates":[], "products":[]}
        except AttributeError:
            self.rcts = {rn:{"substrates":[], "products":[]}}

        #Skipping lines til good one
        i = 0
        while i < len(txt_list) and not txt_list[i].startswith("EQUATION"):
            i += 1

        #In EQUATION field
        if txt_list[i].startswith("EQUATION"):
            line = txt_list[i].rstrip("\n")
            j = 0
            while line[0] == " " or line[0:5] == "EQUAT":  #while still in EQUATION field
                if j > 0:
                    logger.error("Multiple lines in equation for entry %s" %rn)
                    raise SystemExit() #Is this something to handle?
                line = line.lstrip("EQUATION ")
                eq = line.split(" <=> ")
#                print line
                if len(eq) == 1:
                    logger.error("/!\/!\/!\/!\/!\/!\/!\/!\/!\ NO <=> FOR RN %s : %s" %(rn, line))
                    raise SystemExit() #Goes with multiple line problem
                if len(eq) > 2:
                    logger.error("Multiple <=> in reaction equation for RN %s: %s" %(rn, line))
                    raise SystemExit() #Is this something to handle?
                for i_rctnt, key in enumerate(["substrates", "products"]):
                    reactants = eq[i_rctnt]
                    cpds = reactants.split(" + ")
                    for cpd in cpds:
                        st_cpd = cpd.split()
                        if len(st_cpd) > 2:
                            logger.error("Something not handled when finding stoechiometry: %s" %cpd)
                            raise SystemExit()
                        stoechio = "1"

                        for i_cp, cp in enumerate(st_cpd):
                            if cp[0] == "C" and cp[1:6].isdigit():
                                self.rcts[rn][key].append((stoechio, cp[:6]))
                                if len(cp) != 6:
                                    logger.warning("Further information about %s: <%s>, will not be handled. Reaction: %s" %(cp[:6], cp[6:], line))
                            elif cp[0] == "G":
                                logger.debug("%s is a glycan reaction, will not be handled" %rn)
                                del self.rcts[rn]
                                return "No"
                            else:
                                stoechio = cp
                                if i_cp != 0:
                                    logger.error("What is this %s ?? Reactant : %s" %(cp, cpd))
#                                    raise SystemExit()
                                if not cp.isdigit():
                                    logger.debug("Is this stoechiometry? %s" %(cpd))
                i += 1
                j += 1
                line = txt_list[i].rstrip("\n")
        else:
            logger.error("No equation field for entry %s!" %rn)
            raise SystemExit()


    def _get_rn(self, r, invalid=False):
        if not os.path.exists(joinP(self.rn_dir, "rn_" + r + ".txt")):
            txt_raw = MetabolicGraph.k.get("rn:"+r)
            try :
                int(txt_raw)
                logger.debug("No hit for missing reaction %s" %r)
            except ValueError:
                txt = txt_raw.split("\n")
                rct = self._parse_RN_entry(r, txt)
                if not rct:
                    open(joinP(self.rn_dir, "rn_" + r + ".txt"), "w").write(txt_raw)
                elif type(invalid) == list:
                    invalid.append(r)
        else:
            with open(joinP(self.rn_dir, "rn_" + r + ".txt"),"r") as f:
                txt = f.read().rstrip("\n").split("\n")
            rct = self._parse_RN_entry(r, txt)
            if rct and type(invalid) == list:
                invalid.append(r)
        if type(invalid) == list:
            return invalid


    def get_all_reaction_info(self, rn_dir, dir_ec):
        """
        Saves Reaction entries
        """
        if not os.path.exists(rn_dir):
            os.makedirs(rn_dir)
        self.rn_dir = rn_dir

        if not os.path.exists(dir_ec):
            os.makedirs(dir_ec)
        self.dir_ec = dir_ec

        logger.info("Finding all RN codes from available ECs")
        rns = self._get_all_reactions()
        logger.debug("Done")

        if os.path.exists("backup_cpkl/rn_to_subs_prods.cpkl"):
            self.rcts = cpk.load(open("backup_cpkl/rn_to_subs_prods.cpkl", "rb"))
            #Check missing RNs
            if not np.all(np.in1d(np.concatenate(self.ec_rn.values()), self.rcts.keys())):
                missing_rn = set(np.concatenate(self.ec_rn.values())) - set(self.rcts.keys())
                for r in missing_rn:
                    self._get_rn(r)
                cpk.dump(self.rcts, open("backup_cpkl/rn_to_subs_prods.cpkl", "wb"))
        else:
            for r in rns:
                self._get_rn(r)

            try:
                cpk.dump(self.rcts, open("backup_cpkl/rn_to_subs_prods.cpkl", "wb"))
            except AttributeError:
                logger.error("Uh oh... self.rcts does not exist!")
                raise SystemExit()



    def _parse_CPD_entry(self, cpd, txt_list):
        """Get first name in NAME field for compound entry with code cpd, in KEGG."""
        #Skipping lines til good one
        i = 0
        while i < len(txt_list) and not txt_list[i].startswith("NAME"):
            i += 1

        #In NAME field
        if txt_list[i].startswith("NAME"):
            first_name = txt_list[i][4:].rstrip(";\n").lstrip(" ")
            try:
                self.cpds[cpd] = first_name + " [" + cpd + "]"
            except (NameError, AttributeError):
                self.cpds = {}
                self.cpds[cpd] = first_name + " [" + cpd + "]"
        else:
            logger.error("Uh oh, no NAME field for compound %s" %cpd)


    def get_all_cpds_names(self):
        if os.path.exists("backup_cpkl/cpds_to_names.cpkl"):
            self.cpds = cpk.load(open("backup_cpkl/cpds_to_names.cpkl", "rb"))
            subs = np.vstack([np.array(self.rcts[r]["substrates"]) for r in self.rcts.keys()])[:,1]
            prods = np.vstack([np.array(self.rcts[r]["products"]) for r in self.rcts.keys()])[:,1]
            if not np.all(np.in1d(np.concatenate([prods,subs]), self.cpds.keys())):
                missing_cpds = set(np.concatenate([prods,subs])) - set(self.cpds.keys())
                for cpd in missing_cpds:
                    txt_raw = MetabolicGraph.k.get("cpd:"+cpd)
                    try :
                        int(txt_raw)
                        logger.error("No hit for missing compound %s" %cpd)
                    except ValueError:
                        txt = txt_raw.split("\n")
                        self._parse_CPD_entry(cpd, txt)
                cpk.dump(self.cpds, open("backup_cpkl/cpds_to_names.cpkl", "wb"))
        else:
            cpd_codes = set()
            for rctnt in self.rcts.values():
                for k in ["substrates", "products"]:
                    for stoe, cpd in rctnt[k]:
                        cpd_codes.add(cpd)
            for cpd in cpd_codes:
                txt_raw = MetabolicGraph.k.get("cpd:"+cpd)
                try :
                    int(txt_raw)
                except ValueError:
                    txt = txt_raw.split("\n")
                    self._parse_CPD_entry(cpd, txt)
            cpk.dump(self.cpds, open("backup_cpkl/cpds_to_names.cpkl", "wb"))


    def add_new_enzyme(self, e):
        """
        If reconstruction found backup but in backup enzyme e is missing.
        """

        #Get reaction code
        rcts = self._get_reaction_code(e)
        self.ec_rn[e.entry] = rcts
        cpk.dump(self.ec_rn, open("backup_cpkl/ec_to_rn.cpkl", "wb"))

        #Get reaction entries
        invalid_rs = []
        for r in rcts:
            self._get_rn(r, invalid=invalid_rs)
        cpk.dump(self.rcts, open("backup_cpkl/rn_to_subs_prods.cpkl", "wb"))

        #Get compounds associated to reaction
        cpd_codes = set()
        for r in rcts:
            if r in invalid_rs:
                continue
            for k in ["substrates", "products"]:
                for stoe, cpd in self.rcts[r][k]:
                    cpd_codes.add(cpd)
        for cpd in cpd_codes:
            txt_raw = MetabolicGraph.k.get("cpd:"+cpd)
            try :
                int(txt_raw)
            except ValueError:
                txt = txt_raw.split("\n")
                self._parse_CPD_entry(cpd, txt)
        cpk.dump(self.cpds, open("backup_cpkl/cpds_to_names.cpkl", "wb"))



    def build_reaction_graph (self, filtr = True, save = True,
                              gname = "metabolites_reaction.graphml",
                              pklname = 'metabolites_reactions_graph.cpkl',
                              pathways = False):
        """
        Builds a directed reaction graph (substrates -> enzyme -> products).
        Skips enzymes without product and substrate entries.

        INPUT:
        filtr - boolean. If True, filters list of ubiquitous metabolites. Defaults to True.
        save - if True, saves graph as graphml. Defaults to True.
        gname - graph name if save = True. Defaults to "metabolites_reaction.graphml".
        pklname - cpickle graph name if save = True. Defaults to "metabolites_reactions_graph.cpkl".
        pathways - if we only want enzymes known to be in a pathway. Defaults to False.

        OUTPUT:
        graphe - reaction graph
        """
        if len(self.enzs_parsed) == 0 :
            enzs = self.parse_enzymes()
            self.enzs_parsed = enzs
        else: #skips step if already built a reaction graph -> already parsed enzyme files
            enzs = self.enzs_parsed

        count_skip = 0
        count_skip_paths = 0
        count_skip_enzymes = 0

        logger.info("Building graph...")

        if filtr :
            if pathways :
                graphe = self.pathway_reaction_graph
            else:
                graphe = self.reaction_graph
        else:
            if pathways :
                graphe = self.pathway_unfiltered_reaction_graph
            else:
                graphe = self.unfiltered_reaction_graph

        for e in enzs:
            try:
                self.ec_rn[e.entry]
            except KeyError:
                self.add_new_enzyme(e)
            if len(self.ec_rn[e.entry]) == 0 and len(e.substrate) != 0 and len(e.product) != 0:
                count_skip_enzymes += 1
                continue

            #Filtering invalid reactions
            reactions = [rct for rct in self.ec_rn[e.entry] if rct in self.rcts.keys()]

            for r, rct in enumerate(reactions):
                if pathways :
                    if len(e.pathway) == 0:
                        count_skip_paths += 1 #Count each reaction
                        continue

                if len(reactions) > 1: #Multiple reactions per enzyme
                    er = e.entry + "_" + str(r)
                    name = e.name[0] + "_" + str(r)
                else:
                    er = e.entry #EC code
                    name = e.name[0]

                subs = self.rcts[rct]["substrates"]
                prod = self.rcts[rct]["products"]
                if filtr:
                    subs = [cpd for cpd in subs if cpd[1] not in MetabolicGraph.ubi_metab]
                    prod = [cpd for cpd in prod if cpd[1] not in MetabolicGraph.ubi_metab]

                if len(subs) == 0 and len(prod) == 0: #Skipping enzymes without substrate and product entries
                    count_skip += 1
                    continue

                graphe.node_name_equivalence[er] = name

                #Building graph
                graphe.graph.add_node(er, id=name) #takes the first out of all synonymous enzyme names
                for _, s in subs:
                    if s not in graphe.node_name_equivalence.keys():
                        graphe.node_name_equivalence[s] = self.cpds[s]
                    graphe.graph.add_node(s, id=graphe.node_name_equivalence[s])
                    graphe.graph.add_edge(s, er)

                for _, p in prod:
                    if p not in graphe.node_name_equivalence.keys():
                        graphe.node_name_equivalence[p] = self.cpds[p]
                    graphe.graph.add_node(p, id=graphe.node_name_equivalence[p])
                    graphe.graph.add_edge(er, p)

        #Writing output file with name equivalences
        if save:
            if filtr:
                logger.info("Saving graph as %s and as cpickle object %s in directory %s",
                            gname, pklname, self.directory)
                nx.write_graphml(graphe.graph, joinP(self.directory, gname))
                cpk.dump(graphe.graph, open(joinP(self.directory, pklname),'wb'))
            else:
                logger.info("Saving graph as %s and as cpickle object %s in directory %s",
                            "unfiltered_" + gname, "unfiltered_" + pklname, self.directory)
                nx.write_graphml(graphe.graph, joinP(self.directory, "unfiltered_" + gname))
                cpk.dump(graphe.graph, open(joinP(self.directory, "unfiltered_" + pklname), 'wb'))

        if count_skip > 0:
            logger.warning("%d/%d enzyme entries without substrate nor product information have been skipped, and will not appear in graph.", count_skip, len(enzs))
        if count_skip_paths > 0:
            logger.warning("%d/%d enzyme entries without pathway have been skipped, and will not appear in graph", count_skip_paths, len(enzs))
        if count_skip_enzymes > 0:
            logger.warning("%d/%d enzyme entries without RN but with subs and prods in their fields in enzyme entry", count_skip_enzymes, len(enzs))
        return graphe


    def get_reaction_graph(self, filtr = True, save = True,
                           gname="metabolites_reaction.graphml",
                           pklname='metabolites_reactions_graph.cpkl',
                           pathways=False, ko_list=[],
                           dir_ec="", rn_dir="", through_brite=False):
        """
        Global function for building a reaction graph, if you don't want to do every
        step of it (fetching gene entries, fetching enzyme entries, building the graph).
        Once a graph has already been built (gene and enzyme entries already fetched), it
        is recommended to use build_reaction_graph() or build_substrate_product_graph() directly.

        INPUT:
        filtr - boolean. If True, filters list of ubiquitous metabolites. Defaults to True.
        save - if True, saves graph as graphml. Defaults to True.
        gname - graph name if save = True. Defaults to "metabolites_reaction.graphml".
        pklname - cpickle graph name if save = True. Defaults to "metabolites_reactions_graph.cpkl".
        pathways - if we only want enzymes known to be in a pathway. Defaults to False.
        ko_list - list of KOs for when self.KO != False
        dir_ec - global directory where ECs have already been downloaded
        rn_dir - global directory where reaction entries are downloaded
		through_brite - use brite shortcut to get all ECs from brite entry

        OUTPUT :
        graphe - self.reaction_graph or self.unfiltered_reaction_graph
        """
#        if len(self.reaction_graph.node_name_equivalence.keys()) == 0 and len(self.unfiltered_reaction_graph.node_name_equivalence.keys()) == 0 :

        if not self.KO :
            if through_brite:
                self.shortcut_get_ECs(dir_ec)
            else:
                self.get_kegg_genes()
                self.get_ECs(dir_ec)
        else :
            assert len(ko_list) > 0, "ko_list argument must have at least one KO code"
            self.get_ecs_from_KOs(ko_list, dir_ec)

        if not hasattr(self, 'ec_rn') or not hasattr(self, 'rcts') or not hasattr(self, 'cpds')\
        or not self.ec_rn or not self.rcts or not self.cpds:
            self.get_all_reaction_info(rn_dir, dir_ec)
            self.get_all_cpds_names()
        graphe = self.build_reaction_graph (filtr, save, gname, pklname, pathways)
        return graphe




if __name__ == '__main__':
    pass
