# FunctionalPrediction5000species
Code for paper Weber Zendrera, Sokolovska, Soula. Functional prediction of environmental variablesusing metabolic networks. Scientific Reports (2021)

## How to run the code

A Python2.7 environment is provided to generate the graphs, obtain the scope matrix and do the random forest analysis (**requirements2.7.txt**). We then switched to Python3, so the Python3 environment is provided for the other analyses (**requirements3.7.txt**)

## Metabolic networks
### Metabolic networks already generated
We provide the metabolic networks that we generated in the folder **MetabolicNetworks/**. They are provided as adjacency matrices of size *union of all nodes* x *union of all nodes*, in .npy format.

### Metabolic network generation
You can otherwise apply our code to generate the networks :

Species metabolic networks will be generated in a folder called graph\_species/, where each species will have a folder containing its networks in GraphML and cPickle formats, with ubiquitous metabolites filtered or not ("unfiltered_"...). There will also be a folder where KEGG EC entries are saved and another one where KEGG reaction entries are saved: this allows the metabolic network generation.

To generate the metabolic networks, you need to run **build\_KEGG\_prok\_networks.py**

## Scope analysis
Metabolic network scope is obtained by running **scope\_kegg\_prk.py**. The scope matrix obtained as output will be the input for all further analysis.

Scope random forest analysis is performed in **scope\_classifier\_prk.py**.

### The following analyses were performed after switching to Python3, so the Python3 environement must be built and used (requirements3.7.txt).
t-SNE analysis, in **scope\_plotting\_tsne.py**, where we fit t-SNE and plot it colouring species with different metadata variables.

Neural network analysis where we predict growth temperature with the scope through an artificial neural network **neuralnetwork\_temp\_prk.py**.


## Species metadata
**species_metadata.csv** - CSV file with all of the environmental information on the species, collected from multiple databases.

*Be extra careful with species Natronolimnobius sp. AArc1, with KEGG code "nan", as the KEGG code is often interpreted by machines as NaN.*

**Columns**
- sp\_codes: KEGG database species code
- sp\_names: KEGG database species name
- bacdive\_code: species BacDive database code
- tax\_id: NCBI taxonomy ID
- oxygen\_fusion: Oxygen tolerance type from Fusion DB
- oxygen\_gold: Oxygen tolerance type from Gold DB
- merge\_oxygen: Oxygen tolerance merged from Fusion and Gold DB
- habitat: Habitat classes from FusionDB (simplified)
- temp\_range\_deduced: growth temperature class deduced from temp_def (Bacdive DB)
- temp\_def: growth temperature value  (Bacdive DB)
- temp\_type\_def: whether the growth temperature value is deduced from optimal growth or simply from growth temperature
- continent\_def: Continent where sample was found (Bacdive DB)
- habMix: Simplification of habitat variable into 3 classes : "symbiont", "mixed", "environment"
- oxySimpl: Simplification of merge\_oxygen variable into 3 classes : "aerobe", "facultative", "anaerobe"
- life\_domain: Archaea or Bacteria
- clades: From NCBI Taxonomy database, we chose clades with max 800 of our species as leaves, else we separated the clade into its branches
