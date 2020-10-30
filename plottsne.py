import tensorflow as tf
import keras
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers.merge import concatenate
from keras.layers import Input, Dense, Dropout
import os
import sys
def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3 


scRNAseq= pd.read_csv("reconstructed_scRNAseq.txt", sep = " ")
sc_labels = pd.read_csv("scRNAseq.mnn.labels.txt", sep = " ")
#scAPAusage = pd.read_csv("APAusage.txt",  sep = " ")
#APA_labels = pd.read_csv("APAusage.labels.txt", sep = " ")
scAPAusage = pd.read_csv("wlength.txt",  sep = " ")
APA_labels = pd.read_csv("wlength.labels.txt", sep = " ")
d = {"cells":scRNAseq.columns,  "sc_labels":sc_labels["x"]}
df = pd.DataFrame(d)
df


#common = intersection(gene_id, APA_id)
#len(common)
apa_cells = scAPAusage.columns
sc_cells = scRNAseq.columns
common_cells = intersection(apa_cells,sc_cells )
len(common_cells)

scRNAseq = scRNAseq.loc[:, scRNAseq.columns.isin(common_cells)]
scAPAusage = scAPAusage.loc[:, scAPAusage.columns.isin(common_cells)]
df = df.loc[df["cells"].isin(common_cells), :]
df = df.sort_values(by=["cells"])
scRNAseq = scRNAseq.reindex(sorted(scRNAseq.columns), axis=1)
scAPAusage  = scAPAusage.reindex(sorted(scAPAusage.columns), axis=1)

df.to_csv(r'index_wlength.csv', header=True, index=True, sep='\t', mode='a')

X_scRNAseq = scRNAseq.T
X_scProteomics = scAPAusage.T
#X_scRNAseq  =np.log(X_scRNAseq + 1) 
X_scProteomics = np.log(X_scProteomics + 1) 
# Input Layer
ncol_scRNAseq = X_scRNAseq.shape[1]
input_dim_scRNAseq = Input(shape = (ncol_scRNAseq, ), name = "scRNAseq")
ncol_scProteomics = X_scProteomics.shape[1]
input_dim_scProteomics = Input(shape = (ncol_scProteomics, ), name = "scProteomics")
# Dimensions of Encoder for each OMIC
encoding_dim_scRNAseq = 50
encoding_dim_scProteomics = 10
# Encoder layer for each OMIC
encoded_scRNAseq = Dense(encoding_dim_scRNAseq, activation = 'linear', 
                         name = "Encoder_scRNAseq")(input_dim_scRNAseq)
encoded_scProteomics = Dense(encoding_dim_scProteomics, activation = 'linear', 
                             name = "Encoder_scProteomics")(input_dim_scProteomics)
                             # Merging Encoder layers from different OMICs
merge = concatenate([encoded_scRNAseq, encoded_scProteomics])
# Bottleneck compression
bottleneck = Dense(50, kernel_initializer = 'uniform', activation = 'linear', 
                   name = "Bottleneck")(merge)
                   #Inverse merging
merge_inverse = Dense(encoding_dim_scRNAseq + encoding_dim_scProteomics, 
                      activation = 'elu', name = "Concatenate_Inverse")(bottleneck)
                      
# Decoder layer for each OMIC
decoded_scRNAseq = Dense(ncol_scRNAseq, activation = 'sigmoid', 
                         name = "Decoder_scRNAseq")(merge_inverse)
decoded_scProteomics = Dense(ncol_scProteomics, activation = 'sigmoid', 
                             name = "Decoder_scProteomics")(merge_inverse)
                             # Combining Encoder and Decoder into an Autoencoder model
autoencoder = tf.keras.Model(inputs = [input_dim_scRNAseq, input_dim_scProteomics], 
                    outputs = [decoded_scRNAseq, decoded_scProteomics])
# Compile Autoencoder
autoencoder.compile(optimizer = 'adam', 
                    loss={'Decoder_scRNAseq': 'mean_squared_error', 
                          'Decoder_scProteomics': 'mean_squared_error'})
autoencoder.summary()

# Autoencoder training
estimator = autoencoder.fit([X_scRNAseq, X_scProteomics], 
                            [X_scRNAseq, X_scProteomics], 
                            epochs = 100, batch_size = 128, 
                            validation_split = 0.2, shuffle = True, verbose = 1)
print("Training Loss: ",estimator.history['loss'][-1])
print("Validation Loss: ",estimator.history['val_loss'][-1])
# Encoder model
encoder = tf.keras.Model(inputs = [input_dim_scRNAseq, input_dim_scProteomics], 
                outputs = bottleneck)
bottleneck_representation = encoder.predict([X_scRNAseq, X_scProteomics])
bottle = pd.DataFrame(bottleneck_representation)
bottle.to_csv(r'bottle_wlength.csv', header=False, index=False, sep='\t')