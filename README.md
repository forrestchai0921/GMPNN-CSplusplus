# GMPNN-CSplusplus
This research led to my selection as a Regeneron Science Talent Search Scholar. The model is named GMPNN-CS++. It is built onto GMPNN-CS (Nyamabo et al., 2022)'s framework, with a novel dual-contrasting framework and the self-attention and residual memory network modules introduced to enhance model performance. 

**For GMPNN-CS++ implementation, see final DDI codes.**
**For GMPNN-CS++ paper,**   [Download Paper (PDF)](./GMPNN-CS++.pdf)

**Repository Structure & Attribution**
All files in this repository, except for the folder final DDI codes and the file /GMPNN-CS++.pdf, originate from kanz76/GMPNN-CS
.
They are the original source code of GMPNN-CS. Full credit for these files goes to the original authors.

final DDI codes/
This folder contains the complete implementation of GMPNN-CS++, which is based on the original GMPNN-CS.
My modifications include:

ddi_datasets.py – modified to construct the dataset with the novel interaction-mislabeled DDI type.

custom_loss.py – replaced the original loss function with my proposed dual-contrasting loss function.

models.py – extended by adding self-attention and residual memory network modules.

selfattention.py - newly added to incorporate the self-attention module to the model


**Abstract:** Drug-drug interactions (DDIs) occur when multiple drugs react with each other when taken together. They can lead to unintended side effects that may be harmful to patients. Developing an efficient and accurate computational model for DDI predictions is highly important to assist healthcare professionals in making better prescription decisions. 

The proposed GMPNN-CS++ model in this paper employs the Self-attention mechanism and a residual memory network after GMPNN-CS’s message-passing module to enhance the extracted representation of cross-substructure pairs within two interacting drug molecules. Previous neural network models for DDI prediction mainly consider two types of samples: positive samples and negative non-interaction samples. I introduce a novel dual-contrasting sampling approach in GMPNN-CS++ to include a third type, negative mislabeled-interaction samples, representing an overlooked common negative scenario. A dual-contrasting loss function is designed to make the neural network distinguish between positive samples and both types of negative samples, thereby widening the model’s applicability. Through dual-contrasting, the proposed method GMPNN-CS++ demonstrates an ability to capture additional features and improves performance in predicting both positive and negative DDI cases, achieving an overall accuracy of 97%, compared to the baseline GMPNN-CS.

