from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import pandas as pd
from PIL import Image
import os
import cv2
import pickle


# train_list = []
# for i in range(0, 122755):
#     small_list = []
#     path = 'cnn/t_p_t_data.csv'
#     data = pd.read_csv(path)
# # data = list(data)
#     rel = data.iloc[i][3]
#     head_drug_name = data.iloc[i][1]
#     file_name = str(head_drug_name) + '.png'
#     dir = 'mol_graph'
#     path = os.path.join(dir, file_name)
#     head_drug_img = cv2.imread(path, flags=1)
#
#     path = 'cnn/t_p_t_data.csv'
#     data = pd.read_csv(path)
#     tail_drug_name = data.iloc[i][2]
#     file_name = str(tail_drug_name) + '.png'
#     dir = 'mol_graph'
#     path = os.path.join(dir, file_name)
#     tail_drug_img = cv2.imread(path, flags=1)
#
#     small_list.append(head_drug_img)
#     small_list.append(tail_drug_img)
#     small_list.append(rel)
#     train_list.append(small_list)
#     a = 1

# cv2.imshow("Demo1", tail_drug_img)
# cv2.imshow("Demo2", head_drug_img)
# cv2.waitKey(10000)
# cv2.destroyALLWindows()


# ntrain_list = []
# path = 'cnn/t_n_t_data.csv'
# data = pd.read_csv(path)
# total_line = len(open(path).readlines()) - 1
# for i in range(0, total_line):
#     path = 'cnn/t_n_t_data.csv'
#     data = pd.read_csv(path)
#     small_list = []
#     negative_drug_list = data.iloc[i][1].split('$')
#     file_name = str(negative_drug_list[0]) + '.png'
#     dir = 'mol_graph'
#     path = os.path.join(dir, file_name)
#     negative_drug_img = cv2.imread(path, flags=1)
#     path = 'cnn/t_p_t_data.csv'
#     data = pd.read_csv(path)
#     rel = data.iloc[i][3]
#     if negative_drug_list[1] == 'h':
#         tail_drug_name = data.iloc[i][2]
#         file_name = str(tail_drug_name) + '.png'
#         dir = 'mol_graph'
#         path = os.path.join(dir, file_name)
#         tail_drug_img = cv2.imread(path, flags=1)
#         small_list.append(negative_drug_img)
#         small_list.append(tail_drug_img)
#         small_list.append(rel)
#     if negative_drug_list[1] == 't':
#         head_drug_name = data.iloc[i][1]
#         file_name = str(head_drug_name) + '.png'
#         dir = 'mol_graph'
#         path = os.path.join(dir, file_name)
#         head_drug_img = cv2.imread(path, flags=1)
#         small_list.append(head_drug_img)
#         small_list.append(negative_drug_img)
#         small_list.append(rel)
#     ntrain_list.append(small_list)


# path = os.path.join('drugbank', '.csv')
# data = pd.read_csv(path)
# length = len(open(path).readlines())-1
# a = str(data.iloc[0, 1]).split('$')
# img_data = os.listdir('mol_graph')


# F=open(r'data/preprocessed/drugbank/drug_data.pkl','rb')

# content=pickle.load(F)


# a = 1



path = 'data/drugbank.tab'
data = pd.read_csv(path, delimiter='\t')

drug_smile_dict = {}
#
for id1, id2, smiles1, smiles2, relation in zip(data['ID1'], data['ID2'], data['X1'], data['X2'],
                                                data['Y']):
    drug_smile_dict[id1] = smiles1
    drug_smile_dict[id2] = smiles2

for id, smiles in drug_smile_dict.items():
    mol = Chem.MolFromSmiles(smiles)
    img = Draw.MolToImage(mol)
    dir = 'mol_graph'
    file_name = str(id) + '.png'
    save_path = os.path.join(dir, file_name)
    img.save(save_path, "PNG")
    # img.show()


