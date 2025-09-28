from datetime import datetime
import numpy as np
import torch
from torch import optim
import models
import time
from tqdm import tqdm
from ddi_datasets import load_ddi_data_fold, total_num_rel
from custom_loss import SigmoidLoss
from custom_metrics import do_compute_metrics
import argparse
from matplotlib import pyplot as plt



parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, required=True, choices=['drugbank', 'twosides'])
parser.add_argument('-fold', '--fold', type=int, required=True, help='Fold on which to train on')
parser.add_argument('-n_iter', '--n_iter', type=int, required=True, help='Number of iterations/')
parser.add_argument('-drop', '--dropout', type=float, default=0, help='dropout probability')
parser.add_argument('-b', '--batch_size', type=int, default=512, help='Batch size')  # 512

args = parser.parse_args()

print(args)

dataset_name = args.dataset
fold_i = args.fold
dropout = args.dropout
n_iter = args.n_iter
TOTAL_NUM_RELS = total_num_rel(name=dataset_name)
batch_size = args.batch_size
data_size_ratio = 1
# device = 'mps' if torch.backends.mps.is_available() else 'cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
hid_feats = 64
rel_total = TOTAL_NUM_RELS
lr = 1e-3
weight_decay = 5e-4
n_epochs =100
kge_feats = 64

def do_compute(model, batch, device): 

        batch = [t.to(device) for t in batch]
        p_score, n_score, n2_score = model(batch)
        assert p_score.ndim == 2
        assert n_score.ndim == 3
        assert n2_score.ndim == 3
        probas_pred = np.concatenate([torch.sigmoid(p_score.detach()).cpu().mean(dim=-1), torch.sigmoid(n_score.detach()).mean(dim=-1).view(-1).cpu(), torch.sigmoid(n2_score.detach()).mean(dim=-1).view(-1).cpu()])
        ground_truth = np.concatenate([np.ones(p_score.shape[0]), np.zeros(n_score.shape[:2]).reshape(-1), np.zeros(n2_score.shape[:2]).reshape(-1)])
#pscore512,nscore512, probas_pred1024.
#groundtruth就是pscore全部填1，nscore全部填0
        return p_score, n_score, n2_score, probas_pred, ground_truth


def run_batch(model, optimizer, data_loader, epoch_i, desc, loss_fn, device):
        total_loss = 0
        loss_pos = 0
        loss_neg = 0
        loss_neg2 = 0
        probas_pred = []
        ground_truth = []
        
        for batch in tqdm(data_loader, desc= f'{desc} Epoch {epoch_i}'):
            p_score, n_score, n2_score, batch_probas_pred, batch_ground_truth = do_compute(model, batch, device)

            probas_pred.append(batch_probas_pred)
            ground_truth.append(batch_ground_truth)

            loss, loss_p, loss_n, loss_n2 = loss_fn(p_score, n_score, n2_score)
            if model.training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            loss_pos += loss_p.item()
            loss_neg += loss_n.item()
            loss_neg2 += loss_n2.item()
        total_loss /= len(data_loader)
        loss_pos /= len(data_loader)
        loss_neg /= len(data_loader)
        loss_neg2 /= len(data_loader)
        
        probas_pred = np.concatenate(probas_pred)
        ground_truth = np.concatenate(ground_truth)
        

        return total_loss, do_compute_metrics(probas_pred, ground_truth)


def print_metrics(loss, acc, auroc, f1_score, precision, recall, int_ap, ap, p, r, fpr, tpr, thresholds,cm ):
    print(f'loss: {loss:.4f}, acc: {acc:.4f}, roc: {auroc:.4f}, f1: {f1_score:.4f}, ', end='')
    print(f'p: {precision:.4f}, r: {recall:.4f}, int-ap: {int_ap:.4f}, ap: {ap:.4f}')  

    return f1_score

    
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True,
                          i=0
                         ):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions
    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig(f"confusion matrix{i}.jpg",bbox_inches = 'tight')
    # plt.clf() 
    plt.gcf().clear()

def train(model, train_data_loader, val_data_loader, test_data_loader, loss_fn, optimizer, n_epochs, device, scheduler):
    # bestscore = 0
    
    
    train_acc = []
    loss = []
    val_acc = []
    auroc = []
    f1 = []
    precision = []
    recall = []
    ap = []
    

    for epoch_i in range(1, n_epochs+1):
        start = time.time()
        model.train()
        ## Training
        train_loss, train_metrics = run_batch(model, optimizer, train_data_loader, epoch_i,  'train', loss_fn, device)
        train_acc.append(train_metrics[0]*100)

        if scheduler:
            scheduler.step()

        model.eval()
        with torch.no_grad():

            ## Validation 
            if val_data_loader:
                val_loss , val_metrics = run_batch(model, optimizer, val_data_loader, epoch_i, 'val', loss_fn, device)
                loss.append(val_loss)
                val_acc.append(val_metrics[0]*100)
                auroc.append(val_metrics[1])
                f1.append(val_metrics[2])
                precision.append(val_metrics[3])
                recall.append(val_metrics[4])
                ap.append(val_metrics[6])
                p = val_metrics[7]
                r = val_metrics[8]
                fpr = val_metrics[9]
                tpr = val_metrics[10]
                thresholds = val_metrics[11]
                cm = val_metrics[12]
                # ROC Curve图像
                plt.plot(fpr, tpr, lw=2, alpha=0.3, label=f'ROC Curve (AUC = {val_metrics[5]})')
                plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'black')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC')
                plt.legend(loc="lower right")
                plt.savefig(f"ROC Curve{epoch_i}.jpg",bbox_inches = 'tight')
                # plt.clf() 
                plt.gcf().clear()
                
                # precision-recall curve
                plt.plot(r, p, lw=3, alpha=0.3, label=f'PR Curve (AUC = {val_metrics[5]})')
                plt.title('Precision-Recall Curve')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.legend(loc="lower right")
                plt.savefig(f"precision-recall Curve{epoch_i}.jpg",bbox_inches = 'tight')
                # plt.clf() 
                plt.gcf().clear()
                
                # confusion matrix
                plot_confusion_matrix(cm = cm, 
                                      normalize    = False,
                                      target_names = ['0', '1'],
                                      title        = "Confusion Matrix",
                                      i = epoch_i
                                     )
                plot_confusion_matrix(cm = cm, 
                                      normalize    = True,
                                      target_names = ['0', '1'],
                                      title        = "Confusion Matrix Normalized",
                                      i = epoch_i
                                     )
            
                
                
                
                
        # if bestscore <= val_metrics[0]:
        #     bestscore = val_metrics[0]
        #     model_save_path = f"./best_acc{val_metrics[0]}.pt"
        #     torch.save(model.state_dict(), model_save_path)

        if train_data_loader:
            print(f'\n#### Epoch time {time.time() - start:.4f}s')
            print_metrics(train_loss, *train_metrics)

        if val_data_loader:
            print('#### Validation')
            print_metrics(val_loss, *val_metrics)

                 
    print("loss:", loss) 
    print("train_acc", train_acc)
    print("val_acc:", val_acc)
    print("p:", p)
    print("r", r)
    print("ap", ap)
    print("f1:", f1)
    print("auroc:", auroc)
    print("fpr:", fpr)
    print("tpr:", tpr)
    print("cm", cm)

    
    # acc图像
    plt.plot(train_acc, lw=2, alpha=0.3, label='Train accuracy')
    plt.plot(val_acc, lw=2, alpha=0.3,label='valid accuracy')
    plt.title('accuracy')
    plt.ylabel('accuracy unit:%')
    plt.xlabel('epoch#')
    plt.legend(loc="lower right")
    plt.savefig("accuracy.jpg",bbox_inches = 'tight')
    # plt.clf()
    plt.gcf().clear()
    
#     plt.plot(loss)
#     plt.title('validation loss')
#     plt.xlabel('epoch#')
#     plt.savefig("loss.jpg")

#     plt.plot(roc)
#     plt.title('validation ROC')
#     plt.xlabel('epoch#')
#     plt.savefig("roc.jpg")

    plt.plot(f1)
    plt.title('validation F1 Score')
    plt.xlabel('epoch#')
    plt.savefig("F1.jpg",bbox_inches = 'tight')
    # plt.clf() 
    plt.gcf().clear()
    
    
#     plt.plot(p)
#     plt.title('Validation Precision')
#     plt.xlabel('epoch#')
#     plt.savefig("Precision.jpg")

#     plt.plot(r)
#     plt.title('Validation Recall')
#     plt.xlabel('epoch#')
#     plt.savefig("recall.jpg")

    plt.plot(ap)
    plt.title('Validation Average Precision')
    plt.xlabel('epoch#')
    plt.savefig("average_precision.jpg",bbox_inches = 'tight')
    # plt.clf() 
    plt.gcf().clear()
    
    plt.plot(auroc)
    plt.title('Validation AUROC')
    plt.xlabel('epoch#')
    plt.savefig("auroc.jpg",bbox_inches = 'tight')


def test(model, val_data_loader, test_data_loader, device):
    model.eval()
    with torch.no_grad():

        ## Validation
        if val_data_loader:
            probas_pred = []
            ground_truth = []

            for batch in tqdm(val_data_loader):
                p_score, n_score, n2_score, batch_probas_pred, batch_ground_truth = do_compute(model, batch, device)

                probas_pred.append(batch_probas_pred)
                ground_truth.append(batch_ground_truth)

            probas_pred = np.concatenate(probas_pred)
            ground_truth = np.concatenate(ground_truth)

            val_metrics = do_compute_metrics(probas_pred, ground_truth)
            val_loss = 0
            print_metrics(val_loss, *val_metrics)



mode = "train"

train_data_loader, val_data_loader, test_data_loader, NUM_FEATURES, NUM_EDGE_FEATURES = \
    load_ddi_data_fold(dataset_name, fold_i, batch_size=batch_size, data_size_ratio=data_size_ratio)

GmpnnNet = models.GmpnnCSNetDrugBank if dataset_name == 'drugbank' else models.GmpnnCSNetTwosides

model = GmpnnNet(NUM_FEATURES, NUM_EDGE_FEATURES, hid_feats, rel_total, n_iter, dropout)
print(model)

loss_fn = SigmoidLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.96 ** (epoch))

time_stamp = f'{datetime.now()}'.replace(':', '_')


model.to(device=device)
print(f'Training on {device}.')
print(f'Starting fold_{fold_i} at', datetime.now())
if mode == "train":
    train(model, train_data_loader, val_data_loader, test_data_loader, loss_fn, optimizer, n_epochs, device, scheduler)
else:
    load_model_weight_path = ""
    if load_model_weight_path:
        model.load_state_dict(torch.load(load_model_weight_path))
    test(model, val_data_loader, test_data_loader, device)

