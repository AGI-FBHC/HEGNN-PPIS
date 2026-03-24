import sys
import os

import numpy as np
import pandas as pd
from torch.autograd import Variable
from sklearn import metrics
from sklearn.model_selection import KFold
from HEGNNPPIS_mode import *
from dataloader import *
import time
from torch.utils.data import DataLoader
import dhg


# Path
Dataset_Path = "./Dataset/"
Model_Path = "./Model/"
Log_path = "./Log/"
Test_path = './Model/model/'
model_time = None

SEED = 2020
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.cuda.manual_seed(SEED)


def train_one_epoch(model, data_loader):
    epoch_loss_train = 0.0
    n = 0
    alpha = 0.0
    for data in data_loader:
        model.optimizer.zero_grad()
        sequence_name, labels, node_features, virtual_node_features, pos, virtual_pos, edge_index, A2V_edge_index, V2A_edge_index, hypergraph = data

        if torch.cuda.is_available():
            node_features = Variable(node_features.cuda().float())
            virtual_node_features = Variable(virtual_node_features.cuda().float())
            edge_index = Variable(edge_index.cuda().long())
            A2V_edge_index = Variable(A2V_edge_index.cuda().long())
            V2A_edge_index = Variable(V2A_edge_index.cuda().long())
            y_true = Variable(labels.cuda())
            pos = Variable(pos.cuda().float())
            virtual_pos = Variable(virtual_pos.cuda().float())
            hypergraph = hypergraph[0]

        y_true = torch.squeeze(y_true)
        y_true = y_true.long()
        y_pred = model(node_features, pos, virtual_node_features, virtual_pos, edge_index, A2V_edge_index, V2A_edge_index, hypergraph)

        loss = model.criterion(y_pred, y_true)

        loss.backward()
        model.optimizer.step()
        epoch_loss_train += loss.item()
        n += 1
    epoch_loss_train_avg = epoch_loss_train / n
    return epoch_loss_train_avg


def train(model, train_dataframe, valid_dataframe, fold=0):
    train_loader = DataLoader(dataset=ProDataset(train_dataframe, hypernodes=Config.hypernodes),
                              batch_size=Config.batch_size, shuffle=True,
                              num_workers=4,
                              collate_fn=graph_collate,
                              persistent_workers=True, pin_memory=True)

    valid_loader = DataLoader(dataset=ProDataset(dataframe=valid_dataframe,
                                                      hypernodes=Config.hypernodes),
                             batch_size=Config.batch_size,
                             shuffle=True, num_workers=4, collate_fn=graph_collate,
                             persistent_workers=True, pin_memory=True)

    best_epoch = 0
    best_val_auc = 0
    best_val_aupr = 0
    for epoch in range(Config.epochs):
        print("\n========== Train epoch " + str(epoch + 1) + " ==========")
        model.train()
        time1 = time.time()
        _ = train_one_epoch(model, train_loader)
        print("========== Evaluate Valid set ==========")
        epoch_loss_valid_avg, valid_true, valid_pred, _ = evaluate(model, valid_loader)
        result_valid = analysis(valid_true, valid_pred, 0.5)
        print("Valid loss: ", epoch_loss_valid_avg)
        print("Valid AUC: ", result_valid['AUC'])
        print("Valid AUPRC: ", result_valid['AUPRC'])
        if best_val_aupr < result_valid['AUPRC']:
            best_epoch = epoch + 1
            best_val_auc = result_valid['AUC']
            best_val_aupr = result_valid['AUPRC']
            torch.save(model.state_dict(), os.path.join(Model_Path, 'fold' + str(fold) + '_best_model.pkl'))
        model.scheduler.step(result_valid['AUPRC'])
        time2 = time.time()
        print('one epoch cost :', time2 - time1)
    return best_epoch, best_val_auc, best_val_aupr


def evaluate(model, data_loader):
    model.eval()
    epoch_loss = 0.0
    n = 0
    valid_pred = []
    valid_true = []
    pred_dict = {}
    for data in data_loader:
        with torch.no_grad():
            sequence_names, labels, node_features, hyper_node_features, pos, hyper_pos, edge_index, A2V_edge_index, V2A_edge_index, hypergraph = data
            if torch.cuda.is_available():
                node_features = Variable(node_features.cuda().float())
                hyper_node_features = Variable(hyper_node_features.cuda().float())
                edge_index = Variable(edge_index.cuda().long())
                A2V_edge_index = Variable(A2V_edge_index.cuda().long())
                V2A_edge_index = Variable(V2A_edge_index.cuda().long())
                y_true = Variable(labels.cuda())
                pos = Variable(pos.cuda().float())
                hyper_pos = Variable(hyper_pos.cuda().float())
                hypergraph = hypergraph[0]

            y_true = torch.squeeze(y_true)
            y_true = y_true.long()
            y_pred = model(node_features, pos, hyper_node_features, hyper_pos,
                                                       edge_index, A2V_edge_index, V2A_edge_index, hypergraph)
            loss = model.criterion(y_pred, y_true)
            softmax = torch.nn.Softmax(dim=1)
            y_pred = softmax(y_pred)
            y_pred = y_pred.cpu().detach().numpy()
            y_true = y_true.cpu().detach().numpy()

            valid_pred += [pred[1] for pred in y_pred]
            valid_true += list(y_true)
            pred_dict[sequence_names[0]] = [pred[1] for pred in y_pred]
            epoch_loss += loss.item()
            n += 1
    epoch_loss_avg = epoch_loss / n
    return epoch_loss_avg, valid_true, valid_pred, pred_dict


def analysis(y_true, y_pred, best_threshold=None):
    if best_threshold == None:
        best_f1 = 0
        best_threshold = 0
        for threshold in range(0, 100):
            threshold = threshold / 100
            binary_pred = [1 if pred >= threshold else 0 for pred in y_pred]
            binary_true = y_true
            f1 = metrics.f1_score(binary_true, binary_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

    binary_pred = [1 if pred >= best_threshold else 0 for pred in y_pred]
    binary_true = y_true

    # binary evaluate
    binary_acc = metrics.accuracy_score(binary_true, binary_pred)
    precision = metrics.precision_score(binary_true, binary_pred)
    recall = metrics.recall_score(binary_true, binary_pred)
    f1 = metrics.f1_score(binary_true, binary_pred)
    AUC = metrics.roc_auc_score(binary_true, y_pred)
    precisions, recalls, thresholds = metrics.precision_recall_curve(binary_true, y_pred)
    AUPRC = metrics.auc(recalls, precisions)
    mcc = metrics.matthews_corrcoef(binary_true, binary_pred)

    results = {
        'binary_acc': binary_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'AUC': AUC,
        'AUPRC': AUPRC,
        'mcc': mcc,
        'threshold': best_threshold
    }
    return results


def xtest(test_dataframe, psepos_path):
    print("testing------------------------------")
    test_loader = DataLoader(dataset=ProDataset(dataframe=test_dataframe, psepos_path=psepos_path,
                                                      hypernodes=Config.hypernodes),
                             batch_size=Config.batch_size,
                             shuffle=False, num_workers=1, collate_fn=graph_collate,
                             persistent_workers=True, pin_memory=True)
    for model_name in sorted(os.listdir(Test_path)):
        print(model_name)
        model = HEGNNPPIS(in_dim=67, in_edge_dim=1, hidden_dim=67, layers=4)

        if torch.cuda.is_available():
            model.cuda()
        model.load_state_dict(torch.load(Test_path + model_name, map_location='cuda:0'))
        epoch_loss_test_avg, test_true, test_pred, pred_dict = evaluate(model, test_loader)
        result_test = analysis(test_true, test_pred)
        print("========== Evaluate Test set ==========")
        print("Test loss: ", epoch_loss_test_avg)
        print("Test binary acc: ", result_test['binary_acc'])
        print("Test precision:", result_test['precision'])
        print("Test recall: ", result_test['recall'])
        print("Test f1: ", result_test['f1'])
        print("Test AUC: ", result_test['AUC'])
        print("Test AUPRC: ", result_test['AUPRC'])
        print("Test mcc: ", result_test['mcc'])
        print("Threshold: ", result_test['threshold'])


def generate_dataframe(dataset):
    IDs, sequences, labels = [], [], []
    for ID in dataset:
        IDs.append(ID)
        item = dataset[ID]
        sequences.append(item[0])
        labels.append(item[1])
    test_dic = {"ID": IDs, "sequence": sequences, "label": labels}
    test_dataframe = pd.DataFrame(test_dic)
    return test_dataframe


def main():
    if not os.path.exists(Log_path): os.makedirs(Log_path)
    with open("./Dataset/Test_60.pkl", "rb") as f:
        Test_60 = pickle.load(f)
    with open(Config.dataset_path + "Test_315-28.pkl", "rb") as f:
        Test_315_28 = pickle.load(f)
    with open(Config.dataset_path + "UBtest_31-6.pkl", "rb") as f:
        UBtest_31_6 = pickle.load(f)
    Btest_31_6 = {}
    with open(Config.dataset_path + "bound_unbound_mapping31-6.txt", "r") as f:
        lines = f.readlines()[1:]
    for line in lines:
        bound_ID, unbound_ID, _ = line.strip().split()
        Btest_31_6[bound_ID] = Test_60[bound_ID]


    Test_60 = generate_dataframe(Test_60)
    Test_315_28 = generate_dataframe(Test_315_28)
    Btest_31_6 = generate_dataframe(Btest_31_6)
    UBtest_31_6 = generate_dataframe(UBtest_31_6)
    Test60_psepos_Path = './Feature/psepos/Test60_psepos_SC.pkl'
    Test315_28_psepos_Path = './Feature/psepos/Test315-28_psepos_SC.pkl'
    Btest31_psepos_Path = './Feature/psepos/Test60_psepos_SC.pkl'
    UBtest31_28_psepos_Path = './Feature/psepos/UBtest31-6_psepos_SC.pkl'

    print("Evaluate HEGNN-PPIS on Test_60")
    xtest(Test_60, Test60_psepos_Path)

    print("Evaluate HEGNN-PPIS on Test_315-28")
    xtest(Test_315_28, Test315_28_psepos_Path)

    print("Evaluate HEGNN-PPIS on Btest_31-6")
    xtest(Btest_31_6, Btest31_psepos_Path)

    print("Evaluate HEGNN-PPIS on UBtest_31-6")
    xtest(UBtest_31_6, UBtest31_28_psepos_Path)


if __name__ == "__main__":
    main()