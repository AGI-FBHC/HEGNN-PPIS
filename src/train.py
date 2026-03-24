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


def generate_dataset():
    init()
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

    test_data_frame, psepos_path = {}, {}
    if Config.test_type == 1:
        test_data_frame = generate_dataframe(Test_60)
        psepos_path = Config.Test60_psepos_path
    elif Config.test_type == 2:
        test_data_frame = generate_dataframe(Test_315_28)
        psepos_path = Config.Test315_28_psepos_path
    elif Config.test_type == 4:
        test_data_frame = generate_dataframe(UBtest_31_6)
        psepos_path = Config.UBtest31_28_psepos_path
    elif Config.test_type == 3:
        test_data_frame = generate_dataframe(Btest_31_6)
        psepos_path = Config.Btest31_psepos_path
    return test_data_frame, psepos_path


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

def cross_validation(all_dataframe, fold_number=5, layers=4):
    sequence_names = all_dataframe['ID'].values
    sequence_labels = all_dataframe['label'].values
    kfold = KFold(n_splits=fold_number, shuffle=True)
    fold = 0
    best_epochs = []
    valid_aucs = []
    valid_auprs = []

    for train_index, valid_index in kfold.split(sequence_names, sequence_labels):
        print("\n\n========== Fold " + str(fold + 1) + " ==========")
        train_dataframe = all_dataframe.iloc[train_index, :]
        valid_dataframe = all_dataframe.iloc[valid_index, :]
        print("Train on", str(train_dataframe.shape[0]), "samples, validate on", str(valid_dataframe.shape[0]),
              "samples")

        model = HEGNNPPIS(in_dim=67, in_edge_dim=1, hidden_dim=67, layers=layers)
        if torch.cuda.is_available():
            model.cuda()
        best_epoch, valid_auc, valid_aupr = train(model, train_dataframe, valid_dataframe, fold + 1)
        best_epochs.append(str(best_epoch))
        valid_aucs.append(valid_auc)
        valid_auprs.append(valid_aupr)
        fold += 1
    print("\n\nBest epoch: " + " ".join(best_epochs))
    print("Average AUC of {} fold: {:.4f}".format(fold_number, sum(valid_aucs) / fold_number))
    print("Average AUPR of {} fold: {:.4f}".format(fold_number, sum(valid_auprs) / fold_number))
    return round(sum([int(epoch) for epoch in best_epochs]) / fold_number)


def train_full_model(all_dataframe, test_dataframe, test_psepos_path, layers=4):
    print("\nTraining a full model using all training data...\n")
    model = HEGNNPPIS(in_dim=67, in_edge_dim=1, hidden_dim=67, layers=layers)

    if torch.cuda.is_available():
        model.cuda()

    train_loader = DataLoader(dataset=ProDataset(all_dataframe, hypernodes=Config.hypernodes), batch_size=Config.batch_size, shuffle=True,
                              num_workers=4,
                              collate_fn=graph_collate,
                              persistent_workers=True, pin_memory=True)

    test_loader = DataLoader(dataset=ProDataset(dataframe=test_dataframe, psepos_path=test_psepos_path,
                                                      hypernodes=Config.hypernodes),
                             batch_size=Config.batch_size,
                             shuffle=True, num_workers=4, collate_fn=graph_collate,
                             persistent_workers=True, pin_memory=True)
    ans = 0
    best_auprc = -1
    for epoch in range(Config.epochs):
        print("\n========== Train epoch " + str(epoch + 1) + " ==========")
        model.train()
        time1 = time.time()
        epoch_loss_train_avg = train_one_epoch(model, train_loader)
        epoch_loss_test_avg, test_true, test_pred, _ = evaluate(model, test_loader)
        result_test = analysis(test_true, test_pred, 0.5)
        print("Test loss: ", epoch_loss_test_avg)
        print("Test AUC: ", result_test['AUC'])
        print("Test AUPRC: ", result_test['AUPRC'])
        time2 = time.time()
        print('full cost : ', time2 - time1)
        ans += time2 - time1
        if result_test['AUPRC'] > best_auprc:
            best_auprc = result_test['AUPRC']
            torch.save(model.state_dict(), os.path.join(Model_Path, 'best_model_{}.pkl'.format(epoch + 1)))
    print(ans)


def xtest(test_dataframe, psepos_path):
    print("testing------------------------------")
    test_loader = DataLoader(dataset=ProDataset(dataframe=test_dataframe, psepos_path=psepos_path,
                                                      hypernodes=Config.hypernodes),
                             batch_size=Config.batch_size,
                             shuffle=False, num_workers=1, collate_fn=graph_collate,
                             persistent_workers=True, pin_memory=True)
    print(Test_path)
    for model_name in sorted(os.listdir(Test_path)):
        print(model_name)
        model = HEGNNPPIS(in_dim=67, in_edge_dim=1, hidden_dim=67, layers=4)

        if torch.cuda.is_available():
            model.cuda()
        model.load_state_dict(torch.load(Test_path + model_name, map_location='cuda:0'))
        time1 = time.time()
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
        time2 = time.time()
        print(time2 - time1)

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, 'ab', buffering=0)

    def write(self, message):
        self.terminal.write(message)
        try:
            self.log.write(message.encode('utf-8'))
        except ValueError:
            pass

    def close(self):
        self.log.close()
        sys.stdout = self.terminal

    def flush(self):
        pass


def main(layer=2):
    if not os.path.exists(Log_path): os.makedirs(Log_path)
    with open(Dataset_Path + "Train_335.pkl", "rb") as f:
        Train_335 = pickle.load(f)
        Train_335.pop('2j3rA')  # remove the protein with error sequence in the train dataset

    with open(Dataset_Path + "Test_60.pkl", "rb") as f:
        Test_60 = pickle.load(f)
    IDs, sequences, labels = [], [], []
    for ID in Train_335:
        IDs.append(ID)
        item = Train_335[ID]
        sequences.append(item[0])
        labels.append(item[1])

    train_dic = {"ID": IDs, "sequence": sequences, "label": labels}
    train_dataframe = pd.DataFrame(train_dic)
    IDs, sequences, labels = [], [], []
    for ID in Test_60:
        IDs.append(ID)
        item = Test_60[ID]
        sequences.append(item[0])
        labels.append(item[1])
    test_dic = {"ID": IDs, "sequence": sequences, "label": labels}
    test_dataframe, pos = generate_dataset()
    # cross_validation
    # iter = cross_validation(train_dataframe, layers=Config.layers)
    # full
    train_full_model(train_dataframe, test_dataframe, pos, layers=Config.layers)


if __name__ == "__main__":
    if model_time is not None:
        checkpoint_path = os.path.normpath(Log_path + "/" + model_time)
    else:
        localtime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        checkpoint_path = os.path.normpath(Log_path + "/" + localtime)
        os.makedirs(checkpoint_path)
    Model_Path = os.path.normpath(checkpoint_path + '/model')
    if not os.path.exists(Model_Path): os.makedirs(Model_Path)

    sys.stdout = Logger(os.path.normpath(checkpoint_path + '/train.log'))

    main()
    sys.stdout.log.close()
