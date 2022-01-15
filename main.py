import os.path

import torch
import warnings

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from data import German
from args import argument_parser
from model import GCN, Encoder, Predictor, Classifier
from utils import stat_parity_and_equal, sim_loss, sp2sptensor
from sklearn.metrics import f1_score, roc_auc_score

warnings.filterwarnings('ignore')

def vanilla_gcn(args, device, model, optimizer, datagenerator, data, A, labels, loss_fn, trainIdx, valIdx, testIdx, sen_vals):

    best_loss = 100

    for epoch in range(args.num_epoch+1):

        # train
        model.train()
        optimizer.zero_grad()

        out = model(data, A)
        loss = loss_fn(out[trainIdx], labels[trainIdx])

        loss.backward()
        optimizer.step()

        # validation
        model.eval()
        val_out = model(data, A)
        val_loss = loss_fn(val_out[valIdx], labels[valIdx])

        f1 = f1_score(labels[valIdx].cpu().detach().numpy(), torch.argmax(val_out[valIdx], dim=1).cpu().detach().numpy())
        auroc = roc_auc_score(labels[valIdx].cpu().detach().numpy(), F.softmax(val_out[valIdx]).cpu().detach().numpy()[:, 1])
        sp_score, eo_score = stat_parity_and_equal(sen_vals,
                                                   val_out.detach().cpu().numpy(),
                                                   labels.detach().cpu().numpy(),
                                                   valIdx.detach().cpu().numpy())
        if val_loss.item() < best_loss:
            best_loss = val_loss.item()
            torch.save(model.state_dict(), 'best_model.pt')
        if epoch % 100 == 0 and epoch > 0:
            print(f"Epoch {epoch}: val loss: {val_loss.item():.4f}, f1: {100*f1:.2f}, auroc: {100*auroc:.2f}, sp: {100*sp_score:.2f}, eo: {100*eo_score:.2f}")

    # test
    model.load_state_dict(torch.load('best_model.pt'))
    model.eval()

    cf_data = datagenerator.generate_counterfactual_perturbation(data=data).to(device)
    noise_data = datagenerator.generate_node_perturbation(prob=args.data_perturb, sen=False).to(device)

    test_out = model(data, A)
    cf_out = model(cf_data, A)
    noise_out = model(noise_data, A)

    f1 = f1_score(labels[testIdx].cpu().detach().numpy(), torch.argmax(test_out[testIdx], dim=1).cpu().detach().numpy())
    auroc = roc_auc_score(labels[testIdx].cpu().detach().numpy(), F.softmax(test_out[testIdx]).cpu().detach().numpy()[:, 1])
    sp_score, eo_score = stat_parity_and_equal(sen_vals,
                                               test_out.detach().cpu().numpy(),
                                               labels.detach().cpu().numpy(),
                                               testIdx.detach().cpu().numpy())
    unfair = 1.0 - torch.argmax(test_out, dim=1).eq(torch.argmax(cf_out, dim=1))[testIdx].sum().item()/testIdx.shape[0]
    instab = 1.0 - torch.argmax(test_out, dim=1).eq(torch.argmax(noise_out, dim=1))[testIdx].sum().item()/testIdx.shape[0]

    print(f"================================ TEST RESULTS ================================")
    print(f"F1: {100*f1:.2f}\nAUROC: {100*auroc:.2f}\nSP: {100*sp_score:.2f}\nEO: {100*eo_score:.2f}\nUnfairness: {100*unfair:.2f}\nInstability: {100*instab:.2f}")


def nifty_gcn(args, device, encoder, predictor, classifier, optimizer, datagenerator, data, A, labels, loss_fn, trainIdx, valIdx, testIdx, sen_vals):

    def calculate(noise_data, noise_A, cf_data, cf_A, encoder, predictor, classifier, coef, idx):
        # encoder
        noise_enco = encoder(noise_data, noise_A)
        cf_enco = encoder(cf_data, cf_A)

        # prediction
        noise_pre = predictor(noise_enco)
        cf_pre = predictor(cf_enco)

        # cls
        noise_cls = classifier(noise_enco)
        cf_cls = classifier(cf_enco)

        # sim_loss
        similarity = sim_loss(emb_1=noise_enco[idx], pred_1=noise_pre[idx], emb_2=cf_enco[idx], pred_2=cf_pre[idx])

        # cls loss
        noise_cls_loss = loss_fn(noise_cls[idx], labels[idx])
        cf_cls_loss = loss_fn(cf_cls[idx], labels[idx])
        cls_loss = 0.5 * (noise_cls_loss + cf_cls_loss)

        if args.sim:
            loss = coef * similarity + (1.0 - coef) * cls_loss
        else:
            loss = cls_loss
        return loss, noise_cls, cf_cls

    best_loss = 100
    # train data TO-DO: the generation takes too long, this part could be optimized
    if not os.path.exists(os.path.join(args.data_path, 'tmp')):
        os.makedirs(os.path.join(args.data_path, 'tmp'))

    datafiles = os.listdir(os.path.join(args.data_path, 'tmp'))
    suffix = []
    for file in datafiles:
        suffix.append(os.path.splitext(file)[1])

    if '.npy' in suffix:
        print("Data exist, Loading...")
        train_noise_data = np.load(os.path.join(args.data_path, 'tmp', 'train_noise.npy'), allow_pickle=True)
        train_noise_A_data = np.load(os.path.join(args.data_path, 'tmp', 'train_noise_A.npy'), allow_pickle=True)
        train_cf_data = np.load(os.path.join(args.data_path, 'tmp', 'train_cf.npy'), allow_pickle=True)
        train_cf_A_data = np.load(os.path.join(args.data_path, 'tmp', 'train_cf_A.npy'), allow_pickle=True)
        #print(type(train_noise_data), train_noise_data.shape, train_noise_data)

    else:
        print("Generating training data and validation data...")
        train_noise = []
        train_noise_A = []
        train_cf = []
        train_cf_A = []
        for _ in tqdm(range(args.num_epoch+1)):
            train_noise.append(datagenerator.generate_node_perturbation(prob=args.data_perturb, sen=False).numpy())
            train_noise_A.append(torch.tensor(datagenerator.generate_struc_perturbation(drop_prob=args.struc_perturb, tensor=False), dtype=torch.float32))
            train_cf.append(datagenerator.generate_counterfactual_perturbation(data.cpu()).numpy())
            train_cf_A.append(torch.tensor(datagenerator.generate_struc_perturbation(drop_prob=args.struc_perturb, tensor=False), dtype=torch.float32))
        print("Saving data as .npy files...")
        np.save(os.path.join(args.data_path, 'tmp', 'train_noise.npy'), np.array(train_noise))
        np.save(os.path.join(args.data_path, 'tmp', 'train_noise_A.npy'), np.array(train_noise_A))
        np.save(os.path.join(args.data_path, 'tmp', 'train_cf.npy'), np.array(train_cf))
        np.save(os.path.join(args.data_path, 'tmp', 'train_cf_A.npy'), np.array(train_cf_A))
        print("Successfully saving!")

        train_noise_data = train_noise
        train_noise_A_data = train_noise_A
        train_cf_data = train_cf
        train_cf_A_data = train_cf_A

    # val data
    val_noise_data = datagenerator.generate_node_perturbation(prob=args.data_perturb, sen=False).to(device)
    val_noise_A = torch.tensor(datagenerator.generate_struc_perturbation(drop_prob=args.struc_perturb), dtype=torch.float32).to(device)
    val_cf_data = datagenerator.generate_counterfactual_perturbation(data=data).to(device)
    val_cf_A = torch.tensor(datagenerator.generate_struc_perturbation(drop_prob=args.struc_perturb), dtype=torch.float32).to(device)

    for epoch in range(args.num_epoch+1):

        # lipschitz normalization
        if args.lipschitz:
            encoder.lipschitz_norm()
            #predictor.lipschitz_norm()

        # train
        encoder.train()
        predictor.train()
        classifier.train()

        optimizer.zero_grad()

        # generate data
        #print(train_noise[epoch])

        noise_data = torch.tensor(train_noise_data[epoch], dtype=torch.float32).to(device)
        noise_A = torch.tensor(sp2sptensor(train_noise_A_data[epoch]), dtype=torch.float32).to(device)
        cf_data = torch.tensor(train_cf_data[epoch], dtype=torch.float32).to(device)
        # cf_data = datagenerator.generate_node_perturbation(prob=args.data_perturb, sen=True).to(device)
        cf_A = torch.tensor(sp2sptensor(train_cf_A_data[epoch]), dtype=torch.float32).to(device)

        # train
        loss, _, _ = calculate(noise_data=noise_data,
                               noise_A=noise_A,
                               cf_data=cf_data,
                               cf_A=cf_A,
                               encoder=encoder,
                               predictor=predictor,
                               classifier=classifier,
                               idx=trainIdx,
                               coef=args.coef)
        loss.backward()
        optimizer.step()

        # validation
        encoder.eval()
        predictor.eval()
        classifier.eval()

        val_loss, noise_out, _ = calculate(noise_data=val_noise_data,
                                           noise_A=val_noise_A,
                                           cf_data=val_cf_data,
                                           cf_A=val_cf_A,
                                           encoder=encoder,
                                           predictor=predictor,
                                           classifier=classifier,
                                           coef=args.coef,
                                           idx=valIdx)

        f1 = f1_score(y_true=labels[valIdx].detach().cpu().numpy(),
                      y_pred=torch.argmax(noise_out[valIdx], dim=1).detach().cpu().numpy())
        auroc = roc_auc_score(y_true=labels[valIdx].detach().cpu().numpy(),
                              y_score=F.softmax(noise_out[valIdx]).detach().cpu().numpy()[:, 1])
        sp_score, eo_score = stat_parity_and_equal(sen_vals,
                                                   noise_out.detach().cpu().numpy(),
                                                   labels.detach().cpu().numpy(),
                                                   valIdx.detach().cpu().numpy())
        if val_loss.item() < best_loss:
            best_loss = val_loss.item()
            torch.save({'encoder': encoder.state_dict(),
                        'predictor': predictor.state_dict(),
                        'classifier': classifier.state_dict()},
                       'best_nifty_model.pt')
        if epoch % 100 == 0 and epoch > 0:
            #print(f"Epoch {epoch}: val loss: {val_loss.item():.4f}")
            print(f"Epoch {epoch}: val loss: {val_loss.item():.4f}, f1: {100*f1:.2f}, auroc: {100*auroc:.2f}, sp: {100*sp_score:.2f}, eo: {100*eo_score:.2f}")

    # test
    models = torch.load('best_nifty_model.pt')
    encoder.load_state_dict(models['encoder'])
    predictor.load_state_dict(models['predictor'])
    classifier.load_state_dict(models['classifier'])

    encoder.eval()
    predictor.eval()
    classifier.eval()

    test_cf_out = classifier(encoder(datagenerator.generate_counterfactual_perturbation(data).to(device), A))
    test_noise_out = classifier(encoder(datagenerator.generate_node_perturbation(prob=args.data_perturb, sen=False).to(device), A))
    normal_out = classifier(encoder(data, A))
    #print(torch.argmax(normal_out, dim=1))
    f1 = f1_score(y_true=labels[testIdx].detach().cpu().numpy(),
                  y_pred=torch.argmax(normal_out[testIdx], dim=1).detach().cpu().numpy())
    auroc = roc_auc_score(y_true=labels[testIdx].detach().cpu().numpy(),
                          y_score=F.softmax(normal_out[testIdx]).detach().cpu().numpy()[:, 1])
    sp_score, eo_score = stat_parity_and_equal(sen_val=sen_vals,
                                               out=normal_out.detach().cpu().numpy(),
                                               labels=labels.detach().cpu().numpy(),
                                               idx=testIdx.detach().cpu().numpy())

    unfair = 1.0 - torch.argmax(normal_out, dim=1).eq(torch.argmax(test_cf_out, dim=1))[testIdx].sum().item() / testIdx.shape[0]
    instab = 1.0 - torch.argmax(normal_out, dim=1).eq(torch.argmax(test_noise_out, dim=1))[testIdx].sum().item() / testIdx.shape[0]
    print(f"================================ TEST RESULTS ================================")
    print(f"F1: {100*f1:.2f}\nAUROC: {100*auroc:.2f}\nSP: {100*sp_score:.2f}\nEO: {100*eo_score:.2f}\nUnfairness: {100*unfair:.2f}\nInstability: {100*instab:.2f}")


if __name__ == '__main__':

    args = argument_parser().parse_args()

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #print(torch.cuda.is_available())

    # data
    datagenerator = German(path=args.data_path)
    data, A, labels = datagenerator.get_raw_data()
    trainIdx, valIdx, testIdx = datagenerator.get_index()

    data = data.to(device)
    A = torch.tensor(A, dtype=torch.float32).to(device)
    labels = labels.to(device)
    trainIdx = trainIdx.to(device)
    valIdx = valIdx.to(device)
    testIdx = testIdx.to(device)
    sen_vals = np.array(datagenerator.sen_vals)

    # loss fn
    criterion = nn.CrossEntropyLoss()

    if args.model == 'gcn':
        model = GCN(in_channel=data.shape[1],
                    num_hidden=args.num_hidden,
                    out_channel=args.num_class,
                    dropout=args.dropout).to(device)

        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr,
                                     weight_decay=args.weight_decay)

        vanilla_gcn(args=args,
                    model=model,
                    device=device,
                    optimizer=optimizer,
                    datagenerator=datagenerator,
                    data=data, A=A, labels=labels,
                    loss_fn=criterion,
                    trainIdx=trainIdx,
                    valIdx=valIdx,
                    testIdx=testIdx,
                    sen_vals=sen_vals)

    elif args.model == 'niftygcn':
        encoder = Encoder(in_channel=data.shape[1],
                          num_hidden=args.num_hidden,
                          out_channel=args.num_hidden,
                          dropout=args.dropout).to(device)

        predictor = Predictor(in_channel=args.num_hidden,
                              num_hidden=args.num_hidden,
                              out_channel=args.num_hidden).to(device)

        classifier = Classifier(in_channel=args.num_hidden,
                                num_class=args.num_class).to(device)

        optimizer = torch.optim.Adam([{'params': encoder.parameters(), 'lr': args.lr},
                                      {'params': predictor.parameters(), 'lr': args.lr},
                                      {'params': classifier.parameters(), 'lr': args.lr}],
                                     weight_decay=args.weight_decay)
        print("============================== Learning start ==============================")
        if not args.lipschitz:
            print("NO Lipschitz Normalization")

        if not args.sim:
            print("NO Similarity Loss")

        nifty_gcn(args=args,
                  device=device,
                  encoder=encoder,
                  predictor=predictor,
                  classifier=classifier,
                  optimizer=optimizer,
                  datagenerator=datagenerator,
                  data=data,
                  A=A,
                  labels=labels,
                  loss_fn=criterion,
                  trainIdx=trainIdx,
                  valIdx=valIdx,
                  testIdx=testIdx,
                  sen_vals=sen_vals)

    else:
        raise ValueError("Unrecognized model!")