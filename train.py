import copy
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import os

from ContrastLoss import SupConLoss, SupConLoss_new
from model import ModelFedCon, ModelFedCon_noheader, SimpleCNNMNIST
from utils import DATA_nclass, get_dataloader, compute_accuracy

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

def init_nets(net_configs, n_parties, args, device='cpu'):
    nets = {net_i: None for net_i in range(n_parties)}
    n_classes = DATA_nclass[args.dataset]
    if args.alg == 'fedproc':
        for net_i in range(n_parties):
            if args.use_project_head:
                net = ModelFedCon(args.model, args.out_dim, n_classes, net_configs)
            else:
                net = ModelFedCon_noheader(args.model, args.out_dim, n_classes, net_configs)
            if device == 'cpu':
                net.to(device)
            else:
                net = net.cuda()
            nets[net_i] = net

    else:
        if args.normal_model:
            for net_i in range(n_parties):
                if args.model == 'simple-cnn':
                    net = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10)
                if device == 'cpu':
                    net.to(device)
                else:
                    net = net.cuda()
                nets[net_i] = net
        else:
            for net_i in range(n_parties):
                if args.use_project_head:
                    net = ModelFedCon(args.model, args.out_dim, n_classes, net_configs)
                else:
                    net = ModelFedCon_noheader(args.model, args.out_dim, n_classes, net_configs)
                if device == 'cpu':
                    net.to(device)
                else:
                    net = net.cuda()
                nets[net_i] = net

    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)

    return nets, model_meta_data, layer_type


def train_net_fedproc(round, net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args,
                      device="cpu", logger=None, global_class_center=None):
    net = nn.DataParallel(net)
    net.cuda()

    train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    print('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    print('>> Pre-Training Test accuracy: {}'.format(test_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().cuda()
    if args.scloss == 'SupConLoss_new':
        criterion_extra = SupConLoss_new()

    if round < 100:
        alpha = round / 100
    else:
        alpha = 1


    for epoch in range(epochs):
        epoch_loss_collector, epoch_CEloss_collector, epoch_SCloss_collector = [], [], []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.cuda(), target.cuda()

            optimizer.zero_grad()
            x.requires_grad = True
            target.requires_grad = False
            target = target.long()

            _, feature, out = net(x)
            CEloss = criterion(out, target)
            SCloss = criterion_extra(features=feature, labels=target, center=global_class_center)

            # loss = CEloss
            if round == 0:
                loss = CEloss
            else:
                loss = alpha * CEloss + (1 - alpha) * SCloss

            loss.backward()
            optimizer.step()

            epoch_CEloss_collector.append(CEloss.item())
            epoch_SCloss_collector.append(SCloss.item())
            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        epoch_CEloss = sum(epoch_CEloss_collector) / len(epoch_CEloss_collector)
        epoch_SCloss = sum(epoch_SCloss_collector) / len(epoch_SCloss_collector)

        train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
        test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

        print(
            '[Round: %d net_id: %d] Epoch: %d  Loss: %f  CEloss: %f  SCloss: %f  alpha: %.2f  ||  train_acc: %f  test_acc: %f' % (
                round, net_id, epoch, epoch_loss, epoch_CEloss, epoch_SCloss, alpha, train_acc, test_acc))
        logger.info(
            '[Round: %d net_id: %d] Epoch: %d  Loss: %f  CEloss: %f  SCloss: %f  alpha: %.2f  ||  train_acc: %f  test_acc: %f' % (
                round, net_id, epoch, epoch_loss, epoch_CEloss, epoch_SCloss, alpha, train_acc, test_acc))

    net.to('cpu')
    return train_acc, test_acc


def train_net_fedavg(round, net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args,
                     device="cpu", logger=None):
    net = nn.DataParallel(net)
    net.cuda()

    train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    print('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    print('>> Pre-Training Test accuracy: {}'.format(test_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().cuda()

    for epoch in range(epochs):
        epoch_loss_collector, epoch_CEloss_collector, epoch_SCloss_collector = [], [], []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.cuda(), target.cuda()

            optimizer.zero_grad()
            x.requires_grad = True
            target.requires_grad = False
            target = target.long()

            _, feature, out = net(x)

            loss = criterion(out, target)

            loss.backward()
            optimizer.step()

            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)

        train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
        test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

        print(
            '[Round: %d net_id: %d] Epoch: %d  Loss: %f ||  train_acc: %f  test_acc: %f' % (
                round, net_id, epoch, epoch_loss, train_acc, test_acc))
        logger.info(
            '[Round: %d net_id: %d] Epoch: %d  Loss: %f ||  train_acc: %f  test_acc: %f' % (
                round, net_id, epoch, epoch_loss, train_acc, test_acc))

    net.to('cpu')
    return train_acc, test_acc


def train_net_moon(net_id, net, global_net, previous_nets, train_dataloader, test_dataloader, epochs, lr,
                   args_optimizer, mu, temperature, args, round, device="cpu", logger=None):
    net = nn.DataParallel(net)
    net.cuda()

    train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    print('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    print('>> Pre-Training Test accuracy: {}'.format(test_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().cuda()
    global_net.to(device)

    for previous_net in previous_nets:
        previous_net.cuda()
    global_w = global_net.state_dict()

    cnt = 0
    cos = torch.nn.CosineSimilarity(dim=-1)
    # mu = 0.001

    for epoch in range(epochs):
        epoch_loss_collector = []
        epoch_loss1_collector = []
        epoch_loss2_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.cuda(), target.cuda()

            optimizer.zero_grad()
            x.requires_grad = True
            target.requires_grad = False
            target = target.long()

            _, pro1, out = net(x)
            _, pro2, _ = global_net(x)

            posi = cos(pro1, pro2)
            logits = posi.reshape(-1, 1)

            for previous_net in previous_nets:
                previous_net.cuda()
                _, pro3, _ = previous_net(x)
                nega = cos(pro1, pro3)
                logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)

                previous_net.to('cpu')

            logits /= temperature
            labels = torch.zeros(x.size(0)).cuda().long()

            loss2 = mu * criterion(logits, labels)

            loss1 = criterion(out, target)
            loss = loss1 + loss2

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())
            epoch_loss1_collector.append(loss1.item())
            epoch_loss2_collector.append(loss2.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
        epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)
        # logger.info('Epoch: %d Loss: %f Loss1: %f Loss2: %f' % (epoch, epoch_loss, epoch_loss1, epoch_loss2))

        train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
        test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

        print(
            '[Round: %d net_id: %d] Epoch: %d  Loss: %f  CEloss: %f  SCloss: %f  ||  train_acc: %f  test_acc: %f' % (
                round, net_id, epoch, epoch_loss, epoch_loss1, epoch_loss2, train_acc, test_acc))
        logger.info(
            '[Round: %d net_id: %d] Epoch: %d  Loss: %f  CEloss: %f  SCloss: %f  ||  train_acc: %f  test_acc: %f' % (
                round, net_id, epoch, epoch_loss, epoch_loss1, epoch_loss2, train_acc, test_acc))

    return train_acc, test_acc

def train_net_fedprox(round, net_id, net, global_net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, mu, args,
                      device="cpu", logger=None):



    net.cuda()

    train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().cuda()

    cnt = 0
    global_weight_collector = list(global_net.cuda().parameters())


    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.cuda(), target.cuda()

            optimizer.zero_grad()
            x.requires_grad = True
            target.requires_grad = False
            target = target.long()

            _,_,out = net(x)
            loss = criterion(out, target)


            fed_prox_reg = 0.0
            for param_index, param in enumerate(net.parameters()):
                fed_prox_reg += ((mu / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)
            loss += fed_prox_reg

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)

        train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
        test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

        print(
            '[Round: %d net_id: %d] Epoch: %d  Loss: %f ||  train_acc: %f  test_acc: %f' % (
                round, net_id, epoch, epoch_loss, train_acc, test_acc))
        logger.info(
            '[Round: %d net_id: %d] Epoch: %d  Loss: %f ||  train_acc: %f  test_acc: %f' % (
                round, net_id, epoch, epoch_loss, train_acc, test_acc))

    net.to('cpu')
    return train_acc, test_acc


def get_feature(net_id, net, train_dataloader, alg, logdir):
    feature_collector, target_collector = [], []

    for batch_idx, (x, target) in enumerate(train_dataloader):
        x, target = x.cuda(), target.cuda()
        net.cuda()

        _, feature, out = net(x)

        a = feature.cpu().tolist()
        b = target.cpu().tolist()

        feature_collector.extend(a)
        target_collector.extend(b)

    net.cpu()

    f_dir = logdir + '/feature/'
    mkdirs(f_dir)

    features_dir = f_dir + 'client' + str(net_id) + '_features.csv'
    csv_features = pd.DataFrame(data=feature_collector)
    csv_features.to_csv(features_dir, index=None)
    targets_dir = f_dir + '/client' + str(net_id) + '_targets.csv'
    csv_targets = pd.DataFrame(data=target_collector)
    csv_targets.to_csv(targets_dir, index=None)

    return feature_collector, target_collector


def local_train_net(nets, args, net_dataidx_map, train_dl=None, test_dl=None, global_model=None, prev_model_pool=None,
                    server_c=None, clients_c=None, round=None, device="cpu", logger=None, global_class_center=None, logdir=None):
    avg_acc = 0.0
    acc_list = []
    if global_model:
        global_model.cuda()
    if server_c:
        server_c.cuda()
        server_c_collector = list(server_c.cuda().parameters())
        new_server_c_collector = copy.deepcopy(server_c_collector)
    for net_id, net in nets.items():
        dataidxs = net_dataidx_map[net_id]
        print("Training Client %s  n_training: %d" % (str(net_id), len(dataidxs)))
        logger.info("Training Client %s  n_training: %d" % (str(net_id), len(dataidxs)))
        train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs
        if args.alg == 'fedproc':
            trainacc, testacc = train_net_fedproc(round, net_id, net, train_dl_local, test_dl, n_epoch, args.lr,
                                                  args.optimizer, args, device=device, logger=logger,
                                                  global_class_center=global_class_center)
            if args.save_feature and round + 1 == args.comm_round:
                get_feature(net_id, net, train_dl_local, args.alg, logdir)
                print("get feature ....")

        elif args.alg == 'fedavg':
            trainacc, testacc = train_net_fedavg(round, net_id, net, train_dl_local, test_dl, n_epoch, args.lr,
                                                  args.optimizer, args, device=device, logger=logger)
            if args.save_feature and round + 1 == args.comm_round:
                get_feature(net_id, net, train_dl_local, args.alg, logdir)
                print("get feature ....")

        elif args.alg == 'fedprox':
            trainacc, testacc = train_net_fedprox(round, net_id, net, global_model, train_dl_local, test_dl, n_epoch,
                                                  args.lr, args.optimizer, args.mu, args, device=device, logger=logger)


        elif args.alg == 'moon':
            prev_models = []
            for i in range(len(prev_model_pool)):
                prev_models.append(prev_model_pool[i][net_id])
            trainacc, testacc = train_net_moon(net_id, net, global_model, prev_models, train_dl_local, test_dl,
                                               n_epoch, args.lr, args.optimizer, args.mu, args.temperature, args, round,
                                               device=device, logger=logger)

        avg_acc += testacc
        acc_list.append(testacc)
    avg_acc /= args.n_parties
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)
        logger.info("std acc %f" % np.std(acc_list))
    if global_model:
        global_model.to('cpu')
    if server_c:
        for param_index, param in enumerate(server_c.parameters()):
            server_c_collector[param_index] = new_server_c_collector[param_index]
        server_c.to('cpu')
    return nets, acc_list


def get_global_class_center(global_class_center_old, n_party, nets, args, net_dataidx_map, device="cpu", logger=None):
    local_class_center = []
    clsnum = DATA_nclass[args.dataset]
    class_count = np.zeros((n_party, clsnum))
    for net_id, net in nets.items():
        # net = nn.DataParallel(net)
        net.cuda()
        dataidxs = net_dataidx_map[net_id]
        print("Calculate the class center of the Client %s " % (str(net_id)))
        logger.info("Calculate the class center of the Client %s " % (str(net_id)))

        train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs)

        class_feature = {}
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(train_dl_local):
                x, target = x.cuda(), target.cuda()
                _, features, out = net(x)  # features [64,256]

                for label in torch.unique(target):
                    index = torch.eq(target, label)
                    feature = features[index]

                    lab = str(label.cpu().numpy())

                    if lab not in class_feature.keys():
                        class_feature[lab] = torch.sum(feature, 0)
                        class_count[net_id][label] = feature.shape[0]
                    else:
                        class_feature[lab] += torch.sum(feature, 0)
                        class_count[net_id][label] += feature.shape[0]
        net.to('cpu')

        local_class_center.append(class_feature)  # [5,10,256]


    print("Aggregate the global class center")
    logger.info("Aggregate the global class center")

    global_center = [[] for x in range(clsnum)]
    for cls in range(clsnum):
        for id in range(len(local_class_center)):
            if global_center[cls] == []:
                try:
                    global_center[cls] = local_class_center[id][str(cls)]
                    inittype = torch.zeros_like(global_center[cls])
                except:
                    pass
            else:
                try:
                    global_center[cls] += local_class_center[id][str(cls)]
                except:
                    pass

        if global_center[cls] == []:
            if global_class_center_old == None:
                global_center[cls] = inittype
            else:
                global_center[cls] = global_class_center_old[cls]

        else:
            global_center[cls] = global_center[cls] / sum(class_count[:, cls])

    return global_center
