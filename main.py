import argparse
import copy
import os
import time
import datetime
import threading

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from train import init_nets, local_train_net, get_global_class_center
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet50',
                        help='simple-cnn,resnet50,neural network used in training')
    parser.add_argument('--dataset', type=str, default='cifar100', help='dataset: tinyimagenet,cifar100,cifar10')
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--partition', type=str, default='noniid', help='the data partitioning strategy: iid/noniid')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate (default: 0.1)')
    parser.add_argument('--epochs', type=int, default=10, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=10, help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='fedproc',
                        help='communication strategy: fedproc/fedavg/fedprox/moon/local_training/all_in')
    parser.add_argument('--comm_round', type=int, default=100, help='number of maximum communication roun')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--mu', type=float, default=1, help='the mu parameter for fedprox or moon')
    parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')
    parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')
    parser.add_argument('--local_max_epoch', type=int, default=100,
                        help='the number of epoch for local optimal training')
    parser.add_argument('--model_buffer_size', type=int, default=1,
                        help='store how many previous models for contrastive loss')
    parser.add_argument('--pool_option', type=str, default='FIFO', help='FIFO or BOX')
    parser.add_argument('--sample_fraction', type=float, default=1, help='how many clients are sampled in each round')
    parser.add_argument('--load_model_file', type=str, default=None, help='the model to load as global model')
    parser.add_argument('--load_pool_file', type=str, default=None, help='the old model pool path to load')
    parser.add_argument('--load_model_round', type=int, default=0,
                        help='how many rounds have executed for the loaded model')
    parser.add_argument('--load_first_net', type=int, default=1, help='whether load the first net as old net or not')
    parser.add_argument('--normal_model', type=int, default=0, help='use normal model or aggregate model')
    parser.add_argument('--loss', type=str, default='contrastive')
    parser.add_argument('--save_model', type=int, default=1)
    parser.add_argument('--use_project_head', type=int, default=1)
    parser.add_argument('--server_momentum', type=float, default=0, help='the server momentum (FedAvgM)')
    parser.add_argument('--scloss', type=str, default='SupConLoss_new', help='SupConLoss_new')
    parser.add_argument('--save_feature', type=int, default=1)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    device = torch.device(args.device)
    logdir = os.path.join(args.logdir, str(datetime.datetime.now().strftime("%Y-%m-%d/%H.%M.%S")))
    mkdirs(logdir)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        filename=os.path.join(logdir, 'info.log'),
        format='[%(levelname)s](%(asctime)s) %(message)s',
        datefmt='%Y/%m/%d/ %I:%M:%S %p', level=logging.DEBUG, filemode='w')
    logger = logging.getLogger()
    logger.info(device)


    tb_port = 6006
    tb_host = "127.0.0.1"
    writer = SummaryWriter(log_dir=logdir, filename_suffix="info")
    tb_thread = threading.Thread(
        target=launch_tensor_board,
        args=([logdir, tb_port, tb_host])
    ).start()
    time.sleep(3.0)

    print("**Basic Setting...")
    logger.info("**Basic Setting...")
    print('  ', args)
    logging.info(args)

    seed = args.init_seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)

    print("**Partitioning data...")
    logger.info("**Partitioning data...")

    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        args.dataset, args.datadir, args.logdir, args.partition, args.n_parties, beta=args.beta)

    n_party_per_round = int(args.n_parties * args.sample_fraction)
    party_list = [i for i in range(args.n_parties)]
    party_list_rounds = []
    if n_party_per_round != args.n_parties:
        for i in range(args.comm_round):
            party_list_rounds.append(random.sample(party_list, n_party_per_round))
    else:
        for i in range(args.comm_round):
            party_list_rounds.append(party_list)

    train_dl_global, test_dl, train_ds_global, test_ds_global = get_dataloader(args.dataset,
                                                                               args.datadir,
                                                                               args.batch_size,
                                                                               32)

    train_dl = None

    print('**Initializing nets...')
    logger.info("**Initializing nets")

    nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.n_parties, args, device=device)
    global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 1, args, device=device)
    global_model = global_models[0]
    n_comm_rounds = args.comm_round
    start_rounds = 0
    if args.load_model_file and args.alg != 'plot_visual':
        global_model.load_state_dict(torch.load(args.load_model_file))
        start_rounds = args.load_model_round

    if args.server_momentum:
        moment_v = copy.deepcopy(global_model.state_dict())
        for key in moment_v:
            moment_v[key] = 0
    print('   Completed')
    logger.info("Completed")
    print("-" * 100 + '  Start Training  ' + "-" * 100)
    logger.info("-" * 100 + '  Start Training  ' + "-" * 100)
    flag = 0

    if args.alg == 'fedproc':
        for round in range(start_rounds, n_comm_rounds):
            print("**COMMON ROUND:", str(round))
            logger.info("**COMMON ROUND:" + str(round))
            party_list_this_round = party_list_rounds[round]

            global_w = global_model.state_dict()
            if args.server_momentum:
                old_w = copy.deepcopy(global_model.state_dict())

            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)


            if flag == 0:
                global_class_center_old = None
                global_class_center = get_global_class_center(global_class_center_old, args.n_parties, nets_this_round,
                                                              args, net_dataidx_map, logger=logger)
                flag = 1


            _, acc_list = local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_dl, test_dl=test_dl,
                                          round=round,
                                          device=device, logger=logger, global_class_center=global_class_center, logdir=logdir)
            global_class_center_old = copy.deepcopy(global_class_center)
            global_class_center = get_global_class_center(global_class_center_old, args.n_parties, nets_this_round,
                                                          args, net_dataidx_map, logger=logger)

            total_data_points = sum([len(net_dataidx_map[r]) for r in range(args.n_parties)])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in range(args.n_parties)]


            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]

            if args.server_momentum:
                delta_w = copy.deepcopy(global_w)
                for key in delta_w:
                    delta_w[key] = old_w[key] - global_w[key]
                    moment_v[key] = args.server_momentum * moment_v[key] + (1 - args.server_momentum) * delta_w[key]
                    global_w[key] = old_w[key] - moment_v[key]

            global_model.load_state_dict(global_w)
            global_model.cuda()

            train_acc, train_loss = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix, test_loss = compute_accuracy(global_model, test_dl, get_confusion_matrix=True,
                                                                device=device)
            global_model.to('cpu')

            print('[Round: %d] >> Global Model Train loss: %f' % (round, train_loss))
            print('[Round: %d] >> Global Model Train accuracy: %f' % (round, train_acc))
            print('[Round: %d] >> Global Model Test accuracy: %f' % (round, test_acc))
            print('[Round: %d] >> Global Model Test loss: %f' % (round, test_loss))

            logger.info('[Round: %d] >> Global Model Train loss: %f' % (round, train_loss))
            logger.info('[Round: %d] >> Global Model Train accuracy: %f' % (round, train_acc))
            logger.info('[Round: %d] >> Global Model Test accuracy: %f' % (round, test_acc))
            logger.info('[Round: %d] >> Global Model Test loss: %f' % (round, test_loss))

            writer.add_scalar('scalar/Test_Accuracy', test_acc, round)
            writer.add_scalar('scalar/Train_Accuracy', train_acc, round)
            writer.add_scalar('scalar/Test_Loss', test_loss, round)
            writer.add_scalar('scalar/Train_Loss', train_loss, round)

            mkdirs(logdir + '/fedproc/')
            global_model.to('cpu')

            torch.save(global_model.state_dict(), logdir + '/fedproc/' + 'globalmodel' + '.pth')
            torch.save(nets[0].state_dict(), logdir + '/fedproc/' + 'localmodel0' + '.pth')

    if args.alg == 'fedavg':
        for round in range(start_rounds, n_comm_rounds):
            print("**COMMON ROUND:", str(round))
            logger.info("**COMMON ROUND:" + str(round))
            party_list_this_round = party_list_rounds[round]

            global_w = global_model.state_dict()
            if args.server_momentum:
                old_w = copy.deepcopy(global_model.state_dict())

            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)

            _, acc_list = local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_dl, test_dl=test_dl,
                                          round=round, device=device, logger=logger, global_class_center=None)


            total_data_points = sum([len(net_dataidx_map[r]) for r in range(args.n_parties)])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in range(args.n_parties)]

            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]

            if args.server_momentum:
                delta_w = copy.deepcopy(global_w)
                for key in delta_w:
                    delta_w[key] = old_w[key] - global_w[key]
                    moment_v[key] = args.server_momentum * moment_v[key] + (1 - args.server_momentum) * delta_w[key]
                    global_w[key] = old_w[key] - moment_v[key]

            global_model.load_state_dict(global_w)
            global_model.cuda()

            train_acc, train_loss = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix, test_loss = compute_accuracy(global_model, test_dl, get_confusion_matrix=True,
                                                                device=device)
            global_model.to('cpu')

            print('[Round: %d] >> Global Model Train loss: %f' % (round, train_loss))
            print('[Round: %d] >> Global Model Train accuracy: %f' % (round, train_acc))
            print('[Round: %d] >> Global Model Test accuracy: %f' % (round, test_acc))
            print('[Round: %d] >> Global Model Test loss: %f' % (round, test_loss))

            logger.info('[Round: %d] >> Global Model Train loss: %f' % (round, train_loss))
            logger.info('[Round: %d] >> Global Model Train accuracy: %f' % (round, train_acc))
            logger.info('[Round: %d] >> Global Model Test accuracy: %f' % (round, test_acc))
            logger.info('[Round: %d] >> Global Model Test loss: %f' % (round, test_loss))

            writer.add_scalar('scalar/Test_Accuracy', test_acc, round)
            writer.add_scalar('scalar/Train_Accuracy', train_acc, round)
            writer.add_scalar('scalar/Test_Loss', test_loss, round)
            writer.add_scalar('scalar/Train_Loss', train_loss, round)

            mkdirs(logdir + '/fedavg/')
            global_model.to('cpu')

            torch.save(global_model.state_dict(), logdir + '/fedavg/' + 'globalmodel' + '.pth')
            torch.save(nets[0].state_dict(), logdir + '/fedavg/' + 'localmodel0' + '.pth')

    if args.alg == 'moon':
        old_nets_pool = []
        if args.load_pool_file:
            for nets_id in range(args.model_buffer_size):
                old_nets, _, _ = init_nets(args.net_config, args.n_parties, args, device='cpu')
                checkpoint = torch.load(args.load_pool_file)
                for net_id, net in old_nets.items():
                    net.load_state_dict(checkpoint['pool' + str(nets_id) + '_' + 'net' + str(net_id)])
                old_nets_pool.append(old_nets)
        elif args.load_first_net:
            if len(old_nets_pool) < args.model_buffer_size:
                old_nets = copy.deepcopy(nets)
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():
                        param.requires_grad = False

        for round in range(n_comm_rounds):
            print("**COMMON ROUND:", str(round))
            logger.info("**COMMON ROUND:" + str(round))

            party_list_this_round = party_list_rounds[round]

            global_model.eval()
            for param in global_model.parameters():
                param.requires_grad = False
            global_w = global_model.state_dict()

            if args.server_momentum:
                old_w = copy.deepcopy(global_model.state_dict())

            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)

            local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_dl, test_dl=test_dl,
                            global_model=global_model, prev_model_pool=old_nets_pool, round=round, device=device,
                            logger=logger)

            total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]

            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]

            if args.server_momentum:
                delta_w = copy.deepcopy(global_w)
                for key in delta_w:
                    delta_w[key] = old_w[key] - global_w[key]
                    moment_v[key] = args.server_momentum * moment_v[key] + (1 - args.server_momentum) * delta_w[key]
                    global_w[key] = old_w[key] - moment_v[key]

            global_model.load_state_dict(global_w)

            global_model.cuda()
            train_acc, train_loss = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix, test_loss = compute_accuracy(global_model, test_dl, get_confusion_matrix=True,
                                                                device=device)

            print('[Round: %d] >> Global Model Train loss: %f' % (round, train_loss))
            print('[Round: %d] >> Global Model Train accuracy: %f' % (round, train_acc))
            print('[Round: %d] >> Global Model Test accuracy: %f' % (round, test_acc))
            print('[Round: %d] >> Global Model Test loss: %f' % (round, test_loss))

            logger.info('[Round: %d] >> Global Model Train loss: %f' % (round, train_loss))
            logger.info('[Round: %d] >> Global Model Train accuracy: %f' % (round, train_acc))
            logger.info('[Round: %d] >> Global Model Test accuracy: %f' % (round, test_acc))
            logger.info('[Round: %d] >> Global Model Test loss: %f' % (round, test_loss))

            writer.add_scalar('scalar/Test_Accuracy', test_acc, round)
            writer.add_scalar('scalar/Train_Accuracy', train_acc, round)
            writer.add_scalar('scalar/Test_Loss', test_loss, round)
            writer.add_scalar('scalar/Train_Loss', train_loss, round)

            if len(old_nets_pool) < args.model_buffer_size:
                old_nets = copy.deepcopy(nets)
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():
                        param.requires_grad = False
                old_nets_pool.append(old_nets)
            elif args.pool_option == 'FIFO':
                old_nets = copy.deepcopy(nets)
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():
                        param.requires_grad = False
                for i in range(args.model_buffer_size - 2, -1, -1):
                    old_nets_pool[i] = old_nets_pool[i + 1]
                old_nets_pool[args.model_buffer_size - 1] = old_nets

            mkdirs(logdir + '/moon/')
            if args.save_model:
                torch.save(global_model.state_dict(), logdir + '/moon/global_model_' + '.pth')
                torch.save(nets[0].state_dict(), logdir + '/moon/localmodel0' + '.pth')
                for nets_id, old_nets in enumerate(old_nets_pool):
                    torch.save({'pool' + str(nets_id) + '_' + 'net' + str(net_id): net.state_dict() for net_id, net in
                                old_nets.items()}, logdir + '/moon/prev_model_pool_' + '.pth')

    if args.alg == 'fedprox':

        for round in range(n_comm_rounds):
            # logger.info("in comm round:" + str(round))
            print("**COMMON ROUND:", str(round))
            logger.info("**COMMON ROUND:" + str(round))
            party_list_this_round = party_list_rounds[round]
            global_w = global_model.state_dict()
            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)

            
            local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_dl, test_dl=test_dl,
                            global_model=global_model, round=round, device=device,logger=logger)

            global_model.to('cpu')

            total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]

            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * fed_avg_freqs[net_id]
            global_model.load_state_dict(global_w)



            global_model.cuda()
            train_acc, train_loss = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix, test_loss = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)

            print('[Round: %d] >> Global Model Train loss: %f' % (round, train_loss))
            print('[Round: %d] >> Global Model Train accuracy: %f' % (round, train_acc))
            print('[Round: %d] >> Global Model Test accuracy: %f' % (round, test_acc))
            print('[Round: %d] >> Global Model Test loss: %f' % (round, test_loss))

            logger.info('[Round: %d] >> Global Model Train loss: %f' % (round, train_loss))
            logger.info('[Round: %d] >> Global Model Train accuracy: %f' % (round, train_acc))
            logger.info('[Round: %d] >> Global Model Test accuracy: %f' % (round, test_acc))
            logger.info('[Round: %d] >> Global Model Test loss: %f' % (round, test_loss))

            writer.add_scalar('scalar/Test_Accuracy', test_acc, round)
            writer.add_scalar('scalar/Train_Accuracy', train_acc, round)
            writer.add_scalar('scalar/Test_Loss', test_loss, round)
            writer.add_scalar('scalar/Train_Loss', train_loss, round)

            mkdirs(logdir + '/fedprox/')
            global_model.to('cpu')

            torch.save(global_model.state_dict(), logdir + '/fedprox/' + 'globalmodel' + '.pth')


    if args.alg == 'local_training':
        logger.info("Initializing nets")
        local_train_net(nets, args, net_dataidx_map, train_dl=train_dl, test_dl=test_dl, device=device)
        mkdirs(args.modeldir + 'localmodel/')
        for net_id, net in nets.items():
            torch.save(net.state_dict(),
                       args.modeldir + 'localmodel/' + 'model' + str(net_id) + args.log_file_name + '.pth')

    print("hello world")

