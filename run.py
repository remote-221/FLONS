import torch.optim as optim
import argparse
import copy
from torch.utils.tensorboard import SummaryWriter
from utils import *
import random

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, default='simple-cnn', help='neural network used in training')
    parser.add_argument('dataset', type=str, default='mnist', help='dataset used for training')
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--partition', type=str, default='noniid-labeldir', help='the data partitioning strategy')
    parser.add_argument('--batch-size', type=int, default=10, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=5, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=100,  help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='scaffold',help='fl algorithms: fedprox/scaffold/fedcurv/fedmas')
    parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')
    parser.add_argument('--loss', type=str, default='contrastive')
    parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')
    parser.add_argument('--comm_round', type=int, default=50, help='number of maximum communication roun')
    parser.add_argument('--is_same_initial', type=int, default=1, help='Whether initial all the models with the same parameters ')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--beta', type=float, default=0.3, help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--mu', type=float, default=0.1, help='the mu parameter for fedprox')
    parser.add_argument('--rho', type=float, default=0, help='Parameter controlling the momentum SGD')
    parser.add_argument('--sample', type=float, default=0.2, help='Sample ratio for each communication round')
    args = parser.parse_args()
    return args

def init_nets(net_configs, n_parties, args):
    nets = {net_i: None for net_i in range(n_parties)}
    if args.use_projection_head:
        add = ""
        if "mnist" in args.dataset and args.model == "simple-cnn":
            add = "-mnist"
    else:
        for net_i in range(n_parties):
            if args.dataset in ("cifar10"):
                net = CifarNet(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10)
            elif args.dataset in ("mnist"):
                net = SimpleCNN(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10)
            else:
                print("not supported yet")
                exit(1)
            nets[net_i] = net

    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)
    return nets, model_meta_data, layer_type

def train_net(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, device="cpu", masx=None):

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                out = net(x)
                loss = criterion(out, target)
                if masx is not None:
                    masx_loss = masx.penalty(net)
                    loss += 0.2 * masx_loss
                loss.backward()
                optimizer.step()

                cnt += 1
                epoch_loss_collector.append(loss.item())
    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    return train_acc, test_acc

def train_net_fedcurv(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, device="cpu", masx=None):

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]
    ewc = EWC(net, train_dataloader, 'cpu')

    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                out = net(x)
                loss = criterion(out, target)
                if ewc is not None:
                    ewc_loss = ewc.penalty(net)
                    loss += 2.0 * ewc_loss
                loss.backward()
                optimizer.step()

                cnt += 1
                epoch_loss_collector.append(loss.item())
    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    return train_acc, test_acc

def train_net_fedprox(net_id, net, global_net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, mu, device="cpu"):

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)
    cnt = 0
    # mu = 0.001
    global_weight_collector = list(global_net.to(device).parameters())

    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)

            optimizer.zero_grad()
            x.requires_grad = True
            target.requires_grad = False
            target = target.long()

            out = net(x)
            loss = criterion(out, target)

            #for fedprox
            fed_prox_reg = 0.0
            jskjf = list(net.parameters())

            for param_index, param in enumerate(net.parameters()):
                fed_prox_reg += ((mu / 2) * torch.norm((param - global_weight_collector[param_index]))**2)
            loss += fed_prox_reg

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    net.to('cpu')
    return train_acc, test_acc

def train_net_scaffold(net_id, net, global_model, c_local, c_global, train_dataloader, test_dataloader, epochs, lr, args_optimizer, device="cpu"):

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    c_global_para = c_global.state_dict()
    c_local_para = c_local.state_dict()

    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                out = net(x)
                loss = criterion(out, target)

                loss.backward()
                optimizer.step()

                net_para = net.state_dict()
                for key in net_para:
                    net_para[key] = net_para[key] - args.lr * (c_global_para[key] - c_local_para[key])
                net.load_state_dict(net_para)

                cnt += 1
                epoch_loss_collector.append(loss.item())

    c_new_para = c_local.state_dict()
    c_delta_para = copy.deepcopy(c_local.state_dict())
    global_model_para = global_model.state_dict()
    net_para = net.state_dict()
    for key in net_para:
        c_new_para[key] = c_new_para[key] - c_global_para[key] + (global_model_para[key] - net_para[key]) / (cnt * args.lr)
        c_delta_para[key] = c_new_para[key] - c_local_para[key]
    c_local.load_state_dict(c_new_para)

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    net.to('cpu')
    return train_acc, test_acc, c_delta_para

def local_train_net(nets, selected, args, net_dataidx_map, test_dl = None, device="cpu", masx=None):
    avg_acc = 0.0
    print("selected: %s" % selected)
    print("Training network：", end=' ')

    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]
        print("%s" % str(net_id), end = ' ')
        net.to(device)
        train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs
        trainacc, testacc = train_net(net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, device=device,masx=masx)
        avg_acc += testacc

    avg_acc /= len(selected)
    if args.alg == 'local_training':
        print("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list

def local_train_net_fedcurv(nets, selected, args, net_dataidx_map, test_dl = None, device="cpu"):
    avg_acc = 0.0
    print("selected: %s" % selected)
    print("Training network：", end=' ')

    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]
        print("%s" % str(net_id), end = ' ')
        net.to(device)
        train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs

        trainacc, testacc = train_net_fedcurv(net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, device=device)
        avg_acc += testacc
    avg_acc /= len(selected)
    if args.alg == 'local_training':
        print("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list

def local_train_net_fedprox(nets, selected, global_model, args, net_dataidx_map, test_dl = None, device="cpu"):
    avg_acc = 0.0
    print("selected: %s" % selected)
    print("Training network：", end=' ')

    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]
        print("%s" % str(net_id), end=' ')
        net.to(device)
        train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs

        trainacc, testacc = train_net_fedprox(net_id, net, global_model, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, args.mu, device=device)
        avg_acc += testacc

    avg_acc /= len(selected)
    if args.alg == 'local_training':
        print("avg test acc %f" % avg_acc)
    nets_list = list(nets.values())
    return nets_list

def local_train_net_scaffold(nets, selected, global_model, c_nets, c_global, args, net_dataidx_map, test_dl = None, device="cpu"):
    avg_acc = 0.0
    print("selected: %s" % selected)
    print("Training network：", end=' ')

    total_delta = copy.deepcopy(global_model.state_dict())
    for key in total_delta:
        total_delta[key] = 0.0
    c_global.to(device)
    global_model.to(device)
    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]
        print("%s" % str(net_id), end=' ')
        train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs

        trainacc, testacc, c_delta_para = train_net_scaffold(net_id, net, global_model, c_nets[net_id], c_global, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, device=device)

        for key in total_delta:
            total_delta[key] += c_delta_para[key]

        avg_acc += testacc
    c_global_para = c_global.state_dict()
    for key in c_global_para:
        if c_global_para[key].type() == 'torch.LongTensor':
            c_global_para[key] += total_delta[key].type(torch.LongTensor)
        elif c_global_para[key].type() == 'torch.cuda.LongTensor':
            c_global_para[key] += total_delta[key].type(torch.cuda.LongTensor)
        else:
            c_global_para[key] += total_delta[key]
    c_global.load_state_dict(c_global_para)

    avg_acc /= len(selected)
    if args.alg == 'local_training':
        print("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list

def get_partition_dict(dataset, partition, n_parties, init_seed=0, datadir='./data', logdir='./logs', beta=0.5):
    seed = init_seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        dataset, datadir, logdir, partition, n_parties, beta=beta)
    return net_dataidx_map

# EWC implementation
class EWC:
    def __init__(self, model: nn.Module, dataloaders: list, device):
        self.model = model
        self.dataloaders = dataloaders
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self.device = device
        self._precision_matrices = self._calculate_importance()

        for n, p in self.params.items():
            self._means[n] = p.clone().detach()

    def _calculate_importance(self):
        precision_matrices = {}
        for n, p in self.params.items():
            precision_matrices[n] = p.clone().detach().fill_(0)

        self.model.eval()
        dataloader_num = len(self.dataloaders)
        num_data = sum([len(loader) for loader in self.dataloaders])
        for dataloader in self.dataloaders:
            for data in dataloader[0]:
                self.model.zero_grad()
                output = self.model(data.to(self.device))
                output.pow_(2)
                loss = torch.sum(output, dim=0)
                loss = loss.mean()
                loss.backward()

                for n, p in self.model.named_parameters():
                    precision_matrices[n].data += p.grad.abs()
        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss

# mas implementation
class mas(object):

  def __init__(self, model: nn.Module, dataloader, device, prev_guards=[None]):
    self.model = model
    self.dataloader = dataloader
    self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
    self.p_old = {}
    self.device = device
    self.previous_guards_list = prev_guards
    self._precision_matrices = self.calculate_importance()

    for n, p in self.params.items():
      self.p_old[n] = p.clone().detach()

  def calculate_importance(self):
    precision_matrices = {}
    for n, p in self.params.items():
      precision_matrices[n] = p.clone().detach().fill_(0)
      for i in range(len(self.previous_guards_list)):
        if self.previous_guards_list[i]:
          precision_matrices[n] += self.previous_guards_list[i]

    self.model.eval()
    if self.dataloader is not None:
      for data in self.dataloader:
        self.model.zero_grad()
        output = self.model(data[0].to(self.device))
        l2_norm = output.norm(2, dim=0).pow(2).mean()
        l2_norm.backward()

        for n, p in self.model.named_parameters():
          precision_matrices[n].data += p.grad.data * 2

      precision_matrices = {n: p for n, p in precision_matrices.items()}
    return precision_matrices

  def penalty(self, model: nn.Module):
    loss = 0
    for n, p in model.named_parameters():
      _loss = self._precision_matrices[n] * (p - self.p_old[n]) ** 2
      loss += _loss.sum()
    return loss

if __name__ == '__main__':

    args = get_args()
    mkdirs(args.logdir)
    mkdirs(args.modeldir)
    
    device = torch.device(args.device)

    print("分区数据")
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        args.dataset, args.datadir, args.logdir, args.partition, args.n_parties, beta=args.beta)
    n_classes = len(np.unique(y_train))

    train_dl_global, test_dl_global, train_ds_global, test_ds_global = get_dataloader(args.dataset, args.datadir,args.batch_size, 32)

    print("len train_dl_global:", len(train_ds_global))

    data_size = len(test_ds_global)

    train_all_in_list = []
    test_all_in_list = []

    writer = SummaryWriter('runs/%s_%s/%s' % (args.dataset, args.partition, args.alg))

    if args.alg == 'fedcurv':
        print("初始化网络")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config,  args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round in range(args.comm_round):
            print("通信伦次:" + str(round))
            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]
            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            local_train_net_fedcurv(nets, selected, args, net_dataidx_map, test_dl = test_dl_global, device=device)
            
            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

            for idx in range(len(selected)):
                net_para = nets[selected[idx]].cpu().state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            global_model.load_state_dict(global_para)

            global_model.to(device)
            test_acc, conf_matrix = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True,
                                                     device=device)
            print('>> Global Model Train accuracy: %f' % test_acc)
            print('>> Global Model Test accuracy: %f' % test_acc)
            writer.add_scalars('Accuracy/train', {'global': test_acc}, round)
            writer.add_scalars('Accuracy/test', {'global': test_acc}, round)


    elif args.alg == 'fedmas':
        print("初始化网络")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)
        listcilint = [[0, 61, 24, 5, 13, 4,  6, 71, 8, 9],
                      [10, 26, 12, 3, 14, 15, 16, 17, 18, 19],
                      [20, 88, 32, 37, 2, 25, 11, 27, 91, 29],
                      [30, 60, 22, 33, 34, 35, 36, 23, 38, 39],
                      [99, 41, 42, 43, 64, 45, 46, 47, 59, 49],
                      [50, 63, 52, 53, 65, 55, 56, 75, 58, 48],
                      [31, 1, 62, 51, 44, 54, 66, 67, 86, 69],
                      [70, 7, 72, 93, 74, 57, 76, 98, 78, 80],
                      [79, 81, 92, 83, 84, 85, 68, 87, 21, 89],
                      [90, 28, 82, 73, 94, 95, 96, 77, 98, 40]
                      ]
        for round in range(args.comm_round):
            print("通信伦次:" + str(round))
            for listcilintindex in listcilint:
                selected = random.sample(listcilintindex, int(args.sample * len(listcilintindex)))
                if round is 0:
                    masx = None
                global_para = global_model.state_dict()
                if round == 0:
                    if args.is_same_initial:
                        for idx in selected:
                            nets[idx].load_state_dict(global_para)
                else:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)

                local_train_net(nets, selected, args, net_dataidx_map, test_dl = test_dl_global, device=device,masx=masx)

                total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
                fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

                for idx in range(len(selected)):
                    net_para = nets[listcilint[idx]].cpu().state_dict()
                    if idx == 0:
                        for key in net_para:
                            global_para[key] = net_para[key] * fed_avg_freqs[idx]
                    else:
                        for key in net_para:
                            global_para[key] += net_para[key] * fed_avg_freqs[idx]
                global_model.load_state_dict(global_para)

                dataidxs = net_dataidx_map[selected[-1]]
                train_dl_local, _, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32,
                                                                     dataidxs, selected[-1], args.n_parties - 1)
                masx = mas(global_model, train_dl_local, device)

            global_model.to(device)

            test_acc, conf_matrix = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True,
                                                     device=device)
            print('>> Global Model Train accuracy: %f' % test_acc)
            print('>> Global Model Test accuracy: %f' % test_acc)
            writer.add_scalars('Accuracy/train', {'global': test_acc}, round)
            writer.add_scalars('Accuracy/test', {'global': test_acc}, round)

    elif args.alg == 'fedprox':
        print("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config,0, 1, args)
        global_model = global_models[0]

        global_para = global_model.state_dict()

        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)
        for round in range(args.comm_round):
            print("in comm round:" + str(round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            local_train_net_fedprox(nets, selected, global_model, args, net_dataidx_map, test_dl = test_dl_global, device=device)
            global_model.to('cpu')

            # update global model
            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

            for idx in range(len(selected)):
                net_para = nets[selected[idx]].cpu().state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            global_model.load_state_dict(global_para)

            print('global n_training: %d' % len(train_dl_global))
            print('global n_test: %d' % len(test_dl_global))

            global_model.to(device)
            test_acc, conf_matrix = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True,
                                                     device=device)
            print('>> Global Model Train accuracy: %f' % test_acc)
            print('>> Global Model Test accuracy: %f' % test_acc)
            writer.add_scalars('Accuracy/train', {'global': test_acc}, round)
            writer.add_scalars('Accuracy/test', {'global': test_acc}, round)

    elif args.alg == 'scaffold':
        print("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        c_nets, _, _ = init_nets(args.net_config, args.n_parties, args)
        c_globals, _, _ = init_nets(args.net_config, 1, args)
        c_global = c_globals[0]
        c_global_para = c_global.state_dict()
        for net_id, net in c_nets.items():
            net.load_state_dict(c_global_para)

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round in range(args.comm_round):
            print("in comm round:" + str(round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            local_train_net_scaffold(nets, selected, global_model, c_nets, c_global, args, net_dataidx_map, test_dl = test_dl_global, device=device)
            # update global model
            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

            for idx in range(len(selected)):
                net_para = nets[selected[idx]].cpu().state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            global_model.load_state_dict(global_para)

            print('global n_training: %d' % len(train_dl_global))
            print('global n_test: %d' % len(test_dl_global))

            global_model.to(device)
            test_acc, conf_matrix = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True,
                                                     device=device)
            print('>> Global Model Train accuracy: %f' % test_acc)
            print('>> Global Model Test accuracy: %f' % test_acc)
            writer.add_scalars('Accuracy/train', {'global': test_acc}, round)
            writer.add_scalars('Accuracy/test', {'global': test_acc}, round)
