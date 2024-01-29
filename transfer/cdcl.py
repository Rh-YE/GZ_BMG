import random
from torch.backends import cudnn
from dataset.galaxy_dataset import *
from args import *
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import schemas
import argparse
from utils.utils import *
from models.utils import *
from tqdm import tqdm
from scipy.spatial.distance import cdist
import torch
import numpy as np
import torch.nn as nn
import gc

root = "/data/public/renhaoye/mgs/"
import csv
def predict_and_save_to_csv(model, data_loader, num_samples, output_file, schema):
    with open(root+output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        header = ["Image Path"]
        for answer in schema.answers:
            header.extend([f"{answer.text}_meanv", f"{answer.text}_meanf", f"{answer.text}_var", f"{answer.text}_label", f"{answer.text}_fraction"])
        for question in schema.questions:
            header.extend([f"{question.text}_entropy"])
        for answer in schema.answers:
            header.extend([f"{answer.text}_mi"])
        writer.writerow(header)
        model.eval()
        enable_dropout(model)
        for img, label, img_path in data_loader:
            img = img.to("cuda:1")
            expected_votes = []
            expected_probs = []
            for mc_run in range(num_samples):
                with torch.no_grad():
                    output, _ = model(img)
                    expected_votes.append(output)
                    output = vote2prob(output, schema.question_index_groups)
                    expected_probs.append(output)
            mean_probs = torch.mean(torch.stack(expected_probs), dim=0)
            fraction = calculate_fraction(label, schema.question_index_groups)
            Hp = - mean_probs * torch.log2(mean_probs)  # MI = Hp-Ep
            var_probs = torch.var(torch.stack(expected_probs), dim=0)
            mean_votes = torch.mean(torch.stack(expected_votes), dim=0)
            Ep = torch.mean(-torch.stack(expected_probs) * torch.log2(torch.stack(expected_probs)), dim=0)
            MI = Hp - Ep
            entropy = torch.zeros(img.shape[0], 10)
            for q_n in range(len(schema.question_index_groups)):
                q_indices = schema.question_index_groups[q_n]
                q_start = q_indices[0]
                q_end = q_indices[1]
                entropy[:, q_n] = torch.sum(Hp[:, q_start:q_end + 1], dim=1)
            for i in range(img.shape[0]):
                row = [img_path[i]]  # 添加图像路径
                for answer in schema.answers:
                    row.extend([mean_votes[i][answer.index].item(), mean_probs[i][answer.index].item(),
                                var_probs[i][answer.index].item(), label[i][answer.index].item(),fraction[i][answer.index].item()])
                for q_n in range(len(schema.question_index_groups)):
                    row.append(entropy[i, q_n].item())
                for answer in schema.answers:
                    row.append(MI[i][answer.index].item())
                writer.writerow(row)
def pseudo_mask(tensor):
    # mask = tensor[:, 0] == 1
    # tensor[mask, 3:15] = 0
    # tensor[mask, 18:30] = 0
    
    # mask1 = tensor[:,1] == 1
    # tensor[mask1, 15:18] = 0
    
    # mask2 = tensor[:,4] == 1
    # tensor[mask2, 18:21] = 0
    
    # mask3 = tensor[:,3] == 1
    # tensor[mask3, 5:18] = 0
    # tensor[mask3, 21:30] = 0
    
    # mask4 = tensor[:,6] == 1
    # tensor[mask4, 21:30] = 0
    
    # mask5 = tensor[:,2] == 1
    # tensor[mask5,3:] = 0
    
    # artifact = (tensor[:, 2] == 1)
    # smooth = (tensor[:, 0] == 1)  & (tensor[:, 30:34].sum(dim=1) == 1) & (tensor[:, 15:18].sum(dim=1) == 1)
    # edge_on = (tensor[:, 1] == 1) & (tensor[:, 3] == 1) & (tensor[:, 30:34].sum(dim=1) == 1) & (tensor[:, 18:20].sum(dim=1) == 1)
    # nobugledisk = (tensor[:, 1] == 1) & (tensor[:, 4] == 1) & (tensor[:, 6] == 1) & (tensor[:, 30:34].sum(dim=1) == 1)  & (tensor[:, 7:10].sum(dim=1) == 1) & (tensor[:, 10:15].sum(dim=1) == 1)
    # spiral = (tensor[:, 1] == 1) & (tensor[:, 4] == 1) & (tensor[:, 7] == 1) & (tensor[:, 30:34].sum(dim=1) == 1) & (tensor[:, 7:10].sum(dim=1) == 1) & (tensor[:, 10:15].sum(dim=1) == 1) & (tensor[:, 21:24].sum(dim=1) == 1) & (tensor[:, 24:30].sum(dim=1) == 1)
    # tensor[~(artifact|smooth | edge_on | nobugledisk | spiral),:] = 0
    # tensor[:, 2] = 0
    return tensor
class SphericalKMeans:
    def __init__(self, n_clusters, init_centers, max_iters=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = init_centers
        self.question_answer_pairs = gz2_pairs
        self.dependencies = gz2_and_decals_dependencies
        self.schema = schemas.Schema(self.question_answer_pairs, self.dependencies)
    def threshold_and_modify(self, tensor):
        return tensor
    def fit(self, X):
        for iter in range(self.max_iters):
            prev_centroids = self.centroids.clone()
            similarities = (torch.matmul(X, torch.Tensor(prev_centroids.T).to(X.device)))
            similarities = self.threshold_and_modify(similarities.clone())
            cluster_assignments = torch.zeros_like(similarities)
            for q_n in range(len(self.schema.question_index_groups)):
                q_indices = self.schema.question_index_groups[q_n]
                q_start = q_indices[0]
                q_end = q_indices[1]
                input_slice = similarities[:, q_start:q_end + 1]
                max_prob, max_index = torch.max(input_slice, dim=1)
                condition = max_prob > -1
                cluster_assignments[condition, q_start + max_index[condition]] = 1
            if len(X) == 1:
                cluster_assignments = cluster_assignments.squeeze(0)
            cluster_assignments = pseudo_mask(cluster_assignments)
            for i in range(self.n_clusters):
                cluster_points = X[cluster_assignments[:,i] == 1]
                if cluster_points.size(0) > 0:
                    new_centroid = torch.mean(cluster_points, dim=0)
                    self.centroids[i] = new_centroid / new_centroid.norm(p=2)  # 归一化
            centroid_shift = torch.sum((self.centroids - prev_centroids) ** 2)
            if centroid_shift <= self.tol:
                break
def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
def get_center(model):
    classifier_weights = model.module.net.classifier[1].weight.data
    classifier_weights = F.normalize(classifier_weights, p=2, dim=1)
    return classifier_weights
class DomainAdaptation():
    def __init__(self, tau, model, config, optimizer):
        self.tau = tau
        self.model = model
        self.config = config
        self.optimizer = optimizer
        self.question_answer_pairs = gz2_pairs
        self.dependencies = gz2_and_decals_dependencies
        self.schema = schemas.Schema(self.question_answer_pairs, self.dependencies)
        self.device = "cuda:1"
        self.prototypes = None
        self.cluster_centers = None
        # self.scheduler = CosineAnnealingLR(optimizer, T_max=self.config.epochs)
    def compute_infoNCE_loss(self, z_t, W_s, pseudo_labels, t):
        total = torch.mm(z_t, W_s.t()) / t  # e.g. size 8*8
        loss = torch.zeros((total.shape[0], 10))
        for q_n in range(len(self.schema.question_index_groups)):
            q_indices = self.schema.question_index_groups[q_n]
            q_start = q_indices[0]
            q_end = q_indices[1]
            loss_ = torch.log_softmax(total[:, q_start:q_end + 1], dim=1) * pseudo_labels[:, q_start:q_end + 1]
            loss[:, q_n] = loss_.sum(dim=1)
        return -loss.sum(dim=1).mean()
        # return -(loss.sum(dim=1) / (loss != 0).sum(dim=1).float()).mean()
    def extract_feas(self, loader):
        features_list = []
        self.model.eval()
        with torch.no_grad():
            try:
                for X, _, _ in tqdm(loader, dynamic_ncols=True, desc="Extracting features"):
                    X = X.to(self.device)
                    features = self.model(X, get_features=True)
                    features_list.append(features)
            except:
                for X, _ in tqdm(loader, dynamic_ncols=True, desc="Extracting features"):
                    X = X.to(self.device)
                    features = self.model(X, get_features=True)
                    features_list.append(features)
        features_matrix = torch.cat(features_list, dim=0)
        features_matrix = F.normalize(features_matrix, p=2, dim=1)
        return features_matrix
    def clustering(self, features_matrix, init_centers):
        skmeans = SphericalKMeans(n_clusters=34, init_centers=init_centers)
        skmeans.fit(features_matrix)
        cluster_centers =skmeans.centroids
        return features_matrix, torch.Tensor(cluster_centers).to(self.device)
    def initialize_centers(self, train_loader, init_centers, epoch):
        self.model.eval()
        if epoch == 0:
            features_matrix = torch.Tensor(np.load("/data/public/renhaoye/mgs/features_matrix.npy"))
            return self.clustering(features_matrix, init_centers)
        else:
            features_matrix = self.extract_feas(train_loader)
            return self.clustering(features_matrix, init_centers)
    def pseudo_label(self, X, epoch, train=True):
        X_norm = F.normalize(X, p=2, dim=1)
        similarities = (torch.matmul(X_norm, self.cluster_centers.t().to(X_norm.device)))
        percent = 0.1
        # percent = 0.01 + 0.01 * epoch
        top_10_percent = int(percent * similarities.size(0))
        selected_rows = torch.zeros(similarities.size(0), dtype=torch.bool)
        for i in range(similarities.size(1)):
            _, indices = torch.topk(similarities[:, i], top_10_percent, largest=True)
            selected_rows[indices] = True
        similarities[~selected_rows,:] = 0
        result = torch.zeros_like(similarities)
        for q_n in range(len(self.schema.question_index_groups)):
            q_indices = self.schema.question_index_groups[q_n]
            q_start = q_indices[0]
            q_end = q_indices[1]
            input_slice = similarities[:, q_start:q_end + 1]
            max_prob, max_index = torch.max(input_slice, dim=1)
            # percent = 0.1 + 0.02 * epoch
            # max_prob = self.threshold(max_prob, percent)
            # condition = max_prob > -0.9
            condition = max_prob > 0.9 - epoch * 0.005
            # condition = max_prob > 0.7
            result[condition, q_start + max_index[condition]] = 1
        result = pseudo_mask(result)
        # max_prob, max_indices = torch.max(similarities, dim=1)
        # result[range(result.size(0)), max_indices] = 1
        return result
    def train_unsupervised_epoch(self, train_loader, epoch, writer):
        train_loss = 0
        self.model.train()
        with tqdm(train_loader, dynamic_ncols=True) as tqdmDataLoader:
            for X, image_path, y_hat in tqdmDataLoader:
                self.model.eval()
                X, y_hat = X.to(self.device), y_hat.to(self.device)
                self.optimizer.zero_grad()
                features = self.model(X, get_features=True) # 计算z_t^i
                features = features.to(self.device)
                z = F.normalize(features, p=2, dim=1) # 对特征进行l2归一化
                self.model.train()
                loss_value = self.compute_infoNCE_loss(z, self.prototypes.to(self.device), y_hat, self.tau)
                loss_value.backward()
                self.optimizer.step()
                train_loss += loss_value.item()
                tqdmDataLoader.set_postfix(
                    ordered_dict={
                        "epoch": epoch + 1,
                        "loss: ": loss_value.item(),
                        "LR": self.optimizer.param_groups[0]['lr'],
                    }
                )
        avg_train_loss = train_loss / len(train_loader)
        writer.add_scalar('Unsupervised Training loss by steps', avg_train_loss, epoch)
        return avg_train_loss
    def validate(self, valid_loader, writer, epoch):
        valid_loss = 0
        self.model.eval()
        for X, image_path, y_hat in valid_loader:
            X = X.to(self.device)
            y_hat = y_hat.to(self.device)
            enable_dropout(self.model)
            with torch.no_grad():
                features = self.model(X, get_features=True)
                features = features.to(self.device)
                z = F.normalize(features, p=2, dim=1)
                # y_hat = self.pseudo_label(z, epoch) # 计算伪标签
                loss_value = self.compute_infoNCE_loss(z, self.prototypes.to(self.device), y_hat, self.tau)
                valid_loss += loss_value.item()
        avg_valid_loss = valid_loss / len(valid_loader)
        writer.add_scalar('Unsupervised Validation loss by steps', avg_valid_loss, epoch)
        return avg_valid_loss
    def train_unsupervised(self, train_data, valid_data, init_loader):
        os.makedirs(self.config.save_dir + "log/", exist_ok=True)
        writer = SummaryWriter(self.config.save_dir + "log/")
        self.prototypes = get_center(self.model) # 初始化聚类中心 使用源模型的最后一层权重
        self.cluster_centers = get_center(self.model)
        for param in self.model.module.net.classifier.parameters(): # 冻结全连接层
            param.requires_grad = False
        for epoch in range(self.config.epochs):
            if epoch == 0:
                print(f"{self.config.save_dir.split('/')[-2]}_{epoch}.csv")
            features, self.cluster_centers = self.initialize_centers(init_loader, self.prototypes, epoch)  # 使用kmeans初始化聚类中心
            
            pseudo_labels = self.pseudo_label(features, epoch)
            train_data.set_pseudo_labels(pseudo_labels.detach().cpu())
            pseudo_indices = torch.nonzero((pseudo_labels.sum(dim=1)!=0)).squeeze().tolist()
            sub_train_data = Subset(train_data, pseudo_indices)
            train_loader  = DataLoader(dataset=sub_train_data, batch_size=self.config.batch_size,
                              shuffle=True, num_workers=self.config.WORKERS, pin_memory=True, drop_last=True)
            
            train_loss = self.train_unsupervised_epoch(train_loader, epoch, writer)
            
            self.save_checkpoint(epoch)
            print(f"epoch: {epoch}, train_loss: {train_loss}")
            # features = self.
            valid_loader = DataLoader(dataset=valid_data, batch_size=self.config.batch_size,
                              shuffle=False, num_workers=self.config.WORKERS, pin_memory=True, drop_last=True)
            del features
            if epoch == 0:
                features = torch.Tensor(np.load("/data/public/renhaoye/mgs/valid_features.npy"))
            else:
                features = self.extract_feas(valid_loader)
            # features = self.extract_feas(valid_loader)
            pseudo_labels = self.pseudo_label(features, epoch, False)
            valid_data.set_pseudo_labels(pseudo_labels.detach().cpu())
            pseudo_indices = torch.nonzero((pseudo_labels.sum(dim=1)!=0)).squeeze().tolist()
            sub_valid_data = Subset(valid_data, pseudo_indices)
            valid_loader = DataLoader(dataset=sub_valid_data, batch_size=self.config.batch_size,
                              shuffle=False, num_workers=self.config.WORKERS, pin_memory=True, drop_last=True)
            valid_loss = self.validate(valid_loader, writer, epoch)
            print(f"epoch: {epoch}, valid_loss: {valid_loss}")
            # self.scheduler.step()
            del train_loader
            gc.collect()
            print("------------start testing--------------")
            test_data = TestDataset(annotations_file="/data/public/renhaoye/morphics/dataset/overlap_north_raw_noerror.txt",
                                    transform=transforms.Compose([transforms.ToTensor()]), )
            test_loader = DataLoader(dataset=test_data, batch_size=256,
                                    shuffle=False, num_workers=8, pin_memory=True)
            predict_and_save_to_csv(self.model, test_loader, num_samples=25, output_file=f"{self.config.save_dir.split('/')[-2]}_{epoch}.csv", schema=self.schema)
    def save_checkpoint(self, epoch):
        os.makedirs(f'{self.config.save_dir}/checkpoint', exist_ok=True)
        torch.save(self.model, f'{self.config.save_dir}/model_{epoch}.pt')
def main(config):
    train_data = GalaxyDataset(annotations_file=config.train_file, transform=config.transfer)
    init_loader = DataLoader(dataset=train_data, batch_size=512,
                              shuffle=False, num_workers=config.WORKERS, pin_memory=True, drop_last=True)
    valid_data = GalaxyDataset(annotations_file=config.valid_file, transform=config.transfer)
    
    device_ids = [0,1]
    model = torch.load("/data/public/renhaoye/mgs/" + config.model, map_location='cuda:0')
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, betas=config.betas)
    adapter = DomainAdaptation(config.tau, model, config, optimizer)
    adapter.train_unsupervised(train_data, valid_data, init_loader)
def init_rand_seed(rand_seed):
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(rand_seed)  # 为所有GPU设置随机种子
    np.random.seed(rand_seed)
    random.seed(rand_seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
if __name__ == "__main__":
    init_rand_seed(1926)
    data_config = get_data_config()
    info = get_data_config()  # change this line
    os.makedirs(info['save_dir'], exist_ok=True)
    with open(info['save_dir'] + "info.txt", "w") as w:  # change this line
        for each in info.keys():
            attr_name = each
            attr_value = info[each]  # change this line
            w.write(str(attr_name) + ':' + str(attr_value) + "\n")
    os.system(f"cp {os.path.abspath(__file__)} {info['save_dir']}")
    parser = argparse.ArgumentParser(description='Morphics: Galaxy Classification Training')
    parser.add_argument('--train_file', type=str, default=data_config['train_file'],
                        help='Path to the training data annotations file')
    parser.add_argument('--valid_file', type=str, default=data_config['valid_file'],
                        help='Path to the validation data annotations file')
    parser.add_argument('--save_dir', type=str, default=data_config['save_dir'],
                        help='Directory to save logs, checkpoints, and trained models')
    parser.add_argument('--epochs', type=int, default=data_config['epochs'],
                        help='Number of epochs to training')
    parser.add_argument('--batch_size', type=int, default=data_config['batch_size'],
                        help='Batch size for training and validation')
    parser.add_argument('--WORKERS', type=int, default=data_config['WORKERS'],
                        help='Number of workers for data loading')
    parser.add_argument('--betas', type=tuple, default=data_config['betas'],
                        help='Optimizer parameters')
    parser.add_argument('--transfer', type=callable, default=data_config['transfer'],
                        help='Transforms to apply to the input data')
    parser.add_argument('--lr', type=float, default=data_config['lr'],
                        help='Learning rate for training')
    parser.add_argument('--patience', type=int, default=data_config['patience'],
                        help='Patience for early stopping')
    parser.add_argument('--phase', type=str, default=data_config['phase'],
                        help='Phase for training')
    parser.add_argument('--sample', type=int, default=data_config['sample'],
                        help='Sample nums for training')
    parser.add_argument('--dropout_rate', type=float, default=data_config['dropout_rate'],
                        help='Dropout rate for training')
    parser.add_argument('--dist_threshold', type=float, default=data_config['dist_threshold'],
                        help='Distance threshold for training')
    parser.add_argument('--model', type=str, default=data_config['model'],
                        help='Model to training')
    parser.add_argument('--tau', type=float, default=data_config['tau'],
                        help='Tau for training')
    parser.add_argument('--tmax', type=int, default=data_config['tmax'],
                        help='Tmax for training')
    args = parser.parse_args()
    main(args)
