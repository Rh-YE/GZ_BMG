"""跑测试集的代码"""
import csv
root = "/data/public/renhaoye/mgs/"
from dataset.galaxy_dataset import *
from utils import schemas
from utils.utils import *
import random
from torch.backends import cudnn
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
def init_rand_seed(rand_seed):
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)
    np.random.seed(rand_seed)
    random.seed(rand_seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
def get_expected_votes_ml(prob_of_answers, question, votes_for_base_question: int, schema, round_votes):
    prev_q = question.asked_after
    if prev_q is None:
        expected_votes = torch.ones(prob_of_answers.shape[0]).to(prob_of_answers.device) * votes_for_base_question
    else:
        joint_p_of_asked = schema.joint_p(prob_of_answers, prev_q.text)  # prob of getting the answer needed to ask this question
        expected_votes = joint_p_of_asked * votes_for_base_question
    if round_votes:
        return torch.round(expected_votes)
    else:
        return expected_votes
def predict_and_save_to_csv(model, data_loader, num_samples, output_file, schema):
    # features_list = []
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
                    # output = softmax_output(output, schema.question_index_groups) * 99 + 1
                    output = vote2prob(output, schema.question_index_groups)
                    # output = tree_predict(output, schema)
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
def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
if __name__ == '__main__':
    init_rand_seed(1926)
    question_answer_pairs = gz2_pairs
    dependencies = gz2_and_decals_dependencies
    schema = schemas.Schema(question_answer_pairs, dependencies)
    device_ids = [1]
    model = torch.load("/data/public/renhaoye/mgs/ye_2023_sigmoid_raw/model_54.pt", map_location="cuda:1")
    test_data = TestDataset(annotations_file="/data/public/renhaoye/morphics/dataset/overlap_north_raw_noerror.txt",
                            transform=transforms.Compose([transforms.ToTensor()]), )
    test_loader = DataLoader(dataset=test_data, batch_size=256,
                              shuffle=False, num_workers=8, pin_memory=True)
    predict_and_save_to_csv(model, test_loader, num_samples=25, output_file="overlap_south_source_model.csv", schema=schema)
