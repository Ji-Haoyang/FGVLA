import os
import re
import json
import torch
import random
import datetime
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from alisuretool.Tools import Tools
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers.optimization import AdamW
from dataset.randaugment import RandomAugment
from torchvision.transforms import InterpolationMode
from utils import MetricLogger, SmoothedValue
from model import CreateModel
from nltk import pos_tag, word_tokenize
import nltk
nltk.data.path.append('/root/yun project/nltk_data')
from copy import deepcopy
import spacy


class RETrainDataset(Dataset):

    def __init__(self, ann_file, transform, image_root, max_words=30):
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.img_ids = {}

        n = 0
        for ann in self.ann:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1
                pass
            pass
        pass

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]

        image_path = os.path.join(self.image_root, ann['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        caption = self.pre_caption(ann['caption'], self.max_words)
        label = torch.tensor(ann['label'])
        return image, caption, self.img_ids[ann['image_id']], label

    @staticmethod
    def pre_caption(caption, max_words):
        caption = re.sub(r"([,.'!?\"()*#:;~])", '', caption.lower(),).replace(
            '-', ' ').replace('/', ' ').replace('<person>', 'person')

        caption = re.sub(r"\s{2,}", ' ', caption)
        caption = caption.rstrip('\n')
        caption = caption.strip(' ')

        # truncate caption
        caption_words = caption.split(' ')
        if len(caption_words) > max_words:
            caption = ' '.join(caption_words[:max_words])
        if not len(caption):
            raise ValueError("pre_caption yields invalid text")
        return caption

    pass


class REEvalDataset(Dataset):

    def __init__(self, ann_file, transform, image_root, max_words=30):
        self.ann = json.load(open(ann_file, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(RETrainDataset.pre_caption(caption, self.max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1
            pass
        pass

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_root, self.ann[index]['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image, index

    pass


class CreateDataset(object):

    def __init__(self):
        self.normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                              (0.26862954, 0.26130258, 0.27577711))
        self.pretrain_transform = transforms.Compose([
            transforms.RandomResizedCrop(config.image_res, scale=(0.2, 1.0), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                                  'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
            transforms.ToTensor(), self.normalize])
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(config.image_res, scale=(0.5, 1.0),
                                         interpolation=InterpolationMode.BICUBIC), transforms.RandomHorizontalFlip(),
            RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                                  'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
            transforms.ToTensor(), self.normalize])
        self.test_transform = transforms.Compose([transforms.Resize((config.image_res, config.image_res),
                                                                    interpolation=InterpolationMode.BICUBIC),
                                                  transforms.ToTensor(), self.normalize])

        self.train_dataset = RETrainDataset(config.train_file, self.train_transform, config.image_root)
        self.test_dataset = REEvalDataset(config.test_file, self.test_transform, config.image_root) #返回一张图片和它的索引，文本描述可以通过索引单独访问
        self.val_dataset = REEvalDataset(config.val_file, self.test_transform, config.image_root)

        self.train_loader = DataLoader(self.train_dataset, batch_size=config.batch_size_train,
                                       num_workers=4, pin_memory=True, shuffle=True, drop_last=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=config.batch_size_test,
                                      num_workers=4, pin_memory=True, shuffle=False, drop_last=False)
        self.val_loader = DataLoader(self.val_dataset, batch_size=config.batch_size_test,
                                     num_workers=4, pin_memory=True, shuffle=False, drop_last=False)
        pass

    pass


class RSITRBaseline(nn.Module):

    def __init__(self):
        super().__init__()
        create_model = CreateModel()
        self.model = create_model.create_model_and_transforms("ViT-B/32", pretrained=config.pretrain_path_open_clip)
        print(f"Model class: {self.model.__class__}")
        self.tokenize = create_model.tokenize


    def extract_nouns(self, text):
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        nouns = [word for word, tag in tagged if tag.startswith('NN')]
        return nouns if nouns else [tokens[0]]

    def get_vis_emb(self, image):
        features = self.model.encode_image(image, normalize=True)
        if isinstance(features, tuple):
            patch_feats, global_feat = features
            return patch_feats, global_feat
        else:
            return None, features


    def get_txt_emb(self, text_ids, return_word_feats=False):
        if return_word_feats:
            word_feats, sent_feat = self.model.encode_text(text_ids, normalize=True, return_word_feats=True)
            return word_feats, sent_feat
        else:
            sent_feat = self.model.encode_text(text_ids, normalize=True, return_word_feats=False)
            return None, sent_feat  


    def forward(self, image, text_ids, raw_texts, idx=None, label=None):

        patch_feats, global_img_feat = self.get_vis_emb(image)
        word_feats, global_txt_feat = self.get_txt_emb(text_ids, return_word_feats=True)


        noun_feats_list = []  
        for i, text in enumerate(raw_texts):
            text_nouns = self.extract_nouns(text) 
            noun_indices = [j for j, word in enumerate(text.split()) if word in text_nouns]
            if noun_indices:  
                noun_feats_list.append(word_feats[i, noun_indices]) 
            else:
                noun_feats_list.append(word_feats[i, :1])  


        loss_contrastive = self.get_contrastive_loss(
            global_img_feat, global_txt_feat, idx)
        loss_triplet =self.get_triplet_loss(
            global_img_feat, global_txt_feat)
        loss_fine = self.get_fine_grained_loss(
            patch_feats, noun_feats_list)
            
        return loss_contrastive, loss_triplet, loss_fine  

    
    def get_fine_grained_loss(self, patch_feats, noun_feats_list, temperature=0.07, pool_type='mean'):
        """Calculate fine-grained loss, combining contrastive learning and mask prediction
        Args:
            patch_feats: [B, N, D] Patch features 
            noun_feats_list: list of [num_nouns, D] List of noun features corresponding to each sample
            temperature: Temperature coefficient
            pool_type: Aggregation method, options are ['mean', 'max', 'attention', 'weighted', 'topk_mean']
        """
        B, N, D = patch_feats.shape
        device = patch_feats.device
        H = W = int(N**0.5)
        
        # Store the loss for each batch
        contrast_losses = []
        mask_losses = []
        
        for b in range(B):
            patch_feat_b = patch_feats[b]  # [N, D]
            noun_feats_b = noun_feats_list[b]  # [num_nouns, D]
            
            # Calculate similarity matrix
            similarity = torch.matmul(noun_feats_b, patch_feat_b.transpose(0, 1))  # [num_nouns, N]
            similarity_2d = similarity.view(len(noun_feats_b), H, W)  # [num_nouns, H, W]
            
            # 1. Calculate contrastive loss 
            for noun_idx in range(len(noun_feats_b)):
                noun_sim = similarity[noun_idx]  # [N]
                
                # Find the top-k most similar patches as positive samples
                k_pos = 5  
                topk_sim, topk_idx = torch.topk(noun_sim, k_pos)
                
  
                pos_feats = patch_feat_b[topk_idx]  # [k_pos, D]

                mask = torch.ones(N, dtype=torch.bool, device=device)
                mask[topk_idx] = False
                neg_feats = patch_feat_b[mask]  # [N-k_pos, D] 
                
                # Calculate similarity between anchor (noun) and positive/negative samples
                l_pos = torch.einsum('d,kd->k', noun_feats_b[noun_idx], pos_feats)  
                l_neg = torch.einsum('d,nd->n', noun_feats_b[noun_idx], neg_feats)  
                
                # Construct logits and labels
                logits = torch.cat([l_pos, l_neg]) 
                labels = torch.zeros(N, device=device)
                labels[:k_pos] = 1.0
                
                contrast_loss = -torch.sum(F.log_softmax(logits / temperature, dim=0) * labels)
                contrast_losses.append(contrast_loss)
            
            # 2. Calculate mask prediction loss
            # Aggregate similarity based on different pool_type
            if pool_type == 'mean':
                similarity_pooled = torch.mean(similarity_2d, dim=0)  # [H, W]
            elif pool_type == 'max':
                similarity_pooled = torch.max(similarity_2d, dim=0)[0]  # [H, W]

            # Take top-k to construct mask
            k = 5  
            topk_similar = torch.topk(similarity_pooled.view(-1), k)[1]
            pred_mask = torch.zeros_like(similarity_pooled)
            pred_mask.view(-1).scatter_(-1, topk_similar, 1)
            
            # Calculate mask prediction loss
            mask_loss = F.binary_cross_entropy_with_logits(
                similarity_pooled / temperature,
                pred_mask.float()
            )
            mask_losses.append(mask_loss)
        
        # Combine the two types of loss
        contrast_loss = torch.mean(torch.stack(contrast_losses))
        mask_loss = torch.mean(torch.stack(mask_losses))
        
        # Adjust the weights of the two losses  # Only change this for fine-grained loss weights
        lambda_contrast = 0.6
        lambda_mask = 0.4
        total_loss = lambda_contrast * contrast_loss + lambda_mask * mask_loss
        
        return total_loss

    def get_contrastive_loss(self, image_feat, text_feat, idx=None):
        logits = image_feat @ text_feat.t()
        
        logits = self.model.logit_scale *logits
        if idx is None:
            labels = torch.arange(image_feat.shape[0], device=image_feat.device)
            loss_i2t = F.cross_entropy(logits, labels)
            loss_t2i = F.cross_entropy(logits.t(), labels)
        else:
            idx = idx.view(-1, 1)
            pos_idx = torch.eq(idx, idx.t()).float()
            labels = pos_idx / pos_idx.sum(dim=1, keepdim=True)
            loss_i2t = -torch.sum(F.log_softmax(logits, dim=1) * labels, dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(logits.t(), dim=1) * labels, dim=1).mean()
        return (loss_i2t + loss_t2i) / 2.0

    def get_triplet_loss(self, image_feat, text_feat, margin=0.1):
        _scores = image_feat @ text_feat.t()
        _diagonal = _scores.diag().view(image_feat.shape[0], 1)

        def get_cost(diagonal, scores):
            cost = (margin + scores - diagonal.expand_as(scores)).clamp(min=0)
            cost = cost.masked_fill_(Variable(torch.eye(scores.size(0)) > .5).to(scores.device), 0)
            return cost.sum()

        sum_cost_s = get_cost(_diagonal, _scores)
        sum_cost_im = get_cost(_diagonal.t(), _scores)
        return (sum_cost_s + sum_cost_im) / 2.0

    pass


class Runner(object):

    def __init__(self):
        self.device = torch.device(config.device)

        self.model = RSITRBaseline()
        self.tokenize = self.model.tokenize
        self.model = self.model.to(self.device)

        self.set_trainable(self.model)
        Tools.print(f"learnable parameter num = {self.count_trainable_parameters()}", txt_path=config.log_filename)

        self.create_dataset = CreateDataset()

        self.optimizer = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=config.lr,
                               weight_decay=config.weight_decay, eps=1e-8, betas=(0.9, 0.98))
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, config.epochs * len(self.create_dataset.train_loader))

        self.best_weights = None
        self.best_val_score = 0
        self.warm_up_epochs = 2  
 
    @staticmethod
    def set_trainable(model):
        for name, module in model.named_modules():
            module.eval()
            for param in module.parameters():
                param.requires_grad = False
            pass
        for name, module in model.named_modules():
            if 'adapter' in name:
                module.train()
                for param in module.parameters():
                    param.requires_grad = True
                    # Tools.print(name)
                    pass
                pass
            pass
        pass

    @staticmethod
    def load_checkpoint(model, checkpoint):
        if checkpoint != '-1':
            checkpoint_value = torch.load(checkpoint, map_location='cpu')
            state_dict = checkpoint_value['model'] if 'model' in checkpoint_value.keys() else checkpoint_value
            msg = model.load_state_dict(state_dict, strict=False)
            print("missing", msg.missing_keys)
            print("unexp", msg.unexpected_keys)
            pass
        pass

    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def train(self):
        Tools.print("Start training ")
        best_result = self.test()
        for epoch in range(0, config.epochs):
            train_stats = self.train_one_epoch(self.create_dataset.train_loader, epoch)
            test_result = self.test()
            
            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_result.items()},
                'epoch': epoch,
            }
            Tools.print(json.dumps(log_stats), txt_path=config.log_filename)
            
            if test_result['r_mean'] > best_result['r_mean']:
                best_result = test_result
                
        return best_result

    def train_one_epoch(self, data_loader, epoch):
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('loss_contrastive', SmoothedValue(window_size=1, fmt='{value:.4f}'))
        metric_logger.add_meter('loss_triplet', SmoothedValue(window_size=1, fmt='{value:.4f}'))
        metric_logger.add_meter('loss_fine', SmoothedValue(window_size=1, fmt='{value:.4f}'))


        header = 'Train Epoch: [{}]'.format(epoch)

        self.model.train()
        for i, (image, text, idx, label) in enumerate(metric_logger.log_every(data_loader, 50, header)):
            image = image.to(self.device, non_blocking=True)
            idx = idx.to(self.device, non_blocking=True)
            text_input = self.tokenize(text).to(self.device)

            loss_contrastive, loss_triplet, loss_fine = self.model(
                image, text_input,text, idx=idx, label=label)
            loss = loss_contrastive + loss_triplet + loss_fine

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            metric_logger.update(loss_contrastive=loss_contrastive.item())
            metric_logger.update(loss_triplet=loss_triplet.item())
            metric_logger.update(loss_fine=loss_fine.item())
            metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])


        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    def test(self):
        score_test_i2t, score_test_t2i = self.evaluation(self.create_dataset.test_loader)
        test_result = self.itm_eval(score_test_i2t, score_test_t2i, self.create_dataset.test_dataset.txt2img,
                                    self.create_dataset.test_dataset.img2txt)
        Tools.print(f"Start evaluating test_result={test_result}")
        return test_result
    
    @torch.no_grad()
    def evaluation(self, data_loader):
        self.model.eval()
        alpha = 0.9  # Weight for coarse and fine-grained similarity

        def aggregate_similarity(patch_noun_sim, method='topk_mean', temperature=0.07, top_k=5):
            """Aggregate the similarity matrix between patches and nouns
            Args:
                patch_noun_sim: [num_nouns/num_patches, num_patches/num_nouns] Similarity matrix
                method: Aggregation method
                temperature: Temperature coefficient
                top_k: Number of top-k values to take
            """
            if method == 'max':
                return torch.max(patch_noun_sim)
        
            elif method == 'mean':
                return torch.mean(patch_noun_sim)
            

        # Inference img features
        image_patch_embeds = []
        image_global_embeds = []
        for image, img_id in data_loader:
            patch_embed, global_embed = self.model.get_vis_emb(image.to(self.device))
            image_patch_embeds.append(patch_embed)
            image_global_embeds.append(global_embed)

        # Inference text features
        text_word_embeds = []
        text_global_embeds = []
        text_nouns_indices = []  # Store the positions of nouns in each sentence
        texts = data_loader.dataset.text
        num_text = len(texts)
        text_bs = config.batch_size_test_text
    
        for i in range(0, num_text, text_bs):
            text_batch = texts[i: min(num_text, i + text_bs)]
        
            # Get text features
            text_input = self.tokenize(text_batch).to(self.device)
            word_embed, global_embed = self.model.get_txt_emb(text_input, return_word_feats=True)
        
            # Extract the positions of nouns in each sentence
            noun_indices = []
            for text in text_batch:
                nouns = self.extract_nouns(text)
                indices = [j for j, word in enumerate(text.split()) if word in nouns]
                noun_indices.append(indices if indices else [0])  # If no nouns, use the first token
            
            text_word_embeds.append(word_embed)
            text_global_embeds.append(global_embed)
            text_nouns_indices.extend(noun_indices)

        # Concatenate features
        image_patch_embeds = torch.cat(image_patch_embeds, dim=0)   # [N_img, 49, D]
        image_global_embeds = torch.cat(image_global_embeds, dim=0) # [N_img, D]
        text_word_embeds = torch.cat(text_word_embeds, dim=0)       # [N_txt, 77, D]
        text_global_embeds = torch.cat(text_global_embeds, dim=0)   # [N_txt, D]

        # 1. Calculate coarse-grained similarity
        sims_matrix_global = image_global_embeds @ text_global_embeds.t()  # [N_img, N_txt]

        # 2. Calculate fine-grained similarity
        sims_matrix_local_img2txt = torch.zeros_like(sims_matrix_global)  # [N_img, N_txt]
        sims_matrix_local_txt2img = torch.zeros_like(sims_matrix_global)  # [N_img, N_txt]

        # Set aggregation method
        aggregate_method = 'mean'  # Different aggregation methods can be chosen

        # Fine-grained similarity calculation when retrieving images from sentences  
        for i in range(len(text_word_embeds)):  # Iterate over each text
            txt_nouns = text_word_embeds[i, text_nouns_indices[i]]  # [num_nouns, D]
            for j in range(len(image_patch_embeds)):  # Iterate over each image
                img_patches = image_patch_embeds[j]  # [49, D]
                patch_noun_sim = torch.matmul(txt_nouns, img_patches.t())  # [num_nouns, 49]
                sims_matrix_local_txt2img[j, i] = aggregate_similarity(
                    patch_noun_sim,
                    method=aggregate_method,
                    temperature=0.07,
                    top_k=5
                )

        # Fine-grained similarity calculation when retrieving sentences from images
        for i in range(len(image_patch_embeds)):  # Iterate over each image
            img_patches = image_patch_embeds[i]  # [49, D]
            for j in range(len(text_word_embeds)):  # Iterate over each text
                txt_nouns = text_word_embeds[j, text_nouns_indices[j]]  # [num_nouns, D]
                patch_noun_sim = torch.matmul(img_patches, txt_nouns.t())  # [49, num_nouns]
                sims_matrix_local_img2txt[i, j] = aggregate_similarity(
                    patch_noun_sim,
                    method=aggregate_method,
                    temperature=0.07,
                    top_k=5
                )

        # 3. Combine the two types of similarity
        sims_matrix_txt2img = alpha * sims_matrix_global + (1-alpha) * sims_matrix_local_txt2img
        sims_matrix_img2txt = alpha * sims_matrix_global + (1-alpha) * sims_matrix_local_img2txt

        return sims_matrix_img2txt.cpu().numpy(), sims_matrix_txt2img.t().cpu().numpy()
    
    
    def extract_nouns(self, text):

        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)

        nouns = [word for word, tag in tagged if tag in ['NN', 'NNS', 'NNP', 'NNPS']]
    

        if len(nouns) > 1:

            from collections import Counter
            noun_counts = Counter(nouns)

            threshold = max(noun_counts.values()) * 0.5
            nouns = [noun for noun, count in noun_counts.items() if count >= threshold]
    
        return nouns if nouns else [tokens[0]]
        
    @torch.no_grad()
    def itm_eval(self, scores_i2t, scores_t2i, txt2img, img2txt):
        # Images->Text
        ranks = np.zeros(scores_i2t.shape[0])
        for index, score in enumerate(scores_i2t):
            inds = np.argsort(score)[::-1]
            # Score
            rank = 1e20
            for i in img2txt[index]:
                tmp = np.where(inds == i)[0][0]
                if tmp < rank:
                    rank = tmp
            ranks[index] = rank
            pass

        # Compute metrics
        tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

        # Text->Images
        ranks = np.zeros(scores_t2i.shape[0])

        for index, score in enumerate(scores_t2i):
            inds = np.argsort(score)[::-1]
            ranks[index] = np.where(inds == txt2img[index])[0][0]
            pass

        # Compute metrics
        ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

        tr_mean = (tr1 + tr5 + tr10) / 3
        ir_mean = (ir1 + ir5 + ir10) / 3
        r_mean = (tr_mean + ir_mean) / 2

        eval_result = {'txt_r1': round(tr1, 2),
                       'txt_r5': round(tr5, 2),
                       'txt_r10': round(tr10, 2),
                       'img_r1': round(ir1, 2),
                       'img_r5': round(ir5, 2),
                       'img_r10': round(ir10, 2),
                       'r_mean': round(r_mean, 2)}
        return eval_result

    pass


class ConfigCommon(object):

    def __init__(self):
        self.clean_gpu()
        self.gpu_id=0
        torch.cuda.set_device(self.gpu_id)
        #torch.cuda.set_device(self.get_gpu_id())
        self.setup_seed(2024)

        self.device = "cuda"
        self.epochs = 10
        self.lr = 0.001
        self.weight_decay = 0.01

        ################ Model setting  ###########################################################
        # Vision encoder setting
        self.image_res = 224  # no need modify
        self.patch_size = 32  # if use swin, set the patch_size to 32, else 16
        ############################################################################################

        ################ Training setting #########################################################
        self.pretrain_path_open_clip = "/root/open_clip/models--laion--CLIP-ViT-B-32-laion2B-s34B-b79K/open_clip_pytorch_model.bin"
        self.batch_size_train = 256
        self.batch_size_test = 128
        self.batch_size_test_text = 128
        ############################################################################################
        pass

    @staticmethod
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        pass

    @staticmethod
    def get_gpu_id():
        """
        torch.cuda.set_device(get_gpu_id())
        """
        import pynvml

        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        gpu_id, free = 0, 0
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            now_free = (info.free // 1048576) / 1024  # info.total, info.free, info.used
            if now_free > free:
                free = now_free
                gpu_id = i
            pass
        pynvml.nvmlShutdown()
        return gpu_id

    @staticmethod
    def clean_gpu(o=None):
        if o is not None:
            del o
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        pass

    pass

class Config_RSITMD_ViT(ConfigCommon):

    def __init__(self):
        super().__init__()

        self.output_dir = Tools.new_dir("./outputs/test_RSITMD_ViT")
        self.log_filename = os.path.join(self.output_dir, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-log.txt"))

        self.model = 'vit'
        self.lr = 0.001

        ############## The train & val & test set root ############################################
        self.image_root = '/root/rsitmd'
        self.train_file = ['data/finetune/rsitmd_train.json']  # Path to the training data file
        self.val_file = 'data/finetune/rsitmd_val.json'  # Path to the validation data file
        self.test_file = 'data/finetune/rsitmd_test.json'  # Path to the testing data file
        ############################################################################################
        pass

    pass


class Config_RSICD_ViT(ConfigCommon):

    def __init__(self):
        super().__init__()

        self.output_dir = Tools.new_dir("./outputs/test_RSICD_ViT")
        self.log_filename = os.path.join(self.output_dir, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-log.txt"))

        self.model = 'vit'
        self.lr = 0.001

        ############## The train & val & test set root ############################################
        self.image_root = '/root/rsicd'
        self.train_file = ['data/finetune/rsicd_train.json']  # Path to the training data file
        self.val_file = 'data/finetune/rsicd_val.json'  # Path to the validation data file
        self.test_file = 'data/finetune/rsicd_test.json'  # Path to the testing data file
        ############################################################################################
        pass

    pass






if __name__ == '__main__':

    result_list = []
    # for ConfigCLS in [Config_RSITMD_ViT, Config_RSICD_ViT]:
    for ConfigCLS in [Config_RSITMD_ViT]:
        config = ConfigCLS()
        runner = Runner()
        best_result = runner.train()
        result_list.append(best_result)
        pass

    for result_one in result_list:
        Tools.print(result_one)
        pass
    pass
