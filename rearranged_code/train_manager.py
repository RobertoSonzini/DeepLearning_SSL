import torch
import torch.nn.functional as F
from itertools import cycle
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

class BaseTrainer:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    @torch.no_grad()
    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0.
        all_preds = []
        all_labels = []

        for x, y in data_loader:
            x, y = x.to(self.device), y.to(self.device)

            logits = self.model(x)
            loss = F.cross_entropy(logits, y) # verify
            total_loss += loss.item()

            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        acc = (all_preds == all_labels).mean()
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        cm = confusion_matrix(all_labels, all_preds)
        tpr_list = []
        fpr_list = []
        for i in range(len(cm)):
            TP = cm[i, i]
            FN = cm[i, :].sum() - TP
            FP = cm[:, i].sum() - TP
            TN = cm.sum() - (TP + FP + FN)

            TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            FPR = FP / (FP + TN) if (FP + TN) > 0 else 0.0

            tpr_list.append(TPR)
            fpr_list.append(FPR)

        tpr_mean = sum(tpr_list) / len(tpr_list)
        fpr_mean = sum(fpr_list) / len(fpr_list)

        return {
            'loss': total_loss / len(data_loader),
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'TPR': tpr_mean,
            'FPR': fpr_mean
        }

    def util_validate(self, val_loader):
        return self.evaluate(val_loader)

    def util_test(self, test_loader):
        return self.evaluate(test_loader)
    
class SSLTrainer_threshold(BaseTrainer):
    def __init__(self, model, device, lambda_u=1.0, threshold=0.95):
        super().__init__(model, device)
        self.lambda_u = lambda_u
        self.threshold = threshold
        self.conf_history_full = []

    def train_epoch(self, labeled_loader, unlabeled_loader, optimizer):
        self.model.train()
        total_loss = 0.
        sup_loss_total = 0.
        unsup_loss_total = 0.

        conf_history = []

        labeled_iter = cycle(labeled_loader)  # oversampling
        if unlabeled_loader is None:
            total_loss = 0.
            n_batches = 0

            for x_l, y_l in labeled_loader:
                x_l, y_l = x_l.to(self.device), y_l.to(self.device)

                logits = self.model(x_l)
                loss = F.cross_entropy(logits, y_l)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            return {
                'total_loss': total_loss / n_batches,
                'supervised_loss': total_loss / n_batches,
                'unsupervised_loss': 0.0,
                'conf_history': []
            }

        for x_u_weak, x_u_strong, _ in tqdm(unlabeled_loader, desc="SSL training"):
            x_l, y_l = next(labeled_iter)

            x_l, y_l = x_l.to(self.device), y_l.to(self.device)
            x_u_weak, x_u_strong = x_u_weak.to(self.device), x_u_strong.to(self.device)

            logits_l = self.model(x_l)
            loss_sup = F.cross_entropy(logits_l, y_l)

            with torch.no_grad():
                logits_u_weak = self.model(x_u_weak)
                probs = F.softmax(logits_u_weak, dim=1)
                max_probs, pseudo_labels = torch.max(probs, dim=1)
                
                conf_mean = max_probs.mean().item()

                mask = max_probs >= self.threshold

            conf_history.append(conf_mean)

            if mask.sum() > 0:
                logits_u_strong = self.model(x_u_strong)
                loss_unsup = (
                    F.cross_entropy(logits_u_strong, pseudo_labels, reduction='none')
                    * mask
                ).sum() / mask.sum()
            else:
                loss_unsup = torch.tensor(0., device=self.device)

            # Combined loss
            loss = loss_sup + self.lambda_u * loss_unsup

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            sup_loss_total += loss_sup.item()
            unsup_loss_total += loss_unsup.item()

        n_batches = len(unlabeled_loader)

        self.conf_history_full.extend(conf_history)
        
        return {
            'total_loss': total_loss / n_batches,
            'supervised_loss': sup_loss_total / n_batches,
            'unsupervised_loss': unsup_loss_total / n_batches,
            'conf_history': conf_history
        }

class SupervisedTrainer(BaseTrainer):
    def __init__(self, model, device):
        super().__init__(model, device)

    def train_epoch(self, labeled_loader, optimizer):
        self.model.train()
        total_loss = 0.

        for x, y in labeled_loader:
            x, y = x.to(self.device), y.to(self.device)

            logits = self.model(x)
            loss = F.cross_entropy(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return {
            'loss': total_loss / len(labeled_loader)
        }
