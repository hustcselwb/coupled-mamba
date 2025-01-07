import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from utils import MetricsTop, dict_to_str
import logging

logger = logging.getLogger('MMSA')

class mamba_train():
    def __init__(self, device, metrics, batch_size=1024, early_stop=10, key_eval='MAE', update_epochs=10):
        self.KeyEval = key_eval
        self.device = device
        self.metrics = metrics
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.update_epochs = update_epochs
        self.start_time = datetime.now()

    def do_train(self, mutimamba, train_loader, valid_loader, test_loader, return_epoch_results=True):
        optimizer = optim.Adam(mutimamba.parameters(), lr=0.0005, weight_decay=0.0005)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

        # Initialize results storage
        epoch_results = {'train': [], 'valid': [], 'test': []} if return_epoch_results else None
        best_valid = float('inf') if self.KeyEval == 'MAE' else -float('inf')

        epochs = 0
        while True:
            epochs += 1
            mutimamba.train()
            y_pred, y_true, train_loss = self.run_epoch(mutimamba, train_loader, optimizer)
            train_results = self.metrics(y_pred, y_true)

            # Log training results
            logger.info(f"Epoch {epochs} - Training Results: {dict_to_str(train_results)}")

            # Validation step
            val_results = self.do_test(mutimamba, valid_loader, mode="VAL")
            cur_valid = val_results[self.KeyEval]

            if self.is_better(cur_valid, best_valid):
                best_valid = cur_valid
                self.save_model(mutimamba)

            if return_epoch_results:
                epoch_results['train'].append(train_results)
                epoch_results['valid'].append(val_results)
                test_results = self.do_test(mutimamba, test_loader, mode="TEST")
                epoch_results['test'].append(test_results)

            if epochs > self.early_stop:
                return epoch_results if return_epoch_results else None

            if epochs % 30 == 0:
                scheduler.step()

    def run_epoch(self, model, data_loader, optimizer=None):
        y_pred, y_true, loss_total = [], [], 0
        with tqdm(data_loader) as td:
            for batch_data in td:
                audio, text, vision, label_m, id = batch_data
                audio, text, vision, label_m = [x.float().to(self.device) for x in [audio, text, vision, label_m]]

                optimizer.zero_grad() if optimizer else None

                model_out = model(self.batch_size, audio, vision, text, False)
                predict = model_out.float()
                label = label_m.float().to(self.device).view(-1, 1)

                loss = F.l1_loss(predict, label)
                loss_total += loss.item()

                if optimizer:
                    loss.backward()
                    optimizer.step()

                y_pred.append(predict)
                y_true.append(label)

        return torch.cat(y_pred), torch.cat(y_true), loss_total / len(data_loader)

    def do_test(self, model, dataloader, mode="VAL"):
        model.eval()
        y_pred, y_true, eval_loss = [], [], 0.0

        with torch.no_grad():
            for batch_data in dataloader:
                audio, text, vision, label_m, id = batch_data
                audio, text, vision, label_m = [x.float().to(self.device) for x in [audio, text, vision, label_m]]

                model_out = model(self.batch_size, audio, vision, text, False)
                predict = model_out.float()
                label = label_m.float().to(self.device).view(-1, 1)

                loss = F.l1_loss(predict, label)
                eval_loss += loss.item()

                y_pred.append(predict)
                y_true.append(label)

        eval_loss /= len(dataloader)
        logger.info(f"{mode} loss: {eval_loss:.4f}")
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        eval_results = self.metrics(pred, true)
        logger.info(f"{mode} results: {dict_to_str(eval_results)}")

        return eval_results

    def is_better(self, current, best):
        return current <= best if self.KeyEval == 'MAE' else current >= best

    def save_model(self, model):
        torch.save(model.cpu().state_dict(), f"/path/to/save/{self.start_time.strftime('%Y-%m-%d_%H-%M-%S')}_model.pth")
        model.to(self.device)
