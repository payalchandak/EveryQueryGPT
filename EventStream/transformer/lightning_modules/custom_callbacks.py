import numpy as np 
import torch
import lightning as L
import wandb


class MonitorInputCallback(L.Callback): 

    def __init__(self, prefix='input', log_context=False, log_query=True): 
        self.prefix = prefix+'/'
        self.log_context = log_context
        self.log_query = log_query

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        context, query, answer = batch
        log_dict = {}
        if self.log_context:
            for k in context.keys(): 
                log_dict[self.prefix+"context_"+k] = wandb.Histogram(np.array(context[k].tolist()))
        if self.log_query:
            for k in query.keys(): 
                log_dict[self.prefix+"query_"+k] = wandb.Histogram(np.array(query[k].tolist()))
        log_dict[self.prefix+"answer"] = wandb.Histogram(np.array(answer.tolist())),
        trainer.logger.experiment.log(log_dict)


class AnomalyDetectionCallback(L.Callback):

    def __init__(self, action='log', print_batch_on_anomaly=True, checkpoint_on_anomaly=True):
        assert action in {'log', 'zero', 'error'}, "Determines what to do when NaN or Inf detected"
        self.action = action
        self.checkpoint_on_anomaly = checkpoint_on_anomaly
        self.print_batch_on_anomaly = print_batch_on_anomaly
        self.anomaly_detected = False
        self.current_batch = None
        self.current_batch_idx = None  

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx): 
        self.current_batch = batch
        self.current_batch_idx = batch_idx
        self.anomaly_detected = self._check_for_anomaly(pl_module, check_grads=False, trainer=trainer)
        if self.print_batch_on_anomaly and self.anomaly_detected: self._print_current_batch()

    def on_after_backward(self, trainer, pl_module):
        self.anomaly_detected = self._check_for_anomaly(pl_module, check_grads=True, trainer=trainer)
        if self.print_batch_on_anomaly and self.anomaly_detected: self._print_current_batch()

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        if self.anomaly_detected and self.checkpoint_on_anomaly:
            self._checkpoint_model(trainer, pl_module)
        if self.anomaly_detected and self.action == 'zero':
            optimizer.zero_grad()
            print("NaN detected, skipping optimizer step.") 
        self.anomaly_detected = False

    def _check_for_anomaly(self, pl_module, check_grads, trainer):
        anomaly = False 
        for name, param in pl_module.named_parameters():
            target = param.grad if check_grads else param
            if target is not None: 
                if torch.isnan(target).any():
                    anomaly = True
                    print( f"NaN detected in {'gradients' if check_grads else 'parameters'} of {name}." )
                elif torch.isinf(target).any():
                    anomaly = True
                    print( f"Inf detected in {'gradients' if check_grads else 'parameters'} of {name}." )
        return anomaly

    def _print_current_batch(self): 
        context, query, answer = self.current_batch
        print(f"Batch causing nans was idx {self.current_batch_idx} with following info:")
        # for k in context.keys(): 
        #     print(f"context {k} \n {context[k]}")
        for k in query.keys(): 
            print(f"query {k} \n {query[k]}")
        print(f"answer \n {answer}")

    def _checkpoint_model(self, trainer, pl_module): 
        checkpoint_name = f"nan_checkpoint-step={trainer.global_step}.ckpt"
        pl_module.model.save_pretrained(checkpoint_name) 
        print(f"Model checkpointed at step {trainer.global_step} due to NaN detection: {checkpoint_name}.")
