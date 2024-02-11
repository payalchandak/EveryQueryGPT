# do i need to recompile ESGPT or something to add a new file?? 

from lightning.pytorch.callbacks import Callback

class MyPrintingCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is starting")

    def on_train_end(self, trainer, pl_module):
        print("Training is ending")
