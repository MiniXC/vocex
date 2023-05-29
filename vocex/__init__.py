import torch


from .conformer_model import VocexModel

class Vocex():
    """ 
    This is a wrapper class for the vocex model. 
    It is used to load the model and perform inference on given audio file(s).
    """

    @staticmethod
    def from_pretrained(model_file):
        """ 
        Load a pretrained model from a given .pt file.
        Also accepts huggingface model names.
        """
        model = VocexModel.load_from_checkpoint(model_file)
        model.eval()
        return Vocex(model)