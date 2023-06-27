from torch import nn
import torch
import torch.nn.functional as F

class SpeakerLoss(nn.Module):
    def __init__(self, hidden_size, num_speakers):
        super().__init__()
        self.linear = nn.Linear(hidden_size, num_speakers)

    def forward(self, speaker_embedding, speaker, return_pred=False):
        speaker_output = self.linear(speaker_embedding)
        speaker_loss = nn.CrossEntropyLoss()(speaker_output, speaker)
        if not return_pred:
            return speaker_loss
        else:
            return speaker_loss, speaker_output
    
class AngularMarginLoss(nn.Module):
    """
    Generic angular margin loss definition
    (see https://github.com/cvqluu/Angular-Penalty-Softmax-Losses-Pytorch)

    "ElasticFace: Elastic Margin Loss for Deep Face Recognition",
    Boutros et al., https://arxiv.org/abs/2109.09416v2
    """

    def __init__(
        self,
        hidden_size,
        num_speakers,
        scale=None,
        m1=1,
        m2=0,
        m3=0,
        eps=1e-6,
    ):
        super(AngularMarginLoss, self).__init__()
        self.fc = nn.Linear(hidden_size, num_speakers, bias=False)
        self.scale = scale
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.eps = eps

    def forward(self, inputs, targets):
        """
        Compute ArcFace loss for inputs of shape [B, E] and
        targets of size [B]

        B: batch size
        E: embedding size
        """
        # Normalize weights
        self.fc.weight.data = F.normalize(self.fc.weight.data, p=2, dim=1)

        # Normalize inputs
        inputs_norms = torch.norm(inputs, p=2, dim=1)
        normalized_inputs = inputs / inputs_norms.unsqueeze(-1).repeat(
            1, inputs.size(1)
        )

        # Set scale
        scales = (
            torch.tensor([self.scale], device=inputs.device).repeat(inputs.size(0))
            if self.scale is not None
            else inputs_norms
        )

        # Cosine similarity is given by a simple dot product,
        # given that we normalized both weights and inputs
        cosines = self.fc(normalized_inputs).clamp(-1, 1)
        preds = torch.argmax(cosines, dim=1)

        # Recover angles from cosines computed
        # from the previous linear layer
        angles = torch.arccos(cosines)

        # Compute loss numerator by converting angles back to cosines,
        # after adding penalties, as if they were the output of the
        # last linear layer
        numerator = scales.unsqueeze(-1) * (
            torch.cos(self.m1 * angles + self.m2) - self.m3
        )
        numerator = torch.diagonal(numerator.transpose(0, 1)[targets])

        # Compute loss denominator
        excluded = torch.cat(
            [
                scales[i]
                * torch.cat((cosines[i, :y], cosines[i, y + 1 :])).unsqueeze(0)
                for i, y in enumerate(targets)
            ],
            dim=0,
        )
        denominator = torch.exp(numerator) + torch.sum(torch.exp(excluded), dim=1)

        # Compute cross-entropy loss
        loss = -torch.mean(numerator - torch.log(denominator + self.eps))

        return loss