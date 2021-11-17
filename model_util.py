import torch
import torch.nn as nn
import torch.nn.functional as F

class embedding_network(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.backbone = models.resnet34(pretrained=False)
        # modify resnet
        pdb.set_trace()
        self.backbone.fc = torch.nn.Identity()
        self.backbone.conv1 = nn.Conv2d(1,64,7, 2)
        freeze_parameters(self.backbone, train_fc=False)
        self.sequence_size = sequence_size
    def forward(self, rgb):


backbone_out = embedding_network()


image = np.load("/home/jinyoung/speaker_identification/output_short_noisy_VAD/test/fv01/fv01_t04_s43.npy")
pout = backbone_out(image)
print(pout)