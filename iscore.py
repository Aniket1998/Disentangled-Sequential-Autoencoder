import torch
import torch.nn.functional as F
import torch.distributions as dist
from model import *
import classifier as cls

def inception_score(p_logits):
    p = F.softmax(p_logits, dim=1)
    q = torch.mean(p, dim=0)
    kl = torch.sum(p * (F.log_softmax(p_logits, dim=1) - torch.log(q)), dim=1)
    return torch.exp(torch.mean(kl))

sample_size=100
std_f = 1.0
std_z = 1.0
dist_z = dist.Normal(0.0, std_z)
dist_f = dist.Normal(0.0, std_mu)
z_rand = dist_z.sample((sample_size, 8, 32))
f_rand = dist_f.sample((sample_size, 256))
f_rand = f_rand.unsqueeze(1).expand(-1, 8, 256)
zf = torch.cat((z_rand, f_rand), dim=2)

device = torch.device('cuda:0')
classifier = cls.SpriteClassifier()
classifier.load_state_dict(torch.load('./checkpoint_classifier.pth')['state_dict'])
classifier.to(device)
classifier.eval()
vae = DisentangledVAE(f_dim=256, z_dim=32, step=256, num_frames=8, num_lookup=6768)
vae.load_state_dict(torch.load('./model.pth')['state_dict'])
vae.to(device)
vae.eval()

x_rand = vae.decode_frames(zf)
p_logits = classifier(x_rand)
lookup = ('body', 'shirt', 'pants', 'hairstyle', 'action')
for i in range(len(lookup)):
    print('Inception Score of {} : {}'.format(lookup[i], inception_score(p_logits[i]).item()))
