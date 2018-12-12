import torch
import torch.distributions as dist
from model import *
import classifier as cls
from dataset import *
from tqdm import *

def kl_categorical(p_prob, q_prob, reduction='mean',eps=1e-7):
    if reduction == 'sum':
        return torch.sum(p_prob * torch.log((p_prob+eps)/(q_prob+eps)))
    else:
        return torch.sum(p_prob * torch.log((p_prob+eps)/(q_prob+eps)) / (1.0 * p_prob.size(0)))


def kl_uniform(p_prob, reduction='mean'):
    return kl_categorical(torch.ones_like(p_prob) / (1.0 * p_prob.size(1)), p_prob)

device = torch.device('cuda:2')
classifier = cls.SpriteClassifier()
classifier.to(device)
print('Loading Classifer')
classifier.load_state_dict(torch.load('./checkpoint_classifier.pth')['state_dict'])
classifier.eval()
vae = DisentangledVAE(f_dim=256, z_dim=32, frames=8, factorised=True)
vae.to(device)
vae.load_state_dict(torch.load('./factorised.pth')['state_dict'])
print('Loading VAE')
vae.eval()
dataset = Sprites('./dataset/lpc-dataset/test', 790)
std_z = 1.0
std_f = 1.0
dist_z = dist.Normal(0.0, std_z)
dist_f = dist.Normal(0.0, std_f)

num_repeats = 200
total = 0
mismatch = {
        'body': 0,
        'shirt': 0,
        'pants': 0,
        'hairstyle': 0,
        'action': 0,
}
kl_rand = {
        'body': 0,
        'shirt': 0,
        'pants': 0,
        'hairstyle': 0,
        'action': 0,
}
kl_pqs = {
        'body': 0,
        'shirt': 0,
        'pants': 0,
        'hairstyle': 0,
        'action': 0,
}


lookup = ('body', 'shirt', 'pants', 'hairstyle', 'action')
with torch.no_grad():
    for idx in tqdm(range(1,791)):
        _, _, _, _, _, _, item = dataset[idx]
        item = item.unsqueeze(0).to(device)
        _, _, f, _, _, z, _ = vae(item)
        f_expand = f.unsqueeze(1).expand(-1, vae.frames, vae.f_dim)
        p = classifier(item)
        max_p = []
        for i in range(len(p)):
            max_p.append(torch.max(p[i].data, 1)[1])
            kl_rand[lookup[i]] += kl_uniform(p[i]).item()
        for _ in range(num_repeats):
            z_rand = dist_z.sample(z.shape)
            x_rand = vae.decode_frames(torch.cat((z, f_expand), dim=2))
            q = classifier(x_rand)
            for i in range(len(p) - 1):
                kl_pqs[lookup[i]] += kl_categorical(q[i], p[i]).item()
                _, mq = torch.max(q[i], 1)
                mismatch[lookup[i]] += (mq != max_p[i]).sum().item()
            f_rand = dist_f.sample(f.shape)
            f_rand = f_rand.unsqueeze(1).expand(-1, vae.frames, vae.f_dim).to(device)
            x_randnew = vae.decode_frames(torch.cat((z, f_rand), dim=2))
            _, _, _, _, act = classifier(x_randnew)
            kl_pqs['action'] += kl_categorical(act, p[len(p) - 1]).item()
            _, mact = torch.max(act, 1)
            mismatch['action'] += (mact != max_p[len(p) - 1]).sum().item()
        total += num_repeats

print('MISMATCHES')
for name, val in mismatch.items():
    print(name, val / total)
print('KL[p_recon || p_data]')
for name, val in kl_pqs.items():
    print('{} : {}', name, val / total)
print('KL[p_random || p_data]')
for name, val in kl_rand.items():
    print('{} : {}', name, (val * num_repeats) / total)
