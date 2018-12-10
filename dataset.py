import torch.utils.data

class Sprites(torch.utils.data.Dataset):
    def __init__(self,path,size):
        self.path = path
        self.length = size;

    def __len__(self):
        return self.length
        
    def __getitem__(self,idx):
        item = torch.load(self.path+'/%d.sprite' % (idx+1))
        return item['id'], item['body'], item['shirt'], item['pant'], item['hair'], item['action'], item['sprite']