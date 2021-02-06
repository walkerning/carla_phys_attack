from torch.utils.data import Dataset

class myDataset(Dataset):
    def __init__(self,  ):
    pass
    def __len__(self):
        #返回最大长度
        return len
    def __getitem__(self,index):
        #返回每次应读取的单个数据
        return data,label

#例子
class myDataset(Dataset):
    def __init__(self,root,transform=None):
        # 所有图片的绝对路径
        imgs=os.listdir(root)
        #这句话可以使用glob快速加载 见66.
        self.imgs=[os.path.join(root,k) for k in imgs]
        self.transforms=transform

    def __getitem__(self, index):
        img_path = self.imgs[index]
        pil_img = Image.open(img_path)
        pil_img = pil_img.convert("RGB")
    if self.transforms:
            data = self.transforms(pil_img)
        else:
        pil_img = np.asarray(pil_img)
            data = torch.from_numpy(pil_img)
            label = xxxxx(这里省略，总之是得到这个图的标签）
        return data,label

    def __len__(self):
        return len(self.imgs)

#创建数据集实例并初始化
dataSet=FlameSet('./test',transform = transform)
#依然用Dataloader加载数据集
data = torch.utils.data.DataLoader(myDataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=0)

