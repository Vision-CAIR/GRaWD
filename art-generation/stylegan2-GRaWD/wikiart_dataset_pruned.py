from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
import scipy.io
import numpy as np
import random
import skimage
from skimage import filters
from pathlib import Path
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

## Noise Augmentation
class AddNoise(object):
    """Rotate by one of the given angles."""
    def __init__(self, noise_type, **kwargs):
        self.noise_type = noise_type
        self.kwargs = kwargs

    def __call__(self, x):
        x = x/255
        return (skimage.util.random_noise(x, mode=self.noise_type, **self.kwargs)*255).astype(np.uint8)
        
class GaussFilt(object):
    def __call__(self, x):
        x = x/255
        return (filters.gaussian(x, multichannel=True)*255).astype(np.uint8) 

gauss_tfm = transforms.Compose([AddNoise('gaussian', mean=0, var=0.008),
                                  transforms.ToPILImage()
                                 ])

speckle_tfm = transforms.Compose([AddNoise('speckle', mean=0, var=0.008),
                                  transforms.ToPILImage()
                                 ])

poisson_tfm = transforms.Compose([AddNoise('poisson'),
                                  transforms.ToPILImage()
                                 ]) 
                                 
gauss_filt = transforms.Compose([GaussFilt(),
                                  transforms.ToPILImage()
                                 ])     

class wikiart_dataset_pruned(Dataset):
    def __init__(self, opt, transform=None, pruned_file='groundtruth_pruned.mat'):

        self.root = '/tmp/wikiart_data/images' #wikiartimages.zip unzipped
        
        self.mat = scipy.io.loadmat(pruned_file)
        self.pruned_filenames = self.mat['groundtruth_pruned'][0][0][0]
        self.pruned_style_class = self.mat['groundtruth_pruned'][0][0][1]
        self.pruned_genre = self.mat['groundtruth_pruned'][0][0][2]
        self.pruned_artist = self.mat['groundtruth_pruned'][0][0][3]
        self.artist_to_label = {a:l for l,a in enumerate(np.unique(self.pruned_artist))}

        opt.n_styles = len(np.unique(self.pruned_style_class))
        opt.n_genres = len(np.unique(self.pruned_genre))
        opt.n_artists = len(np.unique(self.pruned_artist))

        print('wikiart dataset contains %d images'%(len(self.pruned_filenames)))
        print('wikiart dataset contains %s art_styles'%(len(np.unique(self.pruned_style_class))))
        print('wikiart dataset contains %s genres'%(len(np.unique(self.pruned_genre))))
        print('wikiart dataset contains %s artists'%(len(np.unique(self.pruned_artist))))

        if opt.data_aug == 'matlab':
            print("Data Augmentation Used: ", opt.data_aug)
            self.additional_tfms = [gauss_tfm, speckle_tfm, poisson_tfm, gauss_filt, None, None, None, None]       

        self.opt = opt

    def __getitem__(self, index):

        self.transform = transforms.Compose([
            transforms.Resize((self.resolution,self.resolution), Image.BICUBIC),
            transforms.CenterCrop(self.resolution),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                    (0.5, 0.5, 0.5))
        ])

        wikiartimage_path = os.path.join(self.root, self.pruned_filenames[index][0][0])      
        
        inputs = {}
        img = Image.open(wikiartimage_path).convert('RGB')

        if self.opt.data_aug == 'matlab':
            tfm_id = random.randint(0, len(self.additional_tfms)-1)
            if tfm_id != len(self.additional_tfms) - 1:
                # print('applying' , tfm_id, self.additional_tfms[tfm_id])
                img = self.additional_tfms[tfm_id](np.array(img)) 

        img = self.transform(img)

        art_style = self.pruned_style_class[index][0] - 1 ## -1 because the indexes start from 1 in the mat file.
        genre = self.pruned_genre[index][0] ## Did not do -1 because its already from 0
        artist = self.artist_to_label[self.pruned_artist[index][0]]
        inputs['img'] = img
        inputs['art_style'] = art_style 
        inputs['genre'] = genre
        inputs['artist'] = artist

        return inputs

    def __len__(self):
        return len(self.pruned_genre)

    def name(self):
        return 'wikiart_dataset_pruned'


class WikiartImage:
    def __init__(self, image_name, art_style, path, art_style_num):
        self.image_name = image_name
        self.art_style = art_style
        self.path = path
        self.art_style_num = art_style_num

def is_img(filename):
    exts = {".jpg", "jpeg", ".png"}
    return any(filename.endswith(ext) for ext in exts)