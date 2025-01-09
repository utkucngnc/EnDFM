import cv2
from functools import partial
import logging
import numpy as np
import os
import torch
from torch.utils.data import DataLoader

from . import aux

logger = logging.getLogger('base')

class EnDFM:
    def __init__(self, args):
        self.args = args
        self.experiment_root = args.experiment_root
        
        logger.info(f'EnDFM initialized with experiment root: {self.experiment_root}.')

        self.config = args.config
        self.gpu_ids = self.config['gpu_ids']
        self.device_str = 'cpu' if not self.gpu_ids else [f'cuda:{gpu}' for gpu in self.gpu_ids]
        self.device = torch.device(self.device_str)
        logger.info(f'Device set to {self.device_str}.')

        self.lr_dir = None


    def fuse(self) -> None:
        from .datasets import DDFMDataset as D
        from src.ddfm import guided_diffusion
        logger.info('Selected mode: Image Fusion')

        # Load configurations
        model_config = self.config['ddfm']['model']
        diffusion_config = self.config['ddfm']['diffusion']
        record = self.config['ddfm']['record']
        batch_size, shuffle, num_workers = self.config['dataloader'].values()
        save_lr = self.config['ddfm']['save_lr']
        dirs = self.args.input
        mode = aux.check_mode(self.config['ddfm']['mode'])

        # Check integrity of input directories
        logger.info('Checking integrity of input directories...')
        integrity = aux.check_integrity(dirs)
        logger.info(f'Integrity check complete. {integrity["Integrity Score"]} images found.')

        img_paths = {0: [os.path.join(dirs[0], img) for img in integrity['Images']],
                     1: [os.path.join(dirs[1], img) for img in integrity['Images']]}

        # Logging
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        
        # Load model
        model = guided_diffusion.unet.create_model(**model_config)
        model = model.to(self.device)
        model.eval()

        # Load diffusion sampler
        sampler = guided_diffusion.gaussian_diffusion.create_sampler(**diffusion_config)
        sampler_fn = partial(sampler.p_sample_loop, model=model)

        # Working directory
        save_dir = os.path.join(self.experiment_root, 'LR_fused')
        os.makedirs(save_dir, exist_ok=True)

        if record:
            for img_dir in ['recon', 'progress']:
                os.makedirs(os.path.join(save_dir, img_dir), exist_ok=True)
        
        if save_lr:
            import shutil
            for i, img_dir in enumerate[img_paths.values()]:
                tmp_path = os.path.join(self.experiment_root, f'LR_{i+1}')
                os.makedirs(tmp_path, exist_ok=True)
                for img in img_dir:
                    shutil.copy(img, os.path.join(tmp_path, os.path.basename(img)))
        
        # Load images
        dloader = DataLoader(D(img_paths, mode=mode), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

        # Fusion
        for _, data in enumerate(dloader):
            img_1 = data['M_1'].to(self.device)
            img_2 = data['M_2'].to(self.device)

            img_name = integrity['Images'][data['Index']]

            logger.info(f'Fusing images {img_name}...')

            # Sampling
            seed = 3407
            torch.manual_seed(seed)
            x_start = img_1.repeat(1, 3, 1, 1).shape if mode == 'L' else img_1.shape
            x_start = torch.randn(x_start, device=self.device)

            with torch.no_grad():
                out = sampler_fn(x_start = x_start,
                                record = record, 
                                I = img_1, 
                                V = img_2,
                                save_root = save_dir,
                                img_index = os.path.splitext(img_name)[0],
                                lamb = 0.5,
                                rho = 0.001)
            
            out = out.detach().cpu().squeeze().numpy()
            out = np.transpose(out, (1, 2, 0))
            out = cv2.cvtColor(out, cv2.COLOR_RGB2YCrCb)[:, :, 0]
            out = (out - np.min(out)) / (np.max(out) - np.min(out))
            out = ((out) * 255).astype(np.uint8)
            save_root = os.path.join(save_dir, 'recon') if record else save_dir
            cv2.imwrite(os.path.join(save_root, f'{img_name.split('.')[0]}.png'), out)
        
        logger.info('Fusion complete.')
        self.lr_dir = save_root
    

    def upsample(self):
        from .datasets import SRDataset as S
        from src.diwa import Model
        logger.info('Selected mode: Super Resolution')

        # Load configurations
        model_config = self.config['diwa']
        model_config['phase'] = 'val'
        model_config['gpu_ids'] = self.device_str if self.gpu_ids else None
        model_config['distributed'] = False
        batch_size, shuffle, num_workers = self.config['dataloader'].values()
        save_lr = model_config['save_lr']

        # Arrange input directories
        if self.lr_dir is not None:
            root = self.lr_dir
            save_dir = os.path.join(self.experiment_root, 'SR_fused')
            os.makedirs(save_dir, exist_ok=True)
            logger.info(f'No input directory provided. Using last fusion directory: {root}.')
        else:
            root = self.args.input
            assert os.path.exists(root), f'root path {root} does not exist'
            if save_lr:
                import shutil
                logger.info(f'Copying files from input directory {root} to experiment folder.')
                shutil.copytree(root, os.path.join(self.experiment_root, 'LR'), dirs_exist_ok=True)
            else:
                logger.info(f'Using root directory: {root}')
            self.lr_dir = root
            save_dir = os.path.join(self.experiment_root, 'SR')
            os.makedirs(save_dir, exist_ok=True)


        # Load dataset
        dataset = S(root, **model_config['data'])
        dloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
 

        # Logging
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        # Load model
        model = Model.create_model(model_config)
        model.set_new_noise_schedule(model_config['model']['beta_schedule']['val'], schedule_phase='val')

        # Super Resolution
        for _, data in enumerate(dloader):
            model.feed_data(data)
            model.test(continous=True)
            img = model.get_current_visuals(need_LR=False, sample=True)['SAM']
            img = img[-1]
            img_name = data['Name'][0]

            logger.info(f'Super Resolving image {img_name}...')
            try:
                img = img.squeeze()
                img = img.float().cpu().clamp_(min=-1, max=1)
                img = (img + 1) / 2
                img = img.permute(1, 2, 0).numpy() if img.dim() == 3 else img.numpy()
                img = (img * 255).astype(np.uint8)
            except:
                raise ValueError('Error converting tensor to image.')
            
            cv2.imwrite(os.path.join(save_dir, img_name), img)
        
        logger.info('Super Resolution complete.')