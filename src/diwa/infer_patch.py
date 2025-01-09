import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
import os

def patchify(img: torch.Tensor, patch_size: int = 128):
    assert img.dim() == 4, "Input tensor must be 4D"
    assert img.shape[2] % patch_size == 0, "Image width must be divisible by patch size"
    tot_row = img.shape[2] // patch_size
    tot_col = img.shape[3] // patch_size
    patches = torch.zeros(tot_row * tot_col, img.shape[1], patch_size, patch_size)
    for i in range(tot_row):
        for j in range(tot_col):
            patches[i * tot_col + j] = img[:, :, i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size]
    return patches

def assemble(patches: torch.Tensor, n_patch: int, patch_size: int = 128):
    tot_row = int(n_patch ** 0.5)
    tot_col = patches.shape[0] // tot_row
    h = w = tot_row * patch_size
    img = torch.zeros(1, 1, h, w)
    for i in range(tot_row):
        for j in range(tot_col):
            img[:, :, i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size] = patches[i * tot_col + j]
    assert img.ndim == 4
    return img
	

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_wave_128_256_patch.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['val'], help='val(generation)', default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_infer', action='store_true')
    parser.add_argument('-n','--patch', action='store_true')
    
    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)
    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    logger_val = logging.getLogger('val')


    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')
    
    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule']['val'], schedule_phase='val')
    
    logger.info('Begin Model Inference.')
    current_step = 0
    current_epoch = 0
    idx = 0

    result_path = '{}'.format(opt['path']['results'])
    os.makedirs(result_path, exist_ok=True)
    for _,  val_data in enumerate(val_loader):
        avg_psnr = 0.0
        avg_ssim = 0.0
        if opt["patch"]:
            idx += 1
            hr_data = patchify(val_data["HR"])
            sr_data  = patchify(val_data["SR"])
            n_patches = sr_data.shape[0]
            idx = val_data["Index"]
            fed_data = {"HR": hr_data, "SR": sr_data, "Index": idx}
            print(fed_data['SR'].shape)
            diffusion.feed_data(fed_data)
            diffusion.test(continous=True, patch = opt["patch"])
            visuals = diffusion.get_current_visuals(need_LR=False)
            
            hr_img = assemble(visuals['HR'], n_patches)
            fake_img = assemble(visuals['INF'], n_patches)
            sr_img = torch.zeros_like(sr_data)
            hr_img = Metrics.tensor2img(hr_img)  # uint8
            fake_img = Metrics.tensor2img(fake_img)  # uint8

            sr_img_mode = 'grid'
            if sr_img_mode == 'single':
                # single img series
                print(visuals['SR'].shape)
                sr_img = assemble(visuals['SR'])  # uint8
                sample_num = sr_img.shape[0]
                for iter in range(0, sample_num):
                    Metrics.save_img(
                        Metrics.tensor2img(sr_img[iter]), '{}/{}_{}_sr_{}.png'.format(result_path, current_step, idx, iter))
            else:
                # grid img
                assert visuals['SR'].ndim == 5, "Dimension error in output"
                assert visuals['SR'].shape[0] == n_patches
                for n, p in enumerate(visuals['SR']):
                    current_patch = Metrics.tensor2img(p)  # uint8
                    Metrics.save_img(current_patch, '{}/{}_{}_{}_sr_process.png'.format(result_path, current_step, idx.item(), n))
                    sr_img[n] = p[-1]
                sr_img = Metrics.tensor2img(assemble(sr_img, n_patches))
                Metrics.save_img(sr_img, '{}/{}_{}_sr.png'.format(result_path, current_step, idx.item()))

            Metrics.save_img(hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx.item()))
            Metrics.save_img(fake_img, '{}/{}_{}_inf.png'.format(result_path, current_step, idx.item()))
        else:
            idx += 1
            fed_data = {"HR": val_data["HR"], "SR": val_data["SR"], "Index": val_data["Index"]}
            diffusion.feed_data(fed_data)
            diffusion.test(continous=True, patch = opt["patch"])
            visuals = diffusion.get_current_visuals(need_LR=False)

            hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
            fake_img = Metrics.tensor2img(visuals['INF'])  # uint8

            sr_img_mode = 'grid'
            if sr_img_mode == 'single':
                # single img series
                print(visuals['SR'].shape)
                sr_img = visuals['SR']  # uint8
                sample_num = sr_img.shape[0]
                for iter in range(0, sample_num):
                    Metrics.save_img(
                        Metrics.tensor2img(sr_img[iter]), '{}/{}_{}_sr_{}.png'.format(result_path, current_step, idx, iter))
            else:
                if not opt["patch"]:
                    # grid img
                    sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
                    Metrics.save_img(
                        sr_img, '{}/{}_{}_sr_process.png'.format(result_path, current_step, idx))
                    Metrics.save_img(
                        Metrics.tensor2img(visuals['SR'][-1]), '{}/{}_{}_sr.png'.format(result_path, current_step, idx))
                else:
                    sr_img = Metrics.tensor2img(visuals['SR'])
                    Metrics.save_img(
                        sr_img, '{}/{}_{}_sr.png'.format(result_path, current_step, idx))

            Metrics.save_img(
                hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
            Metrics.save_img(
                fake_img, '{}/{}_{}_inf.png'.format(result_path, current_step, idx))
        avg_psnr += Metrics.calculate_psnr(fake_img, hr_img)
        avg_ssim += Metrics.calculate_ssim(fake_img, hr_img)
    logger_val.info(f'# Validation # PSNR: {avg_psnr / idx}')
    logger_val.info(f'# Validation # SSIM: {avg_ssim / idx}')

