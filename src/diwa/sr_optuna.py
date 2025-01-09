import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
import os
from sqlalchemy import create_engine
import optuna
import optuna.pruners as pruners
from optuna.storages import RetryFailedTrialCallback
import json

ROOT_DIR = '/work/py613215/thesis4'
os.makedirs(ROOT_DIR, exist_ok = True)
CHECKPOINT_DIR = '/work/py613215/thesis4/ckpts'
os.makedirs(CHECKPOINT_DIR, exist_ok = True)
STORAGE = 'sqlite:///thesis4'
STUDY_NAME = 'thesis4'
NUM_TRIALS = 60

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_wave_64_128_final.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')
    parser.add_argument('-patch', '-n', action = 'store_true')

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
    Logger.setup_logger('val_custom', opt['path']['log'], 'val_custom', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    
    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_set = Data.create_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(
                train_set, dataset_opt, phase)
        elif phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')
    if opt['phase'] == 'train':
        engine = create_engine(STORAGE)
        #optuna.delete_study(study_name=STUDY_NAME, storage=STORAGE)
        metric_dict = {}
        def objective(trial):
            # Suggestions for loss function
            #huber_delta = trial.suggest_float('HuberDelta',1.3,1.5,step=0.2)
            encoder_coeff = trial.suggest_categorical('Encoder loss coefficient', [0,10,100])
            encoder_coeff = 0 
            #opt['model']['diffusion']['loss_type'] = [encoder_coeff, huber_delta]
            opt['model']['diffusion']['loss_type'] = [encoder_coeff, 1.3]
            # Suggestions for optimizer
            #opt['train']["optimizer"]["lr"] = trial.suggest_float('LearningRate', 2e-6, 2e-5, step=2e-6)
            # Suggestion for model
            #opt['model']['diffusion']['image_size'] = trial.suggest_categorical('ImageSize', [256,512])
            opt['model']['unet']['dropout'] = trial.suggest_categorical('Dropout', [0.3, 0.5])
            opt['model']['unet']['res_blocks'] = trial.suggest_int('ResBlocks', 1,3)
            ind = trial.suggest_categorical('AttnRes', [1,2])
            opt['model']['unet']['attn_res'] = [16] if ind == 1 else [16,8]
            # model
            diffusion = Model.create_model(opt)
            logger.info('Initial Model Finished')

            # Train
            current_step = diffusion.begin_step
            current_epoch = diffusion.begin_epoch
            n_iter = opt['train']['n_iter']
            '''
            if opt['path']['resume_state']:
                logger.info('Resuming training from epoch: {}, iter: {}.'.format(
                    current_epoch, current_step))
            '''
            diffusion.set_new_noise_schedule(
                opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
            
            while current_step < n_iter:
                current_epoch += 1
                for i, train_data in enumerate(train_loader):
                    current_step += 1
                    if current_step > n_iter:
                        break
                    diffusion.feed_data(train_data)
                    diffusion.optimize_parameters()
                    # log
                    if current_step % opt['train']['print_freq'] == 0:
                        logs = diffusion.get_current_log()
                        message = '<epoch:{:3d}, iter:{:8,d}, loss:{:.4e}> '.format(current_epoch, current_step, logs['l_pix'])
                        logger.info(message)
                    # validation
                    if current_step % opt['train']['val_freq'] == 0:
                        avg_psnr = 0.0
                        avg_ssim = 0.0
                        avg_lpips = 0.0
                        idx = 0
                        result_path = '{}/{}'.format(opt['path']
                                                     ['results'], current_epoch)
                        os.makedirs(result_path, exist_ok=True)

                        diffusion.set_new_noise_schedule(
                            opt['model']['beta_schedule']['val'], schedule_phase='val')
                        for _,  val_data in enumerate(val_loader):
                            idx += 1
                            diffusion.feed_data(val_data)
                            diffusion.test(continous=False)
                            visuals = diffusion.get_current_visuals()
                            sr_img = visuals['SR'] 
                            hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
                            lr_img = Metrics.tensor2img(visuals['LR'])  # uint8
                            fake_img = Metrics.tensor2img(visuals['INF'])  # uint8

                            # generation
                            Metrics.save_img(
                                hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
                            if isinstance(sr_img, dict):
                                psnr = {}
                                ssim = {}
                                logger_val_custom = logging.getLogger('val_custom')
                                for k,v in sr_img.items():
                                    gen_img = Metrics.tensor2img(v)
                                    Metrics.save_img(gen_img, '{}/{}_{}_sr_{}.png'.format(result_path, current_step, idx, str(k)))
                                    psnr[k] = Metrics.calculate_psnr(gen_img, hr_img)
                                    ssim[k] = Metrics.calculate_ssim(gen_img, hr_img)
                                    logger_val_custom.info('<epoch:{:3d}, iter:{:8,d}, type: {}> psnr: {:.4e}, ssim: {:.4e}'.format(
                                        current_epoch, current_step, str(k), psnr[k], ssim[k]))
                                avg_psnr += psnr['mean']
                                avg_ssim += ssim['mean']
                            else:
                                sr_img = Metrics.tensor2img(sr_img)
                                try:
                                    Metrics.save_img(sr_img, '{}/{}_{}_sr.png'.format(result_path, current_step, idx))
                                except:
                                    raise TypeError(f"Type is not supported {type(sr_img)}, {sr_img.shape}")
                                avg_psnr += Metrics.calculate_psnr(sr_img, hr_img)
                                avg_ssim += Metrics.calculate_ssim(sr_img, hr_img)
                                avg_lpips += Metrics.calculate_lpips(sr_img, hr_img)

                            Metrics.save_img(
                                lr_img, '{}/{}_{}_lr.png'.format(result_path, current_step, idx))
                            Metrics.save_img(
                                fake_img, '{}/{}_{}_inf.png'.format(result_path, current_step, idx))

                        avg_psnr = avg_psnr / idx
                        avg_ssim = avg_ssim / idx
                        avg_lpips = avg_lpips / idx
                        
                        diffusion.set_new_noise_schedule(
                            opt['model']['beta_schedule']['train'], schedule_phase='train')
                        # log
                        logger.info('# Validation # PSNR: {:.4e}, SSIM: {:.4e}'.format(avg_psnr, avg_ssim))
                        logger_val = logging.getLogger('val')  # validation logger
                        logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}, ssim: {:.4e}, lpips: {:.4e}'.format(current_epoch, current_step, avg_psnr, avg_ssim, avg_lpips))
                        # set intermediate value
                        #score = -logs['l_pix'] + avg_ssim - avg_lpips + (avg_psnr / 35.0)
                        score = avg_ssim - avg_lpips + (avg_psnr / 35.0)
                        trial.report(score, current_epoch)
                        if trial.should_prune():
                            raise optuna.exceptions.TrialPruned()
            
            # save model
            logger.info('End of training.')
            trial_checkpoint_dir = os.path.join(CHECKPOINT_DIR, str(trial.number))
            os.makedirs(trial_checkpoint_dir, exist_ok = True)
            diffusion.save_network_custom(trial_checkpoint_dir, current_epoch, current_step)
            metric_dict[trial.number] = {'psnr': avg_psnr, 'ssim': round(avg_ssim,3), 'lpips': avg_lpips}
            #return logs['l_pix'], avg_psnr, avg_ssim, avg_lpips
            return score
            
        # Create study
        study = optuna.create_study(study_name = STUDY_NAME, storage = STORAGE, load_if_exists = False, pruner=pruners.MedianPruner(), direction='maximize')
        study.optimize(objective, n_trials = NUM_TRIALS)
        metric_dict['best_trial_id'] = study.best_trial.number
        with open(os.path.join(ROOT_DIR, 'summary_final.txt'), 'w') as f:
            print(metric_dict, file=f)
            f.close()
        
    else:
        logger.info('Begin Model Evaluation.')
        avg_psnr = 0.0
        avg_ssim = 0.0
        avg_lpips = 0.0
        idx = 0
        result_path = '{}'.format(opt['path']['results'])
        os.makedirs(result_path, exist_ok=True)
        for _,  val_data in enumerate(val_loader):
            idx += 1
            diffusion.feed_data(val_data)
            diffusion.test(continous=True)
            visuals = diffusion.get_current_visuals()

            hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
            lr_img = Metrics.tensor2img(visuals['LR'])  # uint8
            fake_img = Metrics.tensor2img(visuals['INF'])  # uint8

            sr_img_mode = 'grid'
            if sr_img_mode == 'single':
                # single img series
                sr_img = visuals['SR']  # uint8
                sample_num = sr_img.shape[0]
                for iter in range(0, sample_num):
                    Metrics.save_img(
                        Metrics.tensor2img(sr_img[iter]), '{}/{}_{}_sr_{}.png'.format(result_path, current_step, idx, iter))
            else:
                # grid img
                sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
                Metrics.save_img(
                    sr_img, '{}/{}_{}_sr_process.png'.format(result_path, current_step, idx))
                Metrics.save_img(
                    Metrics.tensor2img(visuals['SR'][-1]), '{}/{}_{}_sr.png'.format(result_path, current_step, idx))

            Metrics.save_img(
                hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
            Metrics.save_img(
                lr_img, '{}/{}_{}_lr.png'.format(result_path, current_step, idx))
            Metrics.save_img(
                fake_img, '{}/{}_{}_inf.png'.format(result_path, current_step, idx))

            # generation
            eval_psnr = Metrics.calculate_psnr(Metrics.tensor2img(visuals['SR'][-1]), hr_img)
            eval_ssim = Metrics.calculate_ssim(Metrics.tensor2img(visuals['SR'][-1]), hr_img)
            eval_lpips = Metrics.calculate_lpips(Metrics.tensor2img(visuals['SR'][-1]), hr_img)

            avg_psnr += eval_psnr
            avg_ssim += eval_ssim
            avg_lpips += eval_lpips

        avg_psnr = avg_psnr / idx
        avg_ssim = avg_ssim / idx
        avg_lpips = avg_lpips / idx

        # log
        logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
        logger.info('# Validation # SSIM: {:.4e}'.format(avg_ssim))
        logger.info('# Validation # LPIPS: {:.4e}'.format(avg_lpips))
        logger_val = logging.getLogger('val')  # validation logger
        logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}, ssim：{:.4e}, lpips: {:.4e}'.format(
            current_epoch, current_step, avg_psnr, avg_ssim, avg_lpips))
