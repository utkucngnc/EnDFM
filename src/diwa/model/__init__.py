import logging
from . import model as Model

logger = logging.getLogger('base')


def create_model(opt):
    m = Model.DDPM(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
