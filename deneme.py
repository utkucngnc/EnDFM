from util.log import load_yaml
#from core.datasets import create_dataloader

conf = load_yaml('config/EnDFM.yaml')
root = 'input'

#dl = create_dataloader(**conf['dataset'])

a = {0: [1,2,3,4,5], 1: [1,2,3,4,5]}

b,c = a.values()

print(f'{b}\n{c}')