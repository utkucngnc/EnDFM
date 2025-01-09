# Imports here
import argparse
import util.log as Log

def main(args):
    from src import EnDFM
    endfm = EnDFM(args)
    if 'fuse' in args.config.keys():
        endfm.fuse()
    if args.upsample:
        endfm.upsample()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/endfm.yaml', help='Path to the configuration file.')
    parser.add_argument('-i', '--input', nargs='*', type=str, default='input', help='Path to the input directory.')
    parser.add_argument('-o', '--output', type=str, default='results', help='Path to the output images.')
    parser.add_argument('-u', '--upsample', action='store_true', help='Upsample the output images.')
    
    args = parser.parse_known_args()[0]
    args = Log.prepare(args)

    main(args)