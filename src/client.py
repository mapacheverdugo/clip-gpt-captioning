'''
    Script for single prediction on an image. It puts result in the folder.
'''

import argparse

from predict import predict

parser = argparse.ArgumentParser()

parser.add_argument(
    '-C',
    '--checkpoint-name',
    type=str,
    default='model.pt',
    help='Checkpoint name'
)

parser.add_argument(
    '-S',
    '--size',
    type=str,
    default='S',
    help='Model size [S, L]',
    choices=['S', 'L', 's', 'l']
)

parser.add_argument(
    '-I',
    '--img-path',
    type=str,
    default='',
    help='Path to the image'
)

parser.add_argument(
    '-R',
    '--res-path',
    type=str,
    default='./data/result/prediction',
    help='Path to the results folder'
)

parser.add_argument(
    '-T',
    '--temperature',
    type=float,
    default=1.0,
    help='Temperature for sampling'
)

parser.add_argument(
    '-P',
    '--plot',
    type=bool,
    default=True,
    help='Generate plot',
    choices=[True, False]
)

args = parser.parse_args()


if __name__ == '__main__':
    caption = predict(args.img_path, args.size,
                      args.checkpoint_name, args.res_path, args.temperature, args.plot)

    print('Generated Caption: "{}"'.format(caption))
