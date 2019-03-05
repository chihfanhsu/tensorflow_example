#-*- coding: utf-8 -*-
import argparse

model_config = argparse.ArgumentParser()

# hyper parameter
model_config.add_argument('--lr', type=eval, default=1e-4, help='')
model_config.add_argument('--MAX_ITERATION', type=eval, default=1e4, help='')
model_config.add_argument('--batch_size', type=eval, default=128, help='')

# training parameter
model_config.add_argument('--tar_model', type=str, default='model_1', help='')
model_config.add_argument('--gpu', type=eval, default=0, help='')
model_config.add_argument('--training', type=bool, default=True, help='')
model_config.add_argument('--debug', type=bool, default=False, help='')
def get_config():
    config, unparsed = model_config.parse_known_args()
    print(config)
    return config, unparsed
