import os, argparse, torch
from pmc.config import Configurator

parser = argparse.ArgumentParser(description='Autonomous Diffusion Model (ADM)')
parser.add_argument(
    "--config", "-c", 
    type=str, 
    help="Path to config file"
)

if __name__ == '__main__':
    # parse arguments
    args = parser.parse_args()
    
    # configurate and save configuration file
    cc = Configurator(args)
    os.makedirs(cc.cfg.exp_dir, exist_ok=True)
    with open(f'{cc.cfg.exp_dir}/config.yaml', 'w') as f:
        f.write(str(cc.cfg))

    # initialize all modules
    exp, model, dataloader, callbacks = cc.init_all()

    # run
    exp(model, dataloader, callbacks)