import argparse
import torch
from torch.utils.data import DataLoader
from core.models.dcgan import DCGANGenerator, DCGANDiscriminator
from core.models.wgan import WGANGenerator, WGANCritic
from core.models.pix2pix import Pix2PixGenerator, Pix2PixDiscriminator
from core.models.cyclegan import CycleGANGenerator, CycleGANDiscriminator
from core.data.dataset import SingleDomainDataset, PairedDataset, UnpairedDataset
from core.trainers.gan_trainer import GANTrainer
from core.trainers.pix2pix_trainer import Pix2PixTrainer
from core.trainers.cyclegan_trainer import CycleGANTrainer

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.mode == 'dcgan':
        from configs.dcgan_config import config
        gen = DCGANGenerator(config['nz'], config['ngf'], config['out_ch'])
        disc = DCGANDiscriminator(config['in_ch'], config['ndf'])
        dataset = SingleDomainDataset(args.data_root, config['img_size'])
        loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
        trainer = GANTrainer(gen, disc, config, device, mode='dcgan')
        trainer.train(loader, num_epochs=config['epochs'])

    elif args.mode == 'wgan-gp':
        from configs.wgan_config import config
        gen = WGANGenerator(config['nz'], config['ngf'], config['out_ch'])
        critic = WGANCritic(config['in_ch'], config['ndf'])
        dataset = SingleDomainDataset(args.data_root, config['img_size'])
        loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
        trainer = GANTrainer(gen, critic, config, device, mode='wgan-gp')
        trainer.train(loader, num_epochs=config['epochs'])

    elif args.mode == 'pix2pix':
        from configs.pix2pix_config import config
        gen = Pix2PixGenerator(config['in_ch'], config['out_ch'], config['ngf'])
        disc = Pix2PixDiscriminator(config['in_ch'] + config['out_ch'], config['ndf'])
        dataset = PairedDataset(args.sketch_dir, args.color_dir, config['img_size'])
        loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
        trainer = Pix2PixTrainer(gen, disc, config, device)
        trainer.train(loader, num_epochs=config['epochs'])

    elif args.mode == 'cyclegan':
        from configs.cyclegan_config import config
        g_ab = CycleGANGenerator(config['in_ch'], config['out_ch'], config['ngf'], config['n_blocks'])
        g_ba = CycleGANGenerator(config['out_ch'], config['in_ch'], config['ngf'], config['n_blocks'])
        d_a = CycleGANDiscriminator(config['in_ch'], config['ndf'])
        d_b = CycleGANDiscriminator(config['out_ch'], config['ndf'])
        dataset = UnpairedDataset(args.dir_a, args.dir_b, config['img_size'])
        loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
        trainer = CycleGANTrainer(g_ab, g_ba, d_a, d_b, config, device)
        trainer.train(loader, num_epochs=config['epochs'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generative AI Pipeline")
    parser.add_argument("--mode", type=str, required=True, choices=['dcgan', 'wgan-gp', 'pix2pix', 'cyclegan'])
    parser.add_argument("--data_root", type=str, help="Path for DCGAN/WGAN data")
    parser.add_argument("--sketch_dir", type=str, help="Path for Pix2Pix sketches")
    parser.add_argument("--color_dir", type=str, help="Path for Pix2Pix colors")
    parser.add_argument("--dir_a", type=str, help="Path for CycleGAN Domain A")
    parser.add_argument("--dir_b", type=str, help="Path for CycleGAN Domain B")
    
    args = parser.parse_args()
    main(args)
