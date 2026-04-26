config = {
    "project": "wgan_gp_celeba",
    "mode": "wgan-gp",
    "img_size": 64,
    "nz": 100,
    "in_ch": 3,
    "out_ch": 3,
    "ngf": 64,
    "ndf": 64,
    "lr": 0.0001,
    "lambda_gp": 10.0,
    "epochs": 50,
    "batch_size": 64,
    "save_every": 10,
    "out_dir": "outputs/wgan_gp"
}
