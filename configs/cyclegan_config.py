config = {
    "project": "cyclegan_horse_to_zebra",
    "mode": "cyclegan",
    "img_size": 128,
    "in_ch": 3,
    "out_ch": 3,
    "ngf": 64,
    "ndf": 64,
    "n_blocks": 6,
    "lr": 0.0002,
    "lambda_cycle": 10.0,
    "lambda_id": 5.0,
    "buffer_size": 50,
    "epochs": 100,
    "batch_size": 1,
    "save_every": 20,
    "out_dir": "outputs/cyclegan"
}
