## Credits
This project is a modification of Pytorch-UNet by milesial.
- Original Repository: https://github.com/milesial/Pytorch-UNet
- License: GPL-3.0 License
Modifications include:
1. Added a structure with multiple UNETs connected in series.
2. Implemented checkpoints between the connected UNETs to reduce VRAM usage and improve memory efficiency during training.
3. Added output to monitor VRAM usage and plot detailed visualizations of training metrics.

