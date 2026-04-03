from typing import Optional

from .encoder import Encoder
from .encoder_idesplat import EncoderIDESplat, EncoderIDESplatCfg
from .visualization.encoder_visualizer import EncoderVisualizer
from .visualization.encoder_visualizer_idesplat import EncoderVisualizerIDESplat

ENCODERS = {
    "idesplat": (EncoderIDESplat, EncoderVisualizerIDESplat),
}

EncoderCfg = EncoderIDESplatCfg


def get_encoder(cfg: EncoderCfg) -> tuple[Encoder, Optional[EncoderVisualizer]]:
    encoder, visualizer = ENCODERS[cfg.name]
    encoder = encoder(cfg)
    if visualizer is not None:
        visualizer = visualizer(cfg.visualizer, encoder)
    return encoder, visualizer
