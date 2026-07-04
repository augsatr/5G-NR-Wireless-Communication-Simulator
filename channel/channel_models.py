from .awgn import AWGNChannel
from .rayleigh import RayleighChannel
from .rician import RicianChannel
from .doppler import DopplerChannel


class ChannelFactory:
    _channels = {
        "AWGN": AWGNChannel,
        "Rayleigh": RayleighChannel,
        "Rician": RicianChannel,
        "Doppler": DopplerChannel,
    }

    @staticmethod
    def create(channel_type, **kwargs):
        cls = ChannelFactory._channels.get(channel_type)
        if cls is None:
            raise ValueError(f"Unknown channel: {channel_type}. "
                             f"Choose from {list(ChannelFactory._channels.keys())}")
        return cls(**kwargs)
