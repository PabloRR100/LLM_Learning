import logging

import torch


LOG = logging.getLogger(__name__)


def get_auto_device():
    """
    Picks the best device to run torch on.

    If LIT_DEVICE is set in the environment, it will set the device to whatever
    is set in the aforementioned env var.

    If no GPU is available, will return allocate to CPU. With one GPU, with
    automatically allocate to the only GPU. With more than one GPU, allocates
    to the device with the lowest utilization.

    Returns:
        torch.device: The best determined device.
    """
    # TODO (pablo): Add MPS acceleration

    if not torch.cuda.is_available():
        LOG.info("No GPU detected, auto-device is `cpu`")
        return torch.device("cpu")

    nb_devices = torch.cuda.device_count()
    if nb_devices == 1:
        LOG.info("One GPU detected, auto-device is `cuda`")
        return torch.device("cuda")

    LOG.info(f"Detected {nb_devices} GPUs. Allocating based on utilization.")
    device_utilization = [
        (torch.cuda.memory_allocated(device_id), device_id)
        for device_id in range(nb_devices)
    ]

    for mem_util, device in device_utilization:
        LOG.info(f"    --> device(cuda:{device}) = {mem_util} Bytes")

    _, best_device_id = min(device_utilization)

    return torch.device(f"cuda:{best_device_id}")
