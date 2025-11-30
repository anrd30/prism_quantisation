import torch
from torchvision.models import mobilenet_v2

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = mobilenet_v2(weights='DEFAULT').eval().to(device)
dummy_input = torch.randn((10, 3, 224, 224), device=device)

from aimet_common.defs import QuantScheme
from aimet_common.quantsim_config.utils import get_path_for_per_channel_config
from aimet_torch import QuantizationSimModel

sim = QuantizationSimModel(model, 
                           dummy_input,
                           default_param_bw=8,
                           default_output_bw=16)
print(sim)

def forward_pass(model):
    with torch.no_grad():
        model(torch.randn((10, 3, 224, 224), device=device))

sim.compute_encodings(forward_pass)

output = sim.model(dummy_input)
print(output)