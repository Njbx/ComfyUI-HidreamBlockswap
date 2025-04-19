import comfy.model_management
import gc
import torch
from comfy.patcher_extension import CallbacksMP
from comfy.model_patcher import ModelPatcher
from comfy.model_base import HiDream
from tqdm import tqdm

#Based on https://github.com/kijai/ComfyUI-WanVideoWrapper
class HidreamImgBlockSwap:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "double_blocks_to_swap": ("INT", {"default": 10, "min": 0, "max": 15, "step": 1, "tooltip": "Number of double blocks to swap"}),
                "single_blocks_to_swap": ("INT", {"default": 0, "min": 0, "max": 31, "step": 1, "tooltip": "Number of single blocks to swap"}),
                "offload_txt_in": ("BOOLEAN", {"default": False, "tooltip": "Offload txt_in layer"}),
                "offload_img_in": ("BOOLEAN", {"default": False, "tooltip": "Offload img_in layer"}),
            },
        }
    RETURN_TYPES = ("MODEL",)
    CATEGORY = "ComfyUI-HidreamBlockswap"
    FUNCTION = "set_callback"

    def set_callback(self, model: ModelPatcher, double_blocks_to_swap, single_blocks_to_swap, offload_txt_in, offload_img_in):
        
        def swap_blocks(model: ModelPatcher, device_to, lowvram_model_memory, force_patch_weights, full_load):
            base_model = model.model
            if isinstance(base_model, HiDream):
                unet = base_model.diffusion_model
                for b, block in enumerate(unet.double_stream_blocks):
                    if b < double_blocks_to_swap:
                        block.to(model.offload_device)

                for b, block in enumerate(unet.single_stream_blocks):
                    if b < single_blocks_to_swap:
                        block.to(model.offload_device)
                        
                if offload_txt_in:
                    unet.txt_in.to(model.offload_device)
                if offload_img_in:
                    unet.img_in.to(model.offload_device)

            comfy.model_management.soft_empty_cache()
            gc.collect()
        
        model = model.clone()
        model.add_callback(CallbacksMP.ON_LOAD,swap_blocks)

        return (model, )


NODE_CLASS_MAPPINGS = {
    "HidreamBlockSwap": HidreamImgBlockSwap
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HidreamBlockSwap": "HidreamImgBlockSwap"
}