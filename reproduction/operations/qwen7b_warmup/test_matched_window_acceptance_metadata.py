"""Regression for the real16-byte CUDA RNG state found in HPC3 checkpoints."""
import io,unittest
import torch
from .accept_matched_window_checkpoint import serialized_cuda_rng_bytes
class CUDAStateMetadata(unittest.TestCase):
 def test_16_byte_checkpoint_roundtrip_is_accepted(self):
  stream=io.BytesIO();torch.save(torch.arange(16,dtype=torch.uint8),stream);stream.seek(0)
  state=torch.load(stream,map_location='cpu',weights_only=True)
  self.assertEqual(serialized_cuda_rng_bytes(state,torch),16)
 def test_truncated_or_wrong_dtype_or_shape_is_rejected(self):
  for state in (torch.zeros(15,dtype=torch.uint8),torch.zeros(16),torch.zeros((2,8),dtype=torch.uint8)):
   with self.assertRaises(ValueError):serialized_cuda_rng_bytes(state,torch)
if __name__=='__main__':unittest.main()
