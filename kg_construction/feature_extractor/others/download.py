import torch

model = torch.hub.load('harritaylor/torchvggish', 'vggish')
traced_graph = torch.jit.trace(model, torch.randn(1, 3, H, W))
traced_graph.save('vggish.pth')
