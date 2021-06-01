# Print network parameters' shapes
print([param.shape for param in self.network.parameters()])

# Ensure that network parameters are being updated
before = torch.cat([param.clone().detach().flatten() for param in self.network.parameters()])
# forward, loss, backward, step
after = torch.cat([param.clone().detach().flatten() for param in self.network.parameters()])
assert torch.nonzero(after - before).shape[0] == before.shape[0]
