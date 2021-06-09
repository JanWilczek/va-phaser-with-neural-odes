# Print network parameters' shapes
print([param.shape for param in self.network.parameters()])

# Ensure that network parameters are being updated
before = torch.cat([param.clone().detach().flatten() for param in self.network.parameters()])
# forward, loss, backward, step
after = torch.cat([param.clone().detach().flatten() for param in self.network.parameters()])
assert torch.nonzero(after - before).shape[0] == before.shape[0]

# Print currently running diode_clipper scripts of the user
ps -x | grep diode_clipper

# Run profiler on a script and output to a file
python -m cProfile -o diode_ode_numerical_profile.bin diode_clipper\diode_ode_numerical.py -u 38 -l 1 -s 5 -i 0 -m forward_euler -f 22050
