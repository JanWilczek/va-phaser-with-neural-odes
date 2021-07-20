# Print network parameters' shapes

```python
print([param.shape for param in self.network.parameters()])
```

## Ensure that network parameters are being updated

```python
before = torch.cat([param.clone().detach().flatten() for param in self.network.parameters()])
# forward, loss, backward, step
after = torch.cat([param.clone().detach().flatten() for param in self.network.parameters()])
assert torch.nonzero(after - before).shape[0] == before.shape[0]
```

## Print currently running diode_clipper scripts of the user

```bash
ps -x | grep diode_clipper
```

## Run profiler on a script and output to a file

```bash
python -m cProfile -o diode_ode_numerical_profile.bin diode_clipper\diode_ode_numerical.py -u 38 -l 1 -s 5 -i 0 -m forward_euler -f 22050
```

## Count the number of parameters in a model

```python
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
```