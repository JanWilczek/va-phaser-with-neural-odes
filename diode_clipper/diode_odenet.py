import torch.optim as optim
from torchdiffeq import odeint_adjoint
from models import ODENetDerivative


def main():
    ode_rhs = ODENetDerivative()
    optimizer = optim.Adam(ode_rhs.parameters(), lr=0.001)

if __name__ == '__main__':
    main()
