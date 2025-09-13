from dataclasses import dataclass

@dataclass
class DescriptorParams:
    r_o: float    # Outer radial cutoff 
    r_i: float    # Inner radial cutoff
    n_rad: int    # Max radial degree to use 
    n_l: int      # Max angular degree to use
    gamma: float  # Dampening parameter
