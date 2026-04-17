"""Risk layer — trader's equation.

Computes edge = P × reward - (1-P) × risk. Priors are empirical,
updated nightly from realized outcomes in the priors store.
"""
