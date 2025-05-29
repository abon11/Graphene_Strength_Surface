import copulas
from copulas.datasets import sample_trivariate_xyz
from copulas.multivariate import GaussianMultivariate
from copulas.visualization import compare_3d

print(copulas.__version__)

real_data = sample_trivariate_xyz()
# real_data.head()
print(real_data.head())



# copula = GaussianMultivariate()
# copula.fit(real_data)

# synthetic_data = copula.sample(len(real_data))



# compare_3d(real_data, synthetic_data)