import numpy as np
import os
p1_0 = np.array(
    [-0.07211538,0.18988171,0.684  ]
    # [0.,0., 0.02]
)
p1_1 = np.array(
    [ 0.24798712, 0.18904889,  0.681   ]
    # [0.3,0., 0.02]
)
transformation_matrix = np.array([
    [0.34100623, 0.83855584, 0.42489865, -0.34668014],
    [-0.09370125, -0.41941968, 0.90294364, -0.53850522],
    [0.93537951, -0.34772294, -0.06445099, 0.04761582],
    [0., 0., 0., 1.]
])

p1_0_transformed = np.dot(transformation_matrix,p1_0)
p1_1_transformed = np.dot(transformation_matrix,p1_1)
print(p1_0_transformed)
print(p1_1_transformed)