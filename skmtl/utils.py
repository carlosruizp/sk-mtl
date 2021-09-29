'''
Module that implements some useful functions for MTL with kernel methods.
'''

def mtl_kernel(X_1, X_2, mtl_type, ckernel, skernel, cgamma, sgamma, **kwargs):

    # Check parameters

    # Cases
    if 'joint' == mtl_type:
        return mtl_kernel_joint(X_1, X_2)
    elif 'laplacian' == mtl_type:
        return mtl_kernel_laplacian(X_1, X_2)
    else:
        raise ValueError(
            '{} is not a valid mtl_type'.format(mtl_type))

def mtl_kernel_joint(X_1, X_2, ckernel, skernel, cgamma, sgamma):
