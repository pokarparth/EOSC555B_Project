##Implementation for prism parameterization. 
##Input is model parameters (m), mesh spacing and number of prisms (params), and mesh XYZ.
##Use as an input for Simpeg's pytorch mapping.

def torch_transform(m, params, xyz):
    xyz = xyz
    n_shape = params[1]
    mesh_spacing = params[0]
    p_0 = 0
    H_sum,p_sum = 0,0
    num_params = 10
    X = torch.tensor(xyz[:, 0])
    Y = torch.tensor(xyz[:, 1])
    Z = torch.tensor(xyz[:, 2])

    xyz = torch.vstack((X, Y, Z))


    for i in range(n_shape):
        hx, hy, hz, phix, phiy, phiz, x_0, y_0, z_0, p_1 = m[i * num_params:(i + 1) * num_params]

        xyz_0 = torch.vstack((x_0, y_0, z_0))

        Rx = torch.zeros((3, 3), dtype=torch.float64)

        Rx[0, 0] = 1
        Rx[1, 1] = torch.cos(phix)
        Rx[1, 2] = -torch.sin(phix)
        Rx[2, 2] = torch.cos(phix)
        Rx[2, 1] = torch.sin(phix)

        Ry = torch.zeros_like(Rx)
        Ry[1, 1] = 1
        Ry[0, 0] = torch.cos(phiy)
        Ry[2, 0] = -torch.sin(phiy)
        Ry[2, 2] = torch.cos(phiy)
        Ry[0, 2] = torch.sin(phiy)

        Rz = torch.zeros_like(Rx)
        Rz[2, 2] = 1
        Rz[0, 0] = torch.cos(phiz)
        Rz[0, 1] = -torch.sin(phiz)
        Rz[1, 1] = torch.cos(phiz)
        Rz[1, 0] = torch.sin(phiz)

        M = Rx @ Ry @ Rz

        xyz_m_xyz_0 = xyz - xyz_0
        tau = xyz_m_xyz_0.T @ M

        lx = mesh_spacing*2
        ly = mesh_spacing*2
        lz = mesh_spacing*2

        sigma_x = 1 / ((1 + torch.exp(-(tau[:,0] + hx) / lx))) - 1 / ((1 + torch.exp(-(tau[:,0] - hx) / lx)))
        sigma_y = 1 / ((1 + torch.exp(-(tau[:,1] + hy) / ly))) - 1 / ((1 + torch.exp(-(tau[:,1] - hy) / ly)))
        sigma_z = 1 / ((1 + torch.exp(-(tau[:,2] + hz) / lz))) - 1 / ((1 + torch.exp(-(tau[:,2] - hz) / lz)))

        H = sigma_x * sigma_y * sigma_z
        #plt.plot(H,'.')
        #plt.show()

        p = (H) * (p_1 - p_0)

        p_sum = p_sum + p
        H_sum = H_sum + H


    w = torch.minimum(torch.tensor(1.0, dtype=H_sum.dtype), 1 / H_sum)
    p = p_0 +  w*p_sum
    #plt.plot(p,'.')
    #plt.show()
    return p